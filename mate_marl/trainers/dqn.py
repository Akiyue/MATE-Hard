"""DQN-MARL trainer with parameter sharing across cameras.

Off-policy alternative to MAPPO. The key advantages over PPO on this env:
  - Experience replay decorrelates samples and reuses informative transitions
    many times — important when episode-end signal is sparse.
  - Off-policy updates don't suffer from the "iter-1 big update, then
    nothing" plateau we observed with PPO.

Architecture:
  - Param-shared Q network: encoder (shared with PPO baseline encoders) +
    linear head with ``n_actions`` outputs.
  - Standard DQN: Bellman target = r + γ * (1 - done) * max_a' Q_target(s', a').
  - Soft Polyak target update (``tau``).
  - ε-greedy linearly annealed from ``eps_start`` to ``eps_end`` over
    ``eps_decay_steps``.
  - Dict-observation replay buffer: stores per-camera obs as flat per-key
    arrays.

Same env stack as MAPPO except the outermost wrapper is
``DiscreteFlattenForDQN`` (not ``FlattenAgentsForPPO``), and
``DiscreteCamera(levels=5)`` is applied at the base-env level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from common_net.moe import MoE
from mate_marl.nets.encoder import EntitySetEncoder, EntitySetEncoderConfig


@dataclass
class DQNConfig:
    num_envs: int = 8
    n_actions: int = 25  # levels * levels for DiscreteCamera

    buffer_size: int = 100_000
    batch_size: int = 256
    learning_starts: int = 10_000
    train_freq: int = 4         # gradient step every N env steps
    target_update_tau: float = 5e-3  # Polyak

    gamma: float = 0.99
    lr: float = 3e-4
    max_grad_norm: float = 10.0

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 200_000

    # Double DQN: use online net to pick the argmax-action in the next state,
    # then evaluate that action with the target net. Reduces the maximization
    # bias of vanilla DQN, which is exactly what made our v2/v3 runs converge
    # to a sub-optimal greedy policy.
    double_dqn: bool = True

    seed: Optional[int] = None
    device: str = "cpu"

    # Encoder.
    encoder: EntitySetEncoderConfig = field(default_factory=EntitySetEncoderConfig)
    head_hidden: int = 128


class _SyncDiscreteVecEnv:
    """Synchronous vec env for DQN. Each step returns:
        obs : dict of (num_envs, num_cameras, ...)
        rewards : (num_envs, num_cameras)
        term, trunc : (num_envs, num_cameras)
        info : list of length num_envs
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]]) -> None:
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].single_action_space
        self.num_cameras = self.envs[0].unwrapped.num_cameras

    def reset(self, seed: Optional[int] = None):
        obs_list, info_list = [], []
        for i, e in enumerate(self.envs):
            s = None if seed is None else seed + i
            o, info = e.reset(seed=s)
            obs_list.append(o)
            info_list.append(info)
        return self._stack(obs_list), info_list

    def step(self, actions: np.ndarray):
        # actions: (num_envs, num_cameras) int
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
        for i, e in enumerate(self.envs):
            o, r, term, trunc, info = e.step(actions[i])
            if term.any() or trunc.any():
                o, _ = e.reset()
            obs_list.append(o)
            rew_list.append(r)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)
        return (
            self._stack(obs_list),
            np.stack(rew_list, axis=0),
            np.stack(term_list, axis=0),
            np.stack(trunc_list, axis=0),
            info_list,
        )

    def _stack(self, obs_list):
        keys = obs_list[0].keys()
        return {k: np.stack([o[k] for o in obs_list], axis=0) for k in keys}

    def close(self):
        for e in self.envs:
            try:
                e.close()
            except Exception:
                pass


class _ReplayBuffer:
    """Dict-observation per-camera replay buffer. The leading dim of the
    obs values is num_cameras (parameter-shared policy uses it as batch)."""

    def __init__(
        self,
        capacity: int,
        obs_spec: dict[str, tuple[tuple[int, ...], np.dtype]],
        num_cameras: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.capacity = capacity
        self.num_cameras = num_cameras
        self.device = torch.device(device)
        self.pos = 0
        self.full = False

        # All buffers store per-CAMERA samples (flatten env dim).
        self.obs: dict[str, np.ndarray] = {}
        self.next_obs: dict[str, np.ndarray] = {}
        for name, (shape, dtype) in obs_spec.items():
            assert shape[0] == num_cameras, name
            shape_per_camera = shape[1:]
            self.obs[name] = np.zeros((capacity, *shape_per_camera), dtype=dtype)
            self.next_obs[name] = np.zeros((capacity, *shape_per_camera), dtype=dtype)

        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add_batch(
        self,
        obs: dict[str, np.ndarray],     # (num_envs, num_cameras, ...)
        actions: np.ndarray,            # (num_envs, num_cameras)
        rewards: np.ndarray,            # (num_envs, num_cameras)
        next_obs: dict[str, np.ndarray],
        dones: np.ndarray,              # (num_envs, num_cameras)
    ) -> None:
        # Flatten (num_envs, num_cameras) into a single batch dim.
        n = actions.size
        actions = actions.reshape(-1)
        rewards = rewards.reshape(-1).astype(np.float32)
        dones = dones.reshape(-1).astype(np.float32)

        for i in range(n):
            idx = (self.pos + i) % self.capacity
            for k in self.obs.keys():
                # obs[k] has shape (num_envs, num_cameras, ...) — flatten leading.
                v = obs[k].reshape((-1,) + obs[k].shape[2:])[i]
                self.obs[k][idx] = v
                v2 = next_obs[k].reshape((-1,) + next_obs[k].shape[2:])[i]
                self.next_obs[k][idx] = v2
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.dones[idx] = dones[i]

        self.pos = (self.pos + n) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        n = len(self)
        idx = np.random.randint(0, n, size=batch_size)
        d = {
            "actions": torch.as_tensor(self.actions[idx], device=self.device, dtype=torch.long),
            "rewards": torch.as_tensor(self.rewards[idx], device=self.device, dtype=torch.float32),
            "dones": torch.as_tensor(self.dones[idx], device=self.device, dtype=torch.float32),
        }
        d["obs"] = {k: torch.as_tensor(arr[idx], device=self.device) for k, arr in self.obs.items()}
        d["next_obs"] = {k: torch.as_tensor(arr[idx], device=self.device) for k, arr in self.next_obs.items()}
        return d


class _QNet(nn.Module):
    """Shared encoder + linear Q-head. Outputs (B, n_actions) Q-values."""

    def __init__(self, encoder: nn.Module, n_actions: int, head_hidden: int = 128) -> None:
        super().__init__()
        self.encoder = encoder
        E = getattr(encoder, "embed_dim", None)
        if E is None:
            raise RuntimeError("encoder must expose embed_dim")
        self.head = nn.Sequential(
            nn.Linear(E, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, n_actions),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        latent, _, _ = self.encoder(obs)
        return self.head(latent)


class DQN:
    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        num_cameras: int,
        n_actions: int,
        config: Optional[DQNConfig] = None,
        encoder: Optional[nn.Module] = None,
    ) -> None:
        self.config = config if config is not None else DQNConfig(n_actions=n_actions)
        self.config.n_actions = n_actions
        cfg = self.config

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)

        self.device = torch.device(cfg.device)
        self.num_cameras = num_cameras

        self.envs = _SyncDiscreteVecEnv([env_factory for _ in range(cfg.num_envs)])

        if encoder is None:
            encoder = EntitySetEncoder(cfg.encoder)
        target_encoder = type(encoder)(cfg.encoder) if hasattr(encoder, "config") and isinstance(encoder, EntitySetEncoder) else None
        if target_encoder is None:
            # Just instantiate a fresh copy for the target net by matching constructor signature.
            target_encoder = EntitySetEncoder(cfg.encoder)
        self.q_net = _QNet(encoder, n_actions=n_actions, head_hidden=cfg.head_hidden).to(self.device)
        self.q_target = _QNet(target_encoder, n_actions=n_actions, head_hidden=cfg.head_hidden).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()  # ensure MoE doesn't store gate logits on target net
        for p in self.q_target.parameters():
            p.requires_grad_(False)

        # Collect MoE layers for periodic gate-logit cleanup. Without this the
        # MoE._gate_logits list grows unboundedly during training (see
        # common_net/moe.py:_store_gate_logits) and we OOM after ~700k steps.
        self._moe_layers: list[MoE] = [
            m for m in self.q_net.modules() if isinstance(m, MoE)
        ] + [
            m for m in self.q_target.modules() if isinstance(m, MoE)
        ]

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)

        # Replay buffer.
        obs_spec: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
        for name, sp in self.envs.single_observation_space.spaces.items():
            obs_spec[name] = (sp.shape, sp.dtype)
        self.buffer = _ReplayBuffer(
            capacity=cfg.buffer_size,
            obs_spec=obs_spec,
            num_cameras=num_cameras,
            device=self.device,
        )

        # Episodic stats.
        self._ep_returns = np.zeros(cfg.num_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(cfg.num_envs, dtype=np.int64)
        self.episodic_stats: list[dict[str, float]] = []

        self.timesteps = 0  # COUNTED IN env interactions (not including agent batching)
        self.iterations = 0
        self.last_obs: dict[str, np.ndarray] | None = None

    # ---- helpers ----

    def _epsilon(self) -> float:
        cfg = self.config
        frac = min(self.timesteps / max(cfg.eps_decay_steps, 1), 1.0)
        return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac

    def _to_tensor_obs(self, obs_np: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k, v in obs_np.items():
            t = torch.as_tensor(v, device=self.device)
            t = t.reshape((-1,) + t.shape[2:])
            out[k] = t
        return out

    def _select_action(self, obs_np: dict[str, np.ndarray]) -> np.ndarray:
        """ε-greedy action selection. Returns (num_envs, num_cameras) int."""
        cfg = self.config
        eps = self._epsilon()
        if np.random.random() < eps:
            return np.random.randint(
                0, cfg.n_actions, size=(cfg.num_envs, self.num_cameras), dtype=np.int64
            )
        # Greedy.
        obs_t = self._to_tensor_obs(obs_np)
        with torch.no_grad():
            q = self.q_net(obs_t)  # (num_envs * num_cameras, n_actions)
        a = torch.argmax(q, dim=-1).cpu().numpy()
        return a.reshape(cfg.num_envs, self.num_cameras)

    @torch.no_grad()
    def _soft_update(self) -> None:
        tau = self.config.target_update_tau
        for tgt, src in zip(self.q_target.parameters(), self.q_net.parameters()):
            tgt.data.mul_(1.0 - tau).add_(src.data, alpha=tau)

    # ---- main loop ----

    def fit(
        self,
        total_timesteps: int,
        log_interval: int = 1000,
        save_dir: Optional[str | Path] = None,
        save_every_steps: Optional[int] = 250_000,
        progress_bar: bool = True,
        tb_log_dir: Optional[str | Path] = None,
    ) -> None:
        cfg = self.config

        tb_writer = None
        if tb_log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_path = Path(tb_log_dir)
                tb_path.mkdir(parents=True, exist_ok=True)
                tb_writer = SummaryWriter(log_dir=str(tb_path))
                print(f"[tb] writing to {tb_path}", flush=True)
            except ImportError:
                pass

        if progress_bar:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=total_timesteps, initial=self.timesteps)
            except ImportError:
                pbar = None
        else:
            pbar = None

        # Reset.
        if self.last_obs is None:
            self.last_obs, _ = self.envs.reset(seed=cfg.seed)

        wall_start = datetime.now()
        last_loss = float("nan")
        last_q_mean = float("nan")

        env_step_count = 0  # raw env step counter (n env interactions)
        target_steps = total_timesteps  # interpret total_timesteps as raw env steps

        while env_step_count < target_steps:
            # 1. Act (vectorized across envs).
            actions = self._select_action(self.last_obs)  # (num_envs, num_cameras)
            next_obs, rewards, term, trunc, info = self.envs.step(actions)
            done = (term | trunc).astype(np.float32)

            # 2. Store in replay.
            self.buffer.add_batch(self.last_obs, actions, rewards, next_obs, done)

            # 3. Track episodic stats.
            self._ep_returns += rewards[:, 0].astype(np.float64)
            self._ep_lengths += 1
            ep_done = (term | trunc).any(axis=1)
            for i in np.flatnonzero(ep_done):
                self.episodic_stats.append({
                    "return": float(self._ep_returns[i]),
                    "length": int(self._ep_lengths[i]),
                })
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0

            self.last_obs = next_obs
            env_step_count += cfg.num_envs
            self.timesteps += cfg.num_envs

            # 4. Train when warm.
            if (
                len(self.buffer) >= max(cfg.learning_starts, cfg.batch_size)
                and env_step_count % cfg.train_freq == 0
            ):
                batch = self.buffer.sample(cfg.batch_size)
                q_pred = self.q_net(batch["obs"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    if cfg.double_dqn:
                        # Online net picks the action; target net evaluates it.
                        next_actions = self.q_net(batch["next_obs"]).argmax(dim=-1, keepdim=True)
                        q_next = self.q_target(batch["next_obs"]).gather(1, next_actions).squeeze(1)
                    else:
                        q_next = self.q_target(batch["next_obs"]).max(dim=-1).values
                    y = batch["rewards"] + cfg.gamma * (1.0 - batch["dones"]) * q_next
                loss = F.smooth_l1_loss(q_pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                if cfg.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.q_net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                self._soft_update()
                # Reset MoE gate-logit caches to prevent unbounded growth.
                for moe in self._moe_layers:
                    moe._reset_gate_logits()
                last_loss = float(loss.item())
                last_q_mean = float(q_pred.detach().mean().item())
                self.iterations += 1

            # 5. Log.
            if log_interval and env_step_count % log_interval == 0:
                recent = self.episodic_stats[-100:]
                mean_ret = float(np.mean([d["return"] for d in recent])) if recent else float("nan")
                mean_len = float(np.mean([d["length"] for d in recent])) if recent else float("nan")
                elapsed = (datetime.now() - wall_start).total_seconds()
                sps = env_step_count / max(elapsed, 1e-6)
                print(
                    f"[step {env_step_count:7d}] mean_ret_100={mean_ret:.3f} "
                    f"loss={last_loss:.4f} q_mean={last_q_mean:+.3f} "
                    f"eps={self._epsilon():.3f} buf={len(self.buffer):6d} "
                    f"sps={sps:.0f}",
                    flush=True,
                )
                if tb_writer is not None:
                    s = env_step_count
                    tb_writer.add_scalar("rollout/mean_return_100", mean_ret, s)
                    tb_writer.add_scalar("rollout/mean_length_100", mean_len, s)
                    tb_writer.add_scalar("rollout/num_episodes", len(self.episodic_stats), s)
                    tb_writer.add_scalar("loss/q", last_loss, s)
                    tb_writer.add_scalar("policy/q_mean", last_q_mean, s)
                    tb_writer.add_scalar("policy/epsilon", self._epsilon(), s)
                    tb_writer.add_scalar("time/sps", sps, s)
                    tb_writer.flush()

            # Periodic checkpoint so a long run produces usable artifacts
            # even if it OOMs / is killed mid-training.
            if (
                save_dir is not None
                and save_every_steps is not None
                and env_step_count > 0
                and env_step_count // save_every_steps
                != (env_step_count - cfg.num_envs) // save_every_steps
            ):
                try:
                    self.save(save_dir)
                except Exception as e:
                    print(f"[checkpoint] save failed: {e}", flush=True)

            if pbar is not None:
                pbar.update(cfg.num_envs)

        if pbar is not None:
            pbar.close()
        if save_dir is not None:
            self.save(save_dir)
        if tb_writer is not None:
            tb_writer.close()

    def save(self, save_dir: str | Path) -> Path:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = save_dir / f"dqn_{ts}.pt"
        torch.save({
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "timesteps": self.timesteps,
            "iterations": self.iterations,
        }, path)
        return path

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.timesteps = int(ckpt.get("timesteps", 0))
        self.iterations = int(ckpt.get("iterations", 0))
