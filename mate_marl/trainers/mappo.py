"""MAPPO (Multi-Agent PPO) trainer with parameter sharing + centralized critic.

Design choices:
  - Parameter-shared actor across all cameras (one set of weights).
  - Centralized critic V(s) takes per-agent latent + team-pooled latent
    (see MARLActorCritic.forward_critic).
  - Continuous Gaussian policy with state-independent log_std.
  - GAE-Lambda per agent.
  - PPO clipped objective + value loss + entropy bonus + MoE auxiliary loss.

The trainer takes an env_factory that builds ONE env (post-wrapping with
MultiCamera + HeterogeneousCameras + EnergyConstraint + DynamicFog +
MateMARLDictObs + FlattenAgentsForPPO). The trainer instantiates ``num_envs``
of these, optionally in subprocesses.

This is intentionally self-contained — does NOT inherit from gym_agent.PPO,
because the multi-agent + dict-obs + centralized-critic combination does not
fit cleanly into the gym_agent abstractions.
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

from common_net.moe import AllGateLoss, MoEGateLossManager

from mate_marl.nets.actor_critic import MARLActorCritic, MARLActorCriticConfig
from mate_marl.nets.encoder import EntitySetEncoderConfig
from mate_marl.trainers.rollout_buffer import MARLRolloutBuffer


@dataclass
class MAPPOConfig:
    num_envs: int = 4
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 4

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    moe_loss_coef: float = 0.01
    max_grad_norm: float = 0.5

    lr: float = 1e-4
    # PPO's standard target_kl heuristic; epoch terminates when
    # approx_kl > 1.5 * target_kl. None disables.
    target_kl: Optional[float] = 0.05
    normalize_advantage: bool = True
    # SB3-VecNormalize-style: divide rewards by a running estimate of the
    # discounted-return std. Stabilizes the value loss when the absolute
    # reward scale is large or shifts during training.
    normalize_reward: bool = True
    reward_norm_clip: float = 10.0

    seed: Optional[int] = None
    device: str = "cpu"

    # Network config.
    encoder: EntitySetEncoderConfig = field(default_factory=EntitySetEncoderConfig)
    centralized_critic: bool = True
    log_std_init: float = -0.5
    head_hidden: int = 128


class _RunningMeanStd:
    """SB3-style Welford running mean + variance over a 1-D scalar stream.
    Used to normalize the discounted-return signal before it becomes a value
    target."""

    def __init__(self, eps: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return
        bm = float(x.mean())
        bv = float(x.var())
        bc = float(x.size)
        delta = bm - self.mean
        tot = self.count + bc
        new_mean = self.mean + delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        m2 = m_a + m_b + (delta ** 2) * self.count * bc / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 1e-8)))


class _SyncVecEnv:
    """Tiny synchronous vector env. The official gym AsyncVectorEnv pickles
    every step, which is heavy for our nested wrapper stack — for the smoke
    test we keep things in-process. Swap for SubprocVecEnv when scaling up."""

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
        return self._stack_obs(obs_list), info_list

    def step(self, actions: np.ndarray):
        # actions: (num_envs, num_cameras, action_dim)
        obs_list, rew_list, term_list, trunc_list, info_list = [], [], [], [], []
        for i, e in enumerate(self.envs):
            o, r, term, trunc, info = e.step(actions[i])
            if term.any() or trunc.any():
                o2, info2 = e.reset()
                o = o2
                # we keep term/trunc as observed to mark episode boundary.
                info["reset_info"] = info2
            obs_list.append(o)
            rew_list.append(r)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)
        return (
            self._stack_obs(obs_list),
            np.stack(rew_list, axis=0),
            np.stack(term_list, axis=0),
            np.stack(trunc_list, axis=0),
            info_list,
        )

    def _stack_obs(self, obs_list):
        keys = obs_list[0].keys()
        return {k: np.stack([o[k] for o in obs_list], axis=0) for k in keys}

    def close(self):
        for e in self.envs:
            try:
                e.close()
            except Exception:
                pass


class MAPPO:
    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        num_cameras: int,
        action_dim: int,
        config: Optional[MAPPOConfig] = None,
        encoder: Optional[nn.Module] = None,
    ) -> None:
        self.config = config if config is not None else MAPPOConfig()
        cfg = self.config

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)

        self.device = torch.device(cfg.device)
        self.num_cameras = num_cameras

        self.envs = _SyncVecEnv([env_factory for _ in range(cfg.num_envs)])

        # Build network.
        ac_cfg = MARLActorCriticConfig(
            encoder=cfg.encoder,
            action_dim=action_dim,
            centralized_critic=cfg.centralized_critic,
            log_std_init=cfg.log_std_init,
            head_hidden=cfg.head_hidden,
        )
        self.policy = MARLActorCritic(
            num_cameras=num_cameras, config=ac_cfg, encoder=encoder
        ).to(self.device)
        self.moe_loss_manager = MoEGateLossManager(self.policy, criterion=AllGateLoss())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

        # Rollout buffer.
        obs_spec: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
        for name, sp in self.envs.single_observation_space.spaces.items():
            obs_spec[name] = (sp.shape, sp.dtype)

        self.buffer = MARLRolloutBuffer(
            n_steps=cfg.n_steps,
            num_envs=cfg.num_envs,
            num_cameras=num_cameras,
            obs_spec=obs_spec,
            action_dim=action_dim,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            device=self.device,
        )

        # Episodic stats.
        self._ep_returns = np.zeros(cfg.num_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(cfg.num_envs, dtype=np.int64)
        self.episodic_stats: list[dict[str, float]] = []

        # Running-mean-std reward normalization (SB3-VecNormalize-style):
        # tracks the std of the *discounted return* rather than per-step
        # reward. We maintain a running discounted return per env and feed
        # it into the RMS each step.
        self._rew_rms = _RunningMeanStd() if cfg.normalize_reward else None
        self._discounted_running = np.zeros(cfg.num_envs, dtype=np.float64)

        self.timesteps = 0
        self.iterations = 0
        self.last_obs: dict[str, np.ndarray] | None = None
        self.last_dones: np.ndarray | None = None  # (num_envs, num_cameras)

    # ---- helpers ----

    def _to_tensor_obs(self, obs_np: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """obs_np: dict of (num_envs, num_cameras, ...) → tensor (B, ...)
        where B = num_envs * num_cameras."""
        out: dict[str, torch.Tensor] = {}
        for k, v in obs_np.items():
            t = torch.as_tensor(v, device=self.device)
            t = t.reshape((-1,) + t.shape[2:])
            out[k] = t
        return out

    def _action_distribution(self, mean: torch.Tensor) -> torch.distributions.Normal:
        std = self.policy.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    # ---- rollout ----

    def collect_rollout(self) -> None:
        cfg = self.config
        if self.last_obs is None:
            obs_np, _ = self.envs.reset(seed=cfg.seed)
            self.last_obs = obs_np
            self.last_dones = np.zeros(
                (cfg.num_envs, self.num_cameras), dtype=np.float32
            )

        self.buffer.reset()
        self.policy.eval()

        for t in range(cfg.n_steps):
            obs_t = self._to_tensor_obs(self.last_obs)
            with torch.no_grad():
                mean, value = self.policy(obs_t)
                dist = self._action_distribution(mean)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)

            action_np = action.cpu().numpy().reshape(
                cfg.num_envs, self.num_cameras, -1
            )
            log_prob_np = log_prob.cpu().numpy().reshape(
                cfg.num_envs, self.num_cameras
            )
            value_np = value.cpu().numpy().reshape(cfg.num_envs, self.num_cameras)

            next_obs_np, rewards, term, trunc, info = self.envs.step(action_np)
            done = (term | trunc).astype(np.float32)

            # Normalize rewards by the running std of discounted returns,
            # SB3-VecNormalize style. This stabilizes the value loss when
            # the per-step reward magnitude is large or shifting.
            stored_rewards = rewards.astype(np.float32)
            if self._rew_rms is not None:
                # rewards shape: (num_envs, num_cameras). All cameras share the
                # team reward, so just take the per-env scalar (first agent).
                step_rew = rewards[:, 0].astype(np.float64)
                self._discounted_running = (
                    self._discounted_running * cfg.gamma + step_rew
                )
                self._rew_rms.update(self._discounted_running)
                # Reset the running discount when an episode ends.
                ep_done_mask = (term | trunc).any(axis=1)
                self._discounted_running[ep_done_mask] = 0.0
                # Divide by std (SB3 does NOT subtract mean — it preserves sign
                # of reward, which matters for sparse-reward correctness).
                clip = cfg.reward_norm_clip
                stored_rewards = np.clip(
                    rewards.astype(np.float64) / self._rew_rms.std,
                    -clip, clip,
                ).astype(np.float32)

            self.buffer.add(
                obs=self.last_obs,
                actions=action_np,
                log_probs=log_prob_np,
                values=value_np,
                rewards=stored_rewards,
                dones=self.last_dones,  # done flag from previous step
            )

            # Track episodic returns using the RAW reward (so logged
            # mean_ret_100 stays interpretable across runs with/without
            # normalization).
            self._ep_returns += rewards[:, 0]
            self._ep_lengths += 1
            ep_done = (term | trunc).any(axis=1)
            for i in np.flatnonzero(ep_done):
                self.episodic_stats.append(
                    {
                        "return": float(self._ep_returns[i]),
                        "length": int(self._ep_lengths[i]),
                    }
                )
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0

            self.last_obs = next_obs_np
            self.last_dones = done
            self.timesteps += cfg.num_envs * self.num_cameras

        # Bootstrap value for last state.
        with torch.no_grad():
            obs_t = self._to_tensor_obs(self.last_obs)
            _, last_value = self.policy(obs_t)
        last_value_np = last_value.cpu().numpy().reshape(
            cfg.num_envs, self.num_cameras
        )
        self.buffer.compute_gae(last_value_np, self.last_dones)

    # ---- update ----

    def update(self) -> dict[str, float]:
        cfg = self.config
        self.policy.train()

        epoch_logs: list[dict[str, float]] = []
        early_stop = False

        for _ in range(cfg.n_epochs):
            if early_stop:
                break
            for sample in self.buffer.iter_minibatches(cfg.batch_size):
                mean, value = self.policy(sample.obs)
                dist = self._action_distribution(mean)
                new_log_probs = dist.log_prob(sample.actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                advantages = sample.advantages
                if cfg.normalize_advantage and advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = torch.exp(new_log_probs - sample.log_probs)
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, sample.returns)

                # MoE aux loss.
                moe_loss = self.moe_loss_manager()
                if not torch.is_tensor(moe_loss):
                    moe_loss = torch.zeros((), device=self.device)

                loss = (
                    policy_loss
                    + cfg.vf_coef * value_loss
                    - cfg.entropy_coef * entropy
                    + cfg.moe_loss_coef * moe_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                if cfg.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - torch.log(ratio.clamp(min=1e-8))).mean()

                epoch_logs.append(
                    {
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy.item()),
                        "moe_loss": float(moe_loss.item() if torch.is_tensor(moe_loss) else 0.0),
                        "kl": float(approx_kl.item()),
                    }
                )

                if cfg.target_kl is not None and approx_kl > 1.5 * cfg.target_kl:
                    early_stop = True
                    break

        agg = {k: float(np.mean([d[k] for d in epoch_logs])) for k in epoch_logs[0]}
        return agg

    # ---- main loop ----

    def fit(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        save_dir: Optional[str | Path] = None,
        progress_bar: bool = True,
        tb_log_dir: Optional[str | Path] = None,
    ) -> None:
        cfg = self.config

        # ---- TensorBoard ----
        tb_writer = None
        if tb_log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_path = Path(tb_log_dir)
                tb_path.mkdir(parents=True, exist_ok=True)
                tb_writer = SummaryWriter(log_dir=str(tb_path))
                print(f"[tb] writing to {tb_path}", flush=True)
            except ImportError:
                print("[tb] tensorboard not installed; logs disabled", flush=True)

        if progress_bar:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=total_timesteps, initial=self.timesteps)
            except ImportError:
                pbar = None
        else:
            pbar = None

        wall_start = datetime.now()

        while self.timesteps < total_timesteps:
            self.collect_rollout()
            stats = self.update()
            self.iterations += 1

            recent = self.episodic_stats[-100:]
            mean_ret = float(np.mean([d["return"] for d in recent])) if recent else float("nan")
            mean_len = float(np.mean([d["length"] for d in recent])) if recent else float("nan")
            elapsed = (datetime.now() - wall_start).total_seconds()
            sps = self.timesteps / max(elapsed, 1e-6)

            if log_interval and self.iterations % log_interval == 0:
                print(
                    f"[iter {self.iterations:4d}] steps={self.timesteps} "
                    f"mean_ret_100={mean_ret:.3f} pi={stats['policy_loss']:+.4f} "
                    f"v={stats['value_loss']:+.4f} H={stats['entropy']:+.3f} "
                    f"moe={stats['moe_loss']:+.4f} kl={stats['kl']:+.4f} "
                    f"sps={sps:.0f}",
                    flush=True,
                )

            if tb_writer is not None:
                step = self.timesteps
                tb_writer.add_scalar("rollout/mean_return_100", mean_ret, step)
                tb_writer.add_scalar("rollout/mean_length_100", mean_len, step)
                tb_writer.add_scalar("rollout/num_episodes", len(self.episodic_stats), step)
                tb_writer.add_scalar("loss/policy", stats["policy_loss"], step)
                tb_writer.add_scalar("loss/value", stats["value_loss"], step)
                tb_writer.add_scalar("loss/moe", stats["moe_loss"], step)
                tb_writer.add_scalar("policy/entropy", stats["entropy"], step)
                tb_writer.add_scalar("policy/approx_kl", stats["kl"], step)
                tb_writer.add_scalar("policy/log_std_mean",
                                     float(self.policy.log_std.mean().item()), step)
                tb_writer.add_scalar("time/sps", sps, step)
                tb_writer.add_scalar("time/elapsed_seconds", elapsed, step)
                tb_writer.flush()

            # Periodic checkpoint so long runs produce usable artifacts
            # even if they crash / are killed mid-training.
            if (
                save_dir is not None
                and self.iterations > 0
                and self.iterations % 50 == 0
            ):
                try:
                    self.save(save_dir)
                except Exception as e:
                    print(f"[checkpoint] save failed: {e}", flush=True)

            if pbar is not None:
                pbar.update(cfg.num_envs * self.num_cameras * cfg.n_steps)

        if pbar is not None:
            pbar.close()

        if save_dir is not None:
            self.save(save_dir)

        if tb_writer is not None:
            tb_writer.close()

    # ---- io ----

    def save(self, save_dir: str | Path) -> Path:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = save_dir / f"mappo_{ts}.pt"
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "timesteps": self.timesteps,
                "iterations": self.iterations,
            },
            path,
        )
        return path

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.timesteps = int(ckpt.get("timesteps", 0))
        self.iterations = int(ckpt.get("iterations", 0))
