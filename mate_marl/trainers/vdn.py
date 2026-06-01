"""VDN trainer (Value Decomposition Networks, Sunehag et al. 2018).

VDN is the simplest cooperative-MARL value-mixing method: the joint Q is
the *sum* of per-agent Q-values,

    Q_tot(s, a_1..N) = sum_i Q_i(s_i, a_i),

with no hypernetwork. It satisfies IGM by construction and is monotone
trivially. As a baseline, VDN is the strictest "value decomposition without
mixing capacity" baseline — anything that doesn't beat VDN cannot claim a
real mixing-network contribution.

Implementation re-uses the TC-QMIX trainer infrastructure (replay buffer,
encoder, per-agent Q-head, target nets, MoE leak fix, periodic checkpoint).
The only change is the joint-Q computation: ``q_tot = sum_i q_i``.
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
from mate_marl.trainers.tcqmix import (
    _SyncVecEnv,
    _JointReplay,
    _QNet,
)


@dataclass
class VDNConfig:
    num_envs: int = 8
    num_agents: int = 4
    n_actions: int = 25
    num_types: int = 3  # unused for VDN but kept for API symmetry

    buffer_size: int = 50_000
    batch_size: int = 64
    learning_starts: int = 5_000
    train_freq: int = 4
    target_update_tau: float = 5e-3

    gamma: float = 0.99
    lr: float = 3e-4
    max_grad_norm: float = 10.0

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 400_000

    seed: Optional[int] = None
    device: str = "cpu"

    encoder: EntitySetEncoderConfig = field(default_factory=EntitySetEncoderConfig)
    head_hidden: int = 128


class VDN:
    """Parameter-shared VDN trainer. Same data path as TCQMIX, but no mixer."""

    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        num_cameras: int,
        n_actions: int,
        config: Optional[VDNConfig] = None,
        encoder: Optional[nn.Module] = None,
    ) -> None:
        self.config = config if config is not None else VDNConfig(
            num_agents=num_cameras, n_actions=n_actions
        )
        cfg = self.config
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)

        self.device = torch.device(cfg.device)
        self.num_cameras = num_cameras
        self.envs = _SyncVecEnv([env_factory for _ in range(cfg.num_envs)])

        if encoder is None:
            encoder = EntitySetEncoder(cfg.encoder)
        target_encoder = EntitySetEncoder(cfg.encoder)
        self.q_net = _QNet(encoder, n_actions, cfg.head_hidden).to(self.device)
        self.q_target = _QNet(target_encoder, n_actions, cfg.head_hidden).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()
        for p in self.q_target.parameters():
            p.requires_grad_(False)

        self._moe_layers: list[MoE] = (
            [m for m in self.q_net.modules() if isinstance(m, MoE)]
            + [m for m in self.q_target.modules() if isinstance(m, MoE)]
        )

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)

        obs_spec = {n: (sp.shape, sp.dtype)
                    for n, sp in self.envs.single_observation_space.spaces.items()}
        self.buffer = _JointReplay(
            capacity=cfg.buffer_size,
            obs_spec=obs_spec,
            num_cameras=num_cameras,
            device=self.device,
        )

        self._ep_returns = np.zeros(cfg.num_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(cfg.num_envs, dtype=np.int64)
        self.episodic_stats: list[dict[str, float]] = []
        self.timesteps = 0
        self.iterations = 0
        self.last_obs: dict[str, np.ndarray] | None = None

    # ---- helpers ----

    def _epsilon(self) -> float:
        cfg = self.config
        frac = min(self.timesteps / max(cfg.eps_decay_steps, 1), 1.0)
        return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac

    def _to_tensor_obs_flat(self, obs_np):
        out: dict[str, torch.Tensor] = {}
        for k, v in obs_np.items():
            t = torch.as_tensor(v, device=self.device)
            t = t.reshape((-1,) + t.shape[2:])
            out[k] = t
        return out

    def _select_action(self, obs_np):
        cfg = self.config
        eps = self._epsilon()
        if np.random.random() < eps:
            return np.random.randint(0, cfg.n_actions,
                                     size=(cfg.num_envs, self.num_cameras),
                                     dtype=np.int64)
        obs_t = self._to_tensor_obs_flat(obs_np)
        with torch.no_grad():
            q, _ = self.q_net(obs_t)
        a = torch.argmax(q, dim=-1).cpu().numpy().reshape(cfg.num_envs, self.num_cameras)
        return a

    @torch.no_grad()
    def _soft_update(self):
        tau = self.config.target_update_tau
        for tgt, src in zip(self.q_target.parameters(), self.q_net.parameters()):
            tgt.data.mul_(1.0 - tau).add_(src.data, alpha=tau)

    def _joint_q(self, batch_obs, use_target):
        B = batch_obs["self"].shape[0]
        N = self.num_cameras
        flat_obs = {k: v.reshape((B * N,) + v.shape[2:])
                    for k, v in batch_obs.items()
                    if v.dim() >= 2 and v.shape[1] == N}
        flat_obs["self_type"] = batch_obs["self_type"].reshape(B * N)
        net = self.q_target if use_target else self.q_net
        q_flat, _ = net(flat_obs)
        return q_flat.view(B, N, -1)

    # ---- main loop ----

    def fit(
        self,
        total_timesteps: int,
        log_interval: int = 5000,
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

        if self.last_obs is None:
            self.last_obs = self.envs.reset(seed=cfg.seed)

        wall_start = datetime.now()
        last_loss = float("nan")
        last_q_tot = float("nan")
        env_step_count = 0

        while env_step_count < total_timesteps:
            actions = self._select_action(self.last_obs)
            next_obs, rewards, term, trunc = self.envs.step(actions)
            done = (term | trunc).astype(np.float32)
            self.buffer.add_batch(self.last_obs, actions, rewards, next_obs, done)

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

            if (
                len(self.buffer) >= max(cfg.learning_starts, cfg.batch_size)
                and env_step_count % cfg.train_freq == 0
            ):
                batch = self.buffer.sample(cfg.batch_size)
                qs_now = self._joint_q(batch["obs"], use_target=False)
                q_chosen = qs_now.gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)
                # VDN joint Q: sum per-agent Q values.
                q_tot = q_chosen.sum(dim=-1)

                with torch.no_grad():
                    qs_next = self._joint_q(batch["next_obs"], use_target=True)
                    q_next_max = qs_next.max(dim=-1).values
                    q_tot_next = q_next_max.sum(dim=-1)
                    joint_r = batch["rewards"].mean(dim=-1)
                    joint_d = batch["dones"].max(dim=-1).values
                    y = joint_r + cfg.gamma * (1.0 - joint_d) * q_tot_next

                loss = F.smooth_l1_loss(q_tot, y)
                self.optimizer.zero_grad()
                loss.backward()
                if cfg.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.q_net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                self._soft_update()
                for moe in self._moe_layers:
                    moe._reset_gate_logits()
                last_loss = float(loss.item())
                last_q_tot = float(q_tot.detach().mean().item())
                self.iterations += 1

            if log_interval and env_step_count % log_interval == 0:
                recent = self.episodic_stats[-100:]
                mean_ret = float(np.mean([d["return"] for d in recent])) if recent else float("nan")
                mean_len = float(np.mean([d["length"] for d in recent])) if recent else float("nan")
                elapsed = (datetime.now() - wall_start).total_seconds()
                sps = env_step_count / max(elapsed, 1e-6)
                print(
                    f"[step {env_step_count:7d}] mean_ret_100={mean_ret:.3f} "
                    f"loss={last_loss:.4f} q_tot={last_q_tot:+.3f} "
                    f"eps={self._epsilon():.3f} buf={len(self.buffer):6d} sps={sps:.0f}",
                    flush=True,
                )
                if tb_writer is not None:
                    s = env_step_count
                    tb_writer.add_scalar("rollout/mean_return_100", mean_ret, s)
                    tb_writer.add_scalar("rollout/mean_length_100", mean_len, s)
                    tb_writer.add_scalar("rollout/num_episodes", len(self.episodic_stats), s)
                    tb_writer.add_scalar("loss/q", last_loss, s)
                    tb_writer.add_scalar("policy/q_tot_mean", last_q_tot, s)
                    tb_writer.add_scalar("policy/epsilon", self._epsilon(), s)
                    tb_writer.add_scalar("time/sps", sps, s)
                    tb_writer.flush()

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
                    print(f"[ckpt] save failed: {e}", flush=True)

            if pbar is not None:
                pbar.update(cfg.num_envs)

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
        path = save_dir / f"vdn_{ts}.pt"
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
