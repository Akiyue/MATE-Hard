"""Per-agent rollout buffer for MAPPO with dict observations.

Storage layout:

  Each step k of the rollout adds one entry per parallel env per agent.
  Observations are stored as a dict whose values have shape
      (T, num_envs, num_cameras, *feature)
  Actions / rewards / values / log_probs / dones are stored similarly with
  shape (T, num_envs, num_cameras, *).

GAE-Lambda is computed per-agent independently. Since the team reward is
broadcast equally to all agents, the per-agent advantage stream is identical
across cameras of the same env at every step — but per-agent value baselines
differ when the critic is decentralized, so the resulting advantages are
per-agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


@dataclass
class RolloutSample:
    obs: dict[str, torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class MARLRolloutBuffer:
    def __init__(
        self,
        n_steps: int,
        num_envs: int,
        num_cameras: int,
        obs_spec: dict[str, tuple[tuple[int, ...], np.dtype]],
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str | torch.device = "cpu",
    ) -> None:
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.num_cameras = num_cameras
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)

        self.obs: dict[str, np.ndarray] = {}
        for name, (shape, dtype) in obs_spec.items():
            # shape from env is (num_cameras, ...) — drop leading num_cameras
            # since we store it explicitly.
            assert shape[0] == num_cameras, (
                f"obs[{name}] expected leading dim {num_cameras}, got {shape}"
            )
            self.obs[name] = np.zeros(
                (n_steps, num_envs, num_cameras, *shape[1:]), dtype=dtype
            )

        self.actions = np.zeros(
            (n_steps, num_envs, num_cameras, action_dim), dtype=np.float32
        )
        self.log_probs = np.zeros((n_steps, num_envs, num_cameras), dtype=np.float32)
        self.values = np.zeros((n_steps, num_envs, num_cameras), dtype=np.float32)
        self.rewards = np.zeros((n_steps, num_envs, num_cameras), dtype=np.float32)
        self.dones = np.zeros((n_steps, num_envs, num_cameras), dtype=np.float32)

        self.advantages = np.zeros_like(self.rewards)
        self.returns = np.zeros_like(self.rewards)

        self.pos = 0

    def reset(self) -> None:
        self.pos = 0

    def add(
        self,
        obs: dict[str, np.ndarray],
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        for name, arr in obs.items():
            # arr shape: (num_envs, num_cameras, ...)
            self.obs[name][self.pos] = arr
        self.actions[self.pos] = actions
        self.log_probs[self.pos] = log_probs
        self.values[self.pos] = values
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.pos += 1

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray) -> None:
        """last_values, last_dones: (num_envs, num_cameras)."""
        adv = np.zeros((self.num_envs, self.num_cameras), dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = (
                self.rewards[t]
                + self.gamma * next_values * next_non_terminal
                - self.values[t]
            )
            adv = delta + self.gamma * self.gae_lambda * next_non_terminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def iter_minibatches(self, batch_size: int) -> Iterator[RolloutSample]:
        # Flatten (T * num_envs * num_cameras) into a single batch dim.
        N = self.n_steps * self.num_envs * self.num_cameras
        idx = np.random.permutation(N)
        for start in range(0, N, batch_size):
            sel = idx[start : start + batch_size]
            obs_b = {
                name: torch.as_tensor(
                    arr.reshape((N,) + arr.shape[3:])[sel], device=self.device
                )
                for name, arr in self.obs.items()
            }
            yield RolloutSample(
                obs=obs_b,
                actions=torch.as_tensor(
                    self.actions.reshape(N, -1)[sel], device=self.device
                ),
                log_probs=torch.as_tensor(
                    self.log_probs.reshape(N)[sel], device=self.device
                ),
                values=torch.as_tensor(
                    self.values.reshape(N)[sel], device=self.device
                ),
                advantages=torch.as_tensor(
                    self.advantages.reshape(N)[sel], device=self.device
                ),
                returns=torch.as_tensor(
                    self.returns.reshape(N)[sel], device=self.device
                ),
            )
