"""Parameter-shared actor + (optionally centralized) critic for MAPPO.

Actor: takes per-camera obs → continuous action (mean + state-independent log_std).
Critic: two modes,
  - "decentralized": per-camera value = head(self_latent) — equivalent to IPPO.
  - "centralized": joint value V(s) computed by max-pooling all camera latents
    in a single env, then head. Broadcast to every camera.

The centralized critic preserves agent grouping by reshaping the leading B
dimension into (num_envs, num_cameras) at value time. The actor never needs to
know about the grouping — it is purely per-agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from mate_marl.nets.encoder import EntitySetEncoder, EntitySetEncoderConfig


@dataclass
class MARLActorCriticConfig:
    encoder: EntitySetEncoderConfig
    action_dim: int = 2
    log_std_init: float = -0.5
    centralized_critic: bool = True
    head_hidden: int = 128


class MARLActorCritic(nn.Module):
    def __init__(
        self,
        num_cameras: int,
        config: MARLActorCriticConfig,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.config = config

        # Allow callers (training script for ablations) to substitute MLP /
        # DeepSet encoders; default to the EntitySetEncoder + MoE flagship.
        self.encoder = encoder if encoder is not None else EntitySetEncoder(config.encoder)

        E = getattr(self.encoder, "embed_dim", None) or config.encoder.embed_dim
        H = config.head_hidden

        self.actor_head = nn.Sequential(
            nn.Linear(E, H),
            nn.GELU(),
            nn.Linear(H, config.action_dim),
        )
        self.log_std = nn.Parameter(
            torch.full((config.action_dim,), float(config.log_std_init))
        )

        # Critic head: input is either E (per-agent latent) or 2*E
        # (per-agent latent || pooled-team latent).
        critic_in = 2 * E if config.centralized_critic else E
        self.critic_head = nn.Sequential(
            nn.Linear(critic_in, H),
            nn.GELU(),
            nn.Linear(H, 1),
        )

    # ---- Actor ----

    def forward_actor(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        latent, _, _ = self.encoder(obs)
        action_mean = self.actor_head(latent)
        # Apply tanh to bound mean to [-1, 1]; PPO will clip later.
        return torch.tanh(action_mean)

    # ---- Critic ----

    def forward_critic(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        latent, _, _ = self.encoder(obs)
        if self.config.centralized_critic:
            # Reshape (B, E) → (num_envs, num_cameras, E).
            B, E = latent.shape
            if B % self.num_cameras != 0:
                raise RuntimeError(
                    f"Critic centralized expects B divisible by num_cameras="
                    f"{self.num_cameras}; got B={B}"
                )
            grouped = latent.view(B // self.num_cameras, self.num_cameras, E)
            team_pool = grouped.mean(dim=1, keepdim=True).expand_as(grouped)
            joint = torch.cat([grouped, team_pool], dim=-1).view(B, 2 * E)
            return self.critic_head(joint).squeeze(-1)
        return self.critic_head(latent).squeeze(-1)

    # ---- Combined ----

    def forward(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent, _, _ = self.encoder(obs)
        action_mean = torch.tanh(self.actor_head(latent))
        if self.config.centralized_critic:
            B, E = latent.shape
            grouped = latent.view(B // self.num_cameras, self.num_cameras, E)
            team_pool = grouped.mean(dim=1, keepdim=True).expand_as(grouped)
            joint = torch.cat([grouped, team_pool], dim=-1).view(B, 2 * E)
            value = self.critic_head(joint).squeeze(-1)
        else:
            value = self.critic_head(latent).squeeze(-1)
        return action_mean, value
