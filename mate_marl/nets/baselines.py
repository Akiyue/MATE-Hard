"""Baseline policies for the MATE-MARL ablation matrix.

  - MLPEncoder        : flatten the dict obs into a single vector → 2-3 layer MLP.
                        No permutation invariance, no attention. Floor for "did
                        the env learn at all?".
  - DeepSetEncoder    : per-token MLP + sum-pool over typed entities, then MLP.
                        Permutation-invariant but no attention. Sanity check
                        that attention >= deep-sets in our setting.

Both expose the same (latent, tokens, mask) tuple as EntitySetEncoder so
MARLActorCritic can use them interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from mate_marl.wrappers.heterogeneous_cameras import NUM_CAMERA_TYPES
from mate_marl.wrappers.dict_obs import (
    F_SELF,
    F_TEAMMATE,
    F_TARGET,
    F_OBSTACLE,
    F_FOG,
    F_PRESERVED,
)


@dataclass
class MLPEncoderConfig:
    embed_dim: int = 128
    hidden: tuple[int, ...] = (256, 256)
    # Token counts — encoder needs them at construction time so the flatten
    # dim is known before the optimizer is built. Defaults match
    # MATE-4v8-9 (4 cameras, 8 targets, 9 obstacles, 4 fogs).
    num_cameras: int = 4
    num_targets: int = 8
    num_obstacles: int = 9
    num_fogs: int = 4


def _mlp(dims, activation: type[nn.Module] = nn.GELU) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class MLPEncoder(nn.Module):
    """Flatten the dict obs into one vector → MLP → latent.

    No structure exploitation: each token is a flat slice of the input. Used
    as the lower-bound baseline.
    """

    def __init__(self, config: MLPEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else MLPEncoderConfig()
        c = self.config
        flat_dim = (
            F_SELF
            + (max(c.num_cameras - 1, 1)) * F_TEAMMATE
            + c.num_targets * F_TARGET
            + c.num_obstacles * F_OBSTACLE
            + max(c.num_fogs, 1) * F_FOG
            + F_PRESERVED
        )
        self._proj = _mlp([flat_dim, *c.hidden])
        self._head = _mlp(
            [c.hidden[-1] if c.hidden else c.embed_dim, c.embed_dim]
        )
        self._activations = nn.GELU()
        self.embed_dim = c.embed_dim

    def forward(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_dtype = next(self._proj.parameters()).dtype
        parts = []
        for k in ("self", "teammate", "target", "obstacle", "fog", "preserved"):
            t = obs[k].to(dtype=target_dtype)
            parts.append(t.reshape(t.shape[0], -1))
        x = torch.cat(parts, dim=-1)
        h = self._activations(self._proj(x))
        latent = self._head(h)
        # No tokens / no mask — return shape-compatible placeholders.
        tokens = latent.unsqueeze(1)
        mask = torch.ones(latent.shape[0], 1, device=latent.device, dtype=latent.dtype)
        return latent, tokens, mask


@dataclass
class DeepSetEncoderConfig:
    embed_dim: int = 128
    hidden: tuple[int, ...] = (128,)


class DeepSetEncoder(nn.Module):
    """Per-entity-type DeepSet pooling encoder. Permutation-invariant per type.

    Each token (self/teammate/target/obstacle/fog/preserved) is embedded by a
    type-specific MLP; tokens of variable-cardinality types are sum-pooled
    after multiplying by their validity mask. The pooled vectors are
    concatenated with the SELF embedding and the camera-type embedding, then
    passed through a final MLP to produce the latent.
    """

    def __init__(
        self,
        config: DeepSetEncoderConfig | None = None,
        num_camera_types: int = NUM_CAMERA_TYPES,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else DeepSetEncoderConfig()
        h = self.config.hidden
        E = self.config.embed_dim
        self.entity_dims = {
            "self": F_SELF,
            "teammate": F_TEAMMATE,
            "target": F_TARGET,
            "obstacle": F_OBSTACLE,
            "fog": F_FOG,
            "preserved": F_PRESERVED,
        }
        self.entity_proj = nn.ModuleDict(
            {k: _mlp([d, *h, E]) for k, d in self.entity_dims.items()}
        )
        self.camera_type_emb = nn.Embedding(num_camera_types, E)
        # 6 entity blocks + camera-type embedding.
        self.head = _mlp([E * 7, *h, E])
        self.embed_dim = E

    def _embed_one(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,F)
        x = x.to(dtype=next(self.entity_proj[name].parameters()).dtype)
        if name in ("teammate", "target", "obstacle", "fog"):
            mask = x[..., -1:]  # (B,K,1)
        else:
            mask = torch.ones_like(x[..., -1:])
        emb = self.entity_proj[name](x)  # (B,K,E)
        return (emb * mask).sum(dim=1)  # (B,E)

    def forward(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parts = []
        for k in ("self", "teammate", "target", "obstacle", "fog", "preserved"):
            parts.append(self._embed_one(k, obs[k]))
        type_emb = self.camera_type_emb(obs["self_type"].long())
        cat = torch.cat(parts + [type_emb], dim=-1)
        latent = self.head(cat)
        tokens = latent.unsqueeze(1)
        mask = torch.ones(latent.shape[0], 1, device=latent.device, dtype=latent.dtype)
        return latent, tokens, mask
