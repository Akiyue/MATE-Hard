"""Entity-Set Encoder with Type-Conditioned Mixture-of-Experts.

Given a dict observation of typed entity tokens (output of MateMARLDictObs),
this module:

  1. Embeds each entity-type into a shared embed_dim space with a per-type
     linear projection + per-type bias + per-type "entity-class" embedding.
  2. Multiplies each token by its validity mask (mask is the LAST feature
     column of teammate/target/obstacle/fog tokens; self/preserved/fog are
     always valid).
  3. Adds the camera-type embedding (broadcast across all tokens) so the
     MoE FF blocks downstream can route differently per camera type.
  4. Concatenates all tokens into one set, runs ``ChainBlock(MAB)`` self-
     attention with TopKMoE feed-forward, and pools the SELF token as the
     per-camera latent.

Output: a (B, embed_dim) latent for the actor + the full (B, N_total, embed_dim)
hidden state for an optional centralized critic.

The encoder is parameter-shared across agents — at training time the caller
flattens (num_envs * num_cameras) into the leading B dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from common_net.att_block import MAB, ChainBlock
from common_net.attentions.mha import MHAConfig
from common_net.moe import TopKMoE, TopKMoEConfig

from mate_marl.wrappers.heterogeneous_cameras import NUM_CAMERA_TYPES
from mate_marl.wrappers.dict_obs import (
    F_SELF,
    F_TEAMMATE,
    F_TARGET,
    F_OBSTACLE,
    F_FOG,
    F_PRESERVED,
)


ENTITY_TYPES = ("self", "teammate", "target", "obstacle", "fog", "preserved")
NUM_ENTITY_TYPES = len(ENTITY_TYPES)
ENTITY_FEATURE_DIM = {
    "self": F_SELF,
    "teammate": F_TEAMMATE,
    "target": F_TARGET,
    "obstacle": F_OBSTACLE,
    "fog": F_FOG,
    "preserved": F_PRESERVED,
}
# Whether the LAST feature column of each entity is a validity mask (1.0=valid).
ENTITY_HAS_MASK = {
    "self": False,
    "teammate": True,
    "target": True,
    "obstacle": True,
    "fog": True,
    "preserved": False,
}


@dataclass
class EntitySetEncoderConfig:
    embed_dim: int = 128
    d_ff: int = 256
    num_blocks: int = 2
    num_heads: int = 4
    num_experts: int = 4
    top_k: int = 2
    use_shared_expert: bool = True
    moe: bool = True
    type_conditioning: bool = True
    dropout: float = 0.0


class EntitySetEncoder(nn.Module):
    """Set-Transformer encoder over typed entity tokens.

    Args:
        config: encoder hyperparameters.
        num_camera_types: number of camera types for type-conditioning.
                          Defaults to NUM_CAMERA_TYPES.
    """

    def __init__(
        self,
        config: Optional[EntitySetEncoderConfig] = None,
        num_camera_types: int = NUM_CAMERA_TYPES,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else EntitySetEncoderConfig()
        cfg = self.config
        self.embed_dim = cfg.embed_dim

        # Per-entity-type projection.
        self.entity_proj = nn.ModuleDict(
            {
                name: nn.Linear(ENTITY_FEATURE_DIM[name], cfg.embed_dim)
                for name in ENTITY_TYPES
            }
        )
        # Per-entity-type embedding (added to the projected token).
        self.entity_type_emb = nn.Embedding(NUM_ENTITY_TYPES, cfg.embed_dim)

        # Camera-type embedding for type-conditioning.
        self.camera_type_emb = nn.Embedding(num_camera_types, cfg.embed_dim)

        # Set-Transformer body.
        mha_cfg = MHAConfig(num_heads=cfg.num_heads)
        if cfg.moe:
            moe_cfg = TopKMoEConfig(
                num_experts=cfg.num_experts,
                top_k=cfg.top_k,
                use_shared_expert=cfg.use_shared_expert,
                norm_topk_prob=True,
            )
            block_kwargs = dict(
                embed_dim=cfg.embed_dim,
                d_ff=cfg.d_ff,
                mha_config=mha_cfg,
                moe_cls=TopKMoE,
                moe_config=moe_cfg,
            )
        else:
            block_kwargs = dict(
                embed_dim=cfg.embed_dim,
                d_ff=cfg.d_ff,
                mha_config=mha_cfg,
            )

        self.body = ChainBlock(
            num_repeats=cfg.num_blocks,
            block_cls=MAB,
            **block_kwargs,
        )

        # Lookup table: entity name → entity type id.
        self.register_buffer(
            "_entity_type_ids",
            torch.tensor(
                [ENTITY_TYPES.index(n) for n in ENTITY_TYPES], dtype=torch.long
            ),
            persistent=False,
        )

    def _embed_entity(self, name: str, x: torch.Tensor, type_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed one entity type. Returns (tokens, mask) of shapes
        (B, K, E) and (B, K) where mask is 1.0 for valid tokens."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # treat as a single token

        # Match dtype to model parameters (Linear is typically float32).
        x = x.to(dtype=self.entity_proj[name].weight.dtype)

        # Extract mask (last feature column) if applicable.
        if ENTITY_HAS_MASK[name]:
            mask = x[..., -1]  # (B, K)
            features = x  # keep mask column; the projection learns to use it
        else:
            mask = torch.ones(x.shape[:2], dtype=x.dtype, device=x.device)
            features = x

        proj = self.entity_proj[name](features)  # (B, K, E)
        et_id = ENTITY_TYPES.index(name)
        proj = proj + self.entity_type_emb.weight[et_id].view(1, 1, -1)
        # Broadcast camera-type embedding to all tokens of this entity.
        proj = proj + type_emb.unsqueeze(1)
        # Zero-out masked tokens (they will not contribute to attention values
        # meaningfully). We don't mask the softmax because MHA in this repo
        # doesn't support a key_padding_mask — but zeroing the value rows is a
        # reasonable approximation given the LayerNorm at the block input.
        proj = proj * mask.unsqueeze(-1)
        return proj, mask

    def forward(
        self,
        obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a dict observation.

        Args:
            obs: keys 'self', 'teammate', 'target', 'obstacle', 'fog',
                 'preserved', 'self_type'. Leading dim B is num_envs *
                 num_cameras (parameter-sharing across agents).

        Returns:
            self_latent : (B, embed_dim) — pooled latent for the agent.
            tokens      : (B, N_total, embed_dim) — full hidden state.
            mask        : (B, N_total) — token validity mask.
        """
        self_type = obs["self_type"].long()
        type_emb = self.camera_type_emb(self_type)  # (B, E)
        if not self.config.type_conditioning:
            type_emb = torch.zeros_like(type_emb)

        token_list, mask_list = [], []
        # Order matters: SELF token must be at index 0 for pooling.
        for name in ("self", "preserved", "teammate", "target", "obstacle", "fog"):
            tok, m = self._embed_entity(name, obs[name], type_emb)
            token_list.append(tok)
            mask_list.append(m)

        tokens = torch.cat(token_list, dim=1)  # (B, N_total, E)
        mask = torch.cat(mask_list, dim=1)  # (B, N_total)

        tokens = self.body(tokens)
        self_latent = tokens[:, 0, :]  # SELF is index 0
        return self_latent, tokens, mask
