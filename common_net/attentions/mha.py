from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from ..common import Gated, ZeroCenteredRMSNorm
from .base_attn import AttentionBase, ScaledDotProductAttention


@dataclass(frozen=True)
class MHAConfig:
    num_heads: int = 1
    attn_cls: type[AttentionBase] = ScaledDotProductAttention
    attn_kwargs: Optional[dict] = None
    gated: bool = False
    gate_act_fn: str | Callable[[torch.Tensor], torch.Tensor] = "sigmoid"
    gate_operator_fn: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "*"
    num_k_heads: Optional[int] = None
    num_v_heads: Optional[int] = None
    kdim: Optional[int] = None
    vdim: Optional[int] = None
    qk_norm_cls: type[nn.Module] = ZeroCenteredRMSNorm
    qk_norm_kwargs: Optional[dict] = None
    bias: bool = False
    device: Optional[torch.device | str] = None
    dtype: Optional[torch.dtype] = None


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        config: MHAConfig,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = config.num_heads
        self.num_k_heads = (
            config.num_k_heads if config.num_k_heads is not None else config.num_heads
        )
        self.num_v_heads = (
            config.num_v_heads if config.num_v_heads is not None else config.num_heads
        )
        self.kdim = config.kdim if config.kdim is not None else embed_dim
        self.vdim = config.vdim if config.vdim is not None else embed_dim

        factory_kwargs = {"device": config.device, "dtype": config.dtype}

        self.head_dim = embed_dim // self.num_heads

        if embed_dim % self.num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible by the number of heads ({self.num_heads})."
            )

        if self.num_heads == self.num_k_heads and self.num_heads == self.num_v_heads:
            pass  # normal attention
        elif self.num_heads > max(
            self.num_k_heads, self.num_v_heads
        ):  # group query Attention
            if self.num_heads % self.num_k_heads != 0:
                raise ValueError(
                    f"Number of attention heads must be multiple of number of key heads, get H={self.num_heads}, Hk={self.num_k_heads}"
                )
            if self.num_heads % self.num_v_heads != 0:
                raise ValueError(
                    f"Number of attention heads must be multiple of number of value heads, get H={self.num_heads}, Hk={self.num_v_heads}"
                )
        else:  # to be update
            raise NotImplementedError(
                "Only support Group Query Attention for now (Hk == Hv < Hq), get Hq={}, Hk={}, Hv={}".format(
                    self.num_heads, self.num_k_heads, self.num_v_heads
                )
            )

        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=config.bias, **factory_kwargs
        )
        self.k_proj = nn.Linear(
            self.kdim,
            self.num_k_heads * self.head_dim,
            bias=config.bias,
            **factory_kwargs,
        )
        self.v_proj = nn.Linear(
            self.vdim,
            self.num_v_heads * self.head_dim,
            bias=config.bias,
            **factory_kwargs,
        )

        if config.gated:
            self.gate = Gated(
                embed_dim,
                embed_dim,
                bias=config.bias,
                gate_act_fn=config.gate_act_fn,
                gate_operator_fn=config.gate_operator_fn,
            )
        else:
            self.gate = None

        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=config.bias, **factory_kwargs
        )

        attn_kwargs = config.attn_kwargs if config.attn_kwargs is not None else {}
        self.attn = config.attn_cls(**attn_kwargs)

        qk_norm_kwargs = (
            config.qk_norm_kwargs if config.qk_norm_kwargs is not None else {}
        )

        self.q_norm = config.qk_norm_cls(self.head_dim, **qk_norm_kwargs)
        self.k_norm = config.qk_norm_cls(self.head_dim, **qk_norm_kwargs)
        self._reset_parameters()

    def _reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.apply(_init_weights)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal=False):
        # Q: (B, N, E)
        # K: (B, Nkv, E)
        # V: (B, Nkv, E)
        B, N, _ = Q.size()
        H = self.num_heads
        Hk = self.num_k_heads
        Hv = self.num_v_heads
        D = self.head_dim

        # E = H*D

        # Project to multi-head
        Q_proj: torch.Tensor = self.q_proj(Q)  # (B, N, H * D)
        K_proj: torch.Tensor = self.k_proj(K)  # (B, Nkv, Hk * D)
        V_proj: torch.Tensor = self.v_proj(V)  # (B, Nkv, Hv * D)

        Q_proj = Q_proj.view(B, N, H, D).transpose(1, 2)  # (B, H , N, D)
        K_proj = K_proj.view(B, N, Hk, D).transpose(1, 2)  # (B, Hk, Nkv, D)
        V_proj = V_proj.view(B, N, Hv, D).transpose(1, 2)  # (B, Hv, Nkv, D)

        Q_proj = self.q_norm(Q_proj)
        K_proj = self.k_norm(K_proj)

        # Attention
        attn_out, attn_weights = self.attn(
            Q_proj, K_proj, V_proj, causal=causal
        )  # (B, H, N, D)

        attn_out: torch.Tensor = (
            attn_out.transpose(1, 2).contiguous().view(B, N, H * D)
        )  # (B, H, N, D) -> (B, N, H, D) -> (B, N, H*D) or (B, N, E)

        if self.gate is not None:
            attn_out = self.gate(
                Q, attn_out
            )  # attn_out = attn_out o act(Q @ W)  -> (B, N, E)

        out = self.out_proj(attn_out)  # (B, N, E)
        return out, attn_weights
