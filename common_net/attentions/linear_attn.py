from typing import Callable

import torch
import torch.nn.functional as F

from .base_attn import AttentionBase


def _elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


class LinearAttention(AttentionBase):
    """
    Linear Attention (kernel-based, positive feature map) supporting causal and non-causal modes.

    Expected input shapes:
      Q, K, V : tensors of shape (B, H, N, D)
        B = batch, H = heads, N = seq_len, D = head_dim

    Output:
      out : (B, H, N, D_v)  where D_v == V.shape[-1]

    Parameters:
      feature_map: callable tensor->tensor. Must return non-negative values. Default: elu(x)+1.
      eps: small float for numerical stability
    """

    def __init__(
        self,
        feature_map: Callable[[torch.Tensor], torch.Tensor] = _elu_feature_map,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.feature_map = feature_map
        self.eps = eps

    def _forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = False
    ):
        # Feature mapping
        # Q: B, H, N, D
        # K: B, H, Nkv, D
        # V: B, H, Nkv, Dv

        Qf = self.feature_map(Q)  # (B,H,N,Dphi)
        Kf = self.feature_map(K)  # (B,H,Nkv,Dphi)

        if not causal:
            # Non-causal (fast global aggregation)
            # phi(K).T (B, H, Dphi, N) @ V (B, Hv, N, Dv) -> (B, H, Dphi, Dv)
            KV = torch.matmul(  # phi(K).T @ V
                Kf.transpose(-2, -1),  # [B, H, D, Nkv]
                V,  # [B, H, Nkv, Dv]
            )  # (B, H, Dphi, Dv)

            # phi(K).T * I
            Kf_prefix = Kf.sum(dim=2)  # (B, H, Dphi)

            numerator = torch.matmul(  # phi(Q) @ (phi(K).T @ V)
                Qf,  # (B,H,N,Dphi)
                KV,  # (B,H,Dphi,Dv)
            )  # (B,H,N,Dv)

            # Denominator: z = Qf_i dot Kf_prefix -> (B,H,N,1)
            denominator = torch.matmul(  # phi(Q) @ (phi(K) @ I)
                Qf,  # (B,H,N,Dphi)
                Kf_prefix.unsqueeze(-1),  # (B,H,Dphi,1)
            )  # (B,H,N,1)
        else:
            # Causal attention: use prefix sums
            # We need per-position prefix sums of Kf and Kf * V
            # Kf: (B,H,N,Dphi), V: (B,H,N,Dv)
            # Build KV_each_pos = Kf[n].unsqueeze(-1) * V[n].unsqueeze(-2) => (B,H,N,Dphi,Dv)
            KV_each = Kf.unsqueeze(-1) * V.unsqueeze(-2)  # (B,H,N,Dphi,Dv)

            # prefix sums along sequence dim (n)
            KV_prefix = KV_each.cumsum(dim=2)  # (B,H,N,Dphi,Dv)
            Kf_prefix = Kf.cumsum(dim=2)  # (B,H,N,Dphi)

            # Numerator at position n: Qf[n] @ KV[n]
            # einsum: out_num[b,h,n,e] = sum_d Qf[b,h,n,d] * KV[b,h,n,d,e]
            numerator = torch.einsum(
                "b h n d, b h n d e -> b h n e", Qf, KV_prefix
            )  # (B,H,N,Dv)

            # Denominator: denom[b,h,n] = Qf[b,h,n] dot Kf_prefix[b,h,n]
            denominator = torch.einsum(
                "b h n d, b h n d -> b h n", Qf, Kf_prefix
            ).unsqueeze(-1)  # (B,H,N,1)

        out = numerator / (
            denominator + self.eps
        )  # (B,H,N,Dv) / (B,H,N,1) -> (B,H,N,Dv)
        return out, None  # no attention weights
