import torch
import torch.nn as nn


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (B, Hkv, N, D) to (B, H, N, D)
    """
    B, Hkv, N, D = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(B, Hkv, n_rep, N, D)
    return hidden_states.reshape(B, Hkv * n_rep, N, D)


class AttentionBase(nn.Module):
    """
    AttentionBase is an abstract base class for attention head mechanisms.

    This class expects input tensors Q, K, and V with the following shapes:
        - Q: (B, H, N, D)
        - K: (B, Hk, N, D)
        - V: (B, Hv, N, Dv)

    Where:
        B   = batch size
        H  = number of query heads
        Hk  = number of key heads
        Hv  = number of value heads
        N   = sequence length
        D   = head dimension for Q and K
        Dv  = head dimension for V

    The class provides shape checking, head expansion (for Group Query Attention) and a standard forward interface for attention modules.
    Subclasses must implement the `_forward` method to define the specific attention mechanism.
    """

    def shape_check(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        if Q.dim() != 4 or K.dim() != 4 or V.dim() != 4:
            raise ValueError("Expected (B,H,N,D) inputs")

        B, H, N, Dq = Q.shape
        Bk, Hk, Nk, Dk = K.shape
        Bv, Hv, Nv, Dv = V.shape
        if B != Bk or B != Bv:
            raise ValueError("Batch sizes of Q,K,V must match")
        if Nk != Nv:
            raise ValueError("Sequence lengths of K,V must match")
        if Dq != Dk:
            raise ValueError("Q and K must have same head dimension")

    def _forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, *args, **kwargs
    ):
        raise NotImplementedError

    def post_forward(self, out):
        return out

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool = False,
        *args,
        **kwargs,
    ):
        self.shape_check(Q, K, V)

        H = Q.shape[1]
        Hk = K.shape[1]
        Hv = V.shape[1]

        K = repeat_kv(K, H // Hk)
        V = repeat_kv(V, H // Hv)

        if H != K.shape[1] or H != V.shape[1]:
            raise ValueError(
                "Q, K and H must have same number of heads after expansion, got Hq={}, Hk={}, Hv={}".format(
                    H, K.shape[1], V.shape[1]
                )
            )

        out, weights = self._forward(Q, K, V, causal=causal, *args, **kwargs)

        out = self.post_forward(out)

        if out.shape != V.shape:
            raise ValueError(
                f"Output shape {out.shape} does not match V shape {V.shape}"
            )

        return self.post_forward(out), weights


class ScaledDotProductAttention(AttentionBase):
    """
    The original Attention
    Implements the scaled dot-product attention mechanism.

    This module computes attention scores between queries (Q) and keys (K), scales them by the square root of the key dimension,
    applies softmax to obtain attention weights, and then computes a weighted sum of the values (V). Dropout is applied to the
    attention weights for regularization.

        dropout (float, optional): Dropout probability applied to the attention weights. Default is 0.0.

    Methods:
        _forward(Q, K, V, causal=False):
            Computes the scaled dot-product attention.

                Q (torch.Tensor): Query tensor of shape (B, H, N, D).
                K (torch.Tensor): Key tensor of shape (B, H, N, D).
                V (torch.Tensor): Value tensor of shape (B, H, N, Dv).
                causal (bool, optional): If True, applies causal masking to prevent attention to future positions.
                                         Currently not implemented. Default is False.

            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                    - attn_output: Output tensor after applying attention, shape (B, H, N, Dv).
                    - attn_weights: Attention weights, shape (B, H, N, N).
    Raises:
        NotImplementedError: If causal attention is requested (causal=True).
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout

    def _forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = False
    ):
        # Q: (B, H, N, D)
        # K: (B, H, Nkv, D)
        # V: (B, H, Nkv, Dv)

        D = Q.shape[-1]

        if causal:
            raise NotImplementedError("Causal attention is not implemented yet")
        else:
            scaling = float(D) ** -0.5

            attn_weights = (
                torch.matmul(
                    Q,  # (B, H, N, D)
                    K.transpose(-2, -1),  # (B, H, D, Nkv)
                )
                * scaling
            )  # (B, H, N, Nkv)

            # if attention_mask is not None:
            #     causal_mask = attention_mask[:, :, :, : K.shape[-2]]
            #     attn_weights = attn_weights + causal_mask

            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(Q.dtype)  # (B, H, N, Nkv)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=self.training
            )

            attn_output = torch.matmul(
                attn_weights, V
            )  # (B, H, N, Nkv) @ (B, H, Nkv, Dv) -> (B, H, N, Dv)

            return attn_output, attn_weights


def main():
    B, H, L, D = 2, 4, 8, 16
    t = torch.randn(B, H, L, D)
    print(t.shape)

    print(repeat_kv(t, 3).shape)


if __name__ == "__main__":
    main()
