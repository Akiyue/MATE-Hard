from typing import Optional

import torch
import torch.nn as nn

from .attentions import MHAConfig
from .attentions import MultiHeadAttention as MHA
from .common import GatedMLP, ZeroCenteredRMSNorm
from .moe import MoE, MoEConfig


class AttnBlock(nn.Module):
    """Abstract base class for attention blocks."""

    pass


class MAB(AttnBlock):
    """Multihead Attention Block (MAB)

    This class implements a Multihead Attention Block that performs self-attention
    between query (Q), key (K), and value (V) tensors, followed by a feed-forward network.
    The architecture follows the standard transformer block design with normalization,
    attention, and feed-forward layers with residual connections.

    ```
    X -> Norm -> MHA -> + ----> Norm -> FF -> + ---> out
    |                   |   |                 |
    v                   |   v                 |
    -------------------->   ------------------>
    ```


    Parameters
    ----------
    embed_dim : int
        The embedding dimension for the input and output tensors.
    d_ff : int
        The hidden dimension size for the feed-forward network.
    mha_config : MHAConfig
        Configuration object for the multi-head attention module.
    inp_norm : type[nn.Module], default=ZeroCenteredRMSNorm
        The normalization layer type to use for input normalization.
    inp_norm_kwargs : dict, optional
        Additional keyword arguments to pass to the input normalization layer.
    moe_cls : type[MoE], optional
        The Mixture of Experts class to use instead of standard feed-forward network.
        If None, a GatedMLP will be used.
    moe_config : MoEConfig, optional
        Configuration object for the Mixture of Experts module if moe_cls is specified.

    Attributes
    ----------
    embed_dim : int
        The embedding dimension for the input and output tensors.
    inp_norm : nn.Module
        The input normalization layer.
    attn : MHA
        The multi-head attention module.
    ff : nn.Module
        The feed-forward network (either GatedMLP or a Mixture of Experts).

    Notes
    -----
    The forward pass applies layer normalization followed by self-attention,
    a residual connection, then a normalization followed by a feed-forward network with another residual
    connection.
    """

    """Multihead Attention Block (MAB)"""

    def __init__(
        self,
        embed_dim: int,
        d_ff: int,
        mha_config: MHAConfig,
        norm_cls: type[nn.Module] = ZeroCenteredRMSNorm,
        norm_kwargs: Optional[dict] = None,
        moe_cls: Optional[type[MoE]] = None,
        moe_config: Optional[MoEConfig] = None,
    ) -> None:
        super().__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}

        self.inp_norm = norm_cls(embed_dim, **norm_kwargs)

        self.attn = MHA(embed_dim=embed_dim, config=mha_config)

        self.ff_norm = norm_cls(embed_dim, **norm_kwargs)

        if moe_cls is not None:
            self.ff = moe_cls(embed_dim=embed_dim, d_ff=d_ff, config=moe_config)
        else:
            self.ff = GatedMLP(embed_dim, d_ff)

    def forward(
        self, X: torch.Tensor, causal: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Forward pass for the Multihead Attention Block.
        ```
        X -> Norm -> MHA -> + ----> Norm -> FF -> + ---> out
        |                   |   |                 |
        v                   |   v                 |
        -------------------->   ------------------>
        ```

        Args:
            X (torch.Tensor): _description_
            causal (bool, optional): _description_. Defaults to False.
        """

        # ---- Multi-Head Attention ----
        X_norm = self.inp_norm(X)  # (B, N, E)

        attn_out, attn_weights = self.attn(
            X_norm, X_norm, X_norm, causal=causal
        )  # (B, N, E)

        hidden_states = X + attn_out  # Residual

        # ---- Feed Forward ----
        ff_out = self.ff(self.ff_norm(hidden_states))  # (B, N, E)

        out = hidden_states + ff_out  # Residual (B, N, E)

        return out


class CrossMAB(AttnBlock):
    """Cross Multi-head Attention Block module for attention between two different tensors.

    This module implements a cross-attention mechanism where queries come from one tensor (X)
    and keys/values from another tensor (Y), followed by a feed-forward network.
    The architecture follows a pre-norm design with residual connections.

        embed_dim (int): The embedding dimension of the input and output tensors.
        d_ff (int): The dimension of the intermediate feed-forward layer.
        mha_config (MHAConfig): Configuration object for Multi-Head Attention.
        norm_cls (type[nn.Module], optional): Normalization layer class.
            Defaults to ZeroCenteredRMSNorm.
        norm_kwargs (Optional[dict], optional): Additional arguments for normalization layer.
            Defaults to None.
        moe_cls (Optional[type[MoE]], optional): Mixture of Experts class for the feed-forward network.
            If None, uses GatedMLP instead. Defaults to None.
        moe_config (Optional[MoEConfig], optional): Configuration for MoE if moe_cls is provided.
            Defaults to None.
    """
    def __init__(
        self,
        embed_dim: int,
        d_ff: int,
        mha_config: MHAConfig,
        norm_cls: type[nn.Module] = ZeroCenteredRMSNorm,
        norm_kwargs: Optional[dict] = None,
        moe_cls: Optional[type[MoE]] = None,
        moe_config: Optional[MoEConfig] = None,
    ) -> None:
        super().__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}

        self.inp_norm = norm_cls(embed_dim, **norm_kwargs)

        self.kv_norm = norm_cls(embed_dim, **norm_kwargs)

        self.attn = MHA(embed_dim=embed_dim, config=mha_config)

        self.ff_norm = norm_cls(embed_dim, **norm_kwargs)

        if moe_cls is not None:
            self.ff = moe_cls(embed_dim=embed_dim, d_ff=d_ff, config=moe_config)
        else:
            self.ff = GatedMLP(embed_dim, d_ff)

    def forward(
        self, X: torch.Tensor, Y: torch.Tensor, causal: bool = False
    ) -> torch.Tensor:
        """
        X: (B, N, d)  query
        Y: (B, M, d)  key/value
        causal: whether to apply causal masking (not supported here)
        """

        # ---- Multi-Head Attention ----
        X_norm = self.inp_norm(X)  # (B, N, E)
        Y_norm = self.kv_norm(Y)  # (B, M, E)

        attn_out, attn_weights = self.attn(
            X_norm, Y_norm, Y_norm, causal=causal
        )  # (B, N, E)

        hidden_states = X + attn_out  # Residual

        # ---- Feed Forward ----
        ff_out = self.ff(self.ff_norm(hidden_states))  # (B, N, E)

        out = hidden_states + ff_out  # Residual (B, N, E)

        return out


class ISAB(AttnBlock):
    """Inducing Self-Attention Block (ISAB).

    This module implements a self-attention mechanism that uses a set of learnable inducing points
    to reduce the computational complexity of standard self-attention. Instead of direct all-to-all
    attention among all elements in the input set, ISAB first computes attention between inducing points
    and input elements, and then computes attention between input elements and the induced representations.
    This approach reduces the computational complexity from O(N²) to O(NM), where N is the number of
    input elements and M is the number of inducing points (typically M << N).

    References:
        Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"

    `Note`: This implementation is suboptimal after the introduction of Linear Attention, which provides
    better efficiency for large sets. Consider using Linear Attention variants for performance-critical applications.
    """

    def __init__(
        self,
        embed_dim: int,
        d_ff: int,
        num_inducing: int,
        mha_config: MHAConfig,
        norm_cls: type[nn.Module] = ZeroCenteredRMSNorm,
        norm_kwargs: Optional[dict] = None,
        moe_cls: Optional[type[MoE]] = None,
        moe_config: Optional[MoEConfig] = None,
    ):
        super().__init__()

        # Learnable inducing points (m, d)
        self.inducing_points = nn.Parameter(torch.randn(num_inducing, embed_dim))

        self.mab1 = CrossMAB(
            embed_dim=embed_dim,
            d_ff=d_ff,
            mha_config=mha_config,
            norm_cls=norm_cls,
            norm_kwargs=norm_kwargs,
            moe_cls=moe_cls,
            moe_config=moe_config
        )

        self.mab2 = CrossMAB(
            embed_dim=embed_dim,
            d_ff=d_ff,
            mha_config=mha_config,
            norm_cls=norm_cls,
            norm_kwargs=norm_kwargs,
            moe_cls=moe_cls,
            moe_config=moe_config
        )

    def forward(self, X: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """
        x: (B, N, E)  input set
        causal
        """

        B = X.shape[0]


        inducing = self.inducing_points.unsqueeze(0).expand(B, -1, -1)  # (B, N, E)

        H, attn1_weighs = self.mab1(inducing, X, causal=causal)       # (B, M, E)

        out, attn2_weights = self.mab2(X, H)  # (B, N, E)

        return out


class ChainBlock(AttnBlock):
    def __init__(self, num_repeats: int, block_cls: type[AttnBlock], **block_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [block_cls(**block_kwargs) for _ in range(num_repeats)]
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, *args, **kwargs)

        return x
