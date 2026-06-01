from typing import Callable, Optional

import torch
import torch.nn as nn

from .utils import get_activation, get_operator_function


class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X: (B, H, N, D) or (B, N, D)
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class GatedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        bias: bool = False,
        gate_act_fn: str = "sigmoid",
    ):
        super().__init__()
        output_dim = output_dim if output_dim is not None else input_dim  #

        self.up = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.down = nn.Linear(hidden_dim, output_dim, bias=bias)

        self.gate = Gated(
            input_dim,
            hidden_dim,
            bias=bias,
            gate_act_fn=gate_act_fn,
            gate_operator_fn="*",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.up(x)  # y = fc(x)
        gate_output = self.gate(x, y)  # y' = y o act(x@W(+B))
        return self.down(gate_output)  # out = fc(y)


class Gated(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        out_dim: int,
        bias=False,
        gate_act_fn: str | Callable[[torch.Tensor], torch.Tensor] = "sigmoid",
        gate_operator_fn: str
        | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "*",
    ):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(inp_dim, out_dim, bias=bias), get_activation(gate_act_fn)
        )

        self.operator_fn = get_operator_function(gate_operator_fn)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (..., Ex)
        # y: (..., Ey)
        return self.operator_fn(y, self.gate(x))  # y o gate(x)   (..., Exy)
