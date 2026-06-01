from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import GatedMLP


@dataclass(frozen=True)
class MoEConfig:
    num_experts: int = 4
    use_shared_expert: bool = False  # Whether to use shared experts
    gate_act_fn: str | Callable[[torch.Tensor], torch.Tensor] = (
        "sigmoid"  # Activation function for the experts
    )


class MoE(nn.Module):
    config: MoEConfig

    def __init__(
        self, embed_dim: int = 128, d_ff: int = 512, config: Optional[MoEConfig] = None
    ):
        super().__init__()

        self.config = match_config(type(self))() if config is None else config
        assert isinstance(self.config, MoEConfig), (
            "config must be an instance of MoEConfig"
        )

        if self.config.use_shared_expert:
            self.shared_expert = GatedMLP(
                embed_dim, d_ff, gate_act_fn=self.config.gate_act_fn
            )
            self.shared_expert_gate = nn.Linear(embed_dim, 1, bias=False)
        else:
            self.shared_expert = None
            self.shared_expert_gate = None

        self.experts = nn.ModuleList(
            [
                GatedMLP(embed_dim, d_ff, gate_act_fn=self.config.gate_act_fn)
                for _ in range(self.config.num_experts - self.config.use_shared_expert)
            ]
        )

        # Gating network (decides which routed expert to use per token).
        # The shared expert (if any) has its OWN sigmoid gate; it is not part
        # of this softmax routing. So this gate emits len(self.experts) logits
        # — i.e. num_experts when use_shared_expert is False, and num_experts-1
        # when it is True.
        self.gate = nn.Linear(embed_dim, len(self.experts), bias=False)

        self._gate_logits: list[torch.Tensor] = []
        self._last_gate_logits: torch.Tensor = None

    def _get_gate_logits(self) -> torch.Tensor:
        return self._gate_logits

    def _store_gate_logits(self, gate_logits: torch.Tensor):
        if self.training:
            self._gate_logits.append(gate_logits)
        self._last_gate_logits = gate_logits

    def _reset_gate_logits(self):
        self._gate_logits = []

    @property
    def gate_logit(self):
        """Returns the most recent gate logits.

        Returns:
            torch.Tensor: The most recent gate logits.
        """
        return self._last_gate_logits

    def forward_shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.config.use_shared_expert:
            shared_expert_output: torch.Tensor = self.shared_expert(hidden_states)
            shared_expert_output = (
                F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
            )
            return shared_expert_output
        return 0.0

    def forward_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_probs = F.softmax(self.gate_logit, dim=1, dtype=hidden_states.dtype)
        gate_probs = gate_probs.to(hidden_states.dtype)

        expert_outputs = torch.stack(
            [expert(hidden_states) for expert in self.experts], dim=-1
        )  # (*, D, n_experts)

        expert_outputs = torch.matmul(
            expert_outputs,  # (*, D, n_experts)
            gate_probs.unsqueeze(-1),  # (*, n_experts, 1)
        ).squeeze(-1)  # (*, D)

        return expert_outputs

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """

        original_shape = hidden_states.shape
        hidden_dim = original_shape[-1]

        hidden_states = hidden_states.view(-1, hidden_dim)

        self._store_gate_logits(self.gate(hidden_states))

        final_hidden_states = self.forward_experts(
            hidden_states
        ) + self.forward_shared_expert(hidden_states)

        return final_hidden_states.reshape(original_shape)


@dataclass(frozen=True)
class TopKMoEConfig(MoEConfig):
    top_k: int = 1
    norm_topk_prob: bool = False  # Whether to normalize the top-k probabilities


class TopKMoE(MoE):
    config: TopKMoEConfig

    def forward_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (*, D)
        # The router emits len(self.experts) logits — the shared expert (if
        # any) is handled separately by forward_shared_expert.
        num_routed = len(self.experts)

        top_k = self.config.top_k

        if top_k > num_routed:
            raise ValueError(
                f"top_k ({top_k}) cannot be greater than the number of routed "
                f"experts ({num_routed})"
            )

        gate_probs = F.softmax(self.gate_logit, dim=1, dtype=hidden_states.dtype)
        gate_probs = gate_probs.to(hidden_states.dtype)
        gate_probs, selected_experts = torch.topk(gate_probs, top_k, dim=-1)

        if self.config.norm_topk_prob:
            gate_probs /= gate_probs.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=num_routed
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(
                -1, hidden_states.shape[-1]
            )
            current_hidden_states = (
                expert_layer(current_state) * gate_probs[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )

        return final_hidden_states


class ImportanceLoss(nn.Module):
    """
    Encourages balanced importance among experts.
    Formula:
        L_importance = CV(importance)^2
    where importance = sum of gate probabilities per expert.
    """

    def __init__(self, eps=1e-9):
        super(ImportanceLoss, self).__init__()
        self.eps = eps

    def forward(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        gate_logits: [batch_size, num_experts]
        """
        gate_logits = gate_logits.view(
            -1, gate_logits.shape[-1]
        )  # Flatten to 2D if necessary

        gate_probs = F.softmax(gate_logits, dim=-1)  # Softmax over experts
        importance = gate_probs.sum(dim=0)  # Sum over batch

        mean_importance = importance.mean()
        var_importance = ((importance - mean_importance) ** 2).mean()

        cv_squared = var_importance / (mean_importance**2 + self.eps)
        return cv_squared


class LoadLoss(nn.Module):
    """
    Encourages balanced load among experts.
    Formula:
        L_load = CV(load)^2
    where load = sum of gate selections per expert (hard selection).
    """

    def __init__(self, eps=1e-9):
        super(LoadLoss, self).__init__()
        self.eps = eps

    def forward(self, gate_logits):
        """
        gate_logits: [batch_size, num_experts]
        """
        gate_logits = gate_logits.view(
            -1, gate_logits.shape[-1]
        )  # Flatten to 2D if necessary

        gate_probs = F.softmax(gate_logits, dim=-1)
        # For load: sum over batch selections
        load = gate_probs.sum(dim=0)

        mean_load = load.mean()
        var_load = ((load - mean_load) ** 2).mean()

        cv_squared = var_load / (mean_load**2 + self.eps)
        return cv_squared


class CapacityLoss(nn.Module):
    """
    Penalizes experts that exceed their capacity.
    """

    def __init__(self, capacity: int = 10):
        super(CapacityLoss, self).__init__()
        self.capacity = capacity

    def forward(self, expert_assignments: torch.Tensor) -> torch.Tensor:
        """
        expert_assignments: [batch_size] tensor of expert indices (hard assignments)
        """
        num_experts = expert_assignments.max().item() + 1
        load_per_expert = torch.zeros(num_experts, device=expert_assignments.device)

        for expert in range(num_experts):
            load_per_expert[expert] = (expert_assignments == expert).sum()

        overload = torch.clamp(load_per_expert - self.capacity, min=0)
        capacity_loss = (overload**2).mean()

        return capacity_loss


class AllGateLoss(nn.Module):
    """
    Wrapper that combines multiple gate losses.
    """

    def __init__(
        self, importance_weight=1.0, load_weight=1.0, eps=1e-9, return_dict=False
    ):
        super(AllGateLoss, self).__init__()
        self.importance_loss = ImportanceLoss(eps=eps)
        self.load_loss = LoadLoss(eps=eps)

        self.importance_weight = importance_weight
        self.load_weight = load_weight
        self.return_dict = return_dict

    def forward(self, gate_logits):
        imp_loss = self.importance_loss(gate_logits)
        load_loss = self.load_loss(gate_logits)
        loss = self.importance_weight * imp_loss + self.load_weight * load_loss

        if self.return_dict:
            return loss, {"importance_loss": imp_loss, "load_loss": load_loss}
        else:
            return loss


class MoEGateLossManager(nn.Module):
    def __init__(self, module: nn.Module, criterion: Optional[Callable] = None):
        super().__init__()
        self.criterion = criterion if criterion is not None else AllGateLoss()

        self.losses: list[MoE] = []

        self._find_moe_layers(module)

    def _find_moe_layers(self, module: nn.Module):
        """Find all MoE layers in a module recursively."""
        for name, submodule in module.named_modules():
            if isinstance(submodule, MoE):
                self.register_moe(submodule)

    def register_moe(self, moe_layer: MoE):
        self.losses.append(moe_layer)

    def extend(self, other: "MoEGateLossManager"):
        self.losses.extend(other.losses)

    def forward(self):
        total_gate_loss = 0.0
        for moe in self.losses:
            gate_logits = moe._get_gate_logits()
            for logit in gate_logits:
                total_gate_loss += self.criterion(logit)

            moe._reset_gate_logits()

        return total_gate_loss


def match_config(moe_cls: type[MoE]) -> type[MoEConfig] | None:
    if moe_cls is MoE:
        return MoEConfig
    elif moe_cls is TopKMoE:
        return TopKMoEConfig
    else:
        return None


def main():
    config = TopKMoEConfig()
    moe = TopKMoE(config=config)
    x = torch.randn(2, 10, config.embed_dim)
    out = moe(x)
    print(moe.shared_experts)
    print(out.shape)  # Should be (2, 10, embed_dim)


if __name__ == "__main__":
    main()
