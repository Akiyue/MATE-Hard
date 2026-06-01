"""Type-Conditioned QMIX (TC-QMIX) mixing network.

Standard QMIX [Rashid et al., 2018] uses a *hypernetwork* over the global
state to produce non-negative weights for a monotonic mixing of per-agent
Q-values:

    Q_tot(s, a_1..N) = Mixer(Q_1(s,a_1), ..., Q_N(s,a_N); s)

with the constraint  dQ_tot / dQ_i >= 0 (monotonicity / IGM).

**TC-QMIX** is our novel contribution. We condition the hypernetwork on
the **per-agent camera-type vector** in addition to (or instead of) the
global state. This is the right inductive bias for heterogeneous-team
MARL like MATE-Hard: the contribution of a NARROW_LONG camera's Q-value
to the team return is structurally different from that of an OMNI_NOISY
camera, and the mixer should know.

Three hypernet inputs we support:

  - ``state``     : flat global state (concat of per-agent latents).
  - ``types``     : per-agent type one-hot.
  - ``state+types`` : both, concatenated (the default; novel).

The mixing remains monotonic: hypernet outputs are pushed through abs()
(or softplus) to guarantee non-negative weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


HyperInputMode = Literal["state", "types", "state+types"]


@dataclass
class TCQMIXMixerConfig:
    num_agents: int = 4
    num_types: int = 3              # = NUM_CAMERA_TYPES
    state_dim: int = 96             # per-agent latent dim that we pool
    embed_dim: int = 64             # mixing hidden width
    hyper_hidden: int = 64          # hypernet hidden width
    hyper_input: HyperInputMode = "state+types"
    use_abs: bool = True            # abs() on hypernet outputs (canonical QMIX)


class TCQMIXMixer(nn.Module):
    """TC-QMIX mixer.

    Args:
        config: TCQMIXMixerConfig

    Forward:
        agent_qs : (B, N)    per-agent Q values for the chosen joint action
        state    : (B, S)    global state
        types    : (B, N)    per-agent type indices (long)

      Returns:
        q_tot    : (B,)
    """

    def __init__(self, config: TCQMIXMixerConfig) -> None:
        super().__init__()
        self.config = config
        cfg = config

        # Type embedding
        self.type_emb = nn.Embedding(cfg.num_types, cfg.embed_dim)

        # Determine hypernet input dimension
        per_agent_type_feat = cfg.embed_dim * cfg.num_agents
        if cfg.hyper_input == "state":
            hyper_in = cfg.state_dim
        elif cfg.hyper_input == "types":
            hyper_in = per_agent_type_feat
        elif cfg.hyper_input == "state+types":
            hyper_in = cfg.state_dim + per_agent_type_feat
        else:
            raise ValueError(f"unknown hyper_input {cfg.hyper_input!r}")

        # Two-layer monotonic mixer:
        #   h1 = elu( w1 * agent_qs + b1 )          (B, embed)
        #   q_tot = w2 * h1 + b2                    (B,)
        # weights w1 (B, N, embed) and w2 (B, embed, 1) are produced by
        # hypernets; abs() enforces non-negativity.
        self.hyper_w1 = nn.Sequential(
            nn.Linear(hyper_in, cfg.hyper_hidden),
            nn.ReLU(),
            nn.Linear(cfg.hyper_hidden, cfg.num_agents * cfg.embed_dim),
        )
        self.hyper_b1 = nn.Linear(hyper_in, cfg.embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(hyper_in, cfg.hyper_hidden),
            nn.ReLU(),
            nn.Linear(cfg.hyper_hidden, cfg.embed_dim),
        )
        # Final bias is a state-conditioned scalar (the "V(s)" term in QMIX).
        self.hyper_b2 = nn.Sequential(
            nn.Linear(hyper_in, cfg.hyper_hidden),
            nn.ReLU(),
            nn.Linear(cfg.hyper_hidden, 1),
        )

    def _hyper_in(self, state: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """Build hypernet input from state and per-agent types.

        state : (B, S)
        types : (B, N) long
        Returns: (B, hyper_in)
        """
        cfg = self.config
        B, N = types.shape
        t_emb = self.type_emb(types)            # (B, N, embed_dim)
        t_flat = t_emb.reshape(B, N * cfg.embed_dim)
        if cfg.hyper_input == "state":
            return state
        if cfg.hyper_input == "types":
            return t_flat
        return torch.cat([state, t_flat], dim=-1)

    def forward(
        self,
        agent_qs: torch.Tensor,
        state: torch.Tensor,
        types: torch.Tensor,
    ) -> torch.Tensor:
        """
        agent_qs : (B, N)
        state    : (B, S)
        types    : (B, N) long
        Returns  : (B,)
        """
        cfg = self.config
        B, N = agent_qs.shape
        if N != cfg.num_agents:
            raise RuntimeError(
                f"TCQMIXMixer expects {cfg.num_agents} agents, got {N}"
            )

        h_in = self._hyper_in(state, types)  # (B, hyper_in)

        # Layer 1
        w1 = self.hyper_w1(h_in).view(B, N, cfg.embed_dim)
        b1 = self.hyper_b1(h_in).view(B, 1, cfg.embed_dim)
        if cfg.use_abs:
            w1 = torch.abs(w1)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)  # (B,1,E)

        # Layer 2
        w2 = self.hyper_w2(h_in).view(B, cfg.embed_dim, 1)
        b2 = self.hyper_b2(h_in).view(B, 1, 1)
        if cfg.use_abs:
            w2 = torch.abs(w2)
        q_tot = torch.bmm(hidden, w2) + b2  # (B, 1, 1)
        return q_tot.view(B)
