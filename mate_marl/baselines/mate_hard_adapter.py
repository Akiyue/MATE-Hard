"""Rule-based agents adapted to run on MATE-Hard.

MATE's built-in rule-based agents (Random / Naive / Greedy / Heuristic) were
designed for the *base* MATE observation. MATE-Hard appends type one-hot,
energy state, and fog tokens to the per-camera observation. This module
provides two adapters:

  - ``RuleBasedNaiveMateHard`` — slices the observation back to the base MATE
    layout and delegates entirely to the underlying rule-based agent. The
    agent does NOT see type, energy or fog tokens. This is the *floor*
    rule-based baseline on MATE-Hard.

  - ``RuleBasedInformedMateHard`` — same slicing as the naive adapter, but
    additionally consumes the MATE-Hard-specific fields to refine the action
    *post-hoc*: zero out the action when energy is depleted (matching the
    EnergyConstraint forced-idle behaviour), and de-prioritise tracking
    fogged targets. The informed adapter therefore exploits the
    augmentation that the env exposes; it is the *ceiling* rule-based
    baseline on MATE-Hard.

Both adapters work via composition rather than inheritance:
``adapter.act(obs[i], info=info[i])`` returns a 2-D continuous action vector
that the underlying MATE-Hard env can step directly. The adapters are
*single-camera* (per-camera index), matching ``mate.AgentBase``; the eval
harness instantiates ``num_cameras`` of them via ``adapter.spawn(N)``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

import mate.constants as consts
from mate.agents.base import CameraAgentBase
from mate_marl.wrappers.heterogeneous_cameras import NUM_CAMERA_TYPES
from mate_marl.wrappers.energy_constraint import _ENERGY_AUG_DIM
from mate_marl.wrappers.dynamic_fog import _FOG_TOKEN_DIM


def _base_obs_dim(num_cameras: int, num_targets: int, num_obstacles: int) -> int:
    """Total dim of the *base* MATE camera observation, before MATE-Hard
    augmentation."""
    return int(
        consts.camera_observation_indices_of(
            num_cameras, num_targets, num_obstacles
        )[-1]
    )


def _split_mate_hard_obs(
    flat_obs: np.ndarray,
    num_cameras: int,
    num_targets: int,
    num_obstacles: int,
    num_fogs: int,
) -> Dict[str, np.ndarray]:
    """Split a single-camera flat MATE-Hard observation into named blocks.

    Layout (matches the wrapper stack:
        HeterogeneousCameras -> EnergyConstraint -> DynamicFog -> MateMARLDictObs):

        [ base MATE obs ][ type one-hot:NUM_CAMERA_TYPES ]
        [ energy aug:_ENERGY_AUG_DIM ][ fog tokens: num_fogs * _FOG_TOKEN_DIM ]
    """
    base_dim = _base_obs_dim(num_cameras, num_targets, num_obstacles)
    flat = np.asarray(flat_obs, dtype=np.float64).reshape(-1)

    if flat.shape[0] < base_dim:
        raise ValueError(
            f"observation too short ({flat.shape[0]}) for base MATE dim "
            f"{base_dim}"
        )

    base = flat[:base_dim]
    o = base_dim
    type_oh = flat[o : o + NUM_CAMERA_TYPES]
    o += NUM_CAMERA_TYPES
    energy_aug = flat[o : o + _ENERGY_AUG_DIM]
    o += _ENERGY_AUG_DIM
    fog_dim = num_fogs * _FOG_TOKEN_DIM
    fog = flat[o : o + fog_dim].reshape(num_fogs, _FOG_TOKEN_DIM) if num_fogs > 0 else np.zeros((0, _FOG_TOKEN_DIM))
    return {
        "base": base,
        "type_one_hot": type_oh,
        "energy_aug": energy_aug,  # [normalized_energy, charging, idle]
        "fog": fog,
    }


class RuleBasedNaiveMateHard(CameraAgentBase):
    """Wraps a base ``mate.agents`` CameraAgent and feeds it the base-MATE
    slice of a MATE-Hard observation. The agent does NOT see the MATE-Hard
    augmentation."""

    DEFAULT_ACTION = consts.CAMERA_DEFAULT_ACTION

    def __init__(self, inner: CameraAgentBase, num_fogs: int) -> None:
        # Set inner BEFORE super().__init__() because the base ctor calls
        # self.seed(), which our seed() override delegates to self._inner.
        self._inner = inner
        self._num_fogs = int(num_fogs)
        # Forward TEAM before super().__init__() (base ctor reads TEAM via
        # self.agent_id formatting).
        self.TEAM = inner.TEAM
        super().__init__()
        # Forward STATE_CLASS / TEAMMATE_STATE_CLASS / OPPONENT_STATE_CLASS
        for attr in ("STATE_CLASS", "TEAMMATE_STATE_CLASS", "OPPONENT_STATE_CLASS"):
            if hasattr(inner, attr):
                try:
                    setattr(self, attr, getattr(inner, attr))
                except AttributeError:
                    pass

    def clone(self) -> "RuleBasedNaiveMateHard":
        return RuleBasedNaiveMateHard(self._inner.clone(), num_fogs=self._num_fogs)

    def spawn(self, num_agents: int) -> List["RuleBasedNaiveMateHard"]:
        return [self.clone() for _ in range(num_agents)]

    def seed(self, seed: Optional[int] = None) -> List[int]:
        return self._inner.seed(seed)

    def reset(self, observation: np.ndarray) -> None:
        # Slice the augmented observation to its base length, delegate.
        base = self._base_slice_from_flat(observation)
        self._inner.reset(base)
        # forward inner state to outer attributes the eval harness reads
        for attr in (
            "num_cameras", "num_targets", "num_obstacles", "index", "agent_id",
            "observation_indices", "observation_slices", "observation_dim",
            "observation_space", "action_space",
        ):
            if hasattr(self._inner, attr):
                setattr(self, attr, getattr(self._inner, attr))

    def observe(self, observation: np.ndarray, info: Optional[dict] = None) -> None:
        base = self._base_slice_from_flat(observation)
        self._inner.observe(base, info)

    def act(self, observation: np.ndarray, info: Optional[dict] = None,
            deterministic: Optional[bool] = None) -> np.ndarray:
        base = self._base_slice_from_flat(observation)
        return self._inner.act(base, info=info, deterministic=deterministic)

    # Communication forwarding — many MATE agents (Greedy/Heuristic) send and
    # receive messages, so we forward these to the inner agent.
    def send_requests(self):
        return self._inner.send_requests() if hasattr(self._inner, "send_requests") else ()

    def receive_requests(self, messages):
        if hasattr(self._inner, "receive_requests"):
            self._inner.receive_requests(messages)

    def send_responses(self):
        return self._inner.send_responses() if hasattr(self._inner, "send_responses") else ()

    def receive_responses(self, messages):
        if hasattr(self._inner, "receive_responses"):
            self._inner.receive_responses(messages)

    # ----- internals -----

    def _base_slice_from_flat(self, observation: np.ndarray) -> np.ndarray:
        """Return the base-MATE prefix of a MATE-Hard observation."""
        flat = np.asarray(observation, dtype=np.float64).reshape(-1)
        nC = int(np.round(flat[0]))
        nT = int(np.round(flat[1]))
        nO = int(np.round(flat[2]))
        base_dim = _base_obs_dim(nC, nT, nO)
        if flat.shape[0] < base_dim:
            raise ValueError(
                f"observation too short ({flat.shape[0]}) for base MATE dim "
                f"{base_dim}; flat[:4]={flat[:4]}"
            )
        return flat[:base_dim]


class RuleBasedInformedMateHard(RuleBasedNaiveMateHard):
    """Like the naive adapter, plus:

      - If the camera's energy is below ``low_energy_threshold`` OR the env
        has marked the camera idle, the action is forced to zero (matching
        the EnergyConstraint forced-idle behaviour). This avoids the agent
        attempting to slew while depleted.
      - If the strongest fog patch covers more than ``fog_intensity_threshold``
        of the agent's FoV (a coarse proxy), the agent reduces its zoom
        rate so it does not waste energy chasing occluded targets.

    These are intentionally simple heuristics: the goal is to expose a
    "rule-based with MATE-Hard awareness" upper bound, not to engineer the
    optimal solution.
    """

    def __init__(
        self,
        inner: CameraAgentBase,
        num_fogs: int,
        low_energy_threshold: float = 0.15,
        fog_intensity_threshold: float = 0.6,
    ) -> None:
        super().__init__(inner, num_fogs=num_fogs)
        self.low_energy_threshold = float(low_energy_threshold)
        self.fog_intensity_threshold = float(fog_intensity_threshold)

    def clone(self) -> "RuleBasedInformedMateHard":
        return RuleBasedInformedMateHard(
            self._inner.clone(),
            num_fogs=self._num_fogs,
            low_energy_threshold=self.low_energy_threshold,
            fog_intensity_threshold=self.fog_intensity_threshold,
        )

    def act(self, observation: np.ndarray, info: Optional[dict] = None,
            deterministic: Optional[bool] = None) -> np.ndarray:
        # Run the base agent on the base-MATE slice.
        action = super().act(observation, info=info, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float64).copy()

        # Pull MATE-Hard augmentation fields.
        flat = np.asarray(observation, dtype=np.float64).reshape(-1)
        nC = int(np.round(flat[0]))
        nT = int(np.round(flat[1]))
        nO = int(np.round(flat[2]))
        try:
            parts = _split_mate_hard_obs(flat, nC, nT, nO, self._num_fogs)
        except ValueError:
            return action

        # Energy-aware: idle if depleted.
        energy = float(parts["energy_aug"][0]) if parts["energy_aug"].size else 1.0
        idle_flag = float(parts["energy_aug"][2]) if parts["energy_aug"].size >= 3 else 0.0
        if energy < self.low_energy_threshold or idle_flag > 0.5:
            return np.zeros_like(action)

        # Fog-aware: if any fog patch is intense and close, halve zoom-rate.
        if parts["fog"].size:
            max_intensity = float(parts["fog"][:, 3].max()) if parts["fog"].shape[0] else 0.0
            if max_intensity > self.fog_intensity_threshold:
                action[1] = action[1] * 0.5

        return action
