"""Adapter that exposes a multi-agent MATE env as if it were single-agent.

We also compute MATE's ``soft_coverage_score`` directly here when reward
shaping is enabled — it is the per-camera, per-step continuous quantity
that gives PPO a gradient-rich training signal. We re-use the static helper
``mate.wrappers.AuxiliaryCameraRewards.compute_soft_coverage_scores`` for
the math, but bypass that wrapper's RepeatedRewardIndividualDone assertion
(which is incompatible with our pipeline).


The outer MATE env (after MultiCamera + MateMARLDictObs) returns:
  - obs: dict of (num_cameras, ...) tensors
  - reward: scalar float (team reward)
  - terminated/truncated: bool
  - info: list of length num_cameras

We re-shape it so a parameter-shared PPO trainer sees a "single env" of
batch size num_cameras at every step:

  - obs: dict of (num_cameras, ...) tensors  [unchanged in shape]
  - action: expects (num_cameras, action_dim)
  - reward: vector of length num_cameras  (every agent receives the team reward)
  - terminated/truncated: vector of bools

This is intentionally simple. The MAPPO trainer treats each camera as one
"sample" in the batch and shares the policy weights across cameras. The
centralized critic operates on the FULL set of camera tokens (it sees all
agents' "self" entries plus the shared global tokens) — so reward attribution
flows through the shared value baseline rather than per-agent shaping.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlattenAgentsForPPO(gym.Wrapper):
    """Expose a per-agent action_space and per-agent reward for MAPPO.

    The wrapped env's ``observation_space`` is *unchanged* — it is already a
    Dict whose values have leading dim = num_cameras. The trainer is
    responsible for treating the leading dim as the batch dim.

    Reward composition (per camera, broadcast):

        r_t = w_team * normalized_raw_reward
            + w_cov  * coverage_rate
            + w_real * real_coverage_rate
            - w_trans * mean_transport_rate

    All four components live in MATE's per-step ``info[i]`` dict, so the
    shaping is dense and per-agent informative without modifying the
    underlying env. Setting ``reward_shaping=False`` disables the shaping
    terms — useful for the "vanilla reward" ablation in the paper.
    """

    DEFAULT_REWARD_WEIGHTS = {
        "team": 1.0,
        # Shaping weight previously 0.10 → caused Q-learning to over-optimize
        # the shaping term and converge on a sub-optimal greedy policy
        # (mean_ret regression -76 → -80 as ε dropped). Lowered to 0.02 so the
        # shaping merely densifies the gradient without dominating the team
        # objective.
        "soft_coverage_score": 0.02,
        "num_tracked": 0.0,
        "coverage_rate": 0.0,
        "real_coverage_rate": 0.0,
        "mean_transport_rate": 0.0,
    }

    def __init__(
        self,
        env: gym.Env,
        reward_shaping: bool = False,
        reward_weights: Optional[dict] = None,
    ) -> None:
        super().__init__(env)

        u = self.unwrapped
        self.num_cameras = u.num_cameras

        self.reward_shaping = bool(reward_shaping)
        self.reward_weights = (
            dict(reward_weights)
            if reward_weights is not None
            else dict(self.DEFAULT_REWARD_WEIGHTS)
        )

        # Per-agent action space (the underlying camera_action_space).
        single_low = u.camera_action_space.low
        single_high = u.camera_action_space.high
        self.single_action_space = u.camera_action_space
        self.action_space = spaces.Box(
            low=np.tile(single_low, (self.num_cameras, 1)),
            high=np.tile(single_high, (self.num_cameras, 1)),
            dtype=np.float64,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, self._info_dict(info)

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        per_agent_reward = self._compose_reward(reward, info)
        per_agent_term = np.full(self.num_cameras, bool(terminated), dtype=np.bool_)
        per_agent_trunc = np.full(self.num_cameras, bool(truncated), dtype=np.bool_)
        return (
            obs,
            per_agent_reward,
            per_agent_term,
            per_agent_trunc,
            self._info_dict(info),
        )

    def _compose_reward(self, raw_reward, info) -> np.ndarray:
        """Build a per-camera shaped reward vector, shape (num_cameras,)."""
        # Resolve the team reward (shared across cameras).
        team_val = 0.0
        if isinstance(info, list) and info and isinstance(info[0], dict):
            team_val = float(info[0].get("normalized_raw_reward", raw_reward))
        else:
            team_val = float(raw_reward)

        if not self.reward_shaping:
            return np.full(self.num_cameras, team_val, dtype=np.float32)

        w = self.reward_weights
        out = np.full(self.num_cameras, w["team"] * team_val, dtype=np.float64)

        # Add per-camera continuous signals computed from the unwrapped env.
        u = self.unwrapped
        if w.get("soft_coverage_score", 0.0) != 0.0:
            try:
                from mate.wrappers.auxiliary_camera_rewards import (
                    AuxiliaryCameraRewards,
                )
                # (num_cameras, num_targets) score matrix
                scs_matrix = AuxiliaryCameraRewards.compute_soft_coverage_scores(u)
                view_mask = u.camera_target_view_mask  # (num_cameras, num_targets)
                for c in range(self.num_cameras):
                    if view_mask[c].any():
                        out[c] += w["soft_coverage_score"] * float(
                            scs_matrix[c, view_mask[c]].sum()
                        )
                    else:
                        # No tracked targets: weak negative shaping based on
                        # closest-target distance (already negated when not tracked).
                        out[c] += w["soft_coverage_score"] * float(
                            np.tanh(scs_matrix[c, :].max())
                        )
            except Exception:
                # Soft-fail if MATE's static helper can't compute scores
                # (e.g. on the very first step before camera boundaries are
                # populated). The team reward still drives learning.
                pass

        if w.get("num_tracked", 0.0) != 0.0:
            view_mask = u.camera_target_view_mask
            for c in range(self.num_cameras):
                out[c] += w["num_tracked"] * float(view_mask[c].sum())

        # Team-shared shaping (added equally to every camera).
        if isinstance(info, list) and info and isinstance(info[0], dict):
            d = info[0]
            if w.get("coverage_rate", 0.0):
                out += w["coverage_rate"] * float(d.get("coverage_rate", 0.0))
            if w.get("real_coverage_rate", 0.0):
                out += w["real_coverage_rate"] * float(d.get("real_coverage_rate", 0.0))
            if w.get("mean_transport_rate", 0.0):
                out -= w["mean_transport_rate"] * float(d.get("mean_transport_rate", 0.0))

        return out.astype(np.float32)

    def _info_dict(self, info) -> dict:
        """Aggregate per-agent infos into a single dict, keeping per-agent arrays."""
        if isinstance(info, list):
            agg = {}
            for k in info[0].keys() if info else []:
                vals = [d.get(k) for d in info]
                # Stack scalars into arrays; keep object dtype if non-numeric.
                try:
                    agg[k] = np.asarray(vals)
                except Exception:
                    agg[k] = vals
            return agg
        return dict(info) if info else {}
