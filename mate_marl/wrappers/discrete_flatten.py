"""Adapter for DQN-MARL: like FlattenAgentsForPPO but exposes a per-camera
discrete action_space.

The DiscreteCamera wrapper at the BASE env level already converts continuous
camera actions to Discrete(levels * levels). After MultiCamera + the rest of
our stack, the env returns scalar reward for the camera team. This wrapper:

  - Sets ``single_action_space = Discrete(levels * levels)`` per camera.
  - Sets ``action_space = MultiDiscrete([n] * num_cameras)`` for the
    full team — but the trainer treats the leading num_cameras as the
    batch dim with parameter sharing.
  - Returns per-agent reward (broadcast team reward) + per-agent
    (terminated, truncated).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DiscreteFlattenForDQN(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        reward_shaping: bool = False,
        reward_weights: Optional[dict] = None,
    ) -> None:
        super().__init__(env)
        u = self.unwrapped
        self.num_cameras = u.num_cameras

        # The action_space on the underlying SingleTeamMultiAgent is set from
        # the *wrapped* env (post-DiscreteCamera). It is a Tuple of
        # Discrete(N) per camera. We need to walk down to find that.
        single = self._find_discrete_camera_space()
        if single is None:
            raise AssertionError(
                "DiscreteFlattenForDQN: could not locate a Discrete "
                "camera_action_space in the wrapper chain. Did you forget "
                "to apply mate.DiscreteCamera before MultiCamera?"
            )
        self.single_action_space = single
        self.n = int(single.n)
        self.action_space = spaces.MultiDiscrete([self.n] * self.num_cameras)

        # MATE's MultiCamera strips intermediate wrappers (including
        # DiscreteCamera) and re-routes step() directly to MultiAgentTracking.
        # That means the discrete→continuous conversion DiscreteCamera was
        # supposed to do never fires. We replicate it here.
        from mate.wrappers.discrete_action_spaces import DiscreteCamera as _DC
        levels = int(round(self.n ** 0.5))
        assert levels * levels == self.n, (
            f"Discrete action count {self.n} is not a perfect square; "
            f"DiscreteCamera levels are square. n={self.n}"
        )
        self._action_grid = _DC.discrete_action_grid(levels=levels)  # (n, 2)
        cam = u.cameras[0]
        self._action_high = np.asarray(
            [cam.rotation_step, cam.zooming_step], dtype=np.float64
        )

        self.reward_shaping = bool(reward_shaping)
        from mate_marl.wrappers.flatten_for_ppo import FlattenAgentsForPPO
        self.reward_weights = (
            dict(reward_weights)
            if reward_weights is not None
            else dict(FlattenAgentsForPPO.DEFAULT_REWARD_WEIGHTS)
        )

    def _find_discrete_camera_space(self):
        """Walk the wrapper chain looking for a Discrete per-camera action.

        ``MultiCamera`` strips intermediate wrappers when locating the base
        env, so a ``camera_action_space`` attribute on ``DiscreteCamera``
        becomes invisible after wrapping. Instead we inspect each wrapper's
        ``action_space``: when it's a ``spaces.Tuple`` whose elements are
        ``Discrete`` (the post-MultiCamera discrete-team layout) we have it.
        Also accept a per-camera ``camera_action_space`` if present.
        """
        cur = self.env
        while cur is not None:
            cas = getattr(cur, "camera_action_space", None)
            if isinstance(cas, spaces.Discrete):
                return cas
            asp = getattr(cur, "action_space", None)
            if isinstance(asp, spaces.Tuple) and len(asp.spaces) > 0:
                first = asp.spaces[0]
                if isinstance(first, spaces.Discrete):
                    return first
            cur = getattr(cur, "env", None)
        return None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, self._info_dict(info)

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape[0] != self.num_cameras:
            raise RuntimeError(
                f"DiscreteFlattenForDQN: expected action of shape "
                f"({self.num_cameras},), got {action.shape}"
            )
        # Discrete index → continuous (rotation, zoom) per camera.
        cont = self._action_grid[action] * self._action_high  # (num_cameras, 2)
        obs, reward, terminated, truncated, info = self.env.step(cont)
        per_agent_reward = self._compose_reward(reward, info)
        per_agent_term = np.full(self.num_cameras, bool(terminated), dtype=np.bool_)
        per_agent_trunc = np.full(self.num_cameras, bool(truncated), dtype=np.bool_)
        return obs, per_agent_reward, per_agent_term, per_agent_trunc, self._info_dict(info)

    # ---- reward composition (same as FlattenAgentsForPPO) ----

    def _compose_reward(self, raw_reward, info) -> np.ndarray:
        if isinstance(info, list) and info and isinstance(info[0], dict):
            team_val = float(info[0].get("normalized_raw_reward", raw_reward))
        else:
            team_val = float(raw_reward)

        if not self.reward_shaping:
            return np.full(self.num_cameras, team_val, dtype=np.float32)

        w = self.reward_weights
        out = np.full(self.num_cameras, w["team"] * team_val, dtype=np.float64)
        u = self.unwrapped
        if w.get("soft_coverage_score", 0.0) != 0.0:
            try:
                from mate.wrappers.auxiliary_camera_rewards import AuxiliaryCameraRewards
                scs = AuxiliaryCameraRewards.compute_soft_coverage_scores(u)
                vm = u.camera_target_view_mask
                for c in range(self.num_cameras):
                    if vm[c].any():
                        out[c] += w["soft_coverage_score"] * float(scs[c, vm[c]].sum())
                    else:
                        out[c] += w["soft_coverage_score"] * float(np.tanh(scs[c, :].max()))
            except Exception:
                pass
        if w.get("num_tracked", 0.0) != 0.0:
            vm = u.camera_target_view_mask
            for c in range(self.num_cameras):
                out[c] += w["num_tracked"] * float(vm[c].sum())
        return out.astype(np.float32)

    def _info_dict(self, info) -> dict:
        if isinstance(info, list):
            agg = {}
            for k in info[0].keys() if info else []:
                vals = [d.get(k) for d in info]
                try:
                    agg[k] = np.asarray(vals)
                except Exception:
                    agg[k] = vals
            return agg
        return dict(info) if info else {}
