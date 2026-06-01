"""Dict-observation packer for the MARL flagship.

Layered after MultiCamera + HeterogeneousCameras + EnergyConstraint + DynamicFog,
this wrapper rearranges the flat per-camera observation into a ``spaces.Dict``
of fixed-shape tensors, one per entity type:

    {
        "self":      (C, F_self),       # own state + type one-hot + energy
        "teammate":  (C, C-1, F_team),  # other cameras (with mask)
        "target":    (C, T,   F_tgt),   # targets (with visibility mask)
        "obstacle":  (C, O,   F_obs),   # obstacles (with mask)
        "fog":       (C, F,   F_fog),   # dynamic fog tokens
        "preserved": (C, F_pre),        # warehouses + counts
        "self_type": (C,),              # int camera type id (for type-conditioned MoE)
    }

Per-entity feature shapes are chosen so each token has a per-token validity
flag in the LAST dimension (mask). The encoder uses these masks for attention.

This wrapper assumes the input observation has the following layout:

    [ base MATE camera obs ] [ type one-hot ] [ energy aug (3) ] [ fog tokens ]
                ^                  ^                ^                  ^
       camera_observation_slices  NUM_CAMERA_TYPES   3        num_fogs * 4

If you stack the wrappers in a different order, fix the slice math here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mate import constants as consts
from mate_marl.wrappers.heterogeneous_cameras import (
    NUM_CAMERA_TYPES,
    HeterogeneousCameras,
)
from mate_marl.wrappers.energy_constraint import EnergyConstraint, _ENERGY_AUG_DIM
from mate_marl.wrappers.dynamic_fog import DynamicFog, _FOG_TOKEN_DIM


# Per-token feature widths after rescaling. Each token ends with a mask flag.
F_SELF = 9 + NUM_CAMERA_TYPES + _ENERGY_AUG_DIM   # private cam state + type + energy
F_TEAMMATE = consts.CAMERA_STATE_DIM_PUBLIC + 1   # public state + mask
F_TARGET = consts.TARGET_STATE_DIM_PUBLIC + 1
F_OBSTACLE = consts.OBSTACLE_STATE_DIM + 1
F_FOG = _FOG_TOKEN_DIM + 1                         # x,y,sigma,intensity + mask
F_PRESERVED = consts.PRESERVED_DIM


class MateMARLDictObs(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

        # Discover sizes from the underlying env.
        u = self.unwrapped
        self.num_cameras = u.num_cameras
        self.num_targets = u.num_targets
        self.num_obstacles = u.num_obstacles
        self.num_warehouses = consts.NUM_WAREHOUSES

        # Slices into the *base* (pre-augmentation) MATE camera observation.
        self._slices = consts.camera_observation_slices_of(
            self.num_cameras, self.num_targets, self.num_obstacles
        )
        # Total dimension of the base MATE camera obs.
        self._base_dim = consts.camera_observation_indices_of(
            self.num_cameras, self.num_targets, self.num_obstacles
        )[-1]

        # Locate the energy block: appended right after type one-hot.
        # Layout: [ base ][ type_onehot:NUM_CAMERA_TYPES ][ energy:_ENERGY_AUG_DIM ][ fog tokens ]
        # Detect num_fogs from observation_space width.
        flat_dim = int(self.env.observation_space.shape[1])
        expected_no_fog = self._base_dim + NUM_CAMERA_TYPES + _ENERGY_AUG_DIM
        fog_block_dim = flat_dim - expected_no_fog
        if fog_block_dim < 0 or fog_block_dim % _FOG_TOKEN_DIM != 0:
            raise RuntimeError(
                f"MateMARLDictObs: cannot infer fog layout. flat_dim={flat_dim}, "
                f"expected_no_fog={expected_no_fog}, residual={fog_block_dim}"
            )
        self.num_fogs = fog_block_dim // _FOG_TOKEN_DIM

        # Build the dict observation space (per-camera; the env returns a
        # batch of size num_cameras for each key).
        terr = consts.TERRAIN_SIZE
        nC = self.num_cameras
        nT = self.num_targets
        nO = self.num_obstacles
        nF = self.num_fogs

        def _box(shape, low=-1.0, high=1.0, dtype=np.float32):
            return spaces.Box(
                low=np.full(shape, low, dtype=dtype),
                high=np.full(shape, high, dtype=dtype),
                dtype=dtype,
            )

        self.observation_space = spaces.Dict(
            {
                "self": _box((nC, F_SELF)),
                "teammate": _box((nC, max(nC - 1, 1), F_TEAMMATE)),
                "target": _box((nC, nT, F_TARGET)),
                "obstacle": _box((nC, nO, F_OBSTACLE)),
                "fog": _box((nC, nF, F_FOG)) if nF > 0 else _box((nC, 1, F_FOG)),
                "preserved": _box((nC, F_PRESERVED), low=-np.inf, high=np.inf),
                "self_type": spaces.Box(
                    low=0, high=NUM_CAMERA_TYPES - 1, shape=(nC,), dtype=np.int64
                ),
            }
        )

    # ---- gym API ----

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        flat_obs, info = self.env.reset(seed=seed, options=options)
        return self._pack(flat_obs), info

    def step(self, action):
        flat_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._pack(flat_obs), reward, terminated, truncated, info

    # ---- packing ----

    def _pack(self, flat_obs: np.ndarray) -> dict[str, np.ndarray]:
        """Convert (C, flat_dim) flat obs into a dict of typed token arrays."""
        if flat_obs.ndim != 2 or flat_obs.shape[0] != self.num_cameras:
            raise RuntimeError(
                f"MateMARLDictObs expects flat obs of shape "
                f"({self.num_cameras}, *), got {flat_obs.shape}"
            )

        obs = flat_obs.astype(np.float64, copy=False)
        terr = consts.TERRAIN_SIZE

        # 1. Preserved (warehouses + counts) — first PRESERVED_DIM dims.
        preserved = obs[:, self._slices["preserved_data"]].copy()  # (C, F_pre)

        # 2. Self state (private camera state) — 9 dims.
        self_state = obs[:, self._slices["self_state"]].copy()  # (C, 9)

        # 3. Targets with mask.
        op_slice = self._slices["opponent_states_with_mask"]
        targets = obs[:, op_slice].reshape(self.num_cameras, self.num_targets, -1)
        # Last column of each target token is the visibility flag (0/1).
        # Normalize x,y by TERRAIN_SIZE; r by TERRAIN_SIZE.
        targets = targets.copy()
        # public target state: (x, y, R, ...) — see TARGET_STATE_DIM_PUBLIC=4
        targets[..., 0:2] /= terr
        targets[..., 2] /= terr
        # leave the trailing mask flag as-is.

        # 4. Obstacles with mask.
        ob_slice = self._slices["obstacle_states_with_mask"]
        obstacles = obs[:, ob_slice].reshape(self.num_cameras, self.num_obstacles, -1).copy()
        obstacles[..., 0:2] /= terr
        obstacles[..., 2] /= terr

        # 5. Teammates with mask. Original layout includes self too; we drop self.
        tm_slice = self._slices["teammate_states_with_mask"]
        teammates = obs[:, tm_slice].reshape(self.num_cameras, self.num_cameras, -1).copy()
        # Drop the diagonal (self looking at self).
        if self.num_cameras > 1:
            mask_keep = ~np.eye(self.num_cameras, dtype=bool)
            teammates = teammates[mask_keep].reshape(
                self.num_cameras, self.num_cameras - 1, -1
            )
        else:
            # Pad with one zero token so encoder shape is consistent.
            teammates = np.zeros((1, 1, F_TEAMMATE), dtype=np.float64)
        teammates[..., 0:2] /= terr
        # Public camera state contains polar (Rcos, Rsin) at idx 3..4 — they
        # already have units of length; rescale.
        teammates[..., 3:5] /= terr

        # 6. Type one-hot block.
        type_oh = obs[
            :, self._base_dim : self._base_dim + NUM_CAMERA_TYPES
        ].copy()  # (C, NUM_CAMERA_TYPES)
        self_type_id = np.argmax(type_oh, axis=1).astype(np.int64)

        # 7. Energy aug block.
        energy_start = self._base_dim + NUM_CAMERA_TYPES
        energy_block = obs[:, energy_start : energy_start + _ENERGY_AUG_DIM].copy()  # (C,3)

        # 8. Fog tokens (broadcast same fog state to all cameras).
        fog_start = energy_start + _ENERGY_AUG_DIM
        if self.num_fogs > 0:
            fog_block = obs[
                :, fog_start : fog_start + self.num_fogs * _FOG_TOKEN_DIM
            ].reshape(self.num_cameras, self.num_fogs, _FOG_TOKEN_DIM).copy()
            # add mask=1 (fogs always visible globally)
            fog_mask = np.ones((self.num_cameras, self.num_fogs, 1), dtype=np.float64)
            fog = np.concatenate([fog_block, fog_mask], axis=-1)
        else:
            fog = np.zeros((self.num_cameras, 1, F_FOG), dtype=np.float64)

        # Self block: private state + type one-hot + energy aug.
        # Rescale self-state coordinates to roughly [-1,1].
        ss = self_state
        ss = ss.copy()
        # CAMERA_STATE_DIM_PRIVATE=9 layout: (x, y, r, Rcos, Rsin, theta, Rmax, phimax, thetamax)
        ss[:, 0:2] /= terr
        ss[:, 2] /= terr
        ss[:, 3:5] /= terr  # Rcos, Rsin
        ss[:, 5] /= 180.0   # theta deg
        ss[:, 6] /= terr    # Rmax
        ss[:, 7] /= 180.0   # phimax deg
        ss[:, 8] /= 180.0   # thetamax deg
        self_block = np.concatenate([ss, type_oh, energy_block], axis=1)  # (C, F_SELF)

        return {
            "self": self_block.astype(np.float32),
            "teammate": teammates.astype(np.float32),
            "target": targets.astype(np.float32),
            "obstacle": obstacles.astype(np.float32),
            "fog": fog.astype(np.float32),
            "preserved": preserved.astype(np.float32),
            "self_type": self_type_id.astype(np.int64),
        }
