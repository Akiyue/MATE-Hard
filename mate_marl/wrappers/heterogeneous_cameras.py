"""Heterogeneous camera wrapper.

Assigns each camera one of three types with distinct sensing parameters:

  - WIDE_SHORT  : large min_viewing_angle, small max_sight_range. Good area
                  coverage at close range; weak at long range.
  - NARROW_LONG : small min_viewing_angle, large max_sight_range. Good
                  long-range tracking; narrow cone, slow to slew area.
  - OMNI_NOISY  : permanently large viewing angle, medium range, but observed
                  target positions are perturbed by Gaussian noise (cheap
                  omnidirectional sensor with poor angular accuracy).

Type assignment is deterministic from the camera index unless ``type_ids`` is
passed explicitly to ``reset(options=...)``.

The wrapper exposes:
  - env.unwrapped_camera_types : np.ndarray[int] of length num_cameras.
  - obs is augmented with a per-camera one-hot type vector (NUM_CAMERA_TYPES
    columns appended to the flat per-camera observation).
  - Sensing accuracy of OMNI_NOISY cameras is implemented by perturbing the
    target sub-observation with noise on each step.

This wrapper must be applied ON TOP OF a SingleTeam wrapper (e.g. MultiCamera)
so that the observation shape is (num_cameras, obs_dim).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mate import constants as consts


CAMERA_TYPE_NAMES = ("WIDE_SHORT", "NARROW_LONG", "OMNI_NOISY")
NUM_CAMERA_TYPES = len(CAMERA_TYPE_NAMES)

WIDE_SHORT = 0
NARROW_LONG = 1
OMNI_NOISY = 2


CAMERA_TYPE_PARAMS = {
    WIDE_SHORT: dict(
        min_viewing_angle=120.0,
        max_sight_range=350.0,
        rotation_step=8.0,
        zooming_step=4.0,
        target_obs_noise_std=0.0,
    ),
    NARROW_LONG: dict(
        min_viewing_angle=30.0,
        max_sight_range=750.0,
        rotation_step=4.0,
        zooming_step=2.0,
        target_obs_noise_std=0.0,
    ),
    OMNI_NOISY: dict(
        min_viewing_angle=180.0,
        max_sight_range=500.0,
        rotation_step=10.0,
        zooming_step=0.5,
        target_obs_noise_std=20.0,
    ),
}


class HeterogeneousCameras(gym.Wrapper):
    """Inject camera-type heterogeneity into MATE.

    Args:
        env: a SingleTeamMultiAgent (e.g. ``MultiCamera``) wrapped MATE env.
        type_assignment: how to assign types to cameras. Either:
            * ``"round_robin"`` (default) — i % NUM_CAMERA_TYPES.
            * ``"random"`` — uniform random per reset.
            * a ``Sequence[int]`` of length ``num_cameras`` — fixed assignment.
        seed: noise generator seed.
    """

    def __init__(
        self,
        env: gym.Env,
        type_assignment: str | Sequence[int] = "round_robin",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env)

        u = self.unwrapped
        self.num_cameras = u.num_cameras
        self.num_targets = u.num_targets
        self.num_obstacles = u.num_obstacles

        self.type_assignment = type_assignment
        self._rng = np.random.default_rng(seed)

        self.camera_types = self._assign_types(self.type_assignment)
        self._noise_std_per_camera = np.array(
            [CAMERA_TYPE_PARAMS[t]["target_obs_noise_std"] for t in self.camera_types],
            dtype=np.float64,
        )

        # MATE's MultiCamera declares observation_space as a Tuple of per-camera
        # Box spaces, but the actual obs returned from step/reset is a 2-D
        # numpy array of shape (num_cameras, obs_dim). We promote the Tuple to
        # a single tiled Box so downstream wrappers see consistent shapes.
        base_low, base_high = self._tiled_obs_bounds(self.env.observation_space)
        type_low = np.zeros((self.num_cameras, NUM_CAMERA_TYPES), dtype=np.float64)
        type_high = np.ones((self.num_cameras, NUM_CAMERA_TYPES), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, type_low], axis=1),
            high=np.concatenate([base_high, type_high], axis=1),
            dtype=np.float64,
        )

        # Pre-compute slices into the (pre-augmentation) observation.
        self._slices = consts.camera_observation_slices_of(
            self.num_cameras, self.num_targets, self.num_obstacles
        )

    @staticmethod
    def _tiled_obs_bounds(space: spaces.Space) -> tuple[np.ndarray, np.ndarray]:
        """Return (low, high) of shape (num_cameras, obs_dim) for either a
        Tuple-of-Box (MATE multi-team) or a Box (already-flattened) obs space."""
        if isinstance(space, spaces.Tuple):
            lows = np.stack([s.low.astype(np.float64) for s in space.spaces], axis=0)
            highs = np.stack([s.high.astype(np.float64) for s in space.spaces], axis=0)
            return lows, highs
        if isinstance(space, spaces.Box):
            return space.low.astype(np.float64), space.high.astype(np.float64)
        raise TypeError(f"Unsupported observation space type: {type(space).__name__}")

    def _assign_types(self, spec) -> np.ndarray:
        if isinstance(spec, str):
            if spec == "round_robin":
                return np.arange(self.num_cameras, dtype=np.int64) % NUM_CAMERA_TYPES
            if spec == "random":
                return self._rng.integers(
                    low=0, high=NUM_CAMERA_TYPES, size=self.num_cameras, dtype=np.int64
                )
            raise ValueError(f"unknown type_assignment string: {spec!r}")

        types = np.asarray(spec, dtype=np.int64)
        if types.shape != (self.num_cameras,):
            raise ValueError(
                f"type_assignment must have length {self.num_cameras}, got {types.shape}"
            )
        if (types < 0).any() or (types >= NUM_CAMERA_TYPES).any():
            raise ValueError(f"type_assignment values must be in [0, {NUM_CAMERA_TYPES})")
        return types

    @property
    def unwrapped_camera_types(self) -> np.ndarray:
        return self.camera_types.copy()

    def _apply_types_to_cameras(self) -> None:
        """Mutate the underlying Camera entities to match the assigned types."""
        u = self.unwrapped
        for i, cam in enumerate(u.cameras):
            params = CAMERA_TYPE_PARAMS[int(self.camera_types[i])]
            cam.min_viewing_angle = params["min_viewing_angle"]
            cam.max_sight_range = params["max_sight_range"]
            cam.rotation_step = params["rotation_step"]
            cam.zooming_step = params["zooming_step"]
            cam.viewing_angle = params["min_viewing_angle"]
            cam.area_product = params["min_viewing_angle"] * params["max_sight_range"] ** 2
            cam.sight_range = np.sqrt(cam.area_product / cam.viewing_angle)
            cam.action_space = spaces.Box(
                low=np.asarray([-cam.rotation_step, -cam.zooming_step]),
                high=np.asarray([cam.rotation_step, cam.zooming_step]),
                dtype=np.float64,
            )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Allow per-episode override via options.
        type_spec = self.type_assignment
        if options is not None and "camera_types" in options:
            type_spec = options["camera_types"]
        self.camera_types = self._assign_types(type_spec)
        self._noise_std_per_camera = np.array(
            [CAMERA_TYPE_PARAMS[t]["target_obs_noise_std"] for t in self.camera_types],
            dtype=np.float64,
        )

        obs, info = self.env.reset(seed=seed, options=options)
        self._apply_types_to_cameras()
        # The first joint_observation was computed BEFORE we mutated cameras; in
        # practice the env's joint_observation is recomputed lazily on step, but
        # to be safe we apply our augmentation now.
        obs = self._augment(obs)
        info = self._augment_info(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._augment(obs)
        info = self._augment_info(info)
        return obs, reward, terminated, truncated, info

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        """Inject per-camera type one-hot and add observation noise for OMNI_NOISY."""
        if obs.shape[0] != self.num_cameras:
            raise RuntimeError(
                f"HeterogeneousCameras expects per-camera obs (shape "
                f"({self.num_cameras}, *)), got {obs.shape}"
            )

        out = obs.astype(np.float64, copy=True)

        # Apply target-position noise for OMNI_NOISY cameras.
        op_slice = self._slices["opponent_states_with_mask"]
        op_mask_slice = self._slices["opponent_mask"]
        for i in range(self.num_cameras):
            std = float(self._noise_std_per_camera[i])
            if std <= 0.0:
                continue
            block = out[i, op_slice].reshape(self.num_targets, -1)
            mask = out[i, op_mask_slice]  # (num_targets,)
            # Only perturb visible targets (mask==1); unobserved are zero anyway.
            visible = mask > 0.5
            if visible.any():
                noise = self._rng.normal(
                    loc=0.0, scale=std, size=(int(visible.sum()), 2)
                )
                block[visible, 0:2] = block[visible, 0:2] + noise
            out[i, op_slice] = block.ravel()

        # Append per-camera one-hot type.
        type_onehot = np.zeros((self.num_cameras, NUM_CAMERA_TYPES), dtype=np.float64)
        type_onehot[np.arange(self.num_cameras), self.camera_types] = 1.0
        out = np.concatenate([out, type_onehot], axis=1)
        return out

    def _augment_info(self, info):
        if isinstance(info, list):
            for i, d in enumerate(info):
                if isinstance(d, dict):
                    d["camera_type"] = int(self.camera_types[i])
        elif isinstance(info, dict):
            info["camera_types"] = self.camera_types.copy()
        return info
