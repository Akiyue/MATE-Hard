"""Dynamic fog patches.

A fixed number of moving 2-D Gaussian fog patches drift across the terrain.
Each patch has center (x, y), radius (1-sigma), velocity, and intensity ∈ [0,1].
A camera-to-target line that passes within ``sigma`` of any active patch is
*occluded* with probability proportional to the patch's intensity.

This wrapper:
  - Re-evaluates target visibility AFTER the env step, applying fog occlusion
    to ``opponent_states_with_mask`` and ``opponent_mask`` slots in the obs.
  - Appends a fixed-size fog token array to the observation:
      [num_fogs, 4]  (x, y, sigma, intensity)  per camera
    (Same fog state is broadcast to every camera, but laid out per-camera so
    downstream encoders can consume it as a token set.)

Apply ON TOP OF EnergyConstraint (or any wrapper that produces flat per-camera
obs). The wrapper relies on ``mate.constants.camera_observation_slices_of`` to
locate the opponent (target) sub-observation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mate import constants as consts


_FOG_TOKEN_DIM = 4  # [x, y, sigma, intensity]


class DynamicFog(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        num_fogs: int = 4,
        sigma_range: tuple[float, float] = (120.0, 220.0),
        velocity_range: tuple[float, float] = (3.0, 9.0),
        intensity_range: tuple[float, float] = (0.4, 0.9),
        respawn_prob: float = 0.005,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env)

        u = self.unwrapped
        self.num_cameras = u.num_cameras
        self.num_targets = u.num_targets
        self.num_obstacles = u.num_obstacles
        self.num_fogs = int(num_fogs)
        self.sigma_range = sigma_range
        self.velocity_range = velocity_range
        self.intensity_range = intensity_range
        self.respawn_prob = float(respawn_prob)

        self._rng = np.random.default_rng(seed)

        # State arrays.
        self.fog_xy = np.zeros((self.num_fogs, 2), dtype=np.float64)
        self.fog_v = np.zeros((self.num_fogs, 2), dtype=np.float64)
        self.fog_sigma = np.zeros(self.num_fogs, dtype=np.float64)
        self.fog_intensity = np.zeros(self.num_fogs, dtype=np.float64)

        # Pre-compute slices.
        self._slices = consts.camera_observation_slices_of(
            self.num_cameras, self.num_targets, self.num_obstacles
        )

        # Augment obs space: append (num_fogs * _FOG_TOKEN_DIM) per camera.
        base_low = self.env.observation_space.low
        base_high = self.env.observation_space.high
        terr = consts.TERRAIN_SIZE
        token_low = np.tile(
            np.array([-terr, -terr, 0.0, 0.0], dtype=np.float64), (self.num_cameras, self.num_fogs)
        )
        token_high = np.tile(
            np.array([terr, terr, max(self.sigma_range), 1.0], dtype=np.float64),
            (self.num_cameras, self.num_fogs),
        )
        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, token_low], axis=1),
            high=np.concatenate([base_high, token_high], axis=1),
            dtype=np.float64,
        )

    @property
    def fog_token_dim(self) -> int:
        return _FOG_TOKEN_DIM

    def _spawn_fog(self, idx: int) -> None:
        terr = consts.TERRAIN_SIZE
        self.fog_xy[idx] = self._rng.uniform(-terr, terr, size=2)
        ang = self._rng.uniform(-np.pi, np.pi)
        speed = self._rng.uniform(*self.velocity_range)
        self.fog_v[idx] = np.array([np.cos(ang), np.sin(ang)]) * speed
        self.fog_sigma[idx] = self._rng.uniform(*self.sigma_range)
        self.fog_intensity[idx] = self._rng.uniform(*self.intensity_range)

    def _spawn_all_fogs(self) -> None:
        for i in range(self.num_fogs):
            self._spawn_fog(i)

    def _step_fog(self) -> None:
        terr = consts.TERRAIN_SIZE
        self.fog_xy += self.fog_v
        # Bounce off boundaries.
        oob_lo = self.fog_xy < -terr
        oob_hi = self.fog_xy > terr
        if oob_lo.any():
            self.fog_v[oob_lo[:, 0], 0] = np.abs(self.fog_v[oob_lo[:, 0], 0])
            self.fog_v[oob_lo[:, 1], 1] = np.abs(self.fog_v[oob_lo[:, 1], 1])
            np.clip(self.fog_xy, -terr, terr, out=self.fog_xy)
        if oob_hi.any():
            self.fog_v[oob_hi[:, 0], 0] = -np.abs(self.fog_v[oob_hi[:, 0], 0])
            self.fog_v[oob_hi[:, 1], 1] = -np.abs(self.fog_v[oob_hi[:, 1], 1])
            np.clip(self.fog_xy, -terr, terr, out=self.fog_xy)

        # Stochastic respawn (changes weather pattern over the long horizon).
        if self.num_fogs > 0:
            mask = self._rng.random(self.num_fogs) < self.respawn_prob
            for idx in np.flatnonzero(mask):
                self._spawn_fog(int(idx))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self.num_cameras = self.unwrapped.num_cameras
        self.num_targets = self.unwrapped.num_targets
        self.num_obstacles = self.unwrapped.num_obstacles
        self._slices = consts.camera_observation_slices_of(
            self.num_cameras, self.num_targets, self.num_obstacles
        )
        self._spawn_all_fogs()
        obs = self._apply_fog(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_fog()
        obs = self._apply_fog(obs)
        info = self._augment_info(info)
        return obs, reward, terminated, truncated, info

    # ---- core fog occlusion logic ----

    def _camera_xy(self) -> np.ndarray:
        u = self.unwrapped
        return np.asarray([cam.location for cam in u.cameras], dtype=np.float64)

    def _occlusion_prob(self, cam_xy: np.ndarray, tgt_xy: np.ndarray) -> np.ndarray:
        """Per (camera, target) probability of occlusion from any fog patch.

        Approximates each fog as a soft-edged disk; if the *line segment* from
        camera to target passes within ``sigma`` of the fog center, occlusion
        probability scales with the patch intensity and proximity.

        Returns array of shape (num_cameras, num_targets) with values in [0,1].
        """
        if self.num_fogs == 0:
            return np.zeros((cam_xy.shape[0], tgt_xy.shape[0]), dtype=np.float64)

        # For each camera-target pair, compute closest distance from fog center
        # to the line segment.
        c = cam_xy[:, None, None, :]  # (C,1,1,2)
        t = tgt_xy[None, :, None, :]  # (1,T,1,2)
        f = self.fog_xy[None, None, :, :]  # (1,1,F,2)
        seg = t - c  # (C,T,1,2)
        seg_len_sq = np.maximum(np.sum(seg * seg, axis=-1), 1e-8)  # (C,T,1)
        u = np.sum((f - c) * seg, axis=-1) / seg_len_sq  # (C,T,F)
        u_clamped = np.clip(u, 0.0, 1.0)[..., None]  # (C,T,F,1)
        closest = c + u_clamped * seg  # (C,T,F,2)
        d = np.linalg.norm(closest - f, axis=-1)  # (C,T,F)

        # Per-fog occlusion prob: intensity * exp(-(d/sigma)^2 / 2).
        sigma = self.fog_sigma[None, None, :]  # (1,1,F)
        intensity = self.fog_intensity[None, None, :]
        per_fog = intensity * np.exp(-0.5 * (d / np.maximum(sigma, 1e-6)) ** 2)

        # Combine independently across fogs: 1 - prod(1 - p_f).
        p = 1.0 - np.prod(1.0 - per_fog, axis=-1)  # (C,T)
        return p

    def _apply_fog(self, obs: np.ndarray) -> np.ndarray:
        u = self.unwrapped
        out = obs.astype(np.float64, copy=True)

        cam_xy = self._camera_xy()
        tgt_xy = np.asarray([t.location for t in u.targets], dtype=np.float64)

        occ = self._occlusion_prob(cam_xy, tgt_xy)  # (C,T) in [0,1]
        # Sample Bernoulli per (camera, target) for hard occlusion this step.
        rng = self._rng.random(occ.shape)
        occluded = rng < occ  # True = occluded

        op_slice = self._slices["opponent_states_with_mask"]
        for i in range(self.num_cameras):
            block = out[i, op_slice].reshape(self.num_targets, -1)
            mask_col = block.shape[1] - 1
            zero_rows = occluded[i]
            if zero_rows.any():
                block[zero_rows] = 0.0  # drop both state and flag
            out[i, op_slice] = block.ravel()

        # Append fog tokens, broadcast per camera.
        terr = consts.TERRAIN_SIZE
        token = np.empty((self.num_fogs, _FOG_TOKEN_DIM), dtype=np.float64)
        token[:, 0:2] = self.fog_xy / terr  # normalize to [-1,1]
        token[:, 2] = self.fog_sigma / terr
        token[:, 3] = self.fog_intensity
        token_flat = token.ravel()  # (num_fogs * 4,)
        token_per_cam = np.broadcast_to(
            token_flat, (self.num_cameras, token_flat.shape[0])
        )
        out = np.concatenate([out, token_per_cam], axis=1)
        return out

    def _augment_info(self, info):
        if isinstance(info, list):
            for d in info:
                if isinstance(d, dict):
                    d["fog_xy"] = self.fog_xy.copy()
                    d["fog_sigma"] = self.fog_sigma.copy()
                    d["fog_intensity"] = self.fog_intensity.copy()
        return info
