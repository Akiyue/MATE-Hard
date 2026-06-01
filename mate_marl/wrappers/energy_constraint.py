"""Energy-budget wrapper.

Each camera has a finite battery in [0, 1]. Drain per step depends on:

  - operating cost (idle drain, scales with current viewing angle / area),
  - slewing cost (scales with the magnitude of the action delta),
  - zoom-change cost (scales with viewing-angle change).

When energy reaches zero the camera is forced into IDLE: its action is zeroed
out and the viewing angle decays toward MAX_CAMERA_VIEWING_ANGLE (cheapest
state). Cameras inside any warehouse circle recharge at a fixed rate per step.

Augmentation:
  - obs += [energy, charging_flag, idle_flag] per camera (3 dims).
  - info[i]['energy'], info[i]['charging'], info[i]['idle'].

The wrapper modifies the action passed to env.step but does NOT modify the
underlying camera dynamics — it acts as an action-mask + obs-augmentation
layer.

Apply ON TOP OF HeterogeneousCameras (or any other per-camera obs wrapper).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mate import constants as consts


_ENERGY_AUG_DIM = 3  # [energy, charging, idle]


class EnergyConstraint(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        battery_capacity: float = 1.0,
        idle_drain: float = 1.0e-4,
        slew_drain_coef: float = 5.0e-4,
        zoom_drain_coef: float = 1.0e-3,
        recharge_rate: float = 5.0e-3,
        warehouse_recharge_radius_factor: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env)

        u = self.unwrapped
        self.num_cameras = u.num_cameras
        self.battery_capacity = float(battery_capacity)
        self.idle_drain = float(idle_drain)
        self.slew_drain_coef = float(slew_drain_coef)
        self.zoom_drain_coef = float(zoom_drain_coef)
        self.recharge_rate = float(recharge_rate)
        self.warehouse_recharge_radius_factor = float(warehouse_recharge_radius_factor)

        self._rng = np.random.default_rng(seed)

        self.energy = np.full(self.num_cameras, self.battery_capacity, dtype=np.float64)
        self.charging = np.zeros(self.num_cameras, dtype=np.bool_)
        self.idle = np.zeros(self.num_cameras, dtype=np.bool_)

        # Augment observation space.
        base_low = self.env.observation_space.low
        base_high = self.env.observation_space.high
        aug_low = np.zeros((self.num_cameras, _ENERGY_AUG_DIM), dtype=np.float64)
        aug_high = np.ones((self.num_cameras, _ENERGY_AUG_DIM), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, aug_low], axis=1),
            high=np.concatenate([base_high, aug_high], axis=1),
            dtype=np.float64,
        )

    def _camera_locations(self) -> np.ndarray:
        u = self.unwrapped
        return np.asarray([cam.location for cam in u.cameras], dtype=np.float64)

    def _update_charging(self) -> None:
        u = self.unwrapped
        warehouses = consts.WAREHOUSES  # (num_warehouses, 2)
        radius = consts.WAREHOUSE_RADIUS * self.warehouse_recharge_radius_factor
        cam_xy = self._camera_locations()
        # Distance from each camera to each warehouse.
        diffs = cam_xy[:, None, :] - warehouses[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        self.charging = (dists < radius).any(axis=1)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = self.env.reset(seed=seed, options=options)

        # Re-bind in case underlying env count changed.
        self.num_cameras = self.unwrapped.num_cameras
        self.energy = np.full(self.num_cameras, self.battery_capacity, dtype=np.float64)
        self.charging = np.zeros(self.num_cameras, dtype=np.bool_)
        self.idle = np.zeros(self.num_cameras, dtype=np.bool_)

        self._update_charging()
        obs = self._augment(obs)
        info = self._augment_info(info)
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).copy()
        if action.shape[0] != self.num_cameras:
            raise RuntimeError(
                f"EnergyConstraint: expected per-camera action of shape "
                f"({self.num_cameras}, *), got {action.shape}"
            )

        # Force idle for depleted cameras.
        depleted = self.energy <= 0.0
        if depleted.any():
            action[depleted] = 0.0
        self.idle = depleted.copy()

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_charging()

        # Drain: idle + slew + zoom.
        slew_mag = np.abs(action[:, 0])
        zoom_mag = np.abs(action[:, 1])
        drain = (
            self.idle_drain
            + self.slew_drain_coef * slew_mag
            + self.zoom_drain_coef * zoom_mag
        )
        # Apply drain only when not charging.
        drain[self.charging] = 0.0
        self.energy = self.energy - drain
        # Recharge at warehouses.
        self.energy[self.charging] = np.minimum(
            self.energy[self.charging] + self.recharge_rate, self.battery_capacity
        )
        np.clip(self.energy, 0.0, self.battery_capacity, out=self.energy)

        obs = self._augment(obs)
        info = self._augment_info(info)
        return obs, reward, terminated, truncated, info

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        cap = max(self.battery_capacity, 1e-8)
        aug = np.stack(
            [
                self.energy / cap,
                self.charging.astype(np.float64),
                self.idle.astype(np.float64),
            ],
            axis=1,
        )
        return np.concatenate([obs, aug], axis=1)

    def _augment_info(self, info):
        if isinstance(info, list):
            for i, d in enumerate(info):
                if isinstance(d, dict):
                    d["energy"] = float(self.energy[i])
                    d["charging"] = bool(self.charging[i])
                    d["idle"] = bool(self.idle[i])
        return info
