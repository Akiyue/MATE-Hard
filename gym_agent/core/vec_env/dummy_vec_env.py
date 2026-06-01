import warnings
from typing import Any, Callable, Optional, Sequence

import gymnasium as gym
import numpy as np

from gym_agent.core.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
    _maybe_cast_dict,
)


class DummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``Cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    actions: np.ndarray

    def __init__(self, env_fns: list[Callable[[], gym.Env]]):
        super().__init__(env_fns)
        self.envs = [env_fn() for env_fn in env_fns]
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You tried to create multiple environments, but the function to create them returned the same instance "
                "instead of creating different objects. "
                "You are probably using `make_vec_env(lambda: env)` or `DummyVecEnv([lambda: env] * n_envs)`. "
                "You should replace `lambda: env` by a `make_env` function that "
                "creates a new instance of the environment at every call "
                "(using `gym.make()` for instance). You can take a look at the documentation for an example. "
                "Please read https://github.com/DLR-RM/stable-baselines3/issues/1151 for more information."
            )

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        # Avoid circular imports

        stacked_obs = [None for _ in range(self.num_envs)]
        stacked_rews = [None for _ in range(self.num_envs)]
        stacked_terminated = [None for _ in range(self.num_envs)]
        stacked_truncated = [None for _ in range(self.num_envs)]
        stacked_infos = [None for _ in range(self.num_envs)]

        for env_idx in range(self.num_envs):
            (
                stacked_obs[env_idx],
                stacked_rews[env_idx],
                stacked_terminated[env_idx],
                stacked_truncated[env_idx],
                stacked_infos[env_idx],
            ) = self.envs[env_idx].step(self.actions[env_idx])

            if stacked_terminated[env_idx] or stacked_truncated[env_idx]:
                stacked_obs[env_idx], self.reset_infos[env_idx] = self.envs[
                    env_idx
                ].reset()

        return (
            stacked_obs,
            stacked_rews,
            stacked_terminated,
            stacked_truncated,
            stacked_infos,
        )

    def _reset(self) -> VecEnvObs:
        stacked_obs = [None for _ in range(self.num_envs)]
        for env_idx in range(self.num_envs):
            maybe_options = (
                {"options": self._options[env_idx]}
                if isinstance(self._options[env_idx], dict)
                else {}
            )
            stacked_obs[env_idx], self.reset_infos[env_idx] = self.envs[env_idx].reset(
                seed=self._seeds[env_idx], **maybe_options
            )

        return stacked_obs

    def reset(self) -> tuple[VecEnvObs, dict[str, Any]]:
        obs = self._reset()

        self._reset_seeds()
        self._reset_options()
        return _maybe_cast_dict(obs), _maybe_cast_dict(self.reset_infos)

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.envs]
        return [env.render() for env in self.envs]  # type: ignore[misc]

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.

        :param mode: The rendering type.
        """
        return super().render(mode=mode)
