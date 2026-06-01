from typing import Any

from .transforms import Transform, EnvWithTransform

from gymnasium.envs.registration import EnvSpec

import gymnasium as gym

from gym_agent.core.vec_env.subproc_vec_env import SubprocVecEnv
from gym_agent.core.vec_env.dummy_vec_env import DummyVecEnv

def make(
        id: str | EnvSpec,
        max_episode_steps: int | None = None,
        disable_env_checker: bool | None = None,
        observation_transform: Transform = None,
        action_transform: Transform = None,
        reward_transform: Transform = None,
        **kwargs: Any,
    ):
    """
    Creates an gymnasium environment with the specified id and wraps it with EnvTransform.

    To find all available environments use ``gymnasium.envs.registry.keys()`` for all valid ids.

    Args:
        id: A string for the environment id or a :class:`EnvSpec`. Optionally if using a string, a module to import can be included, e.g. ``'module:Env-v0'``.
            This is equivalent to importing the module first to register the environment followed by making the environment.
        max_episode_steps: Maximum length of an episode, can override the registered :class:`EnvSpec` ``max_episode_steps``.
            The value is used by :class:`gymnasium.wrappers.TimeLimit`.
            converts the environment step from a done bool to return termination and truncation bools.
            By default, the argument is None in which the :class:`EnvSpec` ``apply_api_compatibility`` is used, otherwise this variable is used in favor.
        disable_env_checker: If to add :class:`gymnasium.wrappers.PassiveEnvChecker`, ``None`` will default to the
            :class:`EnvSpec` ``disable_env_checker`` value otherwise use this value will be used.
        kwargs: Additional arguments to pass to the environment constructor.

    Returns:
        An instance of the environment with wrappers applied.

    Raises:
        Error: If the ``id`` doesn't exist in the :attr:`registry`
    """

    return EnvWithTransform(
        gym.make(id, max_episode_steps, disable_env_checker, **kwargs),
        observation_transform,
        action_transform,
        reward_transform
    )

def make_vec(
        id: str | EnvSpec,
        num_envs: int = 1,
        max_episode_steps: int | None = None,
        disable_env_checker: bool = True,
        observation_transform: Transform = None,
        action_transform: Transform = None,
        reward_transform: Transform = None,
        vectorization_mode: str = "async",
        vector_kwargs: dict[str, Any] | None = None,
        **kwargs: Any):

    if num_envs <= 0:
        raise ValueError("num_envs must be greater than 0")

    if vector_kwargs is None:
        vector_kwargs = {}

    def env_fn():
        return make(
            id,
            max_episode_steps,
            disable_env_checker,
            observation_transform,
            action_transform,
            reward_transform,
            **kwargs
        )

    envs = [env_fn for _ in range(num_envs)]

    if vectorization_mode == 'async':
        return SubprocVecEnv(envs, **vector_kwargs)
    else:
        return DummyVecEnv(envs, **vector_kwargs)
