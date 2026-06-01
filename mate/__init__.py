"""MATE: The Multi-Agent Tracking Environment."""

import os

# import gymnasium as gym
import gymnasium as gym

from mate import agents, constants, environment, utils, wrappers
from mate.agents import (
    GreedyCameraAgent,
    GreedyTargetAgent,
    HeuristicCameraAgent,
    HeuristicTargetAgent,
    RandomCameraAgent,
    RandomTargetAgent,
)
from mate.environment import ASSETS_DIR, MultiAgentTracking
from mate.wrappers import (
    MultiCamera,
    MultiTarget,
    SingleCamera,
    SingleTarget,
    WrapperSpec,
    group_reset,
    group_step,
    group_act,
    group_communicate,
    group_observe
)

__all__ = [
    "make_environment",
    "MultiCamera",
    "MultiTarget",
    "SingleCamera",
    "SingleTarget",
    "GreedyCameraAgent",
    "GreedyTargetAgent",
    "RandomCameraAgent",
    "RandomTargetAgent",
    "HeuristicCameraAgent",
    "HeuristicTargetAgent",

    "group_reset",
    "group_step",
    "group_act",
    "group_communicate",
    "group_observe",
]
__all__.extend(constants.__all__)
__all__.extend(environment.__all__)
__all__.extend(wrappers.__all__)
__all__.extend(agents.__all__)
__all__.extend(utils.__all__)


def make_environment(config=None, wrappers=(), **kwargs):  # pylint: disable=redefined-outer-name
    """Helper function for creating a wrapped environment."""
    import gymnasium as gym  # Ensure gym is defined in this scope

    env = MultiAgentTracking(config, **kwargs)

    for wrapper in wrappers:
        assert (
            isinstance(wrapper, WrapperSpec)
            or callable(wrapper)
            or issubclass(wrapper, gym.Wrapper)
        ), (
            f"You should provide a wrapper class or an instance of `mate.WrapperSpec`. "
            f"Got wrapper = {wrapper!r}."
        )
        env = wrapper(env)

    return env


gym.register(
    id="MultiAgentTracking-v0", entry_point=make_environment, disable_env_checker=True
)
gym.register(id="MATE-v0", entry_point=make_environment, disable_env_checker=True)

gym.register(
    id="MATE-4v2-9-v0",
    entry_point=make_environment,
    kwargs={"config": (ASSETS_DIR / "MATE-4v2-9.yaml")},
    disable_env_checker=True,
)

gym.register(
    id="MATE-4v2-0-v0",
    entry_point=make_environment,
    kwargs={"config": (ASSETS_DIR / "MATE-4v2-0.yaml")},
    disable_env_checker=True,
)

gym.register(
    id="MATE-4v4-9-v0",
    entry_point=make_environment,
    kwargs={"config": (ASSETS_DIR / "MATE-4v4-9.yaml")},
    disable_env_checker=True,
)

gym.register(
    id="MATE-4v4-0-v0",
    entry_point=make_environment,
    kwargs={"config": (ASSETS_DIR / "MATE-4v4-0.yaml")},
    disable_env_checker=True,
)

gym.register(
    id="MATE-4v8-9-v0",
    entry_point=make_environment,
    kwargs={"config": (ASSETS_DIR / "MATE-4v8-9.yaml")},
    disable_env_checker=True,
)

gym.register(
    id="MATE-4v8-0-v0",
    entry_point=make_environment,
    kwargs={"config": (ASSETS_DIR / "MATE-4v8-0.yaml")},
    disable_env_checker=True,
)

gym.register(
    id="MATE-8v8-9-v0",
    entry_point=make_environment,
    kwargs={"config": (ASSETS_DIR / "MATE-8v8-9.yaml")},
    disable_env_checker=True,
)

gym.register(
    id="MATE-8v8-0-v0",
    entry_point=make_environment,
    kwargs={"config": (ASSETS_DIR / "MATE-8v8-0.yaml")},
    disable_env_checker=True,
)

gym.register(
    id="MATE-Navigation-v0",
    entry_point=make_environment,
    kwargs={"config": (ASSETS_DIR / "MATE-Navigation.yaml")},
    disable_env_checker=True,
)


del os, gym
