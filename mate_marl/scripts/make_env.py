"""Builders for the MARL flagship env stack.

The stack (outermost first):

    FlattenAgentsForPPO
      MateMARLDictObs
        DynamicFog
          EnergyConstraint
            HeterogeneousCameras
              MultiCamera (mate.MultiCamera)
                MultiAgentTracking ('MultiAgentTracking-v0')

The factory accepts a config-name (e.g. "MATE-4v8-9.yaml") and assembles the
stack with sensible defaults. Pass overrides via kwargs.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym

import mate
from mate.agents import GreedyTargetAgent

from mate_marl.wrappers import (
    HeterogeneousCameras,
    EnergyConstraint,
    DynamicFog,
    MateMARLDictObs,
    FlattenAgentsForPPO,
)


def make_marl_env(
    mate_config: str = "MATE-4v8-9.yaml",
    target_agent=None,
    type_assignment: str = "round_robin",
    enable_heterogeneous: bool = True,
    enable_energy: bool = True,
    enable_fog: bool = True,
    num_fogs: int = 4,
    reward_shaping: bool = False,
    seed: Optional[int] = None,
    **mate_env_kwargs,
) -> gym.Env:
    """Build a fully-wrapped flagship MARL env.

    Args:
        mate_config: filename of an asset YAML in mate/assets/.
        target_agent: opponent for the target team. Defaults to GreedyTargetAgent.
        type_assignment: how to assign camera types — see HeterogeneousCameras.
        enable_*: toggle each contribution wrapper for ablations.
        num_fogs: number of dynamic fog patches.
        seed: passed to wrappers that need their own RNG.
    """
    base = gym.make("MultiAgentTracking-v0", config=mate_config, **mate_env_kwargs)
    if target_agent is None:
        target_agent = GreedyTargetAgent()
    env = mate.MultiCamera.make(base, target_agent=target_agent)

    if enable_heterogeneous:
        env = HeterogeneousCameras(env, type_assignment=type_assignment, seed=seed)
    else:
        # Keep one-hot column shape stable so MateMARLDictObs works — pass a
        # zero-effect HeterogeneousCameras wrapper that always assigns type 0.
        env = HeterogeneousCameras(env, type_assignment=[0] * env.unwrapped.num_cameras)

    if enable_energy:
        env = EnergyConstraint(env, seed=seed)
    else:
        # Use a no-op energy: never drains, never charges.
        env = EnergyConstraint(
            env,
            idle_drain=0.0,
            slew_drain_coef=0.0,
            zoom_drain_coef=0.0,
            recharge_rate=0.0,
            seed=seed,
        )

    if enable_fog:
        env = DynamicFog(env, num_fogs=num_fogs, seed=seed)
    else:
        env = DynamicFog(
            env,
            num_fogs=0 if num_fogs == 0 else num_fogs,
            intensity_range=(0.0, 0.0),
            seed=seed,
        )

    env = MateMARLDictObs(env)
    env = FlattenAgentsForPPO(env, reward_shaping=reward_shaping)
    return env


def make_mate_hard_baseline_env(
    mate_config: str = "MATE-4v8-9.yaml",
    target_agent=None,
    type_assignment: str = "round_robin",
    enable_heterogeneous: bool = True,
    enable_energy: bool = True,
    enable_fog: bool = True,
    num_fogs: int = 4,
    seed: Optional[int] = None,
    **mate_env_kwargs,
) -> gym.Env:
    """Build a MATE-Hard env for **rule-based** agents.

    Same wrapper stack as ``make_marl_env`` up to and including the
    environment modifications, but DOES NOT apply ``MateMARLDictObs`` or
    ``FlattenAgentsForPPO`` — the env returns a flat per-camera observation
    of shape ``(num_cameras, base_dim + extras)``. Rule-based adapters slice
    the base portion out and run their stock logic on it.

    Used by ``mate_marl/scripts/eval_rule_mate_hard.py``.
    """
    base = gym.make("MultiAgentTracking-v0", config=mate_config, **mate_env_kwargs)
    if target_agent is None:
        target_agent = GreedyTargetAgent()
    env = mate.MultiCamera.make(base, target_agent=target_agent)

    if enable_heterogeneous:
        env = HeterogeneousCameras(env, type_assignment=type_assignment, seed=seed)
    else:
        env = HeterogeneousCameras(env, type_assignment=[0] * env.unwrapped.num_cameras)

    if enable_energy:
        env = EnergyConstraint(env, seed=seed)
    else:
        env = EnergyConstraint(
            env, idle_drain=0.0, slew_drain_coef=0.0, zoom_drain_coef=0.0,
            recharge_rate=0.0, seed=seed,
        )

    if enable_fog:
        env = DynamicFog(env, num_fogs=num_fogs, seed=seed)
    else:
        env = DynamicFog(env, num_fogs=num_fogs, intensity_range=(0.0, 0.0), seed=seed)

    return env


def make_marl_env_discrete(
    mate_config: str = "MATE-4v8-9.yaml",
    levels: int = 5,
    target_agent=None,
    type_assignment: str = "round_robin",
    enable_heterogeneous: bool = True,
    enable_energy: bool = True,
    enable_fog: bool = True,
    num_fogs: int = 4,
    reward_shaping: bool = False,
    shaping_weight: Optional[float] = None,
    seed: Optional[int] = None,
    **mate_env_kwargs,
) -> gym.Env:
    """Build a DQN-friendly env: same flagship stack but with mate.DiscreteCamera
    applied at the BASE-env level so each camera has Discrete(levels*levels) actions.
    """
    from mate.wrappers import DiscreteCamera
    from mate_marl.wrappers import DiscreteFlattenForDQN

    base = gym.make("MultiAgentTracking-v0", config=mate_config, **mate_env_kwargs)
    # Discretize cameras BEFORE MultiCamera so the single-team action_space is
    # Tuple(Discrete(N) * num_cameras). Use the MultiCamera constructor directly
    # rather than .make(), which would strip+re-apply wrappers and double-wrap
    # DiscreteCamera.
    base = DiscreteCamera(base.unwrapped, levels=levels)
    if target_agent is None:
        target_agent = GreedyTargetAgent()
    env = mate.MultiCamera(base, target_agent=target_agent)

    if enable_heterogeneous:
        env = HeterogeneousCameras(env, type_assignment=type_assignment, seed=seed)
    else:
        env = HeterogeneousCameras(env, type_assignment=[0] * env.unwrapped.num_cameras)

    if enable_energy:
        env = EnergyConstraint(env, seed=seed)
    else:
        env = EnergyConstraint(
            env, idle_drain=0.0, slew_drain_coef=0.0, zoom_drain_coef=0.0,
            recharge_rate=0.0, seed=seed,
        )

    if enable_fog:
        env = DynamicFog(env, num_fogs=num_fogs, seed=seed)
    else:
        env = DynamicFog(env, num_fogs=num_fogs, intensity_range=(0.0, 0.0), seed=seed)

    env = MateMARLDictObs(env)
    rw = None
    if shaping_weight is not None:
        from mate_marl.wrappers.flatten_for_ppo import FlattenAgentsForPPO
        rw = dict(FlattenAgentsForPPO.DEFAULT_REWARD_WEIGHTS)
        rw["soft_coverage_score"] = shaping_weight
    env = DiscreteFlattenForDQN(env, reward_shaping=reward_shaping, reward_weights=rw)
    return env
