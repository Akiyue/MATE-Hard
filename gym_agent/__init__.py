from .core.main import make, make_vec, Transform
from . import core

# Off-policy algorithms
from .algos.off_policy.dqn import DQN, DQNConfig

# On-policy algorithms
from .algos.on_policy.a2c import A2C, A2CConfig
from .algos.on_policy.ppo import PPO, PPOConfig

__all__ = [
    "make",
    "make_vec",
    "Transform",
    "core",
    "DQN", "DQNConfig",

    "A2C", "A2CConfig",
    "PPO", "PPOConfig"
]
