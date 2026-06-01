import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import Tensor

from gym_agent.utils import to_torch
from gym_agent.core.agent_base import ActType, ObsType, OffPolicyAgent, OffPolicyAgentConfig
from gym_agent.core.polices import TargetPolicy


@dataclass(kw_only=True)
class DQNConfig(OffPolicyAgentConfig):
    eps_start: float = 1.0
    eps_decay: float = 0.995
    eps_decay_steps: Optional[int] = None
    eps_end: float = 0.01
    tau: float = 1e-3


class DQN(OffPolicyAgent):
    policy: TargetPolicy

    def __init__(
        self,
        env_id: str,
        policy: TargetPolicy,
        config: Optional[DQNConfig] = None,
    ):
        config = config if config is not None else DQNConfig()
        super().__init__(
            env_id=env_id,
            policy=policy,
            config=config,
            supported_action_spaces=(
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.eps_start = config.eps_start
        self.eps_decay = config.eps_decay
        if config.eps_decay_steps is not None:
            self.eps_decay = (config.eps_end / config.eps_start) ** (
                1 / config.eps_decay_steps
            )
        self.eps_end = config.eps_end

        # Soft update parameter
        self.tau = config.tau

        # Initialize epsilon for epsilon-greedy policy
        self.eps = self.eps_start

        self.eps_hist = [self.eps]

        self.add_save_kwargs("eps")

    def reset(self):
        self.eps = max(self.eps_end, self.eps_decay * self.eps)
        self.eps_hist.append(self.eps)
        return super().reset()

    @torch.no_grad()
    def predict(self, state: ObsType, deterministic=True) -> ActType:
        # Determine epsilon value based on evaluation mode
        if deterministic:
            eps = 0
        else:
            eps = self.eps

        # Epsilon-greedy action selection
        if random.random() >= eps:
            # Convert state to tensor and move to the appropriate device
            state = to_torch(state, self.device).float()

            # Set local model to evaluation mode
            self.policy.eval()
            # Get action values from the local model
            action_value: Tensor = self.policy.forward(state)
            # Set local model back to training mode
            self.policy.train()

            # Return the action with the highest value
            return np.argmax(action_value.cpu().data.numpy(), axis=1)
        else:
            # Return a random action from the action space
            return np.array(
                [self.envs.single_action_space.sample() for _ in range(state.shape[0])]
            )

    def learn(self):
        buffers = self.memory.sample(self.batch_size)

        observations = buffers.observations
        actions = buffers.actions
        rewards = buffers.rewards
        next_observations = buffers.next_observations
        terminals = buffers.terminals

        # Get the maximum predicted Q values for the next states from the target model
        q_targets_next: Tensor = (
            self.policy.target_forward(next_observations).detach().max(1)[0]
        )
        # Compute the Q targets for the current states
        q_targets: Tensor = rewards + (self.gamma * q_targets_next * (~terminals))

        # Get the expected Q values from the local model
        q_expected: Tensor = self.policy.forward(observations)

        # Deep Q-Learning always expects actions is Discrete
        actions = actions.long()
        if actions.dim() == q_expected.dim() - 1:
            actions = actions.unsqueeze(-1)

        q_expected = q_expected.gather(-1, actions).squeeze(-1)

        # Compute the loss
        loss: Tensor = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.policy.zero_grad()
        loss.backward()
        self.policy.optimizers_step()
        self.policy.lr_schedulers_step()

        # Update the target network
        self.policy.soft_update(self.tau)
