from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from gym_agent.utils import to_torch
from gym_agent.core.agent_base import (
    ActorCriticAgentConfig,
    ActorCriticPolicyAgent,
    ActType,
    ObsType,
)
from gym_agent.core.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    make_proba_distribution,
)
from gym_agent.core.polices import ActorCriticPolicy


@dataclass(kw_only=True)
class A2CConfig(ActorCriticAgentConfig):
    max_grad_norm: Optional[float] = 0.5
    vf_coef: float = 0.5
    entropy_coef: float = 0.0
    normalize_advantage: bool = False

    # redefined parameters from base class with different default values
    n_steps: int = 5

    batch_size: int = None # will be set to n_steps * num_envs in the agent init



class A2C(ActorCriticPolicyAgent):
    policy: ActorCriticPolicy

    def __init__(
        self,
        env_id: str,
        policy: ActorCriticPolicy,
        config: Optional[A2CConfig] = None,
    ):
        config = config if config is not None else A2CConfig()
        config.batch_size = config.n_steps * config.num_envs    # for on-policy, batch_size is n_steps * num_envs

        super().__init__(
            env_id=env_id,
            policy=policy,
            config=config,
            supported_action_spaces=(
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
                spaces.Box,
            ),
        )

        self.max_grad_norm = config.max_grad_norm

        self.normalize_advantage = config.normalize_advantage
        self.vf_coef = config.vf_coef
        self.entropy_coef = config.entropy_coef
        self.action_dist = make_proba_distribution(self.action_space)

        # If the action space is continuous, we need to learn the standard deviation
        if isinstance(self.action_dist, DiagGaussianDistribution):
            log_std_init = 0
            self.log_std = nn.Parameter(
                torch.ones(self.action_dist.action_dim) * log_std_init,
                requires_grad=True,
            ).to(self.device)

    def distribution(self, action_logits: torch.Tensor) -> Distribution:
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(action_logits, self.log_std)
        elif isinstance(
            self.action_dist,
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            # Here action_logits are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=action_logits)
        else:
            raise ValueError("Invalid action distribution")

    def predict(self, state: ObsType, deterministic: bool = True) -> ActType:
        action_logits = self.policy.forward_actor(
            to_torch(state, self.device).float()
        )  # Shape or dict of Shape: (num_envs, action_dim) or (num_envs, n_actions)

        dist = self.distribution(action_logits)
        actions = dist.get_actions(deterministic=deterministic)
        return actions.cpu().numpy()

    def learn(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.

        Parameters:
            memory (RolloutBuffer): Buffer containing the rollouts.
        """
        # Sample from the buffer

        for rollout_data in self.memory.get(self.batch_size):
            actions = (
                rollout_data.actions
            )  # Shape: (batch_size, 1) or (batch_size, action_dim)

            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()  # Shape: (batch_size,)

            values, log_probs, entropy = self.evaluate_actions(
                rollout_data.observations, actions
            )

            advantages = rollout_data.advantages

            if self.normalize_advantage and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

            # Actor loss
            # Ensure advantages has the expected shape
            assert advantages.shape == log_probs.shape, (
                f"Advantages shape {advantages.shape} doesn't match log_probs shape {log_probs.shape}"
            )
            policy_loss = -(advantages * log_probs).mean()

            # Critic loss
            assert rollout_data.returns.shape == values.shape, (
                f"return shape {rollout_data.returns.shape} doesn't match values shape {values.shape}"
            )
            value_loss = F.mse_loss(rollout_data.returns, values)

            if entropy is None:
                # If the distribution does not have an entropy method, approximate it
                entropy_loss = -torch.mean(-log_probs)
            else:
                entropy_loss = -torch.mean(entropy)

            loss = (
                policy_loss
                + self.vf_coef * value_loss
                + self.entropy_coef * entropy_loss
            )

            self.policy.zero_grad()
            loss.backward()

            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            self.policy.optimizers_step()
            self.policy.lr_schedulers_step()
