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
class PPOConfig(ActorCriticAgentConfig):
    max_grad_norm: Optional[float] = 0.5
    vf_coef: float = 0.5
    entropy_coef: float = 0.0
    normalize_advantage: bool = True
    n_epochs: int = 4
    clip_range: float = 0.2
    target_kl: Optional[float] = None

    # redefined parameters from base class with different default values
    n_steps: int = 2048


class PPO(ActorCriticPolicyAgent):
    policy: ActorCriticPolicy

    def __init__(
        self,
        env_id: str,
        policy: ActorCriticPolicy,
        config: Optional[PPOConfig] = None,
    ):
        config = config if config is not None else PPOConfig()
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
        self.n_epochs = config.n_epochs
        self.clip_range = config.clip_range
        self.target_kl = config.target_kl

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
        """

        self.policy.train()

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Sample from the buffer
            for rollout_data in self.memory.get(self.batch_size):
                # Shape: (batch_size, 1) or (batch_size, action_dim)
                actions = rollout_data.actions
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

                p_ratio = torch.exp(log_probs - rollout_data.log_prob)

                # Actor loss
                policy_loss_1 = advantages * p_ratio
                policy_loss_2 = advantages * torch.clamp(
                    p_ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Critic loss
                value_loss = F.mse_loss(rollout_data.returns.flatten(), values)

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

                with torch.no_grad():
                    log_ratio = log_probs - rollout_data.log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    break

                self.policy.zero_grad()
                loss.backward()

                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm
                    )

                self.policy.optimizers_step()

            else:  # not break from target_kl
                self.policy.lr_schedulers_step()
                continue
            # break if if target_kl is reached
            break

        # end for n_epochs epochs
