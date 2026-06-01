import torch
import torch.nn as nn
import torch.optim as optim

from typing import Callable, Optional, Self

class BasePolicy(nn.Module):
    optimizers: dict[str, optim.Optimizer]
    lr_schedulers: dict[str, optim.lr_scheduler._LRScheduler]

    def __init__(self):
        super().__init__()
        self.optimizers = {}
        self.lr_schedulers = {}

    def act(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


    def save_info(self):
        info = {
            "model": self.state_dict(),
            "optimizers": {k: opt.state_dict() for k, opt in self.optimizers.items()},
            "lr_schedulers": {k: sch.state_dict() for k, sch in self.lr_schedulers.items()},
        }
        return info

    def load_model(self, model_state: dict):
        self.load_state_dict(model_state)

    def load_optimizers(self, optim_state: dict):
        for k, opt in self.optimizers.items():
            if k in optim_state:
                opt.load_state_dict(optim_state[k])

    def load_lr_schedulers(self, lr_scheduler_state: dict):
        for k, sch in self.lr_schedulers.items():
            if k in lr_scheduler_state:
                sch.load_state_dict(lr_scheduler_state[k])

    def load_info(self, info: dict):
        # assume info is from save_info()
        self.load_model(info["model"])
        self.load_optimizers(info["optimizers"])
        self.load_lr_schedulers(info["lr_schedulers"])

    def zero_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

    def optimizers_step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()

    def lr_schedulers_step(self):
        for lr_scheduler in self.lr_schedulers.values():
            lr_scheduler.step()

class TargetPolicy(BasePolicy):
    target_net: torch.nn.Module

    def __init__(
        self,
        net_factory_func: Callable[..., nn.Module],
        lr: float,
        optim_cls: type[optim.Optimizer] = optim.Adam,
        optim_kwargs: Optional[dict] = None,
        lr_scheduler_cls: Optional[type[optim.lr_scheduler._LRScheduler]] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.net = net_factory_func()
        self.target_net = net_factory_func()

        self.optimizers = {
            "net": optim_cls(self.net.parameters(), lr=lr, **(optim_kwargs or {}))
        }

        if lr_scheduler_cls is not None:
            self.lr_schedulers = {
                "net": lr_scheduler_cls(
                    self.optimizers["net"], **(lr_scheduler_kwargs or {})
                )
            }

        self.hard_update()  # Initialize target network weights to match the main network

        self.target_net.eval()  # Set the target network to evaluation mode

    def train(self, mode: bool = True) -> Self:
        # Override to keep target network in eval mode
        super().train(mode)
        self.target_net.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def target_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_net(x)

    def soft_update(self, tau: float):
        for target_param, local_param in zip(
            self.target_net.parameters(), self.net.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def hard_update(self):
        self.soft_update(1.0)


class ActorCriticPolicy(BasePolicy):
    """Actor-Critic Policy

    Args:
        BasePolicy (_type_): _description_
    """
    actor: nn.Module
    critic: nn.Module
    feature_extractor: Optional[nn.Module]
    def __init__(
        self,
        actor_factory_func: Callable[..., nn.Module],
        critic_factory_func: Callable[..., nn.Module],
        actor_lr: float,
        shared_optimizer: bool = True,
        critic_lr: Optional[float] = None,
        feature_extractor_func: Optional[Callable[..., nn.Module]] = None,
        optim_cls: type[optim.Optimizer] = optim.Adam,
        optim_kwargs: Optional[dict] = None,
        lr_scheduler_cls: Optional[type[optim.lr_scheduler._LRScheduler]] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
    ):
        r"""Constructor for policy networks.

        Initializes a policy network with actor and critic components, and optionally a shared feature extractor.
        Sets up optimizers and learning rate schedulers for both actor and critic networks.

        Args:
            actor_factory_func (Callable[..., nn.Module]): Factory function to create the actor network.
            critic_factory_func (Callable[..., nn.Module]): Factory function to create the critic network.
            actor_lr (float): Learning rate for the actor network.
            critic_lr (float, optional): Learning rate for the critic network. If None, defaults to actor_lr.
            feature_extractor_func (Callable[..., nn.Module], optional): Factory function to create the feature extractor. If None, use Identity.
            shared_feature_extractor (bool): flag to share feature extractor or not. Defaults to True
            optim_cls (type[optim.Optimizer], optional): Optimizer class to use. Defaults to Adam.
            optim_kwargs (dict, optional): Additional keyword arguments for optimizer initialization.
            lr_scheduler_cls (type[optim.lr_scheduler._LRScheduler], optional): Learning rate scheduler class.
            lr_scheduler_kwargs (dict, optional): Additional keyword arguments for learning rate scheduler.
        """
        super().__init__()

        self.feature_extractor = feature_extractor_func() if callable(feature_extractor_func) else None


        self.actor = actor_factory_func()
        self.critic = critic_factory_func()

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr if critic_lr is not None else actor_lr

        self.shared_optimizer = shared_optimizer

        if shared_optimizer is True:
            self.optimizers = {
                "shared": optim_cls(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    lr=self.actor_lr,
                    **(optim_kwargs or {})
                )
            }
        else:
            self.optimizers = {
                "actor": optim_cls(
                    self.actor.parameters(), lr=self.actor_lr, **(optim_kwargs or {})
                ),
                "critic": optim_cls(
                    self.critic.parameters(), lr=self.critic_lr, **(optim_kwargs or {})
                ),
            }

        if lr_scheduler_cls is not None:
            if shared_optimizer is True:
                self.lr_schedulers = {
                    "shared": lr_scheduler_cls(
                        self.optimizers["shared"], **(lr_scheduler_kwargs or {})
                    )
                }
            else:
                self.lr_schedulers = {
                    "actor": lr_scheduler_cls(
                        self.optimizers["actor"], **(lr_scheduler_kwargs or {})
                    ),
                    "critic": lr_scheduler_cls(
                        self.optimizers["critic"], **(lr_scheduler_kwargs or {})
                    ),
                }

    def extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Extract features from the input tensor.
        return as tuple because feature extractor may return multiple features

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, ...]: _description_
        """
        if self.feature_extractor is None:
            return (x, )
        features = self.feature_extractor(x)
        return features if isinstance(features, tuple) else (features, )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass through the actor and critic networks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Action logits and value logits.
        """
        features = self.extract_features(x)
        action_logits = self.actor(*features)
        value_logits = self.critic(*features)

        return action_logits, value_logits

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.actor(*features)

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.critic(*features)

    def act(self, *args, **kwargs):
        return self.forward_actor(*args, **kwargs)


class TargetActorCriticPolicy(ActorCriticPolicy):
    target_actor: nn.Module
    target_critic: nn.Module

    def __init__(
        self,
        actor_factory_func: Callable[..., nn.Module],
        critic_factory_func: Callable[..., nn.Module],
        actor_lr: float,
        critic_lr: float,
        feature_extractor_func: Optional[Callable[..., nn.Module]] = None,
        optim_cls: type[optim.Optimizer] = optim.Adam,
        optim_kwargs: Optional[dict] = None,
        lr_scheduler_cls: Optional[type[optim.lr_scheduler._LRScheduler]] = None,
        lr_scheduler_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            actor_factory_func=actor_factory_func,
            critic_factory_func=critic_factory_func,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            feature_extractor_func=feature_extractor_func,
            optim_cls=optim_cls,
            optim_kwargs=optim_kwargs,
            lr_scheduler_cls=lr_scheduler_cls,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )

        self.target_feature_extractor = feature_extractor_func() if callable(feature_extractor_func) else None

        self.target_actor = actor_factory_func()
        self.target_critic = critic_factory_func()

        self.hard_update()  # Initialize target network weights to match the main network

        self.target_feature_extractor.eval()  # Set the target network to evaluation mode
        self.target_actor.eval()  # Set the target network to evaluation mode
        self.target_critic.eval()  # Set the target network to evaluation mode

    def train(self, mode: bool = True) -> 'Self':
        # Override to keep target networks in eval mode
        super().train(mode)
        self.target_feature_extractor.eval()
        self.target_actor.eval()
        self.target_critic.eval()
        return self

    def target_extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.target_feature_extractor is None:
            return (x, )
        features = self.target_feature_extractor(x)
        return features if isinstance(features, tuple) else (features, )

    def target_forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        features = self.target_extract_features(x)
        return self.target_critic(*features)

    def target_forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        features = self.target_extract_features(x)
        return self.target_actor(*features)

    def target_forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.target_extract_features(x)
        action_logits = self.target_actor(*features)
        value_logits = self.target_critic(*features)
        return action_logits, value_logits

    def soft_update(self, tau_actor: float, tau_critic: Optional[float] = None, tau_feature_from_critic: bool = True):
        for target_param, local_param in zip(
                self.target_actor.parameters(), self.actor.parameters()
            ):
                target_param.data.copy_(
                    tau_actor * local_param.data + (1.0 - tau_actor) * target_param.data
                )


        if tau_critic is None:
            tau_critic = tau_actor

        for target_param, local_param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                tau_actor * local_param.data + (1.0 - tau_actor) * target_param.data
            )

        if self.target_feature_extractor is not None:
            tau_feature = tau_critic if tau_feature_from_critic is True else tau_actor
            for target_param, local_param in zip(
                self.target_feature_extractor.parameters(), self.feature_extractor.parameters()
            ):
                target_param.data.copy_(
                    tau_feature * local_param.data + (1.0 - tau_feature) * target_param.data
                )

    def hard_update(self):
        self.soft_update(1.0)

