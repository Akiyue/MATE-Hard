import random
from abc import ABC, abstractmethod
from typing import Any, Generator, Optional, Literal
from dataclasses import dataclass

import gymnasium.spaces as spaces
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
# from tensordict import TensorDict

from gym_agent import utils

# import psutil

# MAX_MEM_AVAILABLE = psutil.virtual_memory().available




@dataclass(frozen=True)
class BaseBufferSamples:
    observations: Tensor | dict[str, Tensor]
    actions: Tensor | dict[str, Tensor]
    rewards: Tensor
    # agent_rewards: Optional[Tensor]


@dataclass(frozen=True)
class ReplayBufferSamples(BaseBufferSamples):
    next_observations: Tensor | dict[str, Tensor]
    terminals: Tensor


@dataclass(frozen=True)
class RolloutBufferSamples(BaseBufferSamples):
    log_prob: Tensor
    values: Tensor
    advantages: Tensor
    returns: Tensor

    # agent_log_prob: Optional[Tensor]
    # agent_values: Optional[Tensor]
    # agent_advantages: Optional[Tensor]
    # agent_returns: Optional[Tensor]


class BaseBuffer(ABC):
    observations: NDArray | dict[str, NDArray]
    actions: NDArray | dict[str, NDArray]
    rewards: NDArray

    def __init__(
        self,
        /,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device | str,
        num_envs: int,
        seed: Optional[int],
    ):
        if num_envs <= 0:
            raise ValueError("The number of environments must be greater than 0.")

        if buffer_size <= 0:
            raise ValueError("The buffer size must be greater than 0.")

        self.buffer_size = buffer_size
        self.obs_shape = utils.get_shape(observation_space)
        self.action_shape = utils.get_shape(action_space)
        self.num_envs = num_envs
        self.device = utils.get_device(device)

        if type(action_space) is spaces.Discrete:
            self.return_one_hot_action = int(action_space.n)
        elif type(action_space) is spaces.MultiDiscrete:
            self.return_one_hot_action = action_space.nvec.tolist()
        else:
            self.return_one_hot_action = False

        self.seed = random.seed(seed)

        self.mem_cntr = 0
        self.full = False
        self._mem: dict[str, NDArray | dict[str, NDArray]] = {}
        self._mem_shape: dict[str, tuple[int, ...] | dict[str, tuple[int, ...]]] = {}

        self.register_base_memory()

    def register_base_memory(self) -> None:
        self.mem_register("observations", self.obs_shape)
        self.mem_register("actions", self.action_shape)
        self.mem_register("rewards")

    def get_mem(self, mem_name: str) -> dict | tuple[int, ...]:
        if mem_name not in self._mem:
            raise ValueError(f"Memory '{mem_name}' is not registered.")
        return getattr(self, mem_name)

    def add2mem(
        self,
        mem_name: str,
        idx: int,
        mem_data: dict | tuple[int, ...],
    ):
        mem = self.get_mem(mem_name)
        if isinstance(mem, dict):
            for key in mem_data.keys():
                mem[key][idx] = mem_data[key]
        else:
            mem[idx] = mem_data

    def add2mems(
        self,
        data: dict[str, NDArray | dict[str, NDArray]],
        idx: int,
        **kwargs
    ):
        backups = {

        }
        for mem_name, mem_data in data.items():
            backups[mem_name] = self.get_mem(mem_name)
            try:
                self.add2mem(mem_name, idx, mem_data, **kwargs)
            except Exception as e:
                print(f"Error when adding to memory '{mem_name}' at index {idx}. Rolling back changes.")
                raise e
                for name, mem in backups.items():
                    self.get_mem(name)[idx] = mem

                # TODO: using logger instead of print
                e.add_note(f"Error at {mem_name=} with {mem_data=}")
                raise e

    def mem_register(
        self,
        name: str,
        shape: Optional[dict[str, tuple[int, ...]] | tuple[int, ...]] = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        if hasattr(self, name):
            raise ValueError(f"Memory '{name}' is already registered.")

        if shape is None:
            shape = []

        if isinstance(shape, dict):
            mem_value = {
                key: np.zeros([self.buffer_size, self.num_envs, *item_shape], dtype=dtype)
                for key, item_shape in shape.items()
            }
        else:
            mem_value = np.zeros([self.buffer_size, self.num_envs, *shape], dtype=dtype)

        self._mem[name] = mem_value
        self._mem_shape[name] = shape
        setattr(self, name, self._mem[name])

    def reset_mem(self):
        for name, values in self._mem.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    self._mem[name][key] = np.zeros([self.buffer_size, self.num_envs, *self._mem_shape[name][key]], dtype=value.dtype)
            else:
                self._mem[name] = np.zeros([self.buffer_size, self.num_envs, *self._mem_shape[name]], dtype=values.dtype)

            setattr(self, name, self._mem[name])

    def to(self, device: torch.device | str) -> None:
        self.device = utils.get_device(device)

    def to_torch(self, x: NDArray | dict[str, NDArray], return_one_hot: Literal[False] | int | list = False) -> Tensor | dict[str, Tensor]:
        # if x is None:
        #     return x
        # data = utils.to_torch(x, device=self.device)
        # if return_one_hot is False:
        #     return data
        # elif type(return_one_hot) is int:   # discrete action
        #     if isinstance(data, dict):
        #         for key, value in data.items():
        #             if value.shape[-1] == 1 and len(value.shape) > 1:
        #                 value = value.squeeze(-1)
        #             data[key] = torch.nn.functional.one_hot(value.long(), num_classes=return_one_hot).float()
        #     else:
        #         if data.shape[-1] == 1 and len(data.shape) > 1:
        #             data = data.squeeze(-1)
        #         return torch.nn.functional.one_hot(data.long(), num_classes=return_one_hot).float()
        # else:   # multi-discrete action
        #     # TODO: to be implemented
        #     raise NotImplementedError("Multi-discrete action one-hot encoding is not implemented yet.")
        #     # if isinstance(data, dict):
        #     #     for key, value in data.items():
        #     #         if value.shape[-1] == 1 and len(value.shape) > 1:
        #     #             value = value.squeeze(-1)
        #     #         one_hots = []
        #     #         for i, n in enumerate(return_one_hot):
        #     #             one_hots.append(torch.nn.functional.one_hot(value[..., i], num_classes=n).float())
        #     #         data[key] = torch.cat(one_hots, dim=-1)
        #     # else:
        #     #     if data.shape[-1] == 1 and len(data.shape) > 1:
        #     #         data = data.squeeze(-1)
        #     #     one_hots = []
        #     #     for i, n in enumerate(return_one_hot):
        #     #         one_hots.append(torch.nn.functional.one_hot(data[..., i], num_classes=n).float())
        #     #     return torch.cat(one_hots, dim=-1)


        return utils.to_torch(x, device=self.device) if x is not None else None

    def __len__(self):
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size * self.num_envs
        return self.mem_cntr * self.num_envs

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    @abstractmethod
    def sample(self):
        """
        Sample elements from the buffer.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """
        Reset the buffer.
        """

        self.mem_cntr = 0
        self.full = False

        self.reset_mem()

class ReplayBuffer(BaseBuffer):
    terminals: NDArray

    def __init__(
        self,
        /,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device="auto",
        num_envs: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            num_envs=num_envs,
            seed=seed,
        )

        self.mem_register("terminals", dtype=np.bool_)

    def add(
        self,
        observation: NDArray | dict[str, NDArray],
        action: NDArray | dict[str, NDArray],
        reward: NDArray,
        terminal: NDArray,
    ) -> None:
        """Add a new experience to memory."""
        idx = self.mem_cntr

        self.add2mems(
            {
                "observations": observation,
                "actions": action,
                "rewards": reward,
                "terminals": terminal,
            },
            idx,
        )

        self.mem_cntr += 1

        if self.mem_cntr == self.buffer_size:
            self.full = True
            self.mem_cntr = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """Randomly sample a batch of experiences from memory."""

        if self.full:
            batch = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.mem_cntr
            ) % self.buffer_size
        else:
            batch = np.random.randint(0, self.mem_cntr, size=batch_size)

        return self._get_sample(batch)

    def _get_sample(
        self, batch: NDArray[np.uint32]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        env_ind = np.random.randint(0, self.num_envs, size=len(batch))

        if isinstance(self.obs_shape, dict):
            observations = {
                key: obs[batch, env_ind] for key, obs in self.observations.items()
            }
            next_observations = {
                key: obs[(batch + 1) % self.buffer_size, env_ind]
                for key, obs in self.observations.items()
            }
        else:
            observations = self.observations[batch, env_ind]
            next_observations = self.observations[
                (batch + 1) % self.buffer_size, env_ind
            ]

        if isinstance(self.action_shape, dict):
            actions = {key: act[batch, env_ind] for key, act in self.actions.items()}
        else:
            actions = self.actions[batch, env_ind]

        rewards = self.rewards[batch, env_ind]

        terminals = self.terminals[batch, env_ind]

        return ReplayBufferSamples(
            observations=self.to_torch(observations),
            actions=self.to_torch(actions, return_one_hot=self.return_one_hot_action),
            rewards=self.to_torch(rewards),
            next_observations=self.to_torch(next_observations),
            terminals=self.to_torch(terminals),
        )

class RolloutBuffer(BaseBuffer):
    # group 1
    log_probs: NDArray
    values: NDArray
    advantages: NDArray
    returns: NDArray
    episode_starts: NDArray

    # group 2
    # agent_log_probs: Optional[NDArray]
    # agent_values: Optional[NDArray]
    # agent_advantages: Optional[NDArray]
    # agent_returns: Optional[NDArray]

    # at least one of those 2 group must exist

    def __init__(
        self,
        /,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        gamma=0.99,
        gae_lambda=1.0,
        device="auto",
        num_envs: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            num_envs=num_envs,
            seed=seed,
        )

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.processed = False

        self.mem_register("log_probs")
        self.mem_register("values")
        self.mem_register("advantages")
        self.mem_register("returns")
        self.mem_register("episode_starts", dtype=np.bool_)

    def add(
        self,
        observation: NDArray | dict[str, NDArray],
        action: NDArray | dict[str, NDArray],
        reward: NDArray,
        value: NDArray,
        log_prob: NDArray,
        episode_start: NDArray[np.bool_],
    ) -> None:
        if self.processed:
            raise ValueError(
                "Cannot add new experiences to the buffer after processing the buffer."
            )

        if self.mem_cntr >= self.buffer_size:
            raise ValueError("Rollout buffer is full. Please reset the buffer.")

        idx = self.mem_cntr     # idx = 1

        self.add2mems(
            {
                "observations": observation,
                "actions": action,
                "rewards": reward,
                "values": value,
                "log_probs": log_prob,
                "episode_starts": episode_start
            },
            idx=idx,
        )

        self.mem_cntr += 1

    def reset(self) -> None:
        super().reset()
        self.processed = False

    def calc_advantages_and_returns(
        self,
        last_values: NDArray[np.float32],
        last_terminals: NDArray[np.bool_],
    ):
        if self.processed:
            raise ValueError(
                "Cannot calculate advantages and returns after processing the buffer."
            )

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_terminals.astype(np.float32)
                next_values = np.int64(last_values)
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def process_mem(self) -> NDArray:
        if self.processed:
            raise ValueError("Cannot process the buffer again.")

        def array_swap_and_flatten(arr: NDArray) -> NDArray:
            shape = arr.shape
            if len(shape) < 3:  # [buffer_size, num_envs]
                shape = (*shape, 1)

            return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

        def swap_and_flatten(
            arr: NDArray | dict[str, NDArray],
        ) -> NDArray | dict[str, NDArray]:
            """
            Swap and then flatten axes 0 (buffer_size) and 1 (num_envs)
            to convert shape from [n_steps, num_envs, ...] (when ... is the shape of the features)
            to [num_envs * n_steps, ...] (which maintain the order)
            """
            return (
                {key: array_swap_and_flatten(arr[key]) for key in arr}
                if isinstance(arr, dict)
                else array_swap_and_flatten(arr)
            )


        self.observations = swap_and_flatten(self.observations)
        self.actions = swap_and_flatten(self.actions)
        self.rewards = swap_and_flatten(self.rewards).flatten()
        self.values = swap_and_flatten(self.values).flatten()
        self.log_probs = swap_and_flatten(self.log_probs).flatten()
        self.advantages = swap_and_flatten(self.advantages).flatten()
        self.returns = swap_and_flatten(self.returns).flatten()

        self.processed = True

    def get(self, batch_size: int = None) -> Generator[RolloutBufferSamples, Any, None]:
        mem_size = len(self.observations)
        if not self.processed:
            self.process_mem()

        if batch_size is None or batch_size is False:
            yield self._get_sample(np.arange(mem_size))

        else:
            indices = np.random.permutation(mem_size)

            for start_idx in range(0, mem_size, batch_size):
                yield self._get_sample(indices[start_idx : start_idx + batch_size])

    def sample(self, batch_size: int) -> RolloutBufferSamples:
        if not self.processed:
            self.process_mem()

        batch = np.random.randint(0, sum(self.end_mem_pos), size=batch_size)

        return self._get_sample(batch)

    def _get_sample(self, batch: NDArray[np.uint32]) -> RolloutBufferSamples:
        if isinstance(self.obs_shape, dict):
            observations = {
                key: self.observations[key][batch] for key in self.obs_shape.keys()
            }
        else:
            observations = self.observations[batch]

        if isinstance(self.action_shape, dict):
            actions = {
                key: self.actions[key][batch] for key in self.action_shape.keys()
            }
        else:
            actions = self.actions[batch]

        return RolloutBufferSamples(
            observations=self.to_torch(observations),
            actions=self.to_torch(actions, return_one_hot=self.return_one_hot_action),
            rewards=self.to_torch(self.rewards[batch]),
            values=self.to_torch(self.values[batch]),
            log_prob=self.to_torch(self.log_probs[batch]),
            advantages=self.to_torch(self.advantages[batch]),
            returns=self.to_torch(self.returns[batch]),
        )


class MultiRewardBuffer(ReplayBuffer):
    agent_rewards: Optional[NDArray]

    def __init__(
        self,
        /,
        buffer_size: int,
        n_agents: int ,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device | str,
        num_envs: int,
        seed: Optional[int],
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device=device,
            num_envs=num_envs,
            seed=seed,
        )

        self.n_agents = n_agents

        self.mem_register("agent_rewards", (n_agents, ))

    def add(
        self,
        observation: NDArray | dict[str, NDArray],
        action: NDArray | dict[str, NDArray],
        reward: NDArray,
        terminal: NDArray,
        agent_rewards: NDArray
    ) -> None:
        """Add a new experience to memory."""
        idx = self.mem_cntr

        self.add2mems(
            {
                "observations": observation,
                "actions": action,
                "rewards": reward,
                "terminals": terminal,
                "agent_rewards": agent_rewards
            },
            idx,
        )

        self.mem_cntr += 1

        if self.mem_cntr == self.buffer_size:
            self.full = True
            self.mem_cntr = 0


class MultiCriticBuffer(RolloutBuffer):
    pass
