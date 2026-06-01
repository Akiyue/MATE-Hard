# Third-party imports
import torch

import numpy as np
from numpy.typing import NDArray
from typing import Any

import matplotlib.pyplot as plt

import gymnasium.spaces as spaces


def get_device(device: torch.device | str = "auto") -> torch.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


def to_device(*args, device="cuda"):
    for arg in args:
        arg.to(device)

def to_torch(
    x: NDArray | dict[str, NDArray],
    device: str | torch.device,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """
    Convert a numpy array or a dictionary of numpy array to a torch.Tensor.

    :param x: the input numpy array or dictionary of numpy array
    :param device: the device to which the tensor(s) will be moved
    :return: the tensor or dictionary of tensors as torch.Tensor(s)
    """
    if isinstance(x, dict):
        return {key: torch.from_numpy(value).to(device) for key, value in x.items()}
    return torch.from_numpy(x).to(device)

def get_shape(
    space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param space:
    :return:
    """
    if isinstance(space, spaces.Box):
        return space.shape
    elif isinstance(space, spaces.Discrete):
        return ()
    elif isinstance(space, spaces.MultiDiscrete):
        # Number of discrete features
        return (len(space.nvec),)
    elif isinstance(space, spaces.MultiBinary):
        # Number of binary features
        return space.shape
    elif isinstance(space, spaces.Dict):
        return {key: get_shape(subspace) for (key, subspace) in space.spaces.items()}
    else:
        raise NotImplementedError(f"{space} space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (len(observation_space.nvec),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {
            key: get_obs_shape(subspace)
            for (key, subspace) in observation_space.spaces.items()
        }  # type: ignore[misc]

    else:
        raise NotImplementedError(
            f"{observation_space} observation space is not supported"
        )


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return len(action_space.nvec)
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(action_space.n, int), (
            f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        )
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def check_for_nested_spaces(space: spaces.Space) -> None:
    """
    Make sure the observation space does not have nested spaces (Dicts/Tuples inside Dicts/Tuples).
    If so, raise an Exception informing that there is no support for this.

    :param space: an observation space
    """
    if isinstance(space, (spaces.Dict, spaces.Tuple)):
        sub_spaces = (
            space.spaces.values() if isinstance(space, spaces.Dict) else space.spaces
        )
        for sub_space in sub_spaces:
            if isinstance(sub_space, (spaces.Dict, spaces.Tuple)):
                raise NotImplementedError(
                    "Nested observation spaces are not supported (Tuple/Dict space inside Tuple/Dict space)."
                )

def key_consistent(dicts: list[dict[str, Any]]) -> set[str]:
    """
    Ensure all dictionaries have the same keys.
    :param dicts: List of info dictionaries
    :return: dictionary with all keys initialized to numpy empty arrays
    """

    full_keys_dict = {}

    for d in dicts:
        for key, value in d.items():
            if key not in full_keys_dict:
                full_keys_dict[key] = np.empty_like(value)

    return full_keys_dict

def stack_dict(
    dicts: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """
    Stack a list of dictionaries into a single dictionary with lists as values.

    :param dicts: List of info dictionaries
    :return: A single dictionary with lists as values
    """
    stacked: dict[str, np.ndarray] = {}

    full_keys_dict = key_consistent(dicts)


    for key in full_keys_dict.keys():
        mask = np.stack([np.ones_like(full_keys_dict[key], dtype=bool) if d.get(key, None) is None else np.zeros_like(full_keys_dict[key], dtype=bool) for d in dicts])
        stacked_key = np.stack([d.get(key, full_keys_dict[key]) for d in dicts]) # use get to fill missing keys with empty arrays
        stacked[key] = np.ma.array(stacked_key, mask=mask)

    return stacked


def smooth(values, window):
    """Simple moving average smoothing."""
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(values[start:i+1]))
    return smoothed

def plot_rl_style(scores, window, filename=None):
    """
    RL-paper style plot: smoothed mean with a shaded band (mean ± std).
    """
    if filename is not None and not isinstance(filename, str):
        raise ValueError("Filename must be a string or None.")


    n = len(scores)
    means = []
    stds = []

    for i in range(n):
        start = max(0, i - window + 1)
        window_slice = scores[start:i+1]
        means.append(np.mean(window_slice))
        stds.append(np.std(window_slice))

    x = np.arange(n)

    plt.figure(figsize=(10, 6))

    # Smoothed mean line
    plt.plot(x, means, color="blue", linewidth=2, label="Smoothed Mean")

    # Shaded mean ± std
    upper = np.array(means) + np.array(stds)
    lower = np.array(means) - np.array(stds)
    plt.fill_between(x, lower, upper, color="skyblue", alpha=0.3, label="±1 std dev")

    plt.title("RL-style Smoothed Performance Curve")
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
