# Standard library imports
import importlib
import pickle
import tempfile
import time
import warnings
import zipfile
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Generic, Optional, Type, TypeVar

# Third-party imports
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import yaml
from dacite import Config, from_dict
from gymnasium import spaces
from numpy.typing import NDArray
from tqdm import tqdm

import gym_agent.utils as utils
from gym_agent.core.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    make_proba_distribution,
)
from gym_agent.core.vec_env.dummy_vec_env import DummyVecEnv
from gym_agent.core.vec_env.subproc_vec_env import SubprocVecEnv

from .buffers import BaseBuffer, ReplayBuffer, RolloutBuffer
from .callbacks import Callbacks
from .main import make
from .polices import ActorCriticPolicy, BasePolicy
from .logger import configure, Logger

ObsType = TypeVar("ObsType", NDArray, dict[str, NDArray])
ActType = TypeVar("ActType", NDArray, dict[str, NDArray])


def auto_find_path(path, env_id: str, name: str, model_version="last"):
    """
    Automatically find the correct path given partial path information.
    .../env_id/name/start_training_time/model_version*

    Args:
        path (str or Path): Base path (could be incomplete).
        env_id (str): Environment identifier.
        name (str): Name of the model or entity.
        model_version (str): Model version, default "last".

    Returns:
        Path: The fully resolved path.

    Raises:
        PathNotFoundError: If no matching path is found.
    """

    base_path = Path(path).resolve()

    env_id = env_id.strip()
    name = name.strip()
    model_version = model_version.strip()

    if base_path.is_file():
        if base_path.suffix in [".zip", ".tar", ".gz"]:
            return base_path

    # Adjust path based on ending
    if (
        base_path.parts[-2] == name and base_path.parts[-3] == env_id
    ):  # .../env_id/name/start_time
        pass
    elif base_path.name == name:  # .../env_id/name
        pass
    elif base_path.name == env_id:  # .../env_id
        base_path = base_path / name
    else:
        # Search recursively for env_id/name directory structure
        found_paths = list(base_path.rglob(f"{env_id}/{name}"))
        if not found_paths:
            raise FileNotFoundError(
                f"Could not find '{env_id}/{name}' under {base_path}"
            )
        base_path = found_paths[0]  # Pick the first match (you could sort if needed)

    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    matches = [
        f
        for f in base_path.glob(f"**/{model_version}*")
        if f.is_file() and f.suffix in [".zip", ".tar", ".gz"]
    ]

    if matches:
        matches.sort(key=lambda p: p.name, reverse=True)
        return matches[0]

    raise FileNotFoundError(
        f"No matching file found for model_version '{model_version}' in {base_path} with env_id '{env_id}' and name '{name}'."
    )


class Clock:
    def __init__(self):
        self._last_tick_time = time.perf_counter()
        self._frame_rate = 0.0
        self._delta_time = 0.0

    def tick(self, framerate=0):
        """
        Updates the clock and optionally delays to maintain a specific framerate.
        Returns the number of milliseconds passed since the last call to tick().
        """
        current_time = time.perf_counter()
        elapsed_seconds = current_time - self._last_tick_time
        self._last_tick_time = current_time

        # Calculate delta time in milliseconds
        self._delta_time = elapsed_seconds * 1000

        if framerate > 0:
            target_frame_time = 1.0 / framerate
            if elapsed_seconds < target_frame_time:
                delay_seconds = target_frame_time - elapsed_seconds
                time.sleep(delay_seconds)
                # Recalculate delta time after sleep
                current_time = time.perf_counter()
                self._delta_time = (current_time - self._last_tick_time) * 1000
                self._last_tick_time = current_time

        # Calculate current FPS
        if elapsed_seconds > 0:
            self._frame_rate = 1.0 / elapsed_seconds
        else:
            self._frame_rate = float("inf")  # Avoid division by zero

        return int(self._delta_time)

    def get_time(self):
        """
        Returns the time in milliseconds that passed since the last call to tick().
        """
        return int(self._delta_time)

    def get_fps(self):
        """
        Returns the current framerate.
        """
        return self._frame_rate


@dataclass(kw_only=True)
class AgentConfig:
    env_kwargs: dict[str, Any] = None
    num_envs: int = 1
    n_steps: int = 1
    batch_size: int = 64
    async_vectorization: bool = True
    device: str = "auto"
    seed: Optional[int] = None
    model_compile: bool = (
        False  # whether to compile the model using torch.compile (PyTorch 2.0 feature)
    )
    # WARNING: torch.compile is experimental and may not work with all models or environments


@dataclass(kw_only=True)
class OffPolicyAgentConfig(AgentConfig):
    gamma: float = 0.99
    buffer_size: int = int(1e5)


@dataclass(kw_only=True)
class OnPolicyAgentConfig(AgentConfig):
    pass  # no additional parameters for now


@dataclass(kw_only=True)
class ActorCriticAgentConfig(AgentConfig):
    gamma: float = 0.99
    gae_lambda: float = 1.0


class AgentBase(ABC, Generic[ObsType, ActType]):
    memory: BaseBuffer
    envs: DummyVecEnv | SubprocVecEnv
    _logger: Logger

    def __init__(
        self,
        env_id: str,
        policy: BasePolicy,
        config: AgentConfig,
        supported_action_spaces: Optional[
            tuple[type[spaces.Space], ...]
        ],  # using for algorithm define only, not for user to set
    ):
        """
        Initialize the agent.
        This method sets up the agent with its environment, policy, and configuration.
        Args:
            env_id (str): The ID of the environment to create.
            policy (BasePolicy): The policy to use for the agent.
            config (AgentConfig): The configuration for the agent, which includes:
                - env_kwargs (dict, optional): Additional arguments for environment creation.
                - num_envs (int): Number of environments to run in parallel.
                - async_vectorization (bool): Whether to use asynchronous environment vectorization.
                - n_steps (int): Number of steps to run for each environment per update.
                - batch_size (int): Mini-batch size for training updates.
                - device (str): Device to run the model on ('cpu', 'cuda', etc.).
                - seed (int): Random seed for reproducibility.
            supported_action_spaces (Optional[tuple[type[spaces.Space], ...]]):
                The action space types supported by this agent. Used for algorithm validation,
                not for user configuration.
        Raises:
            ValueError: If policy is not an instance of BasePolicy, env_id is not a string,
                       supported_action_spaces is not a tuple, or the action space of the
                       environment is not supported.
        Notes:
            - The environment is automatically vectorized based on the configuration.
            - For continuous action spaces, the agent will learn the standard deviation.
            - The method initializes various tracking metrics like timesteps, episodes, and scores.
        """

        if not isinstance(policy, BasePolicy):
            raise ValueError(
                "policy must be an instance of gym_agent.core.policies.BasePolicy"
            )
        self.config = config

        self.name = self.__class__.__name__

        self.env_id = env_id

        env_kwargs = config.env_kwargs or {}

        env_kwargs["render_mode"] = "rgb_array"  # force rgb_array mode

        self.env_kwargs = env_kwargs

        if isinstance(self.env_id, str):
            env_factory_fn = lambda **kwargs: make(self.env_id, **kwargs)  # noqa: E731

            async_vectorization = config.async_vectorization
            if config.num_envs <= 1:
                async_vectorization = (
                    False  # no need to use async vectorization for single env
                )

            if async_vectorization:
                self.envs = SubprocVecEnv(
                    [
                        lambda: env_factory_fn(**env_kwargs)
                        for _ in range(config.num_envs)
                    ]
                )
            else:
                self.envs = DummyVecEnv(
                    [
                        lambda: env_factory_fn(**env_kwargs)
                        for _ in range(config.num_envs)
                    ]
                )
        else:
            raise ValueError(
                "env_id must be a string. currently not implemented for custom environments."
            )

        self.env_factory_fn = env_factory_fn

        self.observation_space = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space

        if supported_action_spaces is not None:
            if not isinstance(supported_action_spaces, tuple):
                raise ValueError("supported_action_spaces must be a tuple")
            if not isinstance(self.action_space, supported_action_spaces):
                raise ValueError(
                    f"Action space {self.action_space} is not supported. Supported action spaces are: {supported_action_spaces}"
                )

        utils.check_for_nested_spaces(self.observation_space)
        utils.check_for_nested_spaces(self.action_space)

        self.action_dist = make_proba_distribution(self.action_space)

        # If the action space is continuous, we need to learn the standard deviation
        if isinstance(self.action_dist, DiagGaussianDistribution):
            log_std_init = 0
            self.log_std = nn.Parameter(
                torch.ones(self.action_dist.action_dim) * log_std_init,
                requires_grad=True,
            ).to(self.device)

        self.num_envs = self.envs.num_envs

        self.n_steps = config.n_steps

        self.policy = policy

        self.device = utils.get_device(config.device)

        #TODO using log
        print("Using device:", self.device)

        self.seed = config.seed

        self.batch_size = config.batch_size

        self.memory = None

        self.timesteps = 0
        self.episodes = 0
        self.n_updates = 0

        self._mean_score_window = 100
        # history of scores each episode
        self.scores: list[float] = []
        # keep track of each env current running score
        self.current_scores = np.zeros(self.num_envs, dtype=np.float32)

        self.save_kwargs: list[str] = []

        self._last_obs: np.ndarray = None
        self._last_episode_starts: np.ndarray = None

        self.start_time = None
        self.end_time = None

        self.to(self.device)

        if config.model_compile:
            warnings.warn("currently torch.compile is not supported.")

            # placeholder for future use
            if False:
                if hasattr(torch, "compile"):
                    try:
                        self.policy = torch.compile(self.policy)
                    except Exception as e:
                        warnings.warn(
                            f"torch.compile failed:\n{e}\nusing uncompiled model."
                        )
                else:
                    warnings.warn(
                        "torch.compile is not available in this version of PyTorch."
                    )

    def set_mean_score_window(self, window: int):
        if not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer")

        self._mean_score_window = window

    @property
    def mean_scores(self):
        if len(self.scores) == 0:
            return []

        if len(self.scores) < self._mean_score_window:
            return [np.mean(self.scores[: i + 1]) for i in range(len(self.scores))]

        mean_scores = []
        cumsum = np.cumsum(self.scores, dtype=float)
        for i in range(len(self.scores)):
            if i < self._mean_score_window:
                mean_scores.append(cumsum[i] / (i + 1))
            else:
                mean_scores.append(
                    (cumsum[i] - cumsum[i - self._mean_score_window])
                    / self._mean_score_window
                )

        return mean_scores

    @property
    def logger(self):
        return self._logger

    @property
    def info(self):
        return {
            "scores": self.scores,
            "episodes": self.episodes,
            "n_updates": self.n_updates,
            "total_timesteps": self.timesteps,
        }

    def plot_scores(
        self, filename: Optional[str] = None, rolling_window: Optional[int] = None
    ):
        if rolling_window is None:
            rolling_window = self._mean_score_window

        orig_mean_score_window = self._mean_score_window
        self.set_mean_score_window(rolling_window)
        utils.plot_rl_style(self.scores, self._mean_score_window, filename)
        self.set_mean_score_window(orig_mean_score_window)

    def apply(self, fn: Callable[[nn.Module], None]):
        self.policy.apply(fn)

    def to(self, device):
        self.device = device
        if self.memory:
            self.memory.to(device)
        self.policy.to(device)

        return self

    def add_save_kwargs(self, name: str):
        if not isinstance(name, str):
            raise ValueError("name must be a string")

        if not hasattr(self, name):
            raise ValueError(f"{name} is not an attribute of the agent")

        self.save_kwargs.append(name)

    def save(self, save_dir: Path | str, filename: str = "checkpoint"):
        """Save the agent's information to a file.

        Args:
            path (Path | str): The directory to save the file.
            *post_names: Additional strings to append to the filename.
            save_key (list[str], optional): The keys to save. If not provided, defaults to ["policy", "total_timesteps", "scores", "mean_scores", "optimizers"] + self.save_kwargs
        """
        save_dir = Path(save_dir)

        # Get the information to be saved
        policy_info = self.policy.save_info()
        save_info = {
            "config": {
                "env_id": self.env_id,
                "config_class": f"{self.config.__class__.__module__}.{self.config.__class__.__qualname__}",
            }
            | asdict(self.config),
            "training_info": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "n_updates": self.n_updates,
                "scores": self.scores,
                "episodes": self.episodes,
                "total_timesteps": self.timesteps,
            },
            "additional_info": {},
            "model": policy_info["model"],
            "optimizers": policy_info["optimizers"],
            "lr_schedulers": policy_info["lr_schedulers"],
        }

        for name in self.save_kwargs:
            save_info["additional_info"][name] = getattr(self, name)

        # Create a temporary directory to store files before zipping

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save files to temporary directory
            torch.save(save_info["model"], temp_path / "model.pth")
            torch.save(save_info["optimizers"], temp_path / "optimizers.pth")
            torch.save(save_info["lr_schedulers"], temp_path / "lr_schedulers.pth")

            # Save configuration as yaml
            with open(temp_path / "config.yaml", "w") as f:
                yaml.dump(save_info["config"], f, sort_keys=False)

            # Save training information as yaml
            with open(temp_path / "training_info.yaml", "w") as f:
                yaml.dump(save_info["training_info"], f, sort_keys=False)

            # Save additional information as pickle for non-serializable data
            with open(temp_path / "additional_info.pkl", "wb") as f:
                pickle.dump(save_info["additional_info"], f)

            # Create the zip file
            save_dir.mkdir(parents=True, exist_ok=True)
            zip_path = (save_dir / filename).with_suffix(".zip")

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for file in temp_path.glob("*"):
                    zipf.write(file, arcname=file.name)

    def load_model(self, load_dir: Path | str) -> None:
        """Load the agent's information from a file.

        Args:
            path (Path | str): The directory to load the file from.
            *post_names: Additional strings to append to the filename.
            load_key (list[str], optional): The keys to load. If not provided, defaults to ["policy", "total_timesteps", "scores", "mean_scores", "optimizers"] + self.save_kwargs.
        """

        # paths is: .../env_id/agent_name/time/name.pth

        load_dir = Path(load_dir)

        # loading model property
        self.policy.load_model(
            torch.load(load_dir / "model.pth", map_location=self.device)
        )
        self.policy.load_optimizers(
            torch.load(load_dir / "optimizers.pth", map_location=self.device)
        )
        self.policy.load_lr_schedulers(
            torch.load(load_dir / "lr_schedulers.pth", map_location=self.device)
        )

    @staticmethod
    def load_config(load_dir: Path | str) -> AgentConfig:
        """Load the agent's configuration from a file.

        Args:
            path (Path | str): The directory to load the file from.
            *post_names: Additional strings to append to the filename.
        """

        load_dir = Path(load_dir)

        # loading configuration from yaml:
        with open(load_dir / "config.yaml", "r") as f:
            data: dict = yaml.safe_load(f)
            if "config_class" in data and "env_id" in data:
                config_class_str: str = data.pop("config_class")
                env_id = data.pop("env_id")

                module_name, class_name = config_class_str.rsplit(".", 1)

                # Import the module
                module = importlib.import_module(module_name)

                # Get the class from the module
                data_class = getattr(module, class_name)

            config = from_dict(
                data_class=data_class,
                data=data,
                config=Config(strict=True),
            )

        return env_id, config

    @classmethod
    def from_checkpoint(
        cls,
        env_id: str,
        policy: BasePolicy,
        path: Path | str,
        model_version: str = "last",
    ):
        """Create an agent from a checkpoint.

        Args:
            path (Path | str): The directory to load the file from.
            *post_names: Additional strings to append to the filename.

        Returns:
            cls: The created agent.
        """

        path = Path(path)

        load_file = auto_find_path(
            path, env_id=env_id, name=cls.__name__, model_version=model_version
        )

        # TODO: add logging
        print(f"Loading checkpoint from {load_file}")

        # Extract the zip file to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Open and extract the zip file
            with zipfile.ZipFile(load_file, "r") as zip_ref:
                zip_ref.extractall(temp_path)

            # Use the temporary directory as the load directory
            load_dir = temp_path

            # loading configuration from yaml:
            config_env_id, config = cls.load_config(load_dir)
            if config_env_id != env_id:
                warnings.warn(
                    f"Warning: env_id in config ({config_env_id}) does not match the provided env_id ({env_id}). Using the config env_id."
                )
                env_id = config_env_id

            # Create the agent instance
            agent = cls(env_id=env_id, policy=policy, config=config)
            # Load the model parameters
            agent.load_model(load_dir)

            # loading training information from yaml:
            with open(load_dir / "training_info.yaml", "r") as f:
                training_info = yaml.safe_load(f)
                agent.start_time = training_info.get("start_time", None)
                agent.end_time = training_info.get("end_time", None)
                agent.scores = training_info.get("scores", [])
                agent.episodes = training_info.get("episodes", 0)
                agent.timesteps = training_info.get("total_timesteps", 0)

            # loading additional information from pickle
            with open(load_dir / "additional_info.pkl", "rb") as f:
                additional_info: dict = pickle.load(f)
                for name in additional_info:
                    if name not in agent.save_kwargs:
                        agent.add_save_kwargs(name)
                        warnings.warn(
                            f"Warning: {name} is not in save_kwargs, but found in additional_info.pkl."
                        )

                    setattr(agent, name, additional_info[name])

        return agent

    def reset(self):
        r"""
        This method should call at the start of an episode (after :func:``env.reset``)
        """
        ...

    @abstractmethod
    def predict(self, observations: ObsType, deterministic: bool = True) -> ActType:
        """
        Perform an action based on the given observations.

        Parameters:
            observations (ObsType): The input observations which can be either a numpy array or a dictionary
            * ``NDArray`` shape - `[batch, *obs_shape]`
            * ``dict`` shape - `[key: sub_space]` with `sub_space` shape: `[batch, *sub_obs_shape]`
            deterministic (bool, optional): If True, the action is chosen deterministically. Defaults to True.

        Returns:
            ActType: The action to be performed.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def collect_buffer(self, deterministic: bool, callbacks: Type[Callbacks] = None):
        raise NotImplementedError

    @abstractmethod
    def learn(self) -> None:
        raise NotImplementedError

    def _setup_fit(
        self,
        total_timesteps: int,
        callback: Optional[Callbacks],
        reset_timesteps: bool,
        progress_bar: Optional[Type[tqdm]],
        save_dir: str | Path,
        log_dir: Optional[str],
        log_formats: Optional[list[str]],
    ) -> tuple[int, tqdm | None, Callbacks]:
        self._num_timesteps_at_start = 0
        if not reset_timesteps:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.timesteps
            self._num_timesteps_at_start = self.timesteps

        train_begin = datetime.now()

        self.start_time = train_begin.timestamp()

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_timesteps or self._last_obs is None:
            assert self.envs is not None
            self._last_obs = self.envs.reset()[0]
            self._last_episode_starts = np.ones((self.num_envs,), dtype=bool)

        # Create eval callback if needed
        if callback is None:
            callback = Callbacks(self)

        loop = None
        if progress_bar is not None:
            if issubclass(progress_bar, tqdm):
                loop = progress_bar(initial=self.timesteps, total=total_timesteps)
            else:
                warnings.warn("Invalid progress bar type. Disabling progress bar.")

        log_formats = log_formats if log_formats is not None else []

        log_dir = log_dir if log_dir is not None else save_dir

        save_dir: Path = (
            Path(save_dir)
            / self.env_id
            / self.name
            / train_begin.strftime("%Y-%m-%d_%H-%M-%S")
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        log_dir: Path = (
            Path(log_dir)
            / self.env_id
            / self.name
            / train_begin.strftime("%Y-%m-%d_%H-%M-%S")
        )
        log_dir.mkdir(parents=True, exist_ok=True)


        self._logger = configure(
            log_formats=log_formats,
        )
        self.logger.start(str(log_dir))

        return total_timesteps, loop, save_dir, callback

    def fit(
        self,
        total_timesteps: int,
        *,
        deterministic=False,
        reset_timesteps: bool = True,
        save_best=False,
        save_every=False,
        improve_threshold_percent: float = 0.0,
        save_dir="./checkpoints",
        log_dir: Optional[str] = None,
        log_formats: Optional[list[str]] = None,
        log_interval: int = 1,
        progress_bar: Optional[Type[tqdm]] = tqdm,
        callbacks: Type[Callbacks] = None,
    ):
        """
        Train the agent for a certain number of timesteps.

        Args:
            total_timesteps (int): The total number of timesteps to train the agent.
            deterministic (bool, optional): Whether to use deterministic actions during training. Defaults to False.
            reset_timesteps (bool, optional): Whether to reset the timestep counter at the start of training. Defaults to True.
            save_best (bool, optional): Whether to save the best model based on average score. Defaults to False.
            save_every (int | bool, optional): If an integer is provided, saves the model every 'save_every' updates. If True, saves every update. Defaults to False.
            improve_threshold_percent (float, optional): The minimum relative improvement in average score required to consider a new best model. Defaults to 0.0.
            save_dir (str | Path, optional): The directory where models will be saved. Defaults to "./checkpoints".
            log_dir (str | None, optional): The directory where logs will be saved. If None, defaults to save_dir
            log_formats (list[str] | None, optional): List of log formats for logging outputs ```stdout```, ```log```, ```json```, ```csv``` or ```tensorboard```. If None, defaults to [```stdout```].
            progress_bar (Type[tqdm] | None, optional): The progress bar class to use for displaying training progress. If None, no progress bar is shown. Defaults to tqdm.
            callbacks (Type[Callbacks] | None, optional): A callback class for custom behavior during training. If None, a default Callbacks instance is used. Defaults to None.

        """

        total_timesteps, loop, save_dir, callbacks = self._setup_fit(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_timesteps=reset_timesteps,
            progress_bar=progress_bar,
            save_dir=save_dir,
            log_dir=log_dir,
            log_formats=log_formats,
        )


        callbacks.on_train_begin()
        self.policy.train()

        best_score = float("-inf")

        last_save_best = None
        last_best_score = None

        iteration = 0

        while True:
            self.reset()
            timesteps, episodes = self.collect_buffer(deterministic, callbacks)
            iteration += 1

            self.timesteps += timesteps
            self.episodes += episodes

            if len(self.scores) == 0:
                avg_score = None
            else:
                avg_score = float(np.mean(self.scores[-self._mean_score_window :]))

            if save_best and avg_score is not None:
                relative_improvement = (
                    improve_threshold_percent + 1e-10
                )  # to ensure the first time it always saves
                if last_best_score is not None:
                    relative_improvement = (avg_score - last_best_score) / (
                        abs(last_best_score) + abs(avg_score) + 1e-10
                    )

                if (
                    avg_score >= best_score
                    and episodes > 0
                    and relative_improvement >= improve_threshold_percent
                ):
                    # only save if there is an improvement and at least one episode is done
                    # to avoid saving too many times when the score is not improved but one env finished an episode
                    # e.g. when the agent is not learning at all
                    # and the score is always 0, but one env finishes an episode every now and then
                    # we don't want to save the model every time that happens
                    # so we only save when the best score is updated

                    best_score = avg_score
                    if last_save_best is not None:
                        # remove the previous best model to save space
                        best_path = save_dir / last_save_best

                        suffix = [".zip", ".tar", ".gz"]
                        for s in suffix:
                            best_path_s = best_path.with_suffix(s)
                            if best_path_s.exists():
                                best_path_s.unlink(missing_ok=True)

                    save_file = f"best_{best_score:.2f}"

                    self.save(save_dir, save_file)
                    last_save_best = save_file
                    last_best_score = best_score

            if save_every:
                if self.n_updates % save_every == 0:
                    self.save(save_dir, f"{self.n_updates}_{avg_score}")

            if progress_bar:
                loop.update(timesteps if total_timesteps else episodes)
                loop.set_postfix(
                    {
                        "episodes": self.episodes,
                        "timesteps": self.timesteps,
                        "n_updates": self.n_updates,
                        "avg_score": avg_score,
                        "score": self.scores[-1] if len(self.scores) > 0 else None,
                    }
                )

            if self.timesteps >= total_timesteps:
                break

            if len(self.memory) > self.batch_size:
                callbacks.on_learn_begin()
                self.learn()
                callbacks.on_learn_end()
                self.n_updates += 1

            time_elapsed = (datetime.now().timestamp() - self.start_time)   # seconds
            fps = int((self.timesteps - self._num_timesteps_at_start) / time_elapsed)
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
            self.logger.record("time/total_timesteps", self.timesteps, exclude="tensorboard")
            self.logger.record("time/episodes", self.episodes, exclude="tensorboard")
            self.logger.record("time/n_updates", self.n_updates, exclude="tensorboard")
            self.logger.record("time/time_elapsed", time_elapsed, exclude="tensorboard")
            self.logger.record("time/fps", fps)

            self.logger.record("rollout/avg_score", avg_score)
            self.logger.record(
                "rollout/score",
                self.scores[-1] if len(self.scores) > 0 else None,
            )

            if log_interval > 0 and iteration % log_interval == 0:
                self.logger.dump(iteration)

        self.save(
            save_dir, f"last_{np.mean(self.scores[-self._mean_score_window :]):.2f}"
        )

        callbacks.on_train_end()
        self.end_time = datetime.now().timestamp()
        self.logger.close()

    def play(
        self,
        # env: EnvWithTransform = None,
        env_kwargs: dict[str, Any] = None,
        max_episode_steps: int = None,
        FPS: int = 30,
        stop_if_truncated: bool = True,
        deterministic=True,
        seed=None,
        options: Optional[dict[str, Any]] = None,
        jupyter: bool = False,
    ):
        if jupyter:
            from IPython.display import (
                display,  # pyright: ignore[reportMissingModuleSource] # type
            )
            from PIL import Image
        else:
            import pygame

            pygame.init()

        # _env_kwargs = self.env_kwargs | {"render_mode": "rgb_array" if jupyter else "human", "max_episode_steps": max_episode_steps}

        # env_kwargs = _env_kwargs if env_kwargs is None else env_kwargs  # complete remake, not override
        env_kwargs = (
            self.env_kwargs if env_kwargs is None else self.env_kwargs | env_kwargs
        )  # override
        env_kwargs |= {
            "render_mode": "rgb_array" if jupyter else "human",
            "max_episode_steps": None,
        }  # ensure these two keys

        # if env is None:
        env = self.env_factory_fn(**env_kwargs)

        score = 0
        obs = env.reset(seed=seed, options=options)[0]
        self.reset()

        done = False
        clock = Clock()

        time_step = 0

        with torch.no_grad():
            self.policy.eval()
            while not done:
                time_step += 1

                # use here to ensure max_episode_steps is always applied instead of env internal one
                if max_episode_steps and time_step > max_episode_steps:
                    break

                clock.tick(FPS)
                pixel = env.render()

                if jupyter:
                    display(Image.fromarray(pixel), clear=True)

                if isinstance(env.observation_space, gym.spaces.Dict):
                    _obs = {key: np.expand_dims(obs[key], 0) for key in obs}
                else:
                    _obs = np.expand_dims(obs, 0)

                action = self.predict(_obs, deterministic)

                next_obs, reward, terminated, truncated, info = env.step(action[0])

                done = terminated or (truncated and stop_if_truncated)

                obs = next_obs

                score += reward

                if not jupyter:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True

        if not jupyter:
            pygame.quit()

        return score, info

    def play_jupyter(
        self,
        # env: EnvWithTransform = None,
        env_kwargs: dict[str, Any] = None,
        max_episode_steps: int = None,
        FPS: int = 30,
        stop_if_truncated: bool = True,
        deterministic=True,
        seed=None,
        options: Optional[dict[str, Any]] = None,
    ):
        return self.play(
            # env
            env_kwargs,
            max_episode_steps,
            FPS,
            stop_if_truncated,
            deterministic,
            seed,
            options,
            jupyter=True,
        )

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


class OffPolicyAgent(AgentBase[ObsType, ActType]):
    memory: ReplayBuffer

    def __init__(
        self,
        env_id: str,
        policy: BasePolicy,
        config: OffPolicyAgentConfig = None,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
    ):
        config = config if config is not None else OffPolicyAgentConfig()
        super().__init__(
            env_id=env_id,
            policy=policy,
            config=config,
            supported_action_spaces=supported_action_spaces,
        )

        self.gamma = config.gamma

        self.memory = ReplayBuffer(
            buffer_size=config.buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            num_envs=self.num_envs,
            seed=self.seed,
        )

    @abstractmethod
    def learn(
        self,
    ) -> None:
        """
        Abstract method to be implemented by subclasses for learning from a batch of experiences.
        use self.memory.sample(self.batch_size) to get a batch of experiences.
        """
        ...

    def collect_buffer(
        self, deterministic: bool, callbacks: Type[Callbacks]
    ) -> tuple[int, int]:
        # TODO: this is not an episode, so the callbacks should be different
        callbacks.on_episode_begin()

        time_steps = 0
        episodes = 0

        self.policy.train()  # ensure in train mode
        for _ in range(self.n_steps):
            time_steps += 1

            with torch.no_grad():
                actions = self.predict(self._last_obs, deterministic)
            next_obs, rewards, terminated, truncated, info = self.envs.step(actions)

            self.memory.add(self._last_obs, actions, rewards, terminated)

            self._last_obs = next_obs
            self._last_episode_starts = np.array(
                terminated | truncated
            )  # episode starts is just done

            self.current_scores += np.array(rewards, dtype=np.float32)

            # if an env is done, record the score and reset the current score for that env
            self.scores.extend(self.current_scores[self._last_episode_starts].tolist())
            # reset the current score for that env
            self.current_scores[self._last_episode_starts] = 0.0
            # count how many episodes finished
            episodes += np.sum(self._last_episode_starts)

        # TODO: this is not an episode, so the callbacks should be different
        callbacks.on_episode_end()

        return time_steps, int(episodes)


class OnPolicyAgent(AgentBase[ObsType, ActType]):
    """
    This is a base class for on-policy agents.
    Because there are on-policy agents that are not actor-critic, so this is a placeholder for the future on-policy implementations if they arise.
    """

    pass


class ActorCriticPolicyAgent(AgentBase[ObsType, ActType]):
    """This is a base class for actor-critic agents.
    You can consider this class as an on-policy agent with a value function.
    """

    memory: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        env_id: str,
        policy: ActorCriticPolicy,
        config: ActorCriticAgentConfig = None,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
    ):
        config = config if config is not None else ActorCriticAgentConfig()
        super().__init__(
            env_id=env_id,
            policy=policy,
            config=config,
            supported_action_spaces=supported_action_spaces,
        )

        self.memory = RolloutBuffer(
            buffer_size=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            num_envs=self.num_envs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            seed=self.seed,
        )

        self._last_obs = None

    @abstractmethod
    def predict(self, state: ObsType, deterministic: bool = True) -> ActType:
        pass

    @abstractmethod
    def learn(self) -> None:
        """
        Perform learning using the experiences stored in the memory buffer.

        This method should be overridden by subclasses to implement specific learning algorithms.
        The method should utilize the experiences stored in `self.memory` to update the agent's policy.
        use self.memory.get(batch_size) to get a generator that yields batches of experiences.
        """
        raise NotImplementedError

    def evaluate_actions(
        self, obs: ObsType, actions: ActType
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions using the current policy.

        Args:
            obs (ObsType): The input observations which can be either a numpy array or a dictionary
                * ``NDArray`` shape - `[batch, *obs_shape]`
                * ``dict`` shape - `[key: sub_space]` with `sub_space` shape: `[batch, *sub_obs_shape]`
            actions (ActType): The actions to evaluate which can be either a numpy array or a dictionary
                * ``NDArray`` shape - `[batch, *action_shape]`
                * ``dict`` shape - `[key: sub_space]` with `sub_space` shape: `[batch, *sub_action_shape]`
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - values (torch.Tensor): The value estimates for the observations.
                - action_log_probs (torch.Tensor): The log probabilities of the actions.
                - entropy (torch.Tensor): The entropy of the action distribution.
        """
        action_logits, value_logits = self.policy.forward(obs)
        values = value_logits.squeeze(-1)

        distribution = self.distribution(action_logits)
        action_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, action_log_probs, entropy

    def collect_buffer(self, deterministic: bool, callbacks: Type[Callbacks] = None):
        # TODO: this is not an episode, so the callbacks should be different
        callbacks.on_episode_begin()

        self.memory.reset()

        time_steps = 0
        episodes = 0

        self.policy.train()  # ensure in train mode
        for _ in range(self.n_steps):
            time_steps += 1

            with torch.no_grad():
                action_logits, value_logits = self.policy.forward(
                    utils.to_torch(self._last_obs, self.device)
                )

                distribution = self.distribution(action_logits)

                # get the action
                if deterministic:
                    actions = distribution.mode()
                else:
                    actions = distribution.sample()

                log_probs = distribution.log_prob(actions).cpu().numpy()
                actions = actions.cpu().numpy()
                values = value_logits.squeeze(-1).cpu().numpy()

            next_obs, rewards, terminated, truncated, info = self.envs.step(actions)

            self.memory.add(
                self._last_obs,
                actions,
                rewards,
                values,
                log_probs,
                self._last_episode_starts,
            )

            self._last_obs = next_obs
            self._last_episode_starts = np.array(
                terminated | truncated
            )  # episode starts is just done

            self.current_scores += np.array(rewards, dtype=np.float32)

            # if an env is done, record the score and reset the current score for that env
            self.scores.extend(self.current_scores[self._last_episode_starts].tolist())
            # reset the current score for that env
            self.current_scores[self._last_episode_starts] = 0.0
            # count how many episodes finished
            episodes += np.sum(self._last_episode_starts)

        # compute the value of the last observation
        with torch.no_grad():
            values = (
                self.policy.forward_critic(utils.to_torch(self._last_obs, self.device))
                .squeeze(-1)
                .cpu()
                .numpy()
            )

        self.memory.calc_advantages_and_returns(
            last_values=values, last_terminals=self._last_episode_starts
        )

        # TODO: this is not an episode, so the callbacks should be different
        callbacks.on_episode_end()

        return time_steps, int(episodes)
