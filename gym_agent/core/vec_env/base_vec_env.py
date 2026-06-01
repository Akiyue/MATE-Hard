import inspect
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Callable, Optional, Sequence, Union

import cloudpickle
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ...utils import check_for_nested_spaces, stack_dict

# Define type aliases here to avoid circular import
# Used when we want to access one or more VecEnv
# VecEnvObs is what is returned by the reset() method
# it contains the observation for each env
VecEnvObs = Union[np.ndarray, dict[str, np.ndarray], tuple[np.ndarray, ...]]
# VecEnvStepReturn is what is returned by the step() method
# it contains the observation, reward, done, info for each env
VecEnvStepReturn = tuple[
    VecEnvObs,
    np.ndarray,
    np.ndarray[tuple[int, ...], np.bool_],
    np.ndarray[tuple[int, ...], np.bool_],
    list[dict],
]


def tile_images(images_nhwc: Sequence[np.ndarray]) -> np.ndarray:  # pragma: no cover
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    :param images_nhwc: list or array of images, ndim=4 once turned into array.
        n = batch index, h = height, w = width, c = channel
    :return: img_HWc, ndim=3
    """
    img_nhwc = np.asarray(images_nhwc)
    n_images, height, width, n_channels = img_nhwc.shape
    # new_height was named H before
    new_height = int(np.ceil(np.sqrt(n_images)))
    # new_width was named W before
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(
        list(img_nhwc)
        + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)]
    )
    # img_HWhwc
    out_image = img_nhwc.reshape((new_height, new_width, height, width, n_channels))
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape((new_height * height, new_width * width, n_channels))  # type: ignore[assignment]
    return out_image


def batch_space(space: spaces.Space, n: int) -> spaces.Space:
    if isinstance(space, spaces.Box):
        low = np.repeat(space.low[np.newaxis, :], n, axis=0)
        high = np.repeat(space.high[np.newaxis, :], n, axis=0)
        return spaces.Box(low=low, high=high, dtype=space.dtype)  # type: ignore[call-arg]
    elif isinstance(space, spaces.Discrete):
        return spaces.MultiDiscrete([space.n for _ in range(n)])
    elif isinstance(space, spaces.MultiDiscrete):
        return spaces.MultiDiscrete(np.repeat(space.nvec[np.newaxis, :], n, axis=0))
    elif isinstance(space, spaces.MultiBinary):
        if isinstance(space.n, int):
            return spaces.MultiBinary([n, space.n])  # shape (n, space.n)
        else:
            return spaces.MultiBinary([n, *space.n])  # shape (n, *space.n)

    elif isinstance(space, spaces.Dict):
        return spaces.Dict(
            {key: batch_space(subspace, n) for (key, subspace) in space.spaces.items()}
        )
    else:
        raise NotImplementedError(f"{space} space is not supported")


def _maybe_cast_dict(sequence: Sequence[Any]) -> Any:
    if isinstance(sequence[0], dict):
        return stack_dict(sequence)  # type: ignore[return-value]
    elif isinstance(sequence, np.ndarray):
        return sequence
    return np.stack(sequence)


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.

    :param num_envs: Number of environments
    :param observation_space: Observation space of 1 environment
    :param action_space: Action space of 1 environment
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
    ):
        self.env_fns = env_fns
        num_envs = len(env_fns)
        self.num_envs = num_envs
        dummy_env = env_fns[0]()

        self.env_id = dummy_env.spec.id if dummy_env.spec else "unknown"

        self.single_observation_space = dummy_env.observation_space
        self.single_action_space = dummy_env.action_space
        check_for_nested_spaces(self.single_observation_space)
        check_for_nested_spaces(self.single_action_space)
        self.observation_space = batch_space(dummy_env.observation_space, num_envs)
        self.action_space = batch_space(dummy_env.action_space, num_envs)

        # store info returned by the reset method
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        # seeds to be used in the next call to env.reset()
        self._seeds: list[Optional[int]] = [None for _ in range(num_envs)]
        # options to be used in the next call to env.reset()
        self._options: list[dict[str, Any]] = [{} for _ in range(num_envs)]

        try:
            render_modes = [dummy_env.render_mode] * num_envs
        except AttributeError:
            warnings.warn(
                "The `render_mode` attribute is not defined in your environment. It will be set to None."
            )
            render_modes = [None for _ in range(num_envs)]

        assert all(render_mode == render_modes[0] for render_mode in render_modes), (
            "render_mode mode should be the same for all environments"
        )

        self.render_mode = render_modes[0]

        render_modes = []
        if self.render_mode is not None:
            if self.render_mode == "rgb_array":
                # SB3 uses OpenCV for the "human" mode
                render_modes = ["human", "rgb_array"]
            else:
                render_modes = [self.render_mode]

        self.metadata = {"render_modes": render_modes}

    def _reset_seeds(self) -> None:
        """
        Reset the seeds that are going to be used at the next reset.
        """
        self._seeds = [None for _ in range(self.num_envs)]

    def _reset_options(self) -> None:
        """
        Reset the options that are going to be used at the next reset.
        """
        self._options = [{} for _ in range(self.num_envs)]

    @abstractmethod
    def _reset(self) -> VecEnvObs:
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        :return: observation, info
        """
        raise NotImplementedError()

    def reset(self) -> tuple[VecEnvObs, dict[str, Any]]:
        obs = self._reset()
        self._reset_seeds()
        self._reset_options()
        return _maybe_cast_dict(obs), _maybe_cast_dict(self.reset_infos)

    @abstractmethod
    def step_async(self, actions: np.ndarray) -> None:
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        raise NotImplementedError()

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        """
        Wait for the step taken with step_async().

        :return: observation, reward, done, information
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """
        Clean up the environment's resources.
        """
        raise NotImplementedError()

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, terminated, truncated, information
        """
        self.step_async(actions)
        pre_obs, pre_rews, pre_terminated, pre_truncated, pre_infos = self.step_wait()
        return (
            _maybe_cast_dict(pre_obs),
            _maybe_cast_dict(pre_rews),
            _maybe_cast_dict(pre_terminated),
            _maybe_cast_dict(pre_truncated),
            _maybe_cast_dict(pre_infos),
        )

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        """
        Return RGB images from each environment when available
        """
        raise NotImplementedError

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Gym environment rendering

        :param mode: the rendering type
        """

        if mode == "human" and self.render_mode != mode:
            # Special case, if the render_mode="rgb_array"
            # we can still display that image using opencv
            if self.render_mode != "rgb_array":
                warnings.warn(
                    f"You tried to render a VecEnv with mode='{mode}' "
                    "but the render mode defined when initializing the environment must be "
                    f"'human' or 'rgb_array', not '{self.render_mode}'."
                )
                return None

        elif mode and self.render_mode != mode:
            warnings.warn(
                f"""Starting from gymnasium v0.26, render modes are determined during the initialization of the environment.
                We allow to pass a mode argument to maintain a backwards compatible VecEnv API, but the mode ({mode})
                has to be the same as the environment render mode ({self.render_mode}) which is not the case."""
            )
            return None

        mode = mode or self.render_mode

        if mode is None:
            warnings.warn(
                "You tried to call render() but no `render_mode` was passed to the env constructor."
            )
            return None

        # mode == self.render_mode == "human"
        # In that case, we try to call `self.env.render()` but it might
        # crash for subprocesses

        if mode == "rgb_array" or mode == "human":
            # call the render method of the environments
            images = self.get_images()
            # Create a big image by tiling images from subprocesses
            bigimg = tile_images(images)  # type: ignore[arg-type]

            if mode == "human":
                # Display it using OpenCV
                import cv2

                cv2.imshow("vecenv", bigimg[:, :, ::-1])
                cv2.waitKey(1)
            else:
                return bigimg

        return None

    def seed(self, seed: Optional[int] = None) -> Sequence[Union[None, int]]:
        """
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.
        WARNING: since gym 0.26, those seeds will only be passed to the environment
        at the next reset.

        :param seed: The random seed. May be None for completely random seeding.
        :return: Returns a list containing the seeds for each individual env.
            Note that all list elements may be None, if the env does not return anything when being seeded.
        """
        if seed is None:
            # To ensure that subprocesses have different seeds,
            # we still populate the seed variable when no argument is passed
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))

        self._seeds = [seed + idx for idx in range(self.num_envs)]
        return self._seeds

    def set_options(self, options: Optional[Union[list[dict], dict]] = None) -> None:
        """
        Set environment options for all environments.
        If a dict is passed instead of a list, the same options will be used for all environments.
        WARNING: Those options will only be passed to the environment at the next reset.

        :param options: A dictionary of environment options to pass to each environment at the next reset.
        """
        if options is None:
            options = {}
        # Use deepcopy to avoid side effects
        if isinstance(options, dict):
            self._options = deepcopy([options] * self.num_envs)
        else:
            self._options = deepcopy(options)

    @property
    def unwrapped(self) -> "VecEnv":
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

class VecEnvWrapper(VecEnv):
    """
    Vectorized environment base class

    :param venv: the vectorized environment to wrap
    :param observation_space: the observation space (can be None to load from venv)
    :param action_space: the action space (can be None to load from venv)
    """

    def __init__(
        self,
        venv: VecEnv,
        observation_space: Optional[spaces.Space] = None,
        action_space: Optional[spaces.Space] = None,
    ):
        self.venv = venv

        super().__init__(
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space,
        )
        self.class_attributes = dict(inspect.getmembers(self.__class__))

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    @abstractmethod
    def _reset(self) -> VecEnvObs:
        pass

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        pass

    def seed(self, seed: Optional[int] = None) -> Sequence[Union[None, int]]:
        return self.venv.seed(seed)

    def set_options(self, options: Optional[Union[list[dict], dict]] = None) -> None:
        return self.venv.set_options(options)

    def close(self) -> None:
        return self.venv.close()

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        return self.venv.render(mode=mode)

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        return self.venv.get_images()

    def __getattr__(self, name: str) -> Any:
        """Find attribute from wrapped venv(s) if this wrapper does not have it.
        Useful for accessing attributes from venvs which are wrapped with multiple wrappers
        which have unique attributes of interest.
        """
        blocked_class = self.getattr_depth_check(name, already_found=False)
        if blocked_class is not None:
            own_class = f"{type(self).__module__}.{type(self).__name__}"
            error_str = (
                f"Error: Recursive attribute lookup for {name} from {own_class} is "
                f"ambiguous and hides attribute from {blocked_class}"
            )
            raise AttributeError(error_str)

        return self.getattr_recursive(name)

    def _get_all_attributes(self) -> dict[str, Any]:
        """Get all (inherited) instance and class attributes

        :return: all_attributes
        """
        all_attributes = self.__dict__.copy()
        all_attributes.update(self.class_attributes)
        return all_attributes

    def getattr_recursive(self, name: str) -> Any:
        """Recursively check wrappers to find attribute.

        :param name: name of attribute to look for
        :return: attribute
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes:  # attribute is present in this wrapper
            attr = getattr(self, name)
        elif hasattr(self.venv, "getattr_recursive"):
            # Attribute not present, child is wrapper. Call getattr_recursive rather than getattr
            # to avoid a duplicate call to getattr_depth_check.
            attr = self.venv.getattr_recursive(name)
        else:  # attribute not present, child is an unwrapped VecEnv
            attr = getattr(self.venv, name)

        return attr


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)
