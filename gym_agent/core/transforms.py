import gymnasium as gym
import numpy as np

from typing import Any, Type


class Transform:
    """
    A base class for transformations.
    Methods
    -------
    reset(**kwargs)
        Resets the transformation parameters. This method should be overridden by subclasses.
    __call__(*args, **kwargs)
        Applies the transformation. This method must be implemented by subclasses.
    __repr__()
        Returns a string representation of the transformation class.
    """

    def reset(self, **kwargs): ...

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Transform must be implemented")

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Compose(Transform):
    """
    Compose multiple Transform objects into a single transform.
    Args:
        *args: Variable length argument list of Transform objects.
    Attributes:
        args (list[Transform]): List of Transform objects.
    Methods:
        append(tfm: Transform):
            Append a Transform object to the list of transforms.
        __call__(X: np.ndarray):
            Apply all the transforms in sequence to the input array X.
        __repr__():
            Return a string representation of the Compose object.
    """

    def __init__(self, *args):
        """
        Initialize the object with a list of Transform instances.
        Args:
            *args: Variable length argument list, each should be an instance of Transform.
        Raises:
            TypeError: If any argument is not an instance of Transform.
        """
        self.args: list[Transform] = []
        for tfm in args:
            self.append(tfm)

    def append(self, tfm: Transform):
        """
        Appends a Transform object to the args list.
        Parameters:
            tfm (Transform): The Transform object to be appended.
        Raises:
            TypeError: If the provided tfm is not an instance of Transform.

        """
        if not isinstance(tfm, Transform):
            raise TypeError("tfm must be Transform")

        self.args.append(tfm)

    def __call__(self, X: np.ndarray):
        for tfm in self.args:
            X = tfm(X)

        return X

    def __repr__(self):
        res = "Compose(\n"
        for tfm in self.args:
            res += f"\t{tfm}\n"
        res += ")"
        return res


class Normalize(Transform):
    """
    A class used to normalize a numpy array by subtracting the mean and dividing by the standard deviation.

    Attributes
    ----------
    mean : float
        The mean value to subtract from the array.
    std : float
        The standard deviation value to divide the array by.

    Methods
    -------
    __call__(X: np.ndarray) -> np.ndarray
        Applies normalization to the input numpy array.

    __repr__()
        Returns a string representation of the Normalize object.
    """

    def __init__(self, mean, std):
        """
        Initializes the instance with mean and standard deviation.

        Args:
            mean (float): The mean value.
            std (float): The standard deviation value.
        """
        self.mean = mean
        self.std = std

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def __repr__(self):
        return f"Normalize(mean={self.mean}, std={self.std})"


class EnvWithTransform(gym.Wrapper):
    """
    A wrapper for gym environments that allows for transformations on observations, actions, and rewards.

    Attributes
    =====
        observation_transform (Transform): A transformation to apply to observations.
        action_transform (Transform): A transformation to apply to actions.
        reward_transform (Transform): A transformation to apply to rewards.

    Methods
    =====
        set_observation_transform(tfm: Type[Transform]):
            Sets the transformation to apply to observations.
        set_action_transform(tfm: Type[Transform]):
            Sets the transformation to apply to actions.
        set_reward_transform(tfm: Type[Transform]):
            Sets the transformation to apply to rewards.
        observation(observation):
            Applies the observation transformation if set.
        action(action):
            Applies the action transformation if set.
        reward(reward):
            Applies the reward transformation if set.
        step(action):
            Steps the environment with the transformed action and returns the transformed observation, reward, and other info.
        reset(**kwargs) -> tuple[Any, dict[str, Any]]:
            Resets the environment and the transformations if set.
    """

    def __init__(
        self,
        env: gym.Env,
        observation_transform: Transform = None,
        action_transform: Transform = None,
        reward_transform: Transform = None,
    ):
        """
        Initializes the utility class with the given environment.

        Args:
            env: The environment to be used by the utility class.

        Attributes:
            observation_transform: A transformation function for observations, initialized to None.
            action_transform: A transformation function for actions, initialized to None.
            reward_transform: A transformation function for rewards, initialized to None.
        """
        super().__init__(env)

        self._observation_transform = observation_transform
        if hasattr(self._observation_transform, "observation_space"):
            self.observation_space = self._observation_transform.observation_space

        self._action_transform = action_transform
        if hasattr(self._action_transform, "action_space"):
            self.action_space = self._action_transform.action_space

        self._reward_transform = reward_transform
        if hasattr(self._reward_transform, "reward_range"):
            self.reward_range = self._reward_transform.reward_range

    def add_wrapper(self, wrapper: Type[gym.Wrapper], *args, **kwargs):
        self.env = wrapper(self.env, *args, **kwargs)

    def set_observation_transform(self, tfm: Transform):
        """
        Sets the observation transform for the environment.
        Parameters:
            tfm (Transform): An instance of the Transform class. This transform will be applied to the observations.
        Raises:
            TypeError: If the provided tfm is not an instance of the Transform class.
        Notes:
            If the provided transform has an 'observation_space' attribute, it will be assigned to the environment's observation_space.
        """
        if not isinstance(tfm, Transform):
            raise TypeError("observation_transform must be Transform")

        self._observation_transform = tfm
        if hasattr(self._observation_transform, "observation_space"):
            self.observation_space = self._observation_transform.observation_space

    def set_action_transform(self, tfm: Transform):
        """
        Sets the action transform for the object.

        Parameters:
            tfm (Transform): The transform to be set. Must be an instance of the Transform class.

        Raises:
            TypeError: If the provided tfm is not an instance of the Transform class.

        Notes:
            If the provided transform has an 'action_space' attribute, it will be assigned to the object's 'action_space'.
        """
        if not isinstance(tfm, Transform):
            raise TypeError("action_transform must be Transform")

        self._action_transform = tfm
        if hasattr(self._action_transform, "action_space"):
            self.action_space = self._action_transform.action_space

    def set_reward_transform(self, tfm: Transform):
        """
        Sets the reward transformation function for the environment.
        Parameters:
            tfm (Transform): An instance of the Transform class that defines the reward transformation.
        Raises:
            TypeError: If the provided tfm is not an instance of the Transform class.
        Notes:
            If the provided Transform instance has a 'reward_range' attribute, it will be used to set the environment's reward range.
        """
        if not isinstance(tfm, Transform):
            raise TypeError("reward_transform must be Transform")

        self._reward_transform = tfm
        if hasattr(self._reward_transform, "reward_range"):
            self.reward_range = self._reward_transform.reward_range

    def observation(self, observation):
        if self._observation_transform:
            return self._observation_transform(observation)
        return observation

    def action(self, action):
        if self._action_transform:
            return self._action_transform(action)
        return action

    def reward(self, reward):
        if self._reward_transform:
            return self._reward_transform(reward)
        return reward

    def step(self, action):
        """Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        .. versionchanged:: 0.26

            The Step API was changed removing ``done`` in favor of ``terminated`` and ``truncated`` to make it clearer
            to users when the environment had terminated or truncated which is critical for reinforcement learning
            bootstrapping algorithms.

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
            done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """
        observation, reward, terminated, truncated, info = self.env.step(
            self.action(action)
        )

        observation = self.observation(observation)
        reward = self.reward(reward)

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset.

        Therefore, :meth:`reset` should (in the typical use case) be called with a seed right after initialization and then never again.

        For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.

        .. versionchanged:: v0.25

            The ``return_info`` parameter was removed and now info is expected to be returned.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)

        Returns:
            observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        if self._observation_transform:
            self._observation_transform.reset(seed=seed, options=options)

        if self._action_transform:
            self._action_transform.reset(seed=seed, options=options)

        if self._reward_transform:
            self._reward_transform.reset(seed=seed, options=options)

        obs, info = self.env.reset(seed=seed, options=options)

        self.env_reward = None

        return self.observation(obs), info
