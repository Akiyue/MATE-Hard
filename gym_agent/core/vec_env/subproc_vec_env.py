import multiprocessing as mp
import warnings
from collections.abc import Sequence
from multiprocessing import Process
from multiprocessing.connection import Connection
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np

from gym_agent.core.transforms import EnvWithTransform
from gym_agent.core.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)


def _worker(
    remote: Connection,
    parent_remote: Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    parent_remote.close()
    env: EnvWithTransform = env_fn_wrapper.var()
    reset_info: Optional[dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                # convert to SB3 VecEnv api

                if terminated or truncated:
                    # save final observation where user can get it, then reset
                    info["final_observation"] = observation
                    observation, reset_info = env.reset()

                remote.send(
                    (observation, reward, terminated, truncated, info, reset_info)
                )

            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            # elif cmd == "get_spaces":
            # remote.send((env.observation_space, env.action_space))
            # elif cmd == "env_method":
            #     method = env.get_wrapper_attr(data[0])
            #     remote.send(method(*data[1], **data[2]))
            # elif cmd == "get_attr":
            #     remote.send(env.get_wrapper_attr(data))
            # elif cmd == "has_attr":
            #     try:
            #         env.get_wrapper_attr(data)
            #         remote.send(True)
            #     except AttributeError:
            #         remote.send(False)
            # elif cmd == "set_attr":
            #     remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            # elif cmd == "is_wrapped":
            #     remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break
        except KeyboardInterrupt:
            break


class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    remotes: list[Connection]
    work_remotes: list[Connection]

    def __init__(
        self, env_fns: list[Callable[[], gym.Env]], start_method: Optional[str] = None
    ):
        super().__init__(env_fns=env_fns)

        self.waiting = False
        self.closed = False

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.num_envs)]
        )
        self.processes: list[Process] = []
        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process: Process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, terminated, truncated, infos, self.reset_infos = zip(*results)
        return obs, rews, terminated, truncated, infos

    def _reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        obs, self.reset_infos = zip(*results)
        return obs

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remotes]
        for pipe in self.remotes:
            # gather render return from subprocesses
            pipe.send(("render", None))
        outputs = [pipe.recv() for pipe in self.remotes]
        return outputs

    # def has_attr(self, attr_name: str) -> bool:
    # """Check if an attribute exists for a vectorized environment. (see base class)."""
    # target_remotes = self._get_target_remotes(indices=None)
    # for remote in target_remotes:
    #     remote.send(("has_attr", attr_name))
    # return all([remote.recv() for remote in target_remotes])

    # def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
    #     """Return attribute from vectorized environment (see base class)."""
    #     target_remotes = self._get_target_remotes(indices)
    #     for remote in target_remotes:
    #         remote.send(("get_attr", attr_name))
    #     return [remote.recv() for remote in target_remotes]

    # def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
    #     """Set attribute inside vectorized environments (see base class)."""
    #     target_remotes = self._get_target_remotes(indices)
    #     for remote in target_remotes:
    #         remote.send(("set_attr", (attr_name, value)))
    #     for remote in target_remotes:
    #         remote.recv()

    # def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
    #     """Call instance methods of vectorized environments."""
    #     target_remotes = self._get_target_remotes(indices)
    #     for remote in target_remotes:
    #         remote.send(("env_method", (method_name, method_args, method_kwargs)))
    #     return [remote.recv() for remote in target_remotes]

    # def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
    #     """Check if worker environments are wrapped with a given wrapper"""
    #     target_remotes = self._get_target_remotes(indices)
    #     for remote in target_remotes:
    #         remote.send(("is_wrapped", wrapper_class))
    #     return [remote.recv() for remote in target_remotes]

    # def _get_target_remotes(self, indices: VecEnvIndices) -> list[Any]:
    #     """
    #     Get the connection object needed to communicate with the wanted
    #     envs that are in subprocesses.

    #     :param indices: refers to indices of envs.
    #     :return: Connection object to communicate between processes.
    #     """
    #     indices = self._get_indices(indices)
    #     return [self.remotes[i] for i in indices]
