import mate
from mate.agents import GreedyTargetAgent
from mate.constants import TERRAIN_SIZE
from mate.entities import Obstacle
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium import spaces

import numpy as np

class MateCameraDictObsWrapper(ObservationWrapper):
    def __init__(self, env, warehouse_include_R=False, target_include_R=False, target_include_loaded=False, expand_preserved_and_obstacle=True, see_all_cameras = True, add_flag_if_see_all_cameras = False):
        super().__init__(env)
        self.target_include_R = target_include_R
        self.target_include_loaded = target_include_loaded
        self.expand_preserved_and_obstacle = expand_preserved_and_obstacle
        self.see_all_cameras = see_all_cameras
        self.add_flag_if_see_all_cameras = add_flag_if_see_all_cameras
        self.warehouse_include_R = warehouse_include_R

        self.num_cameras = self.unwrapped.num_cameras

        preserved_shape = self.unwrapped.num_warehouses*2 + (1 if warehouse_include_R else 0)
        preserved_shape = (self.num_cameras, preserved_shape) if expand_preserved_and_obstacle else (preserved_shape,)

        preserved_space = spaces.Box(low=-1.0, high=1.0, shape=preserved_shape, dtype=np.float64)
        self_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_cameras, 8), dtype=np.float64)   # x, y, R, phi, theta, Rmax, phimax, thetamax

        target_dim = 3
        if target_include_R:
            target_dim += 1
        if target_include_loaded:
            target_dim += 1
        target_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_cameras, self.unwrapped.num_targets*target_dim,), dtype=np.float64)

        obstacle_shape = self.unwrapped.num_obstacles*3
        obstacle_shape = (self.num_cameras, obstacle_shape) if expand_preserved_and_obstacle else (obstacle_shape,)

        obstacle_space = spaces.Box(low=-1.0, high=1.0, shape=obstacle_shape, dtype=np.float64)   # x, y, R

        d = 6 if not see_all_cameras or add_flag_if_see_all_cameras else 5
        teammate_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_cameras, d*(self.num_cameras-1),), dtype=np.float64)   # x, y, R, phi, theta, (flag)

        self.observation_space = spaces.Dict({
            "preserved": preserved_space,
            "self": self_space,
            "targets": target_space,
            "obstacles": obstacle_space,
            "teammate": teammate_space,
        })

        self.obstacles_state = None
        self.preserved_state = None

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ):
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""
        self.obstacles_state = None
        self.preserved_state = None
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info


    def observation(self, obs):
        # PRESERVE OBSERVATION NORMALIZATION
        ignore_dim = 4 # Nc, Nt, No, self_state_index
        preserved_dim = self.unwrapped.num_warehouses*2+1
        offset = 0 if self.warehouse_include_R else -1
        if self.preserved_state is None:
            self.preserved_state = obs[:, ignore_dim:ignore_dim + preserved_dim + offset] / TERRAIN_SIZE  # (ware_house_x, ware_house_y)*4 (+ warehouse_radius)
            if not self.expand_preserved_and_obstacle:
                self.preserved_state = self.preserved_state[0]
        ignore_dim += preserved_dim     # increase ignore_dim for camera state

        # CAMERA STATE OBSERVATION NORMALIZATION
        self_state_dim = 9    # x, y, r, Rcos, Rsin, theta, Rmax, phimax, thetamax
        self_state_x, self_state_y, self_state_r, Rcos, Rsin, self_state_theta, self_state_Rmax, self_state_phimax, self_state_thetamax = obs[:, ignore_dim:ignore_dim+self_state_dim].T

        self_state_x /= TERRAIN_SIZE
        self_state_y /= TERRAIN_SIZE
        self_state_R = np.sqrt(Rcos**2 + Rsin**2) / TERRAIN_SIZE
        self_state_phi = np.arctan2(Rsin, Rcos) / np.pi
        self_state_theta = self_state_theta / 180
        self_state_phimax /= 180
        self_state_Rmax /= TERRAIN_SIZE
        self_state_thetamax /= 180
        self_state = np.stack([self_state_x, self_state_y, self_state_R, self_state_phi, self_state_theta, self_state_Rmax, self_state_phimax, self_state_thetamax], axis=1)      # exclude r
        ignore_dim += self_state_dim   # increase ignore_dim for target state

        # TARGET STATE OBSERVATION NORMALIZATION
        target_state_dim = 5 * self.unwrapped.num_targets    # x, y, R, loaded, flag
        target_state = obs[:, ignore_dim:ignore_dim+target_state_dim]

        target_state[:, np.arange(0, target_state_dim, 5)] /= TERRAIN_SIZE  # x normalization
        target_state[:, np.arange(1, target_state_dim, 5)] /= TERRAIN_SIZE  # y normalization
        target_state[:, np.arange(2, target_state_dim, 5)] /= TERRAIN_SIZE  # R normalization

        exclude_indice = []
        if not self.target_include_R:
            exclude_indice += np.arange(2, target_state_dim, 5).tolist()    # R
        if not self.target_include_loaded:
            exclude_indice += np.arange(3, target_state_dim, 5).tolist()    # loaded

        target_state = np.delete(target_state, exclude_indice, axis=1)
        ignore_dim += target_state_dim   # increase ignore_dim for obstacle state

        # OBSTACLE STATE OBSERVATION NORMALIZATION
        if self.obstacles_state is None:
            self.obstacles_state = np.zeros(self.unwrapped.num_obstacles*3, dtype=np.float64)

            for i in range(self.unwrapped.num_obstacles):
                assert isinstance(self.unwrapped.obstacles[i], Obstacle)
                self.obstacles_state[i*3:(i+1)*3] = self.unwrapped.obstacles[i].state()

            self.obstacles_state /= TERRAIN_SIZE

            if self.expand_preserved_and_obstacle:
                self.obstacles_state = np.repeat(self.obstacles_state, self.num_cameras, axis=0)

        ignore_dim += self.unwrapped.num_obstacles*4   # x, y, R, flag

        # TEAMMATE STATE OBSERVATION NORMALIZATION
        if not self.see_all_cameras:
            teammate_dim = 7 * self.num_cameras   # x, y, r, Rcos, Rsin, theta, flag
            teammate_state = np.zeros((self.num_cameras, teammate_dim - 7), dtype=np.float64)

            for i in range(self.num_cameras):
                s: np.ndarray = obs[i, ignore_dim:ignore_dim + teammate_dim]
                current_indice = np.arange(i * 7, (i + 1) * 7)
                teammate_state[i] = np.delete(s, current_indice)


                teammate_state[i, np.arange(0, teammate_dim - 7, 7)] /= TERRAIN_SIZE    # x
                teammate_state[i, np.arange(1, teammate_dim - 7, 7)] /= TERRAIN_SIZE    # y

                Rcos = teammate_state[i, np.arange(3, teammate_dim - 7, 7)] # Rcos
                Rsin = teammate_state[i, np.arange(4, teammate_dim - 7, 7)] # Rsin
                R = np.sqrt(Rcos**2 + Rsin**2) / TERRAIN_SIZE
                phi = np.arctan2(Rsin, Rcos) / np.pi
                teammate_state[i, np.arange(3, teammate_dim - 7, 7)] = R    # Rcos -> R
                teammate_state[i, np.arange(4, teammate_dim - 7, 7)] = phi  # Rsin -> phi
                teammate_state[i,  np.arange(5, teammate_dim - 7, 7)] = teammate_state[i, np.arange(5, teammate_dim - 7, 7)] / 180  # theta

            r_indice = np.arange(2, teammate_dim - 7, 7)
            teammate_state = np.delete(teammate_state, r_indice, axis=1)   # remove r -> x, y, R, phi, theta, flag
        else:
            d = 6 if self.add_flag_if_see_all_cameras else 5
            teammate_state = np.zeros((self.num_cameras, d*(self.num_cameras-1)), dtype=np.float64)   # x, y, R, phi, theta, (flag)

            indice = np.ones(self.num_cameras, dtype=np.bool_)
            for i in range(self.num_cameras):
                indice[i] = False

                others = self_state[indice, :5]
                if self.add_flag_if_see_all_cameras:
                    flags = np.ones((self.num_cameras-1, 1), dtype=np.float64)
                    others = np.concatenate([others, flags], axis=1)

                teammate_state[i] = others.flatten()

                indice[i] = True


        return {
            "preserved": self.preserved_state,
            "self": self_state,
            "targets": target_state,
            "obstacles": self.obstacles_state,
            "teammate": teammate_state,
        }


MAX_EPISODE_STEPS = 4000


def main():
    base_env = gym.make('MultiAgentTracking-v0', config = "MATE-4v8-9.yaml", render_mode='rgb_array')

    env: mate.MultiAgentTracking = mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())
    env = MateCameraDictObsWrapper(env, expand_preserved_and_obstacle=False)

    print(env.observation_space)
    obs, _ = env.reset()

    # for _ in range(100):
    #     env.step(np.zeros([env.unwrapped.num_cameras, 2]))
    #     if _ > 50:
    #         arr = env.render()

if __name__ == '__main__':
    main()


