import mate
from mate.agents import GreedyCameraAgent, GreedyTargetAgent
import gymnasium as gym


# from agent import Agent


MAX_EPISODE_STEPS = 4000


def main():
	base_env = gym.make('MultiAgentTracking-v0', config = "MATE-8v8-9.yaml")

	env = mate.MultiCamera.make(base_env, target_agent=GreedyTargetAgent())


	camera_agents = GreedyCameraAgent().spawn(env.unwrapped.num_cameras)

	camera_joint_observation, _ = env.reset()

	mate.group_reset(camera_agents, camera_joint_observation)
	camera = None

	run = True

	for i in range(MAX_EPISODE_STEPS):

		camera_joint_action = mate.group_step(
			env.unwrapped, camera_agents, camera_joint_observation, camera
		)

		results = env.step(camera_joint_action)

		camera_joint_observation, target_team_reward, done, truncated, camera = results

		run = env.render()
		# arr = env.render(mode='rgb_array')

		# plt.imshow(arr)
		# plt.show()
		# arr: np.ndarray

		# a = input()


		if not run or done:
			env.close()
			break

	# print(t[0][-1, ])


if __name__ == '__main__':
	main()
