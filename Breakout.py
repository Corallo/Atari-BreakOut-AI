import time
import gym
import numpy as np


def decreaseObservationSpace(observationSpace_old):
	blocks = np.zeros(shape=(6, 18))
	ball = np.zeros(2)
	slider = -1

	i = 0
	j = 0

	# check if the pixels at the konwn block positions are black (block gone) or not (block still there)
	for x in range(57, 88, 6):
		for y in range(8, 152, 8):
			blocks[i][j] = observationSpace_old[x][y][0] != 0
			j += 1
		i += 1
		j = 0

	# get the ball position by finding a pixel with the ball color and checking if either left and right are other colors or below and above it
	# this should always be the case for the ball, since there is no corner in the game where on both sides you have the ball color
	ballFound = False
	for x in range(209, 32, -1):
		if ballFound: break
		for y in range(8, 151):
			if ballFound: break
			if np.array_equal(observationSpace_old[x][y][:], np.array([200, 72, 72])):
				if (not np.array_equal(observationSpace_old[x][y - 1][:], np.array([200, 72, 72])) and np.array_equal(observationSpace_old[x][y + 1][:], np.array([200, 72, 72])) and not np.array_equal(observationSpace_old[x][y + 2][:], np.array([200, 72, 72]))) or (not np.array_equal(observationSpace_old[x + 1][y][:], np.array([200, 72, 72])) and np.array_equal(observationSpace_old[x - 1][y][:], np.array([200, 72, 72])) and not np.array_equal(observationSpace_old[x - 2][y][:], np.array([200, 72, 72]))):
					ball = np.array([x - 1, y + 1])
					ballFound = True

	# get the slider position
	sliderFound = False
	for y in range(8, 151):
		if sliderFound: break
		if np.array_equal(observationSpace_old[190][y][:], np.array([200, 72, 72])):
			b = True
			for k in range(1, 15):
				b = b and np.array_equal(observationSpace_old[190][y + i][:], np.array([200, 72, 72]))
			if b:
				slider = y + 8
				sliderFound = True


	return blocks, ball, slider


if __name__ == '__main__':
	env = gym.make('Breakout-v0')

	# observation = env.reset()
	# print(observation[32][8][:])
	# print(decreaseObservationSpace(observation))

	for i_episode in range(20):
		observation_raw = env.reset()
		observation = decreaseObservationSpace(observation_raw)

		for t in range(1000):
			env.render()
			action = env.action_space.sample()
			observation_raw, reward, done, info = env.step(action)
			observation = decreaseObservationSpace(observation_raw)
			print(observation[0][:][5])
			print(observation[1])
			print(observation[2])
			print("")
			time.sleep(2)

			if done:
				print("Episode finished after {} timesteps".format(t + 1))
				break
	env.close()
