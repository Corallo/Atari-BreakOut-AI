import gym
import numpy as np
import time
env = gym.make('Breakout-v0')

for i_episode in range(1):
    observation = env.reset()
    observation, reward, done, info = env.step(1)

        
    
    observation = np.sum(observation, axis=2)
    blocks = observation[57:88:6,8:152:8].flatten() #try to avoid flatten and use ravel for performance
    blocks = np.where(blocks > 0, 1, -1).astype(float) # try ti avoid this for performance
    

    sliderLine = observation[190,8:152]
    slider = float(np.nonzero(sliderLine)[0][0] - 72)

    
    ballSpace = observation[93:189,8:152]
    ball=np.nonzero(ballSpace)
    if(ball[0].shape[0]>0):
        ballx=ball[0][0] - 48 #normalizing
        bally=ball[1][0] - 72 
    else:
        ballx = 0
        bally = 0

    out = np.zeros(108+1+2)
    out[0:108]=blocks
    out[108]=slider
    out[109]=float(ballx)
    out[110]=float(bally)
    return out
    """
    for t in range(1000):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(1)
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    """
    """
    ball is 4 hight and 2 large
    """
env.close()