import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import time
from replayMemory import *
from epsilonGreedyStrategy import *
from agent import *
from EnvManager import *
from Qvalues import *
from utils import *
from DQN import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T   




"""
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display
"""

    
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)
      
"""
Initialize replay memory capacity.
Initialize the policy network with random weights.
Clone the policy network, and call it the target network.
For each episode:

    Initialize the starting state.
    For each time step:
        Select an action.
            Via exploration or exploitation
        Execute selected action in an emulator.
        Observe reward and next state.
        Store experience in replay memory.
        Sample random batch from replay memory.
        Pass batch of preprocessed states to policy network.
        Calculate loss between output Q-values and target Q-values.
            Requires a pass to the target network for the next state
        Gradient descent updates weights in the policy network to minimize loss.
            After x

time steps, weights in the target network are updated to the weights in the policy network.
"""
    
        
batch_size = 32
gamma = 0.999
#Policy var
eps_start = 1
eps_end = 0.01
eps_decay = 0.0001

target_update = 4
memory_size = 100000
lr = 0.0005
num_episodes = 100000

start_time = time.time()
# your code



device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
envM = EnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

#1) Initialize replay memory capacity.
memory = ReplayMemory(memory_size)

#2) Initialize the policy network with random weights.
policy_net = DQN().to(device)
target_net = DQN().to(device)

optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_start = 0
episode_durations = []
scores = []

bestScore = 0
bestModel = target_net
bestGamePlayed = 0

try:
	checkpoint = torch.load("saved_state_dict.pt")
	policy_net.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	episode_start = checkpoint['episode'] + 1
	episode_durations = checkpoint['episode_durations']
	scores = checkpoint['scores']
	current_step = checkpoint['current_step']
	agent = Agent(strategy, envM.num_actions_available(), device, current_step)
	policy_net.train()
	
	print(current_step)
	
	print("Found saved state_dict")
except:	
    agent = Agent(strategy, envM.num_actions_available(), device, 0)
    print("No state_dict found")
	
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

for episode in range(episode_start, num_episodes):
    episode_time=time.time()
    envM.reset()
    img = envM.get_state()
    state = decreaseObservationSpace(img)
    state = addDirection(state,None)
    score = 0;
    # print("Initial State",state)
    assert (state.shape[0]==113)
    
    for timestep in count():
       # s_time = time.time()
        envM.render()
        #e_time = time.time()
        #print("Time to render: ",e_time-s_time)
        #s_time = time.time()
        action = agent.select_action(state, policy_net)
        #e_time = time.time()
        #print("Time to get action: ",e_time-s_time)
        #s_time = time.time()
        reward = envM.take_action(action)
        #e_time = time.time()
        #print("Time to execute action: ",e_time-s_time)
        
        score+=reward
        

        next_img = envM.get_state()
        next_state = decreaseObservationSpace(next_img)
        next_state = addDirection(next_state,state)

       
        reward*=10
        if(next_state[110]==0 and next_state[109]==0):
             reward=torch.tensor([float(-100.0)], device=device) # This should not b ehere
             #print(reward)
        else:
            reward += float(8.0 - (np.absolute(next_state[110]-(next_state[108]))))
            #print(next_state[108]+8,next_state[110],reward)
            reward=torch.tensor([reward], device=device) # This should not b ehere
        #time.sleep(0.5)
        memory.push(Experience(state, action, next_state, reward))
        state = next_state
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences,device)
            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if envM.done:
            if(score>bestScore):
                bestScore= score
                bestModel = target_net
                bestGamePlayed = episode
                target_net.load_state_dict(policy_net.state_dict())
                torch.save({
                    'episode': episode,
                	'model_state_dict': target_net.state_dict(),
                	'optimizer_state_dict': optimizer.state_dict(),
                	'episode_durations': episode_durations,
                	'scores': scores,
                	'current_step': agent.current_step
                }, "saved_state_dict_best.pt")
                
            print("Done! frames: ", timestep," Game time ",int(time.time() - episode_time), " Score: ", score, " Game played: ", episode, " Total Time: ",int(time.time() - start_time), "BestScore: ", bestScore, "after", bestGamePlayed, "games" )
            episode_durations.append(timestep)
            scores.append(score)
            plot(episode_durations, scores)
            break
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
        torch.save({
        	'episode': episode,
        	'model_state_dict': target_net.state_dict(),
        	'optimizer_state_dict': optimizer.state_dict(),
        	'episode_durations': episode_durations,
        	'scores': scores,
        	'current_step': agent.current_step
        }, "saved_state_dict.pt")
    if episode == 100 or episode == 200 or (episode%500) == 0 :
        torch.save({
        	'episode': episode,
        	'model_state_dict': target_net.state_dict(),
        	'optimizer_state_dict': optimizer.state_dict(),
        	'episode_durations': episode_durations,
        	'scores': scores
        }, "saved_state_dict_" + str(episode) + ".pt")
    	
envM.close()


