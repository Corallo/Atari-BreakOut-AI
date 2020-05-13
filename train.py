
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
    
        
batch_size = 256
gamma = 0.999
#Policy var
eps_start = 1
eps_end = 0.01
eps_decay = 0.0001

target_update = 5
memory_size = 100000
lr = 0.001
num_episodes = 1000


start_time = time.time()
# your code

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
envM = EnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, envM.num_actions_available(), device)

#1) Initialize replay memory capacity.
memory = ReplayMemory(memory_size)

#2) Initialize the policy network with random weights.
policy_net = DQN().to(device)
target_net = DQN().to(device)

optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_start = 0

try:
	checkpoint = torch.load("saved_state_dict.pt")
	policy_net.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	episode_start = checkpoint['episode'] + 1
	memory = checkpoint['memory']
	policy_net.train()
	
	print("Found saved state_dict")
except:	
	print("No state_dict found")
	
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

episode_durations = []
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
        #reward=torch.tensor([reward*100 - 0.1], device=device) # This should not b ehere
        #Add higher penalty for each frame the ball is not in the map
        #Maybe add another penalty when you loose the ball (this may be harder)
        #s_time = time.time()
        next_img = envM.get_state()
        #e_time = time.time()
        #print("Time to get state: ",e_time-s_time)
        
        #s_time = time.time()
        
        next_state = decreaseObservationSpace(img)
        #e_time = time.time()
        #print("Time to compress state: ",e_time-s_time)
        
        #s_time = time.time()
        
        next_state = addDirection(next_state,state)
        #e_time = time.time()
        #print("Time to add direction: ",e_time-s_time)
        

        reward=torch.tensor([reward*100 - 0.1], device=device) # This should not b ehere
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
            
            print("Game Finished! frames: ", timestep," Game time ",int(time.time() - episode_time), " Score: ", score, " Game played: ", episode, " Total Time: ",int(time.time() - start_time) )
            #episode_durations.append(timestep)
            #plot(episode_durations, 100)
            break
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
        torch.save({
        	'episode': episode,
        	'model_state_dict': target_net.state_dict(),
        	'optimizer_state_dict': optimizer.state_dict(),
        	'memory': memory
        }, "saved_state_dict.pt")
envM.close()


