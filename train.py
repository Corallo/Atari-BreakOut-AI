import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

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


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display


    
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
eps_decay = 0.001

target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
envM = EnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, envM.num_actions_available(), device)
#1) Initialize replay memory capacity.
memory = ReplayMemory(memory_size)

#2) Initialize the policy network with random weights.
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)




episode_durations = []
for episode in range(num_episodes):
    envM.reset()
    img = envM.get_state()
    state = decreaseObservationSpace(img)
    state = addDirection(state,None)
    # print("Initial State",state)
    assert (state.shape[0]==113)
    
    for timestep in count():
        action = agent.select_action(state, policy_net)
        reward = envM.take_action(action)
        next_img = envM.get_state()
        
        next_state = decreaseObservationSpace(img)
        
        next_state = addDirection(next_state,state)
        memory.push(Experience(state, action, next_state, reward))
        state = next_state
        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)
            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if envM.done:
            print("Environment Done!")
            episode_durations.append(timestep)
            plot(episode_durations, 100)
            break
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
envM.close()


