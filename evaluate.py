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

policy_net = DQN()

try:
	checkpoint = torch.load("saved_state_dict.pt")
	policy_net.load_state_dict(checkpoint['model_state_dict'])
	episode_start = checkpoint['episode'] + 1
	episode_durations = checkpoint['episode_durations']
	current_step = checkpoint['current_step']
	policy_net.eval()
	
	print("Found saved state_dict")
except:
    print("No state_dict found")
    exit(1)
    
    
print("\nNetwork:")
for param_tensor in policy_net.state_dict():
    print(param_tensor, "\t", policy_net.state_dict()[param_tensor].size())
    
print("\nNetwork trained for " + str(sum(episode_durations)) + " frames\n")

envM = EnvManager("cpu")

bestScore = 0
bestGame = -1
episode = 1
scores = []

while True:
    envM.reset()
    img = envM.get_state()
    state = decreaseObservationSpace(img)
    state = addDirection(state,None)
    score = 0
    
    for timestep in count():
        envM.render()
        with torch.no_grad():
            action = torch.tensor([policy_net(torch.from_numpy(state).float()).argmax()])
        reward = envM.take_action(action)
        
        score+=reward
        
        next_img = envM.get_state()
        next_state = decreaseObservationSpace(next_img)
        next_state = addDirection(next_state,state)
        state = next_state
            
        if envM.done:
            if(score > bestScore):
                bestScore= score
                bestGamePlayed = episode
            episode += 1
            scores.append(score)
            print("Game finished with score " + str(int(score)) + "! Average score for evaluation is %.2f" % (sum(scores) / len(scores)))
            break
envM.close()
