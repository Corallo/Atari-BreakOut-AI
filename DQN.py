
import torch.nn as nn
import torch.nn.functional as F 

class DQN(nn.Module):
    def __init__(self, inputlen=113):
        super().__init__()
            
        self.fc1 = nn.Linear(in_features=inputlen, out_features=256)   
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=8)
        self.out = nn.Linear(in_features=8, out_features=4)
        
    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = self.out(t)
        return t
