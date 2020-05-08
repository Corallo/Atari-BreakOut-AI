import torch
import random

class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
        
    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            print("EXPLORING")
            return torch.tensor([action]).to(self.device) # explore      
        else:
            print("EXPLOITING")
            with torch.no_grad():
               return policy_net(torch.from_numpy(state).to(self.device)).argmax(dim=1).to(self.device) # exploit
        