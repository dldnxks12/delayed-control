import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utils

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate, device):
        super(Critic, self).__init__()
        #self.apply(utils.weight_init)  # Weight initialize

        self.device        = device
        self.state_dim     = state_dim
        self.action_dim    = action_dim
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def _forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def train(self, states, td_targets):

        values = self._forward(states)

        self.optimizer.zero_grad()
        loss   = F.smooth_l1_loss(values ,td_targets.detach())
        loss.backward()
        self.optimizer.step()




