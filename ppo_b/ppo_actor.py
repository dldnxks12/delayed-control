import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import utils

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, learning_rate, eps, device):
        super(Actor, self).__init__()
        #self.apply(utils.weight_init)  # Weight initialize

        self.device     = device
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.action_bound  = action_bound
        self.learning_rate = learning_rate

        self.eps = eps  # clipping ratio

        self.std_bound = [1e-2, 1.0] # Bound of standard deviation

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.mu  = nn.Linear(16, action_dim)
        self.std = nn.Linear(16, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)


    def _forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # mu value check
        mu  = torch.tanh(self.mu(x))*self.action_bound
        std = F.softplus(self.std(x))

        return mu, std

    def get_policy(self, state):
        mu, std = self._forward(state)
        return mu, std

    def log_pdf(self, mu, std):
        std = torch.clamp(std, self.std_bound[0], self.std_bound[1])
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return log_prob

    def train(self, states, GAEs, old_log_prob):

        mu_new, std_new = self._forward(states)
        new_log_prob    = self.log_pdf(mu_new, std_new)

        ratio      = torch.exp(torch.log(new_log_prob) - torch.log(old_log_prob))

        surrogate1 = ratio * GAEs
        surrogate2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * GAEs
        self.optimizer.zero_grad()
        loss = -torch.min(surrogate1, surrogate2)
        self.optimizer.step()


