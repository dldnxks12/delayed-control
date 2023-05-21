import sys
import utils
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, device):
        super(ActorCritic, self).__init__()

        self.device       = device
        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.action_bound = action_bound

        self.data   = []
        self.epochs = 10
        self.eps    = 0.1  # clipping ratio
        self.lamd   = 0.9
        self.gamma  = 0.9
        self.learning_rate = 0.0003

        self.fc1    = nn.Linear(state_dim, 128)
        self.fc_mu  = nn.Linear(128, action_dim)
        self.fc_std = nn.Linear(128, action_dim)
        self.fc_v   = nn.Linear(128, 1)

        self.optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)

    def pi(self, state):
        x   = F.relu(self.fc1(state))
        mu  = torch.tanh(self.fc_mu(x)) * self.action_bound
        std = F.softplus(self.fc_std(x))
        return mu, std

    def v(self, state):
        x = F.relu(self.fc1(state))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        b_state, b_action, b_reward, b_next_state, b_done = [], [], [], [], []
        b_log_old_probs = []

        for state, action, reward, next_state, prob, done in self.data:

            b_state.append(state)
            b_action.append(action)
            b_reward.append([reward])
            b_next_state.append(next_state)
            done_mask = 0 if done else 1
            b_done.append([done_mask])
            b_log_old_probs.append(prob)

        states  = b_state
        actions = b_action
        rewards = b_reward
        next_states = b_next_state
        dones = b_done
        probs = b_log_old_probs

        states      = np.array(np.stack(states,      axis=0))
        next_states = np.array(np.stack(next_states, axis=0))

        states  = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)
        probs = torch.tensor(probs)

        self.data = []

        return states, actions, rewards, next_states, probs, dones

    def gae_td_target(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            td_targets = rewards + self.gamma*self.v(next_states)*dones
            deltas     = td_targets - self.v(states)
            deltas     = deltas.detach().numpy()

        advantage_list = []
        advantage = 0.0
        for delta_t in deltas[::-1]:
            advantage = self.gamma * self.lamd * advantage + delta_t[0]
            advantage_list.append([advantage])

        advantage_list.reverse()
        GAEs = torch.tensor(advantage_list, dtype = torch.float)
        return GAEs, td_targets

    def train(self):
        states, actions, rewards, next_states, old_log_probs, dones = self.make_batch()

        for _ in range(self.epochs):
            # Calc GAE / td_target inside here
            GAEs, td_targets = self.gae_td_target(states, actions, rewards, next_states, dones)

            actions = actions.unsqueeze(1)
            old_log_probs = old_log_probs.unsqueeze(1)

            mu_new, std_new  = self.pi(states)  # ok
            new_dist         = Normal(mu_new, std_new)
            new_log_probs    = new_dist.log_prob(actions)

            # torch.Size([5, 1]) torch.Size([5]) torch.Size([5, 1])
            #print(new_log_probs.shape, old_log_probs.shape, GAEs.shape)
            #print(new_log_probs, old_log_probs, GAEs)
            #sys.exit()

            ratio      = torch.exp(new_log_probs - old_log_probs)
            surrogate1 = ratio * GAEs
            surrogate2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * GAEs
            loss = -torch.min(surrogate1, surrogate2) + F.smooth_l1_loss(self.v(states), td_targets.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
