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
        self.eps    = 0.2  # clipping ratio
        self.lamd   = 0.9
        self.gamma  = 0.9
        self.learning_rate = 0.0003

        self.buffer_size    = 30
        self.minibatch_size = 32

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
        data = []

        for j in range(self.buffer_size):
            for i in range(self.minibatch_size):
                rollout = self.data.pop()
                s_list, a_list, r_list, s_prime_list, prob_list, done_list = [], [], [], [], [], []

                for transition in rollout:
                    state, action, reward, next_state, prob, done = transition

                    s_list.append(state)
                    a_list.append(action)
                    r_list.append([reward])
                    s_prime_list.append(next_state)
                    prob_list.append(prob)
                    done_mask = 0 if done else 1
                    done_list.append([done_mask])

                b_state.append(s_list)
                b_action.append(a_list)
                b_reward.append(r_list)
                b_next_state.append(s_prime_list)
                b_log_old_probs.append(prob_list)
                b_done.append(done_list)

            states  = torch.tensor(b_state, dtype=torch.float)
            actions = torch.tensor(b_action)
            rewards = torch.tensor(b_reward)
            next_states = torch.tensor(b_next_state, dtype=torch.float)
            probs = torch.tensor(b_log_old_probs)
            dones = torch.tensor(b_done, dtype=torch.float)

            mini_batch = states, actions, rewards, next_states, probs, dones

            data.append(mini_batch)

        return data

    def td_target_advantage(self, data):
        data_with_adv = []

        for mini_batch in data:
            states, actions, rewards, next_states, probs, dones = mini_batch
            with torch.no_grad():
                td_targets = rewards + self.gamma*self.v(next_states)*dones
                deltas     = td_targets - self.v(states)
            deltas     = deltas.numpy()

            advantage_list = []
            advantage = 0.0
            for delta_t in deltas[::-1]:
                advantage = self.gamma * self.lamd * advantage + delta_t[0]
                advantage_list.append([advantage])

            advantage_list.reverse()
            advantage = torch.tensor(advantage_list, dtype = torch.float)
            data_with_adv.append((states, actions, rewards, next_states, probs, dones, td_targets, advantage))

        return data_with_adv

    def train(self):
        if len(self.data) == self.minibatch_size * self.buffer_size:
            data = self.make_batch()
            data = self.td_target_advantage(data)

            for _ in range(self.epochs):
                for mini_batch in data:
                    states, actions, rewards, next_states, old_log_probs, dones, td_targets, advantage = mini_batch

                    actions = actions.unsqueeze(1)
                    old_log_probs = old_log_probs.unsqueeze(1)

                    mu_new, std_new  = self.pi(states)  # ok
                    new_dist         = Normal(mu_new, std_new)
                    new_log_probs    = new_dist.log_prob(actions)

                    ratio      = torch.exp(new_log_probs - old_log_probs)
                    surrogate1 = ratio * advantage
                    surrogate2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
                    loss = -torch.min(surrogate1, surrogate2) + F.smooth_l1_loss(self.v(states), td_targets.detach())

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0) # 효과가 괜찮다.
                    self.optimizer.step()

