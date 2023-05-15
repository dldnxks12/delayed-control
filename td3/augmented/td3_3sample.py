import os
import sys
import math
import time
import copy
import random
import collections
import numpy as np
import environment
import gymnasium as gym

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque


class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen=20000)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)

        Is, actions, rewards, I_nexts, terminateds = [], [], [], [], []

        for I, action, reward, I_next, terminated in samples:
            Is.append(I.cpu().numpy())
            actions.append(action.cpu().detach().numpy())
            rewards.append(reward)
            I_nexts.append(I_next.cpu().numpy())
            terminateds.append(terminated)

        Is          = torch.tensor(Is, device = device)
        actions     = torch.tensor(actions, device=device)
        rewards     = torch.tensor(rewards, device=device)
        I_nexts     = torch.tensor(I_nexts, device=device)
        terminateds = torch.tensor(terminateds, device=device)

        return Is, actions, rewards, I_nexts, terminateds

    def size(self):
        return len(self.buffer)



class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc_s   = nn.Linear(10, 128)
        self.fc_a   = nn.Linear(1, 128)

        self.fc1    = nn.Linear(256, 128)
        self.fc2    = nn.Linear(128, 1)

    def forward(self, state, action):

        h1 = F.relu(self.fc_s(state))
        h2 = F.relu(self.fc_a(action))

        concatenate = torch.cat([h1, h2], dim = -1)

        q  = F.relu(self.fc1(concatenate))
        q  = self.fc2(q)

        return q

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state):
        state  = F.relu(self.fc1(state))
        action = torch.tanh(self.fc2(state)) # torque range : [-1 ~ 1]

        return action

# Add noise to Action
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

def train(Buffer, Q1, Q1_target, Q2, Q2_target, Pi, Pi_target, Q1_optimizer, Q2_optimizer, Pi_optimizer):
    Is, actions, rewards, I_nexts, terminateds = Buffer.sample(batch_size)

    Is      = Is.squeeze(1)
    I_nexts = I_nexts.squeeze(1)
    terminateds = torch.unsqueeze(terminateds.type(torch.FloatTensor).to(device), dim = 1)
    rewards     = torch.unsqueeze(rewards, dim = 1)

    Q1_loss, Q2_loss, pi_loss = 0, 0, 0

    with torch.no_grad():

        action_bar = Pi_target(I_nexts)

        q1_value = Q1_target(I_nexts, action_bar)
        q2_value = Q2_target(I_nexts, action_bar)

        y = rewards + ( gamma * torch.minimum(q1_value, q2_value) * (1 - terminateds))

    Q1_loss = ( (y - Q1(Is, actions)) ** 2 ).mean()
    Q2_loss = ( (y - Q2(Is, actions)) ** 2 ).mean()

    Q1_optimizer.zero_grad()
    Q1_loss.backward()
    Q1_optimizer.step()

    Q2_optimizer.zero_grad()
    Q2_loss.backward()
    Q2_optimizer.step()

    for p in Q1.parameters():
        p.requires_grad = False

    pi_loss = - Q1(Is, Pi(Is)).mean()

    Pi_optimizer.zero_grad()
    pi_loss.backward()
    Pi_optimizer.step()

    for p in Q1.parameters():
        p.requires_grad = True

    # Soft update (not periodically update, instead soft update !!)
    soft_update(Q1, Q1_target, Q2, Q2_target, Pi, Pi_target)

def soft_update(Q1, Q1_target, Q2, Q2_target, P, P_target):
    for param, target_param in zip(Q1.parameters(), Q1_target.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * (tau))

    for param, target_param in zip(Q2.parameters(), Q2_target.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * (tau))

    for param, target_param in zip(P.parameters(), P_target.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * (tau))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("")
print(f"On {device}")
print("")

lr_pi = 0.0001
lr_q  = 0.001
tau   = 0.001
gamma = 0.99
batch_size = 64

# Q function
Q1 = QNetwork().to(device)
Q2 = QNetwork().to(device)

Q1_optimizer = optim.Adam(Q1.parameters(), lr = lr_q)
Q2_optimizer = optim.Adam(Q2.parameters(), lr = lr_q)

Q1_target = QNetwork().to(device)
Q1_target.load_state_dict(Q1.state_dict())

Q2_target = QNetwork().to(device)
Q2_target.load_state_dict(Q2.state_dict())

# Policy
Pi = PolicyNetwork().to(device)
Pi_target = PolicyNetwork().to(device)
Pi_optimizer = optim.Adam(Pi.parameters(), lr = lr_pi)
Pi_target.load_state_dict(Pi.state_dict())

Buffer = ReplayBuffer()
noise  = OrnsteinUhlenbeckNoise(mu = np.zeros(1))

xml_file = os.getcwd()+"/environment/assets/inverted_single_pendulum.xml"
env      = gym.make("InvertedSinglePendulum-v4" , model_path=xml_file)
env_visual = gym.make("InvertedSinglePendulum-v4" , render_mode = 'human', model_path=xml_file)
current_env = env

MAX_EPISODE   = 300
max_time_step = env._max_episode_steps

Y = []

for episode in range(MAX_EPISODE):

    # Basic Learning Setting
    state, _   = current_env.reset()
    state      = torch.tensor(state).float().to(device)
    terminated = False
    total_reward = 0

    # Delayed Action Setting
    Action_buffer         = []

    # Generate Episodes ...
    for time_step in range(max_time_step):
        with torch.no_grad():
            if len(Action_buffer) < 4: # Padding with 0
                I = torch.concatenate([state.unsqueeze(0), torch.FloatTensor([0, 0, 0]).unsqueeze(0).to(device)], dim=1)
            else:
                I = torch.concatenate([state.unsqueeze(0), torch.FloatTensor([Action_buffer[-3], Action_buffer[-2], Action_buffer[-1]]).unsqueeze(0).to(device)] , dim = 1)

            #action = torch.clamp( (Pi(I) + noise()[0]), -1, 1 )[0]
            action = Pi(I)[0]

        if len(Action_buffer) < 4: # gathering actions
            next_state, reward, terminated, _, _ = current_env.step(action.cpu().detach().numpy())
            next_state = torch.tensor(next_state).float().to(device)
            state = next_state
            Action_buffer.append(action)
            continue

        else: # apply delayed action
            next_state, reward, terminated, _, _ = current_env.step(Action_buffer[-3].cpu().detach().numpy())
            next_state = torch.tensor(next_state).float().to(device)
            I_next = torch.concatenate([next_state.unsqueeze(0), torch.FloatTensor([Action_buffer[-2], Action_buffer[-1], action]).unsqueeze(0).to(device)],dim=1)
            Action_buffer.append(action)

        terminated = not terminated if time_step == max_time_step - 1 else terminated

        # s0, s1, s2, s3, a1, r, s4
        Buffer.put([I, action, reward, I_next, terminated])
        total_reward += reward

        if Buffer.size() > 500: # Train Q, Pi
            #for _ in range(5):
            train(Buffer, Q1, Q1_target, Q2, Q2_target, Pi, Pi_target, Q1_optimizer, Q2_optimizer, Pi_optimizer)

        if terminated :
            break

        state = next_state

    Y.append(total_reward)
    print(f"Episode : {episode} | TReward : {total_reward}")

env.close()

Y = np.array(Y)
np.save('./TD3_delayed_3sample', Y)

