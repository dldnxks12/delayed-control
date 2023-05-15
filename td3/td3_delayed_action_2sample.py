# 메인 창

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
        bbstates, bstates, states, actions, rewards, next_states, terminateds, truncateds = [], [], [], [], [], [], [], []

        for bbstate, bstate, state, action, reward, next_state, terminated, truncated in samples:

            bbstates.append(bbstate.cpu().numpy())
            bstates.append(bstate.cpu().numpy())
            states.append(state.cpu().numpy())
            actions.append(action.cpu().detach().numpy())
            rewards.append(reward)
            next_states.append(next_state.cpu().numpy())
            terminateds.append(terminated)
            truncateds.append(truncated)

        bbstates = torch.tensor(bbstates, device=device)
        bstates     = torch.tensor(bstates, device=device)
        states      = torch.tensor(states, device = device)
        actions     = torch.tensor(actions, device=device)
        rewards     = torch.tensor(rewards, device=device)
        next_states = torch.tensor(next_states, device=device)
        terminateds = torch.tensor(terminateds, device=device)
        truncateds  = torch.tensor(truncateds, device=device)

        return bbstates, bstates, states, actions, rewards, next_states, terminateds, truncateds

    def size(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc_s   = nn.Linear(7*History, 256)
        self.fc_a   = nn.Linear(1, 256)

        self.fc1    = nn.Linear(512, 256)
        self.fc2    = nn.Linear(256, 64)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, state, action):
        h1 = F.relu(self.fc_s(state))
        h2 = F.relu(self.fc_a(action))

        concatenate = torch.cat([h1, h2], dim = -1)

        q  = F.relu(self.fc1(concatenate))
        q  = F.relu(self.fc2(q))

        return self.fc_out(q)

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(7*History, 256) # Input : State 3
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Output : Action 1

    def forward(self, state):
        state  = F.relu(self.fc1(state))
        state  = F.relu(self.fc2(state))
        action = torch.tanh(self.fc3(state)) # torque range : [-1 ~ 1]

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
    bbstates, bstates, states, actions, rewards, next_states, terminateds, truncateds = Buffer.sample(batch_size)

    terminateds = torch.unsqueeze(terminateds.type(torch.FloatTensor).to(device), dim = 1)
    rewards     = torch.unsqueeze(rewards, dim = 1)

    Q1_loss, Q2_loss, pi_loss = 0, 0, 0

    """
    actions     : 0
    bbstates    : 0
    bstates     : 1
    states      : 2
    next_states : 3
    """
    with torch.no_grad():
        action_bar = Pi_target(bstates) # next_state : 32x3 , action_bar : 32 x 1

        # action_bar : 1

        q1_value = Q1_target(next_states, action_bar) # next_states : 2, action_bar : 1
        q2_value = Q2_target(next_states, action_bar)

        y = rewards + ( gamma * torch.minimum(q1_value, q2_value) * (1 - terminateds))

    Q1_loss = ( (y - Q1(states, actions)) ** 2 ).mean() # states : 1 , actions : 0
    Q2_loss = ( (y - Q2(states, actions)) ** 2 ).mean()

    Q1_optimizer.zero_grad()
    Q1_loss.backward()
    Q1_optimizer.step()

    Q2_optimizer.zero_grad()
    Q2_loss.backward()
    Q2_optimizer.step()


    for p in Q1.parameters():
        p.requires_grad = False

    pi_loss = - Q1(states, Pi(bbstates)).mean() # bstates : 0

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

lr_pi = 0.0001 # Learning rate
lr_q  = 0.001
tau   = 0.001  # Soft update rate
gamma = 0.99  # Discount Factor
batch_size = 32
History = 20

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
#current_env = env_visual

MAX_EPISODE   = 400
max_time_step = env._max_episode_steps

X = np.arange(0, MAX_EPISODE, 1)
Y = []

import time
for episode in range(MAX_EPISODE):

    # Basic Learning Setting
    state, _ = current_env.reset()
    state    = torch.tensor(state).float().to(device)
    terminated, truncated = False, False
    total_reward = 0

    # Delayed Action Setting
    init_flag             = 0 # delay flag
    DelayCounter          = 0
    Action_buffer         = [] # generating part delay
    State_buffer          = []

    # Generate Episodes ...
    for time_step in range(max_time_step):
        with torch.no_grad():
            if episode < 2:
                action = current_env.action_space.sample()
                action = torch.FloatTensor(action).to(device)
            else:
                action = torch.clamp( (Pi(state) + noise()[0]), -1, 1 )

        Action_buffer.append(action)
        State_buffer.append(state)
        if init_flag < 2: # s0 -> a0 -> s1
            next_state, reward, terminated, truncated, _ = current_env.step(action.cpu().detach().numpy())
            next_state = torch.tensor(next_state).float().to(device)

        else: # delayed action # s1 -> a0 -> s2 -> a1
            next_state, reward, terminated, truncated, _ = current_env.step(Action_buffer[-3].cpu().detach().numpy())
            next_state = torch.tensor(next_state).float().to(device)

        terminated = not terminated if time_step == max_time_step - 1 else terminated

        if init_flag < 2:
            state = next_state
            init_flag += 1
            continue # Appending transition into replay buffer -> Skip

        Buffer.put([State_buffer[-3], State_buffer[-2], state, Action_buffer[-3], reward, next_state, terminated, truncated])
        total_reward += reward

        if Buffer.size() > 50: # Train Q, Pi
            train(Buffer, Q1, Q1_target, Q2, Q2_target, Pi, Pi_target, Q1_optimizer, Q2_optimizer, Pi_optimizer)

        if terminated or truncated:
            break

        state = next_state

    Y.append(total_reward)
    print(f"Episode : {episode} | TReward : {total_reward}")

env.close()

Y = np.array(Y)
np.save('./TD3_delayed_history_2sample', Y)

