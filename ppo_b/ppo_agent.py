import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torch.distributions import Normal
import ppo_actor
import ppo_critic
import utils


class PPOagent(nn.Module):

    def __init__(self, env, device):
        super(PPOagent, self).__init__()
        self.gamma   = 0.95
        self.lamd    = 0.9
        self.epochs  = 5
        self.device  = device
        self.ratio_clipping = 0.2

        self.batch_size = 64
        self.Actor_lr   = 0.0001
        self.Critic_lr  = 0.001

        self.env = env
        self.state_dim    = env.observation_space.shape[0]
        self.action_dim   = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        # Define Actor & Critic
        self.actor  = ppo_actor.Actor(self.state_dim, self.action_dim, self.action_bound, self.Actor_lr, self.ratio_clipping, self.device).to(device)
        self.critic = ppo_critic.Critic(self.state_dim, self.action_dim, self.Critic_lr, self.device).to(device)

        self.save_epi_reward = []


    def GAE_target(self, rewards, states, next_states, dones):
        GAE            = torch.zeros_like(rewards)
        GAE_cumulative = 0
        forward_val    = 0

        with torch.no_grad():
            values      = self.critic._forward(states)
            next_values = self.critic._forward(next_states)
            next_value  = self.critic._forward(next_states[-1])

        if not dones[-1]:
            forward_val = next_value

        for k in reversed(range(0, len(rewards))):
            delta          = rewards[k] + self.gamma * forward_val - values[k]
            GAE_cumulative = self.gamma * self.lamd * GAE_cumulative + delta
            GAE[k]         = GAE_cumulative
            forward_val    = values[k]

        td_targets = rewards + self.gamma * next_values * dones

        return GAE, td_targets

    def train(self, max_episode_num):

        b_state, b_action, b_reward, b_next_state, b_done = [], [], [], [], []
        b_log_old_probs = []

        for ep in range(max_episode_num):

            # Init episode
            time, episode_reward, done = 0, 0, False
            state = self.env.reset()[0]

            while not done:
                # old policy의 mean, std 계산하고 action 뽑기

                with torch.no_grad():
                    mu_old, std_old = self.actor.get_policy(torch.FloatTensor(state).to(self.device))
                dist = Normal(mu_old, std_old)
                action = dist.sample()
                old_log_prob = dist.log_prob(action)

                next_state, reward, done, _, _ = self.env.step(action)

                b_state.append(state)
                b_action.append([action])
                b_reward.append([reward/10])
                b_next_state.append(next_state)
                done_mask = 0 if done else 1
                b_done.append([done_mask])
                b_log_old_probs.append(old_log_prob)

                if len(b_state) < self.batch_size:
                    state = next_state
                    episode_reward += reward
                    time += 1
                    continue

                states        = b_state
                actions       = b_action
                rewards       = b_reward
                next_states   = b_next_state
                dones         = b_done
                log_old_probs = b_log_old_probs

                states        = torch.tensor(states, dtype=torch.float)
                actions       = torch.tensor(actions)
                rewards       = torch.tensor(rewards)
                next_states   = torch.tensor(next_states, dtype=torch.float)
                dones         = torch.tensor(dones, dtype=torch.float)
                log_old_probs = torch.tensor(log_old_probs)

                # clear buffer
                b_state, b_action, b_reward, b_next_state, b_done = [], [], [], [], []
                b_log_old_probs = []

                GAEs, td_targets = self.GAE_target(rewards, states, next_states, dones)

                for _ in range(self.epochs):
                    self.actor.train(states, GAEs, log_old_probs)
                    self.critic.train(states, td_targets)

                state = next_state
                time += 1
                episode_reward += reward

            print(f"Episode : {ep} | Time : {time} | Reward : {episode_reward}")
            self.save_epi_reward.append(episode_reward)

    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()





