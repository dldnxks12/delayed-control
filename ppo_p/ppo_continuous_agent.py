import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Normal
import utils
import ppo_network

class PPOagent(nn.Module):
    def __init__(self, env, device):
        super(PPOagent, self).__init__()
        self.device  = device
        self.T       = 3

        self.env          = env
        self.state_dim    = env.observation_space.shape[0]
        self.action_dim   = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        # Define Actor & Critic
        self.network  = ppo_network.ActorCritic(self.state_dim, self.action_dim, self.action_bound, self.device).to(device)
        self.save_epi_reward = []

    def train(self, max_episode_num):
        episode_reward = 0
        for ep in range(max_episode_num):
            state = self.env.reset()[0]
            done, terminated, truncated = False, False, False

            while not done:
                for _ in range(self.T):
                    with torch.no_grad():
                        mu_old, std_old = self.network.pi(torch.FloatTensor(state).to(self.device))

                    dist = Normal(mu_old, std_old)
                    action = dist.sample()
                    old_log_prob = dist.log_prob(action)

                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done  = terminated or truncated

                    self.network.put_data((state, action, reward/10.0, next_state, old_log_prob, done))

                    state = next_state
                    episode_reward += reward

                    if done:
                        break

                self.network.train()

            # self.save_epi_reward.append(episode_reward)
            if ep % 10 == 0 and ep != 0:
                print(f"Episode : {ep} | Reward : {episode_reward / 10}")
                episode_reward = 0


    def plot_result(self):
        self.env.close()
        plt.plot(self.save_epi_reward)
        plt.show()
