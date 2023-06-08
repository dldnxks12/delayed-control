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
    def __init__(self, env, env_eval, device):
        super(PPOagent, self).__init__()
        self.device  = device
        self.rollout_len = 3

        self.env          = env
        self.env_eval     = env_eval
        self.state_dim    = env.observation_space.shape[0]
        self.action_dim   = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]

        # Define Actor & Critic
        self.network  = ppo_network.ActorCritic(self.state_dim, self.action_dim, self.action_bound, self.device).to(device)
        self.save_epi_reward = []

    def train(self, max_episode_num):

        episode_reward = 0
        rollout = []
        for ep in range(max_episode_num):
            state = self.env.reset()[0]
            done, terminated, truncated = False, False, False

            while not done:
                for _ in range(self.rollout_len):
                    with torch.no_grad():
                        mu_old, std_old = self.network.pi(torch.FloatTensor(state).to(self.device))

                    dist = Normal(mu_old, std_old)
                    action = dist.sample() # grad x
                    old_log_prob = dist.log_prob(action) # grad o

                    # print(action, [action.item()])
                    # action : -0.9288
                    # [action.item()] : [-0.928832471370697]

                    next_state, reward, terminated, truncated, _ = self.env.step(action.numpy())
                    done  = terminated or truncated

                    rollout.append((state, action.numpy(), reward/10.0, next_state, old_log_prob.numpy(), done))

                    if len(rollout) == self.rollout_len:
                        self.network.put_data(rollout)
                        rollout = []

                    state = next_state
                    episode_reward += reward

                    if done:
                        break

                self.network.train()

            # self.save_epi_reward.append(episode_reward)
            if ep % 10 == 0 and ep != 0:
                print(f"Episode : {ep} | Reward : {episode_reward / 10}")
                self.save_epi_reward.append(episode_reward / 10)

                episode_reward = 0

    def plot_result(self):
        self.env.close()
        plt.plot(self.save_epi_reward)
        sv = np.array(self.save_epi_reward)
        np.save("./ppo_halfcheetah3.npy", sv)
        plt.show()


