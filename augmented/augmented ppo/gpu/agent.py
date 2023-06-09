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
    def __init__(self, env, state_dim, action_dim, action_bound, d_sample, device):
        super(PPOagent, self).__init__()
        self.device  = device
        self.rollout_len = 3

        self.env          = env
        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.action_bound = action_bound[1]

        self.d_sample = d_sample

        # Define Actor & Critic
        self.network  = ppo_network.ActorCritic(self.state_dim, self.action_dim, self.action_bound, self.device).to(device)
        self.save_epi_reward = []

    def train(self, max_episode_num):
        episode_reward = 0
        rollout = []
        for ep in range(max_episode_num):
            state = self.env.reset()[0]
            state = torch.FloatTensor(state)
            done, terminated, truncated = False, False, False

            act_buf = []
            d = self.d_sample
            for _ in range(self.d_sample):
                act_buf.append(torch.FloatTensor([1e-10]))

            while not done:
                for _ in range(self.rollout_len):
                    I = torch.concatenate([torch.FloatTensor(state), torch.FloatTensor(act_buf[-d:])], dim = 0)

                    with torch.no_grad():
                        mu_old, std_old = self.network.pi(torch.FloatTensor(I).to(self.device))
                    dist = Normal(mu_old, std_old)
                    action = dist.sample()
                    action = torch.clamp(action, -self.action_bound, self.action_bound)
                    old_log_prob = dist.log_prob(action) # grad o
                    action = action.detach().cpu()

                    # act_buf[-d] -> .item() test 해보기
                    next_state, reward, terminated, truncated, _ = self.env.step([act_buf[-d].item()])

                    act_buf.append(action)
                    I_next = torch.concatenate([torch.FloatTensor(next_state), torch.FloatTensor(act_buf[-d:])], dim=0)

                    done  = terminated or truncated

                    rollout.append((I, action, reward/10.0, I_next, [old_log_prob.item()], done))

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
        np.save("./ppo.npy", sv)
        plt.show()


