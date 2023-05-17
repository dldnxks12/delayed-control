# ref : https://github.com/dldnxks12/minimalRL/blob/master/ppo.py

import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

from torch.distributions import Categorical

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPO, self).__init__()
        self.data = []    # buffer
        self.gamma = 0.98 # discount factor
        self.labda = 0.95 # GAE parameter -n
        self.eps   = 0.1  # for clipping
        self.K     = 3    # Epochs - T step 동안 모은 데이터를 몇 번 재탕해서 학습할 지


        # fc1, fc2 layer parameter를 공유 (꼭 공유안해도 된다.)
        self.fc1   = nn.Linear(state_dim, 256)
        self.fc_pi = nn.Linear(256, action_dim)
        self.fc_v  = nn.Linear(256, 1)

        self.optimizir = optim.Adam(self.parameters(), lr = 0.0005)

        # softmax_dim ?
        # state 1개만 들어갈 때 (시뮬레이션할 때)       - softmax_dim = 0
        # state 여러 개가 batch로 들어갈 때 (학습할 때) - softmax_dim = 1
    def pi(self, state, softmax_dim = 0):
        x            = F.relu(self.fc1(state))
        x            = self.fc_pi(x)
        probability  = F.softmax(x, dim = softmax_dim)
        return probability

    def v(self, state):
        x = F.relu(self.fc1(state))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        states, actions, rewards, next_states, probs, dones = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            states.append(s)
            actions.append([a])
            rewards.append([r])
            next_states.append(s_prime)
            probs.append([prob_a])
            done_mask = 0 if done else 1
            dones.append([done_mask])

        s    = torch.tensor(states, dtype=torch.float)
        a    = torch.tensor(actions)
        r    = torch.tensor(rewards)
        n_s  = torch.tensor(next_states, dtype=torch.float)
        d    = torch.tensor(dones, dtype=torch.float)
        p    = torch.tensor(probs)

        self.data = []
        return s, a, r, n_s, d, p

    def train(self):

        # 2017 슐만이 낸 논문에 따라 기본 advantage function말고 GAE를 쓸 때 성능이 더 좋다고 한다.
        # GAE - batch 처리안하면 network를 40번을 호출해야한다. (2 x T)
        # batch 처리하면 2번만 호출
        states, actions, rewards, next_states, dones, probs = self.make_batch()

        for i in range(self.K): # K : Epoch
            td_target  = rewards + self.gamma*self.v(next_states)*dones
            deltas     = td_target - self.v(states)
            deltas     = deltas.detach().numpy()

            advantage_list = []
            advantage      = 0.0
            for delta_t in deltas[::-1]: # 뒤에서 부터 하나씩
                advantage = self.gamma * self.labda * advantage + delta_t[0]
                advantage_list.append([advantage])

            # GAE Get !
            advantage_list.reverse()
            advantage = torch.tensor(advantage_list, dtype = torch.float)

            pi_new   = self.pi(states, softmax_dim=1)
            pi_a_new = pi_new.gather(1, actions) # 선택한 action에 대한 확률들

            # K epoch 동안 probs 고정
            ratio    = torch.exp(torch.log(pi_a_new) - torch.log(probs))

            surrogate1 = ratio * advantage
            surrogate2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # Policy loss + Value loss
            # td_target.detach()이 매우 중요 - detach() 안하면 v(next_states)도 같이 학습이 되버림
            loss = -torch.min(surrogate1, surrogate2) + F.smooth_l1_loss(self.v(states), td_target.detach())

            self.optimizir.zero_grad()
            loss.mean().backward()
            self.optimizir.step()


def main():

    env   = gym.make('CartPole-v1')
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo   = PPO(state_dim=state_dim, action_dim=action_dim)
    """
    N ? Episode를 진행할 Actor 개수 (Here, N = 1 , 즉 Actor 1개만 사용) 
    T ? 몇 time-step동안 data를 모을지 
    PPO는 On policy 이기 때문에 중간 중간에 policy가 계속 업데이트 된다.
        -> T step 마다 update하겠다~ 
    """
    T     = 20
    score = 0.0

    for n_epi in range(10000):
        state = env.reset()[0]
        done  = False
        while not done:

            # while loop 안에 아래의 for loop가 하나 더 들어왔다 -> T step 만큼 data모아서 update
            for t in range(T):
                act_probability  = ppo.pi(torch.from_numpy(state).float())
                act_distribution = Categorical(act_probability)
                action           = act_distribution.sample().item()
                next_state, reward, done, _, _ = env.step(action)

                # π_old(a|s) <- probability[action].item() ...
                # 나중에 r(θ) 계산할 때 사용하기 위해서 넣어준다.
                ppo.put_data( (state, action, reward/100.0, next_state, act_probability[action].item(), done))
                state = next_state

                score += reward
                if done:
                    break

            # T time-step 마다 학습
            ppo.train()

        if n_epi%20==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/20))
            score = 0.0

    env.close()
if __name__ == "__main__":
    main()
