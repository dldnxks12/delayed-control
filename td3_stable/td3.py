import torch
import torch.nn.functional as F

import sys
import utils
import numpy as np
import buffer
import network
import buffer


class TD3:
    def __init__(self, state_dim, action_dim, action_bound, device):

        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.action_bound = action_bound

        self.device       = device
        self.capacity     = 1000000
        #self.memory       = buffer.ReplayBuffer(self.state_dim, self.action_dim, device, self.capacity)
        self.memory = buffer.ReplayMemory(self.state_dim, self.action_dim, device, self.capacity)
        self.batch_size   = 256

        self.act_noise_scale = 0.1
        self.actor_lr        = 0.0001
        self.critic_lr       = 0.0001
        self.gamma           = 0.99
        self.tau             = 0.005


        # Js - simple network
        self.actor         = network.Actor(self.state_dim, self.action_dim, self.action_bound)
        self.critic        = network.Critic(self.state_dim, self.action_dim, self.device)
        self.target_actor  = network.Actor(self.state_dim, self.action_dim, self.action_bound)
        self.target_critic = network.Critic(self.state_dim, self.action_dim, self.device)

        # Jangs network
        #self.actor         = network.Policy(self.state_dim,  self.action_dim, self.action_bound, self.device)
        #self.critic        = network.Twin_Q_net(self.state_dim, self.action_dim, self.device)
        #self.target_actor  = network.Policy(self.state_dim,  self.action_dim, self.action_bound, self.device)
        #self.target_critic = network.Twin_Q_net(self.state_dim, self.action_dim, self.device)

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr = self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = self.critic_lr)

        utils.hard_update(self.actor, self.target_actor)
        utils.hard_update(self.critic, self.target_critic)

    def get_action(self, state, add_noise = True):
        with torch.no_grad():
            if add_noise:
                noise = np.random.normal(loc=0, scale=abs(self.action_bound[1]) * self.act_noise_scale, size=self.action_dim)
                action = self.actor(state).cpu().numpy()[0] + noise
                action = np.clip(action, self.action_bound[0], self.action_bound[1])
            else:
                action = self.actor(state).cpu().numpy()
        return action

    def train_actor(self, states): # OK
        actor_loss = -self.critic.Q_A(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def train_critic(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            target_act_noise   = (torch.randn_like(actions) * 0.2).clamp(-0.2, 0.2).to(self.device)
            next_target_action = (self.target_actor(next_states) + target_act_noise).clamp(self.action_bound[0], self.action_bound[1])

            NQA, NQB = self.target_critic(next_states, next_target_action)
            next_q_values   = torch.min(NQA, NQB)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        QA, QB = self.critic(states, actions)
        critic_loss = ((QA - target_q_values)**2).mean() + ((QB - target_q_values)**2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        self.critic_optimizer.step()

    def train(self, option = 'both'):
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        if option == 'both':
            self.train_actor(states)
            self.train_critic(states, actions, rewards, next_states, dones)

            utils.soft_update(self.actor, self.target_actor, self.tau)
            utils.soft_update(self.critic, self.target_critic, self.tau)

        elif option == 'critic_only':
            self.train_critic(states, actions, rewards, next_states, dones)

        else:
            raise Exception("Wrong option")
