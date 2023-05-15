import torch
import numpy as np
import random
from collections import deque 

# buffer 2
class ReplayMemory:
    def __init__(self, state_dim, action_dim, device, capacity=1e6):
        self.device = device
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0

        self.state_buffer = np.empty(shape=(self.capacity, state_dim), dtype=np.float32)
        self.action_buffer = np.empty(shape=(self.capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.empty(shape=(self.capacity, state_dim), dtype=np.float32)
        self.done_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.size = min(self.size + 1, self.capacity)

        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer[self.position] = done

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        states = torch.FloatTensor(self.state_buffer[idxs]).to(self.device)
        actions = torch.FloatTensor(self.action_buffer[idxs]).to(self.device)
        rewards = torch.FloatTensor(self.reward_buffer[idxs]).to(self.device)
        next_states = torch.FloatTensor(self.next_state_buffer[idxs]).to(self.device)
        dones = torch.FloatTensor(self.done_buffer[idxs]).to(self.device)

        return states, actions, rewards, next_states, dones

      
# buffer 1      

class ReplayBuffer:
    def __init__(self, state_dim, action_dim , device, capacity):

        self.device   = device
        self.buffer = deque(maxlen=int(capacity))

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for state, action, reward, next_state, done in transitions:

            states.append(state.cpu().numpy())
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state.cpu().numpy())
            dones.append(done)

        states      = torch.tensor(states)
        actions     = torch.tensor(actions)
        rewards     = torch.tensor(rewards)
        next_states = torch.tensor(next_states)
        dones       = torch.tensor(dones)

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)
      
