import gymnasium as gym
import sys
import numpy as np
import random
from time import sleep

def behavior_policy(random_policy, state, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(random_policy[state, :])

    return action


def Sarsa(env, random_policy,  Y, epsilon, n_epsiodes):
    Q = np.zeros([nS, nA])
    eligibility = np.zeros([nS, nA])
    lam = 0.95

    for e in range(n_epsiodes):
        state = env.reset()[0]  # Initial State

        gamma = 0.9
        alpha = 0.1

        total_reward = 0
        done = False
        truncated = False

        init_flag = True

        action_buffer = []
        state_buffer  = []
        while True:
            action = np.argmax(Q[state, :]) # state : 0 , action : 0

            action_buffer.append(action)
            state_buffer.append(state)

            if init_flag == True:
                next_state, reward, done, truncated, info = env.step(action_buffer[-1]) # action : 0

            else:
                next_state, reward, done, truncated, info = env.step(action_buffer[-2]) # action : 1

                eligibility = eligibility * lam * gamma
                eligibility[state][action_buffer[-2]] = eligibility[state][action_buffer[-2]] + 1

            if init_flag == True:
                state = next_state
                init_flag = False
                continue

            total_reward += reward

            TD_error = (reward + gamma * Q[next_state][action_buffer[-1]]) - Q[state][action_buffer[-2]]
            Q = Q + alpha * TD_error * eligibility

            state = next_state

            if done == True or truncated == True:
                break

        Y.append(total_reward)
        print(f"Episode : {e} || total_reward : {total_reward}")

    Y = np.array(Y)
    np.save('./dSarsa(λ)', Y)
    return Q

Y = []

env = gym.make('Taxi-v3').env
env = gym.wrappers.TimeLimit(env, max_episode_steps = 30)

nS = env.observation_space.n
nA = env.action_space.n
random_policy = np.ones([nS, nA]) / nA

Q = Sarsa(env, random_policy, Y , epsilon = 0.1, n_epsiodes = 5000)

