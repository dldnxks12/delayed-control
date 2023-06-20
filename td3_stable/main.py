import os
import sys
import gym
import torch
import numpy as np
import utils

import td3
import trainer

def main():

    device        = 'cuda' if torch.cuda.is_available() else 'cpu'
    device        = 'cpu'
    random_seed   = 1234
    env_name      = "HalfCheetah-v4"
    seed          = utils.set_seed(random_seed)
    env, env_eval = utils.make_env(env_name, seed)

    print(f"Device : {device} Random Seed : {seed} Environment : {env_name}")

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = [env.action_space.low[0], env.action_space.high[0]]

    agent = td3.TD3(state_dim, action_dim, action_bound, device)

    train = trainer.Trainer(env, env_eval, agent, env_name, device)
    train.run()

    return env, env_eval, agent, env_name, device

if __name__ == "__main__":
    env, env_eval, agent, env_name, device = main()
