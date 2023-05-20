import sys
import gym
import torch.cuda

import ppo_agent

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # before validation process, use cpu
    device = 'cpu'
    print(f"device : {device}")

    max_episodes = 1000
    env_name     = 'Pendulum-v1'
    env          = gym.make(env_name)
    agent        = ppo_agent.PPOagent(env, device)

    agent.train(max_episodes)
    agent.plot_result()

if __name__ == "__main__":
    main()