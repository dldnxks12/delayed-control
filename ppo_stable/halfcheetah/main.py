import os
import sys
import gymnasium as gym
import torch.cuda
import ppo_agent
import environment

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # before validation process, use cpu
    device = 'cpu'
    print(f"device : {device}")

    # xml_file = os.getcwd() + "/environment/assets/inverted_single_pendulum.xml"
    # env_name     = "InvertedSinglePendulum-v4"
    # env = gym.make(env_name, model_path=xml_file)

    env_name     = "HalfCheetah-v4"
    env          = gym.make(env_name)
    env_eval     = gym.make(env_name , render_mode = 'human')
    agent        = ppo_agent.PPOagent(env, env_eval, device)

    max_episodes = 3000
    agent.train(max_episodes)
    agent.plot_result()

if __name__ == "__main__":
    main()
