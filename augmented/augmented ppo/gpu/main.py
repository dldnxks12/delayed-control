import os
import sys
import gymnasium as gym
import torch.cuda
import ppo_agent
#import environment

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # xml_file = os.getcwd() + "/environment/assets/inverted_single_pendulum.xml"
    # env_name     = "InvertedSinglePendulum-v4"
    # env = gym.make(env_name, model_path=xml_file)
    env_name  = "Pendulum-v1"
    env       = gym.make(env_name)

    d_sample  = 1
    s_time    = "50ms"
    print(f"Device : {device} \nDelayed Sample : {d_sample} | Sampling Time : {s_time}  | Environment : {env_name} " )
    print("-----------------------------------------------------------------------------------------------------------")

    state_dim = env.observation_space.shape[0] + d_sample
    action_dim = env.action_space.shape[0]
    action_bound = [env.action_space.low[0], env.action_space.high[0]]

    agent        = ppo_agent.PPOagent(env, state_dim, action_dim, action_bound, d_sample, device)

    max_episodes = 100000
    agent.train(max_episodes)
    agent.plot_result()

if __name__ == "__main__":
    main()
