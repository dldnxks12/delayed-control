import random
import torch
import torch.nn as nn
import environment

import numpy as np

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers.
        Reference: https://github.com/MishaLaskin/rad/blob/master/curl_sac.py"""

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

def hard_update(network, target_network):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(param.data)

def soft_update(network, target_network, tau):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def set_seed(random_seed):
    if random_seed <= 0:
        seed = np.random.randint(1, 9999)
    else:
        seed = random_seed

    torch.manual_seed(seed)
    np.random.seed(seed)

    return seed

def make_env(env_name, seed):
    import os
    import gymnasium as gym

    xml_file = os.getcwd() + "/environment/assets/inverted_single_pendulum.xml"

    env = gym.make("InvertedSinglePendulum-v4", model_path=xml_file)
    env.action_space.seed(seed)

    env_eval = gym.make("InvertedSinglePendulum-v4", model_path=xml_file)
    env_eval.action_space.seed(seed)

    return env, env_eval
