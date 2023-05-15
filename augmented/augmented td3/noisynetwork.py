import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLayer(nn.Linear):
    def __init__(self, in_features, out_features, sigma_zero = 0.017):
        super(NoisyLayer, self).__init__( in_features, out_features, bias = True)

        sigma_init = sigma_zero / math.sqrt(in_features)

        w = torch.full((out_features, in_features), sigma_init)

        self.sigma_weight = nn.Parameter(w)
        self.register_buffer('epsilon_weight', torch.zeros((out_features, in_features)))

        b = torch.full((out_features, ), sigma_init)
        self.sigma_bias = nn.Parameter(b)
        self.register_buffer('epsilon_bias', torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.bound = math.sqrt( 3 / self.in_features )
        self.weight.data.uniform_(-self.bound, self.bound)
        self.bias.data.uniform_(-self.bound, self.bound)

    def forward(self, x):
        self.epsilon_weight.normal_()
        self.epsilon_bias.normal_()

        # torch.normal_() -> torch.normal_(mean = 0, std = 1)
        noisy_weight = self.sigma_weight * self.epsilon_weight.data + self.weight
        noisy_bias   = self.sigma_bias   * self.epsilon_bias.data + self.bias

        return F.linear(x, weight = noisy_weight, bias = noisy_bias)






