import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import sys
from noisynetwork import *

class NOISYCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(NOISYCritic, self).__init__()

        self.device = device
        self.apply(utils.weight_init)

        self.fc_A1   = NoisyLayer(state_dim + action_dim, 256)
        self.fc_A2   = NoisyLayer(256, 256)
        self.fc_Aout = NoisyLayer(256, 1)

        self.fc_B1   = NoisyLayer(state_dim + action_dim, 256)
        self.fc_B2   = NoisyLayer(256, 256)
        self.fc_Bout = NoisyLayer(256, 1)

    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            u = u.unsqueeze(0)

        return x, u

    def forward(self, state, action):
        s, a = self._format(state, action)

        X = torch.cat([s, a], dim = 1) #.to(torch.float32)

        A = F.relu(self.fc_A1(X))
        A = F.relu(self.fc_A2(A))
        QA = self.fc_Aout(A)

        B = F.relu(self.fc_B1(X))
        B = F.relu(self.fc_B2(B))
        QB = self.fc_Bout(B)

        return QA, QB

    def Q_A(self, state, action):
        s, a = self._format(state, action)

        X = torch.cat([s, a], dim = 1)
        A = F.relu(self.fc_A1(X))
        A = F.relu(self.fc_A2(A))
        QA = self.fc_Aout(A)

        return QA


class NOISYActor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, device):
        super(NOISYActor, self).__init__()
        self.apply(utils.weight_init)
        self.device = device
        self.action_bound = action_bound
        self.fc1 = NoisyLayer(state_dim, 256)
        self.fc2 = NoisyLayer(256, 256)
        self.fc3 = NoisyLayer(256, action_dim)

        self.action_rescale      = torch.as_tensor((action_bound[1] - action_bound[0]) / 2., dtype=torch.float32)
        self.action_rescale_bias = torch.as_tensor((action_bound[1] + action_bound[0]) / 2., dtype=torch.float32)


    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        return x

    def forward(self, state):
        x = self._format(state)

        x = x.to(self.device)
        x  = F.relu(self.fc1(x))
        x  = F.relu(self.fc2(x))
        x  = self.fc3(x)
        x  = F.tanh(x)
        x  = x * self.action_rescale + self.action_rescale_bias

        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(Critic, self).__init__()

        self.device = device
        self.apply(utils.weight_init)

        self.fc_A1   = nn.Linear(state_dim + action_dim, 256)
        self.fc_A2   = nn.Linear(256, 256)
        self.fc_Aout = nn.Linear(256, 1)

        self.fc_B1   = nn.Linear(state_dim + action_dim, 256)
        self.fc_B2   = nn.Linear(256, 256)
        self.fc_Bout = nn.Linear(256, 1)

    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            u = u.unsqueeze(0)

        return x, u

    def forward(self, state, action):
        s, a = self._format(state, action)

        X = torch.cat([s, a], dim = 1) #.to(torch.float32)

        A = F.relu(self.fc_A1(X))
        A = F.relu(self.fc_A2(A))
        QA = self.fc_Aout(A)

        B = F.relu(self.fc_B1(X))
        B = F.relu(self.fc_B2(B))
        QB = self.fc_Bout(B)

        return QA, QB

    def Q_A(self, state, action):
        s, a = self._format(state, action)

        X = torch.cat([s, a], dim = 1)
        A = F.relu(self.fc_A1(X))
        A = F.relu(self.fc_A2(A))
        QA = self.fc_Aout(A)

        return QA


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, device):
        super(Actor, self).__init__()
        self.apply(utils.weight_init)
        self.device = device
        self.action_bound = action_bound
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

        self.action_rescale      = torch.as_tensor((action_bound[1] - action_bound[0]) / 2., dtype=torch.float32)
        self.action_rescale_bias = torch.as_tensor((action_bound[1] + action_bound[0]) / 2., dtype=torch.float32)


    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        return x

    def forward(self, state):
        x = self._format(state)

        x = x.to(self.device)
        x  = F.relu(self.fc1(x))
        x  = F.relu(self.fc2(x))
        x  = self.fc3(x)
        x  = F.tanh(x)
        x  = x * self.action_rescale + self.action_rescale_bias

        return x


################## -------------------- from jang ----------------- #################

class Twin_Q_net(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dims=(256, 256), activation_fc=F.relu):
        super(Twin_Q_net, self).__init__()
        self.device = device
        self.apply(utils.weight_init)

        self.activation_fc = activation_fc

        self.input_layer_A = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers_A = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer_A = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers_A.append(hidden_layer_A)
        self.output_layer_A = nn.Linear(hidden_dims[-1], 1)

        self.input_layer_B = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers_B = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer_B = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers_B.append(hidden_layer_B)
        self.output_layer_B = nn.Linear(hidden_dims[-1], 1)

    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            u = u.unsqueeze(0)

        return x, u

    def forward(self, state, action):
        import sys
        x, u = self._format(state, action)
        x = torch.cat([x, u], dim=1).to(torch.float32)

        x_A = self.activation_fc(self.input_layer_A(x))
        for i, hidden_layer_A in enumerate(self.hidden_layers_A):
            x_A = self.activation_fc(hidden_layer_A(x_A))
        x_A = self.output_layer_A(x_A)

        x_B = self.activation_fc(self.input_layer_B(x))
        for i, hidden_layer_B in enumerate(self.hidden_layers_B):
            x_B = self.activation_fc(hidden_layer_B(x_B))
        x_B = self.output_layer_B(x_B)

        return x_A, x_B

    def Q_A(self, state, action):
        x, u = self._format(state, action)
        x = torch.cat([x, u], dim=1)

        x_A = self.activation_fc(self.input_layer_A(x))
        for i, hidden_layer_A in enumerate(self.hidden_layers_A):
            x_A = self.activation_fc(hidden_layer_A(x_A))
        x_A = self.output_layer_A(x_A)

        return x_A


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, device, hidden_dims=(256, 256), activation_fc=F.relu, out_activation=F.tanh):
        super(Policy, self).__init__()
        self.device = device
        self.apply(utils.weight_init)

        self.activation_fc = activation_fc
        self.out_activation_fc = out_activation
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim)

        self.action_rescale = torch.as_tensor((action_bound[1] - action_bound[0]) / 2., dtype=torch.float32)
        self.action_rescale_bias = torch.as_tensor((action_bound[1] + action_bound[0]) / 2., dtype=torch.float32)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        return x

    def forward(self, state): # state : cuda
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        x = self.out_activation_fc(x)
        x = x * self.action_rescale + self.action_rescale_bias

        return x


