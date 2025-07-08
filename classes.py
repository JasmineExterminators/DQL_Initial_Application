# https://ale.farama.org/environments/breakout/
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    
    

class DQNEncoder(nn.Module):
    def __init__(self):
        super(DQNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # Compute output size after conv layers for a (3, 210, 160) input
        dummy_input = torch.zeros(1, 3, 210, 160)
        with torch.no_grad():
            conv_out = self._forward_conv(dummy_input)
        self.output_dim = conv_out.view(1, -1).size(1)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))  # -> [batch, 32, 51, 39]
        x = F.relu(self.conv2(x))  # -> [batch, 64, 24, 18]
        x = F.relu(self.conv3(x))  # -> [batch, 64, 22, 16]
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        return x.view(x.size(0), -1)  # flatten
