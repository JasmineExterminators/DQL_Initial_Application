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

def getPreprocessNormalizedLong1D(state, device):
    # look at the format of the matrix atm
    # the format is currently a 210 x 160 np array with rgb each an unsigned 8 bit int from 0-255
    # convert to grayscale 
    gray_state = np.dot(state[...,:3], [0.2989, 0.5870, 0.1140])

    #normalizing 0-255 into -1 to 1
    normalized_state = (gray_state.astype(np.float32) / 127.5) - 1
    # import ipdb; ipdb.set_trace()
    flat_state = normalized_state.flatten()
    flat_state_tensor = torch.tensor(flat_state, dtype=torch.float32, device=device).unsqueeze(0) #notice we need format float32 cuz we gonna do matrix multiplication and all the weights and stuff are floats

    return flat_state_tensor
    # make it into one long matrix

def normalize_rgb(image):
    return image.float() / 127.5 - 1


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
