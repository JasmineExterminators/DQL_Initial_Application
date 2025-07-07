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

    #normalizing 0-255 into -1 to 1
    normalized_state = (state.astype(np.float32) / 127.5) - 1
    import ipdb; ipdb.set_trace()
    flat_state = normalized_state.flatten()
    flat_state_tensor = torch.tensor(flat_state, dtype=torch.float32, device=device).unsqueeze(0) #notice we need format float32 cuz we gonna do matrix multiplication and all the weights and stuff are floats

    return flat_state_tensor
    # make it into one long matrix
    


