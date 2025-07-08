import gymnasium as gym
from ale_py import ALEInterface
import ale_py

import math
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm
from einops import rearrange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from preprocessingLong1DVector import DQNEncoder, DQN



# DEFINE STUFF

# hyperparameters
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 25000
TAU = 0.0005
LR = 3e-4
BUFFER_SIZE = 1000000
FRAME_GAP = 5

# transition
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# replay buffer
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
        
# FUNCTIONS

def normalize_rgb(image):
    return image.float() / 127.5 - 1

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def plot_rewards(eps_reward, show_result=False):
        plt.figure(1)
        # durations_t = torch.tensor(episode_durations, dtype=torch.float)
        rewards_t = torch.tensor(eps_reward, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(rewards_t.numpy())
        # Take 100 episode averages and plot them too
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.savefig('reward_plot.png')
    # then put the results through relu
    # then through last layer (multiply matrix such that the output is batch size x actionSpace)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None], dim=0)
    non_final_next_states = non_final_next_states.to(device)
    state_batch = torch.cat(batch.state, dim=0).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_batch = encoder(normalize_rgb(state_batch))
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        non_final_next_states = encoder(normalize_rgb(non_final_next_states))
        next_state_values[non_final_mask] = target_net(
            non_final_next_states
        ).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if __name__ == "__main__":
    # initialize everything
    ale = ALEInterface()
    gym.register_envs(ale_py)
    env = gym.make('ALE/Tetris', obs_type="rgb")
    plt.ion()
    n_actions = env.action_space.n
    observation, info = env.reset()

    # choosing if GPU gonna be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    

    # this part allows reproducibility
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
   
    

    encoder = DQNEncoder()
    encoded_obs = encoder(torch.randn(1,3,210,160))  # Output shape: [32, output_dim]
    encoder = encoder.to(device)
    n_observations = encoded_obs.shape[-1]

    # n_observations = state.shape[1]

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW([
        {'params': encoder.parameters()},
        {'params': policy_net.parameters()},
    ], lr=LR)
    memory = ReplayMemory(BUFFER_SIZE)
    steps_done = 0

    # episode_durations = []
    eps_total_reward_list = []


    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    bar = tqdm(range(1000000))
    for i_episode in range(num_episodes):
        eps_total_reward = 0
        # Initialize the environment and get its state
        observation, info = env.reset()
        observation = torch.tensor(observation)
        observation = rearrange(observation, "h w c -> c h w").unsqueeze(0)
        # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        while True:
            encoded_obs = encoder(normalize_rgb(observation).to(device))
            action = select_action(encoded_obs)
            # action = select_action(state)
            next_observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
            next_observation = torch.tensor(next_observation)
            next_observation = rearrange(next_observation, "h w c -> c h w").unsqueeze(0)

            eps_total_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            bar.update(1)

            # Store the transition in memory
            memory.push(observation, action, next_observation, reward)

            # Move to the next state
            observation = next_observation

            # Perform one step of the optimization (on the policy network)
            if steps_done % FRAME_GAP == 0:
                optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                # episode_durations.append(t + 1)
                eps_total_reward_list.append(eps_total_reward)
                plot_rewards(eps_total_reward_list)
                break

    print('Complete')
    plot_rewards(eps_total_reward_list, show_result=True)
    plt.ioff()
    plt.show()
    env.close()