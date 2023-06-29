"""
Part of the DQN algorithm code is from https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
"""
from dataclasses import dataclass
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from collections import namedtuple, deque
from src import utils

from src.agents.base import Base
from src.vm_gym.envs.env import VmEnv
from src.vm_gym.envs.preprocess import PreprocessEnv

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, obs_dims, n_action, hidden_size):
        super(DQN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(obs_dims, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_action),
        )

    def forward(self, x):
        qsa = self.linear_relu_stack(x)
        return qsa

    def select_action(self, x): 
        return self.linear_relu_stack(x).max(1)[1].view(1, 1)

@dataclass
class DQNConfig(object):
    episodes: int = 2000
    batch_size: int = 100
    memory_size: int = 10000
    gamma: float = 0.99
    eps_start: float = 1
    eps_end: float = 0.1
    eps_decay: int = 400
    target_update: int = 40
    training_progress_bar: bool = True
    hidden_size: int = 256
    lr: float = 1e-4
    device: str = "cpu"

class DQNAgent(Base):

    def __init__(self, env: VmEnv, config: DQNConfig):
        super().__init__(type(self).__name__, env, config)
        self.env = PreprocessEnv(self.env)
        self.device = self.config.device
        
        obs_dims = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.policy_net = DQN(obs_dims, n_actions, self.config.hidden_size).to(self.device)
        self.policy_net = torch.compile(self.policy_net)
        self.target_net = DQN(obs_dims, n_actions, self.config.hidden_size).to(self.device)
        self.target_net = torch.compile(self.target_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)#optim.SGD(self.policy_net.parameters(), momentum=0.9, lr=self.config.lr) # optim.RMSprop(self.policy_net.parameters()) # Alternative: 
        self.memory = ReplayMemory(self.config.memory_size)
        self.steps_done = 0
    
    def save_model(self, modelpath):
        if modelpath: 
            torch.save(self.policy_net.state_dict(), modelpath)

    def load_model(self, modelpath):
        self.policy_net.load_state_dict(torch.load(modelpath))
        self.policy_net.eval()

    def learn(self):
        ep_returns = np.zeros(self.config.episodes)
        pbar = tqdm(range(int(self.config.episodes)), disable=not bool(self.config.training_progress_bar))
        return_factor = int(self.config.episodes*0.01 if self.config.episodes >= 100 else 1)
        step = 0
        for i_episode in pbar:
            self.env.random_seed() # get different sequence
            current_ep_reward = 0
            # Initialize the environment and state
            previous_obs, info = self.env.reset(self.env.config.seed)
            done = False

            while not done:
                # Select and perform an action
                action = self._select_action(previous_obs)
                obs, reward, done, truncated, info = self.env.step(action)
                reward = torch.tensor([reward], device=self.device)
                self.memory.push(previous_obs, action, obs, reward)
                previous_obs, info = obs
                current_ep_reward += reward.item()  # For logging
                loss = self._optimize_model() # Perform one step of the optimization (on the policy network)
                if self.writer and loss: 
                    self.writer.add_scalar('Training/loss', loss, step)
                step += 1

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.config.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            ep_returns[i_episode] = current_ep_reward

            if self.writer: 
                self.writer.add_scalar('Training/ep_return', current_ep_reward, i_episode)

            if i_episode > return_factor: 
                pbar.set_description("Return %.2f" % np.median(ep_returns[i_episode-return_factor:i_episode]))

        ep_returns_median = np.zeros_like(ep_returns)
        for r in range(ep_returns.size):
            ep_returns_median[r] = np.median(ep_returns[r-return_factor:r])

    def act(self, observation):
        return self.policy_net.select_action(observation)

    def _select_action(self, observation):
        EPS_START = self.config.eps_start
        EPS_END = self.config.eps_end
        EPS_DECAY = self.config.eps_decay
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        sample = random.random()
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():                
                return self.act(observation)
        else:
            n_actions = self.env.action_space.n
            return torch.tensor([[random.randrange(n_actions)]], device=self.device, dtype=torch.long)

    def _optimize_model(self):
        BATCH_SIZE = self.config.batch_size
        GAMMA = self.config.gamma

        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch 

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

