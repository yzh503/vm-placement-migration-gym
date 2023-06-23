import gymnasium as gym
import torch
import numpy as np

class PreprocessEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    @property
    def n_actions(self):
        return self.env.n_actions

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return torch.from_numpy(obs).unsqueeze(dim=0), info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = torch.flatten(action).numpy()
        if action.size == 1: 
            action = action.item()
        observation, reward, done, truncated, info = self.env.step(action)
        observation = torch.from_numpy(observation).unsqueeze(dim=0)
        return observation, reward, done, truncated, info