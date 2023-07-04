import gymnasium as gym
import torch
import numpy as np

class PreprocessEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    @property
    def config(self):
        return self.env.config
    
    @property
    def n_actions(self):
        return self.env.n_actions

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return torch.from_numpy(obs).unsqueeze(dim=0), info

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = torch.from_numpy(observation).unsqueeze(dim=0)
        return observation, reward, terminated, truncated, info
    
    def convert_obs_to_dict(self, obs: torch.Tensor) -> dict:
        return self.env.convert_obs_to_dict(obs)