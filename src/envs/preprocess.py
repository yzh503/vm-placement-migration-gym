import gym, torch

class PreprocessEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    @property
    def n_actions(self):
        return self.env.n_actions

    def reset(self):
        obs = self.env.reset()
        return torch.from_numpy(obs).unsqueeze(dim=0).float()

    def step(self, action, eval_mode=False):
        if isinstance(action, torch.Tensor):
            action = torch.flatten(action).cpu().numpy()
        if action.size == 1: 
            action = action.item()
        observation, reward, done, info = self.env.step(action, eval_mode=eval_mode)
        observation = torch.from_numpy(observation).unsqueeze(dim=0).float()
        return observation, reward, done, info