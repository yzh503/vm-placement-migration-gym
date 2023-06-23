from dataclasses import dataclass
from src.agents.base import Base

from stable_baselines3 import PPO

@dataclass
class BaselinePPOConfig: 
    pass

class BaselinePPOAgent(Base):
    def __init__(self, env, config):
        super().__init__(type(self).__name__, env, config)
        self.model = PPO("MlpPolicy", env, verbose=1)
        
    def learn(self):
        self.model.learn(total_timesteps=10000)

    def load_model(self, modelpath):
        pass

    def save_model(self, modelpath):
        pass

    def act(self, observation):
        action, _ = self.model.predict(observation)
        return action