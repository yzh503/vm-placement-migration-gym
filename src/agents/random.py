from dataclasses import dataclass
from src.agents.base import Base

@dataclass
class RandomConfig: 
    pass

class RandomAgent(Base):
    def __init__(self, env, config: RandomConfig):
        super().__init__(type(self).__name__, env, config)
    
    def load_model(self, modelpath):
        pass

    def save_model(self, modelpath):
        pass
    
    def learn(self):
        pass
    
    def act(self, observation):
        return int(self.rng.integers(low=0, high=self.env.action_space.n, size=1)[0])
