from src.agents.base import Base
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from dataclasses import dataclass

@dataclass
class PPOSBConfig:
    episodes: int = 2000
    progress_bar: bool = True
    learning_rate: float = 3e-5
    batch_size: int = 100
    ent_coef: float = 0.01
    hidden_size: int = 256
    device: str = "cpu"

class PPOSBAgent(Base):
    def __init__(self, env, config, tensorboard_log):
        super().__init__(type(self).__name__, env, config)
        self.model = PPO(policy="MlpPolicy",
                                  env=env, 
                                  tensorboard_log=tensorboard_log,
                                  ent_coef=self.config.ent_coef,
                                  batch_size=self.config.batch_size, 
                                  n_steps=self.env.config.training_steps, 
                                  learning_rate=self.config.learning_rate, 
                                  device=self.config.device,
                                  policy_kwargs=dict(net_arch=[dict(pi=[self.config.hidden_size, self.config.hidden_size], 
                                                                    vf=[self.config.hidden_size, self.config.hidden_size])]))

        self.lstm_states = None
        
    def learn(self):
        self.model.learn(total_timesteps=self.config.episodes * self.env.config.training_steps, 
                         progress_bar=self.config.progress_bar)

    def load_model(self, modelpath):
        self.model = PPO.load(modelpath)

    def save_model(self, modelpath):
        self.model.save(modelpath)

    def act(self, observation):
        action, _ = self.model.predict(observation)
        return action