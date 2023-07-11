from src.agents.base import Base
from sb3_contrib import RecurrentPPO
from dataclasses import dataclass

@dataclass
class RecurrentPPOConfig:
    episodes: int = 2000
    progress_bar: bool = True
    learning_rate: float = 3e-5
    batch_size: int = 100
    ent_coef: float = 0.01
    hidden_size: int = 256
    n_lstm_layers: int = 1
    device: str = "cpu"

class RecurrentPPOAgent(Base):
    def __init__(self, env, config, tensorboard_log):
        super().__init__(type(self).__name__, env, config)
        self.model = RecurrentPPO(policy="MlpLstmPolicy",
                                  env=env, 
                                  tensorboard_log=tensorboard_log,
                                  ent_coef=self.config.ent_coef,
                                  batch_size=self.config.batch_size, 
                                  n_steps=self.env.config.training_steps, 
                                  learning_rate=self.config.learning_rate, 
                                  device=self.config.device,
                                  policy_kwargs=dict(lstm_hidden_size=self.config.hidden_size, n_lstm_layers=self.config.n_lstm_layers))

        self.lstm_states = None
        
    def learn(self):
        self.model.learn(total_timesteps=self.config.episodes * self.env.config.training_steps, 
                         progress_bar=self.config.progress_bar)

    def load_model(self, modelpath):
        self.model = RecurrentPPO.load(modelpath)

    def save_model(self, modelpath):
        self.model.save(modelpath)

    def act(self, observation):
        action, lstm_states = self.model.predict(observation, state=self.lstm_states, deterministic=False)
        self.lstm_states = lstm_states
        return action