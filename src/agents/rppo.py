from src.agents.base import Base
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from dataclasses import dataclass
@dataclass
class RPPOConfig:
    n_episodes: int 
    progress_bar: bool
    learning_rate: float
    batch_size: int
    ent_coef: float

    def __post_init__(self):
        self.n_episodes = int(self.n_episodes)
        self.progress_bar = bool(self.progress_bar)
        self.learning_rate = float(self.learning_rate)
        self.batch_size = int(self.batch_size)
        self.ent_coef = float(self.ent_coef)

class RecurrentPPOAgent(Base):
    def __init__(self, env, config):
        super().__init__(type(self).__name__, env, config)
        policy = RecurrentActorCriticPolicy(observation_space=self.env.observation_space, 
                                            action_space=self.env.action_space, 
                                            lr_schedule=lambda r: self.config.learning_rate,
                                            use_sde=False,
                                            lstm_hidden_size=512)
        self.model = RecurrentPPO(policy="MlpLstmPolicy",
                                  env=env, 
                                  learning_rate=self.config.learning_rate, 
                                  n_steps=self.env.config.training_steps, 
                                  batch_size=self.config.batch_size, 
                                  ent_coef=self.config.ent_coef,
                                  device="cpu")

        self.lstm_states = None
        
    def learn(self):
        self.model.learn(total_timesteps=self.config.n_episodes * self.env.config.training_steps, 
                         progress_bar=self.config.progress_bar)

    def load_model(self, modelpath):
        self.model = RecurrentPPO.load(modelpath)

    def save_model(self, modelpath):
        self.model.save(modelpath)

    def act(self, observation):
        action, lstm_states = self.model.predict(observation, state=self.lstm_states, deterministic=False)
        self.lstm_states = lstm_states
        return action