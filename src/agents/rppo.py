from src.agents.base import Base
from sb3_contrib import RecurrentPPO

class RecurrentPPOAgent(Base):
    def __init__(self, env, config):
        super().__init__(type(self).__name__, env, config)
        self.model = RecurrentPPO("MlpLstmPolicy", env)
        
    def learn(self):
        self.model.learn(
            total_timesteps=self.config.n_episodes * self.env.config.training_steps, 
            progress_bar=self.config.show_training_progress)

    def load_model(self, modelpath):
        pass

    def save_model(self, modelpath):
        pass

    def act(self, observation):
        action, _ = self.model.predict(observation)
        return action