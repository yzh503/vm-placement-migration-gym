from dataclasses import dataclass
import numpy as np
from src.agents.base import Base
import src.utils as utils

@dataclass
class BestFitConfig: 
    pass

class BestFitAgent(Base):
    def __init__(self, env, config):
        super().__init__(type(self).__name__, env, config)

    def learn(self):
        pass

    def load_model(self, modelpath):
        pass

    def save_model(self, modelpath):
        pass

    def act(self, observation):
        observation = utils.convert_obs_to_dict(self.env.config.v_num, self.env.config.p_num, observation)
        vm_placement = np.array(observation["vm_placement"], copy=True)
        cpu = np.array(observation["cpu"], copy=True)
        memory = np.array(observation["memory"], copy=True)
        vm_cpu = np.array(observation["vm_cpu"])
        vm_memory = np.array(observation["vm_memory"])
        prod = np.multiply(cpu, memory)

        action = np.copy(vm_placement)

        for v in range(self.env.config.v_num):
            if vm_placement[v] == -1: 
                for best_pm in np.flip(np.argsort(prod)): 
                    valid = cpu[best_pm] + vm_cpu[v] < 1 and memory[best_pm] + vm_memory[v] < 1
                    if valid: 
                        action[v] = best_pm # first status is waiting 
                        cpu[best_pm] += vm_cpu[v]
                        memory[best_pm] += vm_memory[v]
                        break

        action += 1 # first status is waiting
        return action