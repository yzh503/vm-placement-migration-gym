from dataclasses import dataclass
import numpy as np
from src.agents.base import Base
from src.utils import convert_obs_to_dict

class BestFitAgent(Base):
    def __init__(self, env):
        super().__init__(type(self).__name__, env, None)

    def learn(self):
        pass

    def load_model(self, modelpath):
        pass

    def save_model(self, modelpath):
        pass

    def act(self, observation):
        observation = convert_obs_to_dict(self.env.config, observation)
        vm_placement = np.array(observation["vm_placement"], copy=True)
        cpu = np.array(observation["cpu"], copy=True)
        memory = np.array(observation["memory"], copy=True)
        vm_cpu = np.array(observation["vm_cpu"])
        vm_memory = np.array(observation["vm_memory"])

        action = np.copy(vm_placement)

        for v in range(self.env.config.vms):
            if vm_placement[v] == self.env.config.pms:  
                for best_pm in np.flip(np.argsort(cpu + memory)): 
                    valid = cpu[best_pm] + vm_cpu[v] <= 1 and memory[best_pm] + vm_memory[v] <= 1
                    if valid: 
                        action[v] = best_pm # first status is waiting 
                        cpu[best_pm] += vm_cpu[v]
                        memory[best_pm] += vm_memory[v]
                        break

        return action