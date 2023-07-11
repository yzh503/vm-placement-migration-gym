from dataclasses import dataclass
import numpy as np
from src.agents.base import Base
from src.utils import convert_obs_to_dict

@dataclass
class FirstFitConfig: 
    pass

class FirstFitAgent(Base):
    def __init__(self, env, config):
        super().__init__(type(self).__name__, env, config)
        
    def learn(self):
        pass

    def load_model(self, modelpath):
        pass

    def save_model(self, modelpath):
        pass

    def act(self, observation):
        observation = convert_obs_to_dict(self.env.config, observation)
        vm_placement = np.array(observation["vm_placement"])
        cpu = np.array(observation["cpu"])
        memory = np.array(observation["memory"])
        vm_cpu = np.array(observation["vm_cpu"])
        vm_memory = np.array(observation["vm_memory"])

        action = np.copy(vm_placement)

        for v in range(self.env.config.vms):
            if vm_placement[v] == self.env.WAIT_STATUS: 
                for p in range(len(cpu)): 
                    if cpu[p] + vm_cpu[v] <= 1 and memory[p] + vm_memory[v] <= 1:
                        action[v] = p # first status is waiting 
                        cpu[p] += vm_cpu[v]
                        break

        return action