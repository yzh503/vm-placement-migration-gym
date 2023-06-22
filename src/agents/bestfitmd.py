from dataclasses import dataclass
import numpy as np
from src.agents.base import Base
import src.utils as utils

@dataclass
class BestFitMDConfig: 
    pass

class BestFitMDAgent(Base):
    def __init__(self, env, config):
        super().__init__(type(self).__name__, env, config)

    def learn(self):
        pass

    def load_model(self, modelpath):
        pass

    def save_model(self, modelpath):
        pass

    def act(self, observation):
        obsdict = utils.convert_obs_to_dict(self.env.config.v_num, observation)
        vm_placement = np.array(obsdict["vm_placement"], copy=True)
        cpu = np.array(obsdict["cpu"], copy=True)
        vm_resource = np.array(obsdict["vm_resource"])

        action = np.copy(vm_placement)

        for v in range(self.env.config.v_num):
            if vm_placement[v] == -1: 
                for best_pm in np.flip(np.argsort(cpu)): 
                    if cpu[best_pm] + vm_resource[v] <= 1: 
                        action[v] = best_pm # first status is waiting 
                        cpu[best_pm] += vm_resource[v]
                        break

        action += 1 # first status is waiting
        return action