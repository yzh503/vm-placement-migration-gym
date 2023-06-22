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
        obsdict = utils.convert_obs_to_dict(self.env.config.v_num, observation)
        vm_placement = np.array(obsdict["vm_placement"])
        cpu = np.array(obsdict["cpu"])
        vm_resource = np.array(obsdict["vm_resource"])
        action_vms = np.argwhere(vm_placement == -1)
        action_vm = 0 if action_vms.size == 0 else int(action_vms[0])
        action_pm = 0

        for best_pm in np.flip(np.argsort(cpu)): 
            if cpu[best_pm] + vm_resource[action_vm] <= 1: 
                action_pm = int(best_pm)
                cpu[action_pm] += vm_resource[action_vm]
                break
 
        return utils.get_action(action_vm, action_pm, self.env.config.p_num)