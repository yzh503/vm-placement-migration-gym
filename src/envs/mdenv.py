from gym import spaces
import numpy as np
from src.envs.env import EnvConfig, VmEnv

WAIT_STATUS = -1
EMPTY_SLOT = -2

class MultiDiscreteVmEnv(VmEnv):

    metadata = {'render.modes': ['ansi']}

    def __init__(self, config: EnvConfig):
        super(MultiDiscreteVmEnv, self).__init__(config)

        self.config = config
        
        # [vm_placement, vm_remaining_runtime, vm_resource, pm_utilisation]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.config.v_num * 3 + self.config.p_num,1), dtype=np.float32) 
        self.action_space = spaces.MultiDiscrete(np.full(self.config.v_num , self.config.p_num + 1))  # Every VM has (PMs + wait status) actions
        self.reset()

    def _placement_valid(self, pm, vm):
        return self.pm_utilisation[pm] + self.vm_resource[vm] > 1

    def _free_pm(self, pm, vm):
        self.pm_utilisation[pm] -= self.vm_resource[vm]
    
    def _place_vm(self, pm, vm):
        self.pm_utilisation[pm] += self.vm_resource[vm]

    def step(self, action, eval_mode=False):
        
        action = np.copy(action) - 1 # -1 denotes waiting
        actions_valid = np.zeros_like(action)

        if eval_mode: 
            self.last_vm_placement = np.copy(self.vm_placement)

        for vm, move_to_pm in enumerate(action): 
            current_pm = self.vm_placement[vm]
            action_valid = True
            action_valid = action_valid and not (move_to_pm == EMPTY_SLOT)                                          # No direct termination
            action_valid = action_valid and not (current_pm == EMPTY_SLOT)                       # VM has to be waiting or running
            action_valid = action_valid and not (current_pm == move_to_pm)                               # No same spot moving
            action_valid = action_valid and not (current_pm > WAIT_STATUS and move_to_pm > WAIT_STATUS)  # No direct swap
            action_valid = action_valid and not (self._placement_valid(move_to_pm, vm))        # PM has to be available

            actions_valid[vm] = int(action_valid)

            if action_valid: 
                self.vm_placement[vm] = move_to_pm
                if move_to_pm == WAIT_STATUS:  # Free up PM
                    self._free_pm(current_pm, vm)
                    self.vm_suspended[vm] = 1
                    self.suspend_action += 1
                elif move_to_pm > WAIT_STATUS: # Allocate
                    self._place_vm(move_to_pm, vm)
                    if self.vm_suspended[vm] == 0:
                        self.served_requests += 1
                    self.vm_suspended[vm] = 0
                    self.place_action += 1
                else: 
                    pass # do not change PM utilisation 

        obs, reward, done, info = self._process_action(eval_mode)

        info = info | {
            "action": action,
            "valid": actions_valid
        }

        if eval_mode:
            self.last_validity = actions_valid
            self.last_reward = np.round(reward, 3)
            self.last_action = action

        self.timestep += 1
        return obs, reward, done, info  