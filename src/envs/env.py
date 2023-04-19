from dataclasses import dataclass
import gym
from gym import spaces
import numpy as np
import src.utils as utils
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

WAIT_STATUS = -1
EMPTY_SLOT = -2

@dataclass
class EnvConfig(object):
    arrival_rate: float
    service_rate: float
    p_num: int
    v_num: int
    var: float # std deviation of normal in KL divergence 
    training_steps: int
    eval_steps: int
    seed: int
    reward_function: int
    sequence: str
    cap_target_util: bool

    def __post_init__(self):
        self.arrival_rate = float(self.arrival_rate)
        self.service_rate = float(self.service_rate)
        self.p_num = int(self.p_num)
        self.v_num = int(self. v_num)
        self.var = float(self.var)
        self.training_steps = int(self.training_steps)
        self.eval_steps = int(self.eval_steps)
        self.sequence = str(self.sequence)
        self.seed = int(self.seed)
        self.reward_function = int(self.reward_function)
        self.cap_target_util = bool(self.cap_target_util)

class VmEnv(gym.Env):

    metadata = {'render.modes': ['ansi']}

    def __init__(self, config: EnvConfig):
        super(VmEnv, self).__init__()

        self.config = config

        self.rng1 = np.random.default_rng(self.config.seed)
        self.rng2 = np.random.default_rng(self.config.seed)
        self.rng3 = np.random.default_rng(self.config.seed)
        self.rng4 = np.random.default_rng(self.config.seed)

        
        # [vm_placement, vm_remaining_runtime, vm_resource, pm_utilisation]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.config.v_num * 3 + self.config.p_num,1), dtype=np.float32) 
        self.action_space = spaces.Discrete(self.config.v_num * (self.config.p_num + 1)) # VMs * (PMs + wait status)

        self.reset()
    
    @property
    def n_actions(self):
        if isinstance(self.action_space, spaces.Discrete):
            return self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            return self.action_space.nvec.sum()

   
    def step(self, action, eval_mode=False):

        vm_index, vm_action = utils.get_action_pair(action, self.config.p_num)

        if eval_mode: 
            self.last_vm_placement = np.copy(self.vm_placement)
        
        action_valid = True
        action_valid = action_valid and not (vm_action == EMPTY_SLOT)
        action_valid = action_valid and not (self.vm_placement[vm_index] == EMPTY_SLOT) 
        action_valid = action_valid and not (self.vm_placement[vm_index] == vm_index)
        action_valid = action_valid and not (self.vm_placement[vm_index] > WAIT_STATUS and vm_action > WAIT_STATUS)
        action_valid = action_valid and not (self.pm_utilisation[vm_action] + self.vm_resource[vm_index] > 1)


        if action_valid: 
            previous_pm = self.vm_placement[vm_index] 
            self.vm_placement[vm_index] = vm_action
            if vm_action == -1:  # Free up PM
                self.pm_utilisation[previous_pm] -= self.vm_resource[vm_index]
                self.vm_suspended[vm_index] = 1
                self.suspend_action += 1
            else:
                self.pm_utilisation[vm_action] += self.vm_resource[vm_index]
                if self.vm_suspended[vm_index] == 0:
                    self.served_requests += 1
                self.vm_suspended[vm_index] = 0
                self.place_action += 1

        obs, reward, done, info = self._process_action(eval_mode)

        info = info | {
            "action": (vm_index, vm_action),
            "valid": action_valid
        }

        self.timestep += 1
        return obs, reward, done, info  
  
    def _process_action(self, eval_mode: bool):
        # Action is predicted against the observation, so update arrival and terminatino after the action.
        self._run_vms()
        self._accept_vm_requests() 
        
        vms_arrived = np.count_nonzero(self.vm_placement > EMPTY_SLOT)
        waiting_ratio = np.count_nonzero(self.vm_placement == WAIT_STATUS) / vms_arrived if vms_arrived > 0 else 0
        used_pm_ratio = np.count_nonzero(self.pm_utilisation > 0) / self.config.p_num
        target_util_mean = np.sum(self.vm_resource[self.vm_placement != EMPTY_SLOT]) / self.config.p_num

        if self.config.cap_target_util and target_util_mean > 1: 
            target_util_mean = 1.0

        if self.config.reward_function == 1: # KL divergence between from approximator to true
            std = np.std(self.pm_utilisation) 
            current = Normal(np.mean(self.pm_utilisation), std if std > 0 else self.config.var)
            target = Normal(target_util_mean, np.sqrt(self.config.var))
            if target_util_mean == 0:
                reward = 0.0
            else:
                reward = - kl_divergence(target,current).item()      
        elif self.config.reward_function == 2: 
            util = self.pm_utilisation[self.pm_utilisation > 0]
            if util.size > 0: 
                reward = np.mean(util)
            else:
                reward = 0.0
        elif self.config.reward_function == 3:
            reward = - waiting_ratio 
        else: 
            assert False, f'Function does not exist: {self.config.reward_function}'

        obs = self._observation()

        if eval_mode: 
            done = self.timestep >= self.config.eval_steps
        else:
            done = self.timestep >= self.config.training_steps

        info = {
            "waiting_ratio": waiting_ratio, 
            "used_pm_ratio": used_pm_ratio,
            "served_requests": self.served_requests,
            'suspend_actions': self.suspend_action,
            'place_actions': self.place_action,
            "dropped_requests": self.dropped_requests, 
            "total_requests": self.total_requests, 
            "timestep": self.timestep,
            "vm_arrival_steps": self.vm_arrival_steps,
            "vm_placement": self.vm_placement.copy(), 
            "pm_utilisation": self.pm_utilisation.copy(),
            "target_util_mean": target_util_mean,
            'total_resource_requested': self.total_resource_requested,
        }

        return obs, reward, done, info

    def random_seed(self):
        self.config.seed += 1
        self.rng1 = np.random.default_rng(self.config.seed)
        self.rng2 = np.random.default_rng(self.config.seed)
        self.rng3 = np.random.default_rng(self.config.seed)
        self.rng4 = np.random.default_rng(self.config.seed)

    def reset(self):
        # Observable
        self.vm_placement = np.full(self.config.v_num, EMPTY_SLOT) # -1 is a VM request. -2 is an empty slot. 0... are PM indices. 
        self.vm_resource = np.zeros(self.config.v_num) 
        self.pm_utilisation = np.zeros(self.config.p_num)
        self.vm_remaining_runtime = np.zeros(self.config.v_num, dtype=int)
        # Not in observation
        self.timestep = 1
        self.total_requests = 0
        self.served_requests = 0
        self.suspend_action = 0
        self.place_action = 0
        self.dropped_requests = 0
        self.vm_planned_runtime = np.zeros(self.config.v_num, dtype=int)
        self.vm_waiting_time = np.zeros(self.config.v_num, dtype=int)
        self.vm_suspended = np.zeros(self.config.v_num, dtype=int)
        self.vm_arrival_steps = [[] for _ in range(self.config.v_num)] # Make sure the inner arrays are not references to the same array
        self.target_mean = []
        self.total_resource_requested = 0

        max_steps = max(self.config.training_steps, self.config.eval_steps)
        if self.config.sequence == 'ffworst':
            self.vm_sequence = np.tile(np.concatenate((np.repeat(0.15, 6 * self.config.eval_steps // 100), np.repeat(0.34, 8 * self.config.eval_steps // 100), np.repeat(0.51, 6 * self.config.eval_steps // 100))), max_steps // 10).tolist()
        elif self.config.sequence == 'multinomial':
            self.vm_sequence = self.rng4.choice([0.125,0.25,0.375,0.5,0.675,0.75,0.875], p=[0.148,0.142,0.142,0.142,0.142,0.142,0.142], size=max_steps+1, replace=True) # uniform discrete
        elif self.config.sequence == 'uniform':
            self.vm_sequence = np.around(self.rng1.uniform(low=0.1, high=1, size=max_steps*2), decimals=2).tolist() # mean 0.55
        elif self.config.sequence == 'lowuniform':
            self.vm_sequence = np.around(self.rng1.uniform(low=0.1, high=0.65, size=max_steps*2), decimals=2).tolist() # mean 0.375
        elif self.config.sequence == 'highuniform':
            self.vm_sequence = np.around(self.rng1.uniform(low=0.25, high=1, size=max_steps*2), decimals=2).tolist() # mean 0.625
        elif self.config.sequence == 'sqrtuniform':
            self.vm_sequence = np.around(np.sqrt(self.rng1.uniform(low=0.1, high=1, size=max_steps*2)), decimals=2).tolist()
        
        # If requests drop, it will require a seq length longer than max_steps. This will only work when drop rate < 50%

        return self._observation()

    def render(self, mode='ansi', close=False):
        np.set_printoptions(linewidth=np.inf)
        print(f"Timestep: \t\t{self.timestep}")
        print(f"VM request: \t\t{np.count_nonzero(self.vm_placement == -1)}, dropped: {self.dropped_requests}")
        print(f"VM placement: \t\t{self.vm_placement}")
        print(f"VM suspended: \t\t{self.vm_suspended}")
        print(f"VM resources (%): \t{np.array(self.vm_resource*100, dtype=int)} {np.round(np.sum(self.vm_resource), 3)}")
        print(f"PM utilisation (%): \t{np.array(self.pm_utilisation*100, dtype=int)} {np.round(np.sum(self.pm_utilisation), 3)}")
        print(f"VM waiting time: \t{self.vm_waiting_time}")
        print(f"VM planned runtime: \t{self.vm_planned_runtime}")
        print(f"VM remaining runtime: \t{self.vm_remaining_runtime}")

    def close(self):
        pass

    def _run_vms(self):
        vm_running = np.argwhere(np.logical_and(self.vm_remaining_runtime > 0, self.vm_placement > WAIT_STATUS))
        if vm_running.size > 0:
            self.vm_remaining_runtime[vm_running] -= 1  

        vm_waiting = np.argwhere(self.vm_placement == WAIT_STATUS)
        self.vm_waiting_time[vm_waiting] += 1
            
        vm_to_terminate = np.argwhere(np.logical_and(self.vm_remaining_runtime == 0, self.vm_placement > WAIT_STATUS)).flatten()

        if vm_to_terminate.size > 0:
            pms_to_free_up = self.vm_placement[vm_to_terminate]
            self.vm_placement[vm_to_terminate] = EMPTY_SLOT

            # Multiple VMs could be on the same PM, so use a loop to free up iteratively
            for vm, pm in zip(vm_to_terminate, pms_to_free_up): 
                self.pm_utilisation[pm] -= self.vm_resource[vm]

            self.vm_resource[vm_to_terminate] = 0
            self.vm_planned_runtime[vm_to_terminate] = 0
            self.vm_waiting_time[vm_to_terminate] = 0
            self.vm_remaining_runtime[vm_to_terminate] = 0
            self.vm_suspended[vm_to_terminate] = 0

        self.pm_utilisation[self.pm_utilisation < 1e-7] = 0 # precision problem 
        
    def _accept_vm_requests(self):
        arrivals = self.rng3.poisson(self.config.arrival_rate)
        self.total_requests += arrivals
        placed_arrivals = min(arrivals, self.vm_placement[self.vm_placement ==  EMPTY_SLOT].size)
        to_accept = np.argwhere(self.vm_placement == EMPTY_SLOT).flatten()[:placed_arrivals]
        self.vm_placement[to_accept] = WAIT_STATUS
        n_vms = self.vm_sequence[:to_accept.size]
        self.total_resource_requested += np.sum(n_vms)
        self.vm_sequence = self.vm_sequence[to_accept.size:]
        self.vm_resource[to_accept] = n_vms

        self.vm_planned_runtime[to_accept] =  self.rng2.poisson(self.config.service_rate, size=to_accept.size) + 1 
        self.vm_remaining_runtime[to_accept] = self.vm_planned_runtime[to_accept] # New request start counting
        self.dropped_requests += arrivals - placed_arrivals
        for i in to_accept: 
            self.vm_arrival_steps[i].append(self.timestep + 1) # Arrival at next step

    def _observation(self):
        return np.concatenate((self.vm_placement, self.vm_resource, self.vm_remaining_runtime, self.pm_utilisation))