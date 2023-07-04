from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
import torch

WAIT_STATUS = -1
EMPTY_SLOT = -2

@dataclass
class EnvConfig(object):
    arrival_rate: float = 0.182 # 100% system load: pms / distribution expectation / service rate 
    service_length: float = 100
    pms: int = 10
    vms: int = 30   
    var: float = 0.01 # std deviation of normal in KL divergence 
    training_steps: int = 500
    eval_steps: int = 100000
    seed: int = 0
    reward_function: str = "waiting_ratio"
    sequence: str = "uniform"
    cap_target_util: bool = True
    beta: int = 0.5

class VmEnv(gym.Env):

    metadata = {'render_modes': ['ansi']}

    def __init__(self, config: EnvConfig):
        self.config = config
        self.eval_mode = False
        self.observation_space = spaces.Box(low=-2, high=self.config.pms, shape=(self.config.vms * 3 + self.config.pms * 2,)) 
        self.action_space = spaces.MultiDiscrete(np.full(self.config.vms , self.config.pms + 1))  # Every VM has (PMs + wait status) actions
        self.reset(self.config.seed)

        print("Environment initialized with config: ", self.config)

    def _placement_valid(self, pm, vm):
        return self.cpu[pm] + self.vm_cpu[vm] <= 1 and self.memory[pm] + self.vm_memory[vm] <= 1

    def _free_pm(self, pm, vm):
        self.cpu[pm] -= self.vm_cpu[vm]
        self.memory[pm] -= self.vm_memory[vm]
    
    def _place_vm(self, pm, vm):
        self.cpu[pm] += self.vm_cpu[vm]
        self.memory[pm] += self.vm_memory[vm]

    def step(self, action):
        action = action.copy()
        action -= 1 # -1 for wait status
        actions_valid = np.zeros_like(action)

        for vm, move_to_pm in enumerate(action): 
            current_pm = self.vm_placement[vm]
            action_valid = True
            action_valid = action_valid and not (move_to_pm == EMPTY_SLOT)                                          # No direct termination
            action_valid = action_valid and not (current_pm == EMPTY_SLOT)                       # VM has to be waiting or running
            action_valid = action_valid and not (current_pm == move_to_pm)                               # No same spot moving
            action_valid = action_valid and not (current_pm > WAIT_STATUS and move_to_pm > WAIT_STATUS)  # No direct swap
            action_valid = action_valid and self._placement_valid(move_to_pm, vm)        # PM has to be available

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

        obs, reward, terminated, info = self._process_action()

        info = info | {
            "action": action.tolist(),
            "valid": actions_valid
        }

        if self.eval_mode:
            self.last_validity = actions_valid
            self.last_reward = np.round(reward, 3)
            self.last_action = action

        self.timestep += 1
        truncated = False
        return obs, reward, terminated, truncated, info  

    @property
    def n_actions(self):
        if isinstance(self.action_space, spaces.Discrete):
            return self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            return self.action_space.nvec.sum()
        
    def eval(self, eval_mode=True):
        self.eval_mode = eval_mode  
  
    def _process_action(self):
        # Action is predicted against the observation, so update arrival and terminatino after the action.
        self._run_vms()
        self._accept_vm_requests() 
        
        vms_arrived = np.count_nonzero(self.vm_placement > EMPTY_SLOT)
        self.waiting_ratio = np.count_nonzero(self.vm_placement == WAIT_STATUS) / vms_arrived if vms_arrived > 0 else 0
        self.used_cpu_ratio = np.count_nonzero(self.cpu > 0) / self.config.pms
        self.target_cpu_mean = np.sum(self.vm_cpu[self.vm_placement != EMPTY_SLOT]) / self.config.pms
        self.target_memory_mean = np.sum(self.vm_memory[self.vm_placement != EMPTY_SLOT]) / self.config.pms

        if self.config.cap_target_util and self.target_cpu_mean > 1: 
            self.target_cpu_mean = 1.0

        if self.config.cap_target_util and self.target_memory_mean > 1: 
            self.target_memory_mean = 1.0

        if self.config.reward_function == "kl": # KL divergence between from approximator to true
            # std = np.std(self.cpu) 
            # target_sd = np.sqrt(self.config.var)
            current = MultivariateNormal(loc=torch.tensor([np.mean(self.cpu), np.mean(self.memory)]), covariance_matrix=torch.eye(2))
            target = MultivariateNormal(loc=torch.tensor([self.target_cpu_mean, self.target_memory_mean]), covariance_matrix=torch.eye(2))
            if self.target_cpu_mean == 0 or self.target_memory_mean == 0:
                reward = 0.0
            else:
                reward = - kl_divergence(target,current).item()      
        elif self.config.reward_function == "utilisation": 
            cpu_rate = self.cpu[self.cpu >= 0]
            memory_rate = self.memory[self.memory >= 0]

            if cpu_rate.size > 0 or memory_rate.size > 0: 
                reward = self.config.beta * np.mean(cpu_rate) + (1 - self.config.beta) * np.mean(memory_rate)
            else:
                reward = 0.0
        elif self.config.reward_function == "waiting_ratio":
            reward = - self.waiting_ratio 
        elif self.config.reward_function == "waiting_time":
            if self.vm_waiting_time[self.vm_waiting_time > 0].size == 0: 
                reward = 0.0
            else:
                reward = - np.mean(self.vm_waiting_time[self.vm_waiting_time > 0])
        else: 
            assert False, f'Function does not exist: {self.config.reward_function}'

        obs = self._get_obs()

        if self.eval_mode: 
            terminated = self.timestep >= self.config.eval_steps
        else:
            terminated = self.timestep >= self.config.training_steps

        info = self._get_info()

        return obs, reward, terminated, info

    def seed(self, seed):
        if seed is None:
            seed = self.config.seed
        self.rng1 = np.random.default_rng(seed)
        self.rng2 = np.random.default_rng(seed+1)
        self.rng3 = np.random.default_rng(seed+2)
        self.rng4 = np.random.default_rng(seed+3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seed(seed)
        # Observable
        self.vm_placement = np.full(self.config.vms, EMPTY_SLOT) # -1 is a VM request. -2 is an empty slot. 0... are PM indices. 
        self.vm_cpu = np.zeros(self.config.vms) 
        self.vm_memory = np.zeros(self.config.vms)
        self.cpu = np.zeros(self.config.pms)
        self.memory = np.zeros(self.config.pms)
        self.vm_remaining_runtime = np.zeros(self.config.vms, dtype=int)
        self.waiting_ratio = np.zeros(self.config.vms)
        self.used_cpu_ratio = np.zeros(self.config.vms)
        self.target_cpu_mean = 0
        self.target_memory_mean = 0
        # Not in observation
        self.timestep = 1
        self.total_requests = 0
        self.served_requests = 0
        self.suspend_action = 0
        self.place_action = 0
        self.dropped_requests = 0
        self.vm_planned_runtime = np.zeros(self.config.vms, dtype=int)
        self.vm_waiting_time = np.zeros(self.config.vms, dtype=int)
        self.vm_suspended = np.zeros(self.config.vms, dtype=int)
        self.vm_arrival_steps = [[] for _ in range(self.config.vms)] # Make sure the inner arrays are not references to the same array
        self.target_mean = []
        self.total_cpu_requested = 0
        self.total_memory_requested = 0

        max_steps = max(self.config.training_steps, self.config.eval_steps)
        if self.config.sequence == 'ffworst':
            self.vm_cpu_sequence = np.tile(np.concatenate((np.repeat(0.15, 6 * self.config.eval_steps // 100), np.repeat(0.34, 8 * self.config.eval_steps // 100), np.repeat(0.51, 6 * self.config.eval_steps // 100))), max_steps // 10).tolist()
            self.vm_memory_sequence = np.tile(np.concatenate((np.repeat(0.15, 6 * self.config.eval_steps // 100), np.repeat(0.34, 8 * self.config.eval_steps // 100), np.repeat(0.51, 6 * self.config.eval_steps // 100))), max_steps // 10).tolist()
        elif self.config.sequence == 'multinomial':
            self.vm_cpu_sequence = self.rng1.choice([0.125,0.25,0.375,0.5,0.675,0.75,0.875], p=[0.148,0.142,0.142,0.142,0.142,0.142,0.142], size=max_steps+1, replace=True) # uniform discrete
            self.vm_memory_sequence = self.rng2.choice([0.125,0.25,0.375,0.5,0.675,0.75,0.875], p=[0.148,0.142,0.142,0.142,0.142,0.142,0.142], size=max_steps+1, replace=True) # uniform discrete
        elif self.config.sequence == 'uniform':
            self.vm_cpu_sequence = np.around(self.rng1.uniform(low=0.1, high=1, size=max_steps*2), decimals=2).tolist() # mean 0.55
            self.vm_memory_sequence = np.around(self.rng2.uniform(low=0.1, high=1, size=max_steps*2), decimals=2).tolist() # mean 0.55
        elif self.config.sequence == 'lowuniform':
            self.vm_cpu_sequence = np.around(self.rng1.uniform(low=0.1, high=0.65, size=max_steps*2), decimals=2).tolist() # mean 0.375
            self.vm_memory_sequence = np.around(self.rng2.uniform(low=0.1, high=0.65, size=max_steps*2), decimals=2).tolist() # mean 0.375
        elif self.config.sequence == 'highuniform':
            self.vm_cpu_sequence = np.around(self.rng1.uniform(low=0.25, high=1, size=max_steps*2), decimals=2).tolist() # mean 0.625
            self.vm_memory_sequence = np.around(self.rng2.uniform(low=0.25, high=1, size=max_steps*2), decimals=2).tolist() # mean 0.625
        
        # If requests drop, it will require a seq length longer than max_steps. This will only work when drop rate < 50%

        return self._get_obs(), self._get_info()

    def render(self, mode='ansi', close=False):
        np.set_printoptions(linewidth=np.inf)
        print(f"Timestep: \t\t{self.timestep}")
        print(f"VM request: \t\t{np.count_nonzero(self.vm_placement == -1)}, dropped: {self.dropped_requests}")
        print(f"VM placement: \t\t{self.vm_placement}")
        print(f"VM suspended: \t\t{self.vm_suspended}")
        print(f"CPU (%): \t\t{np.array(self.cpu*100, dtype=int)} {np.round(np.sum(self.cpu), 3)}")
        print(f"Memory (%): \t\t{np.array(self.memory*100, dtype=int)} {np.round(np.sum(self.memory), 3)}")
        print(f"VM CPU (%): \t\t{np.array(self.vm_cpu*100, dtype=int)} {np.round(np.sum(self.vm_cpu), 3)}")
        print(f"VM Memory (%): \t\t{np.array(self.vm_memory*100, dtype=int)} {np.round(np.sum(self.vm_memory), 3)}")
        print(f"VM waiting time: \t{self.vm_waiting_time}")
        print(f"VM planned runtime: \t{self.vm_planned_runtime}")
        print(f"VM remaining runtime: \t{self.vm_remaining_runtime}")

    def close(self):
        pass

    def convert_obs_to_dict(self, observation: np.ndarray) -> dict:
        return dict(
            vm_placement=observation[:self.config.vms].astype(int),  
            vm_cpu=observation[self.config.vms:self.config.vms*2], 
            vm_memory=observation[self.config.vms*2:self.config.vms*3], 
            cpu=observation[self.config.vms*3:self.config.vms*3 + self.config.pms],
            memory=observation[self.config.vms*3 + self.config.pms:],
        )

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
                self.cpu[pm] -= self.vm_cpu[vm]
                self.memory[pm] -= self.vm_memory[vm]

            self.vm_cpu[vm_to_terminate] = 0
            self.vm_memory[vm_to_terminate] = 0
            self.vm_planned_runtime[vm_to_terminate] = 0
            self.vm_waiting_time[vm_to_terminate] = 0
            self.vm_remaining_runtime[vm_to_terminate] = 0
            self.vm_suspended[vm_to_terminate] = 0

        self.cpu[self.cpu < 1e-7] = 0 # precision problem 
        self.memory[self.memory < 1e-7] = 0 # precision problem
        
    def _accept_vm_requests(self):
        arrivals = self.rng3.poisson(self.config.arrival_rate)
        self.total_requests += arrivals
        placed_arrivals = min(arrivals, self.vm_placement[self.vm_placement ==  EMPTY_SLOT].size)
        to_accept = np.argwhere(self.vm_placement == EMPTY_SLOT).flatten()[:placed_arrivals]
        self.vm_placement[to_accept] = WAIT_STATUS

        vm_cpu_list = self.vm_cpu_sequence[:to_accept.size]
        self.total_cpu_requested += np.sum(vm_cpu_list)
        self.vm_cpu_sequence = self.vm_cpu_sequence[to_accept.size:]
        self.vm_cpu[to_accept] = vm_cpu_list

        vm_memory_list = self.vm_memory_sequence[:to_accept.size]
        self.total_memory_requested += np.sum(vm_memory_list)
        self.vm_memory_sequence = self.vm_memory_sequence[to_accept.size:]
        self.vm_memory[to_accept] = vm_memory_list

        self.vm_planned_runtime[to_accept] =  self.rng4.poisson(self.config.service_length, size=to_accept.size) + 1 
        self.vm_remaining_runtime[to_accept] = self.vm_planned_runtime[to_accept] # New request start counting
        self.dropped_requests += arrivals - placed_arrivals
        for i in to_accept: 
            self.vm_arrival_steps[i].append(self.timestep + 1) # Arrival at next step

    def _get_obs(self):
        return np.hstack([self.vm_placement, self.vm_cpu, self.vm_memory, self.cpu, self.memory]).astype(np.float32)
    
    def _get_info(self):
        return {
            "waiting_ratio": self.waiting_ratio, 
            "used_cpu_ratio": self.used_cpu_ratio,
            "served_requests": self.served_requests,
            'suspend_actions': self.suspend_action,
            'place_actions': self.place_action,
            "dropped_requests": self.dropped_requests, 
            "total_requests": self.total_requests, 
            "timestep": self.timestep,
            "vm_arrival_steps": self.vm_arrival_steps,
            "vm_placement": self.vm_placement.copy(), 
            "cpu": self.cpu.copy(),
            "memory": self.memory.copy(),
            "vm_cpu": self.vm_cpu.copy(),
            "vm_memory": self.vm_memory.copy(),
            "target_cpu_mean": self.target_cpu_mean,
            "target_memory_mean": self.target_memory_mean,
            'total_cpu_requested': self.total_cpu_requested,
            'total_memory_requested': self.total_memory_requested,
        }