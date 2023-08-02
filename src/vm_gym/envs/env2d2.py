import gymnasium as gym
from gymnasium import spaces
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from src.vm_gym.envs.config import Config

class VmEnv(gym.Env):

    metadata = {'render.modes': ['ansi']}

    def __init__(self, config: Config):
        super(VmEnv, self).__init__()

        self.config = config
        self.config.reward_function = 6
        self.eval_mode = False

        self.rng1 = np.random.default_rng(self.config.seed)
        self.rng2 = np.random.default_rng(self.config.seed)
        self.rng3 = np.random.default_rng(self.config.seed)
        self.rng4 = np.random.default_rng(self.config.seed)

        # [vm_placement, vm_cpu, vm_memory, cpu, memory]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.config.vms * 3 + self.config.pms * 2,), dtype=np.float32) 
        self.action_space = spaces.MultiDiscrete(np.full(self.config.vms , self.config.pms + 1))  # Every VM has (PMs + wait status) actions
        self.reset()
    
    @property
    def n_actions(self):
        if isinstance(self.action_space, spaces.Discrete):
            return self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            return self.action_space.nvec.sum()

    def eval(self, mode=True):
        self.eval_mode = mode
   
    def _placement_invalid(self, pm, vm):
        return self.cpu[pm] + self.vm_cpu[vm] > 1 or self.memory[pm] + self.vm_memory[vm] > 1

    def _free_pm(self, pm, vm):
        self.cpu[pm] -= self.vm_cpu[vm]
        self.memory[pm] -= self.vm_memory[vm]
    
    def _place_vm(self, pm, vm):
        self.cpu[pm] += self.vm_cpu[vm]
        self.memory[pm] += self.vm_memory[vm]

    def step(self, action):
        
        action = np.copy(action) - 1
        actions_valid = np.zeros_like(action)

        if self.eval_mode: 
            self.last_vm_placement = np.copy(self.vm_placement)

        for vm, move_to_pm in enumerate(action): 
            current_pm = self.vm_placement[vm]
            action_valid = True
            action_valid = action_valid and not (move_to_pm == -2)                                          # No direct termination
            action_valid = action_valid and not (current_pm == -2)                       # VM has to be waiting or running
            action_valid = action_valid and not (current_pm > -1 and move_to_pm > -1)  # No direct swap
            action_valid = action_valid and not (self._placement_invalid(move_to_pm, vm))        # PM has to be available

            actions_valid[vm] = int(action_valid)

            if action_valid: 
                self.vm_placement[vm] = move_to_pm
                if current_pm == move_to_pm: 
                    pass
                elif move_to_pm == -1:  # Free up PM
                    self._free_pm(current_pm, vm)
                    self.vm_suspended[vm] = 1
                    self.suspend_action += 1
                elif move_to_pm >= -1: # Allocate
                    self._place_vm(move_to_pm, vm)
                    self.vm_suspended[vm] = 0
                    self.place_action += 1
                else: 
                    pass # do not change PM utilisation 

        obs, reward, done, _, info = self._process_action()

        info = info | {
            "action": action,
            "valid": actions_valid
        }

        if self.eval_mode:
            self.last_validity = actions_valid
            self.last_reward = np.round(reward, 3)
            self.last_action = action

        self.timestep += 1
        return obs, reward, done, False, info  
  
    def _process_action(self):
        # Action is predicted against the observation, so update arrival and terminatino after the action.
        self._run_vms()
        self._accept_vm_requests() 
        
        vms_arrived = np.count_nonzero(self.vm_placement >= -1)
        waiting_ratio = np.count_nonzero(self.vm_placement == -1) / vms_arrived if vms_arrived > 0 else 0
        used_pm_ratio = np.count_nonzero(np.logical_and(self.cpu > 0, self.memory > 0)) / self.config.pms
        target_cpu_mean = np.sum(self.vm_cpu[self.vm_placement >= -1]) / self.config.pms

        reward = - self.waiting_ratio 

        obs = self._observation()

        if self.eval_mode: 
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
            "cpu": self.cpu.copy(),
            "memory": self.memory.copy(),
            "target_cpu_mean": target_cpu_mean,
            "target_memory_mean": target_cpu_mean,
            'total_cpu_requested': self.total_resource_requested,
            'total_memory_requested': self.total_resource_requested,
            'rank': 0
        }

        return obs, reward, done, False, info

    def seed(self, seed = None):
        if seed is None:
            seed = self.config.seed
        self.rng1 = np.random.default_rng(seed)
        self.rng2 = np.random.default_rng(seed+1)
        self.rng3 = np.random.default_rng(seed+2)
        self.rng4 = np.random.default_rng(seed+3)

    def reset(self, seed=None):
        if seed is None:
            super().reset()
        else: 
            super().reset(seed=int(seed))
            self.seed(seed)
        # Observable
        self.vm_placement = np.full(self.config.vms, -2) # -1 is a VM request. -2 is an empty slot. 0... are PM indices. 
        self.vm_cpu = np.zeros(self.config.vms) 
        self.vm_memory = np.zeros(self.config.vms)
        self.cpu = np.zeros(self.config.pms)
        self.memory = np.zeros(self.config.pms)
        self.vm_remaining_runtime = np.zeros(self.config.vms, dtype=int)
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
        self.total_resource_requested = 0

        max_steps = max(self.config.training_steps, self.config.eval_steps)
        if self.config.sequence == 'ffworst':
            self.cpu_sequence = np.tile(np.concatenate((np.repeat(0.15, 6 * self.config.eval_max_steps // 100), np.repeat(0.34, 8 * self.config.eval_max_steps // 100), np.repeat(0.51, 6 * self.config.eval_max_steps // 100))), max_steps // 10).tolist()
        elif self.config.sequence == 'multinomial':
            self.cpu_sequence = self.rng4.choice([0.125,0.25,0.375,0.5,0.675,0.75,0.875], p=[0.148,0.142,0.142,0.142,0.142,0.142,0.142], size=max_steps+1, replace=True) # uniform discrete
        elif self.config.sequence == 'uniform':
            self.cpu_sequence = np.around(self.rng1.uniform(low=0.1, high=1, size=max_steps*2), decimals=2).tolist() # mean 0.55
            self.memory_sequence = np.around(self.rng2.uniform(low=0.1, high=1, size=max_steps*2), decimals=2).tolist() # mean 0.55
        elif self.config.sequence == 'lowuniform':
            self.cpu_sequence = np.around(self.rng1.uniform(low=0.1, high=0.65, size=max_steps*2), decimals=2).tolist() # mean 0.375
        elif self.config.sequence == 'highuniform':
            self.cpu_sequence = np.around(self.rng1.uniform(low=0.25, high=1, size=max_steps*2), decimals=2).tolist() # mean 0.625
        elif self.config.sequence == 'sqrtuniform':
            self.cpu_sequence = np.around(np.sqrt(self.rng1.uniform(low=0.1, high=1, size=max_steps*2)), decimals=2).tolist()
        
        # If requests drop, it will require a seq length longer than max_steps. This will only work when drop rate < 50%

        return self._observation(), {}

    def render(self, mode='ansi', close=False):
        np.set_printoptions(linewidth=np.inf)
        print(f"Timestep: \t\t{self.timestep}")
        print(f"VM request: \t\t{np.count_nonzero(self.vm_placement == -1)}, dropped: {self.dropped_requests}")
        print(f"VM placement: \t\t{self.vm_placement}")
        print(f"VM suspended: \t\t{self.vm_suspended}")
        print(f"VM resources (%): \t{np.array(self.vm_cpu*100, dtype=int)} {np.round(np.sum(self.vm_cpu), 3)}")
        print(f"PM utilisation (%): \t{np.array(self.cpu*100, dtype=int)} {np.round(np.sum(self.cpu), 3)}")
        print(f"VM waiting time: \t{self.vm_waiting_time}")
        print(f"VM planned runtime: \t{self.vm_planned_runtime}")
        print(f"VM remaining runtime: \t{self.vm_remaining_runtime}")

    def close(self):
        pass

    def _run_vms(self):
        vm_running = np.argwhere(np.logical_and(self.vm_remaining_runtime > 0, self.vm_placement >= -1))
        if vm_running.size > 0:
            self.vm_remaining_runtime[vm_running] -= 1  

        vm_waiting = np.argwhere(self.vm_placement == -1)
        self.vm_waiting_time[vm_waiting] += 1
            
        vm_to_terminate = np.argwhere(np.logical_and(self.vm_remaining_runtime == 0, self.vm_placement >= -1)).flatten()

        if vm_to_terminate.size > 0:
            pms_to_free_up = self.vm_placement[vm_to_terminate]
            self.vm_placement[vm_to_terminate] = -2

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

            self.served_requests += vm_to_terminate.size

        self.cpu[self.cpu < 1e-7] = 0 # sometimes it will have a very small number
        self.memory[self.memory < 1e-7] = 0
        
        #assert np.sum(self.vm_cpu) - np.sum(self.vm_cpu[self.vm_placement > EMPTY_SLOT]) < 1e-5
        #assert np.sum(self.vm_cpu[self.vm_placement > -1]) - np.sum(self.cpu) < 1e-5, f'{np.sum(self.vm_cpu[self.vm_placement > -1])} != {np.sum(self.cpu)}'
        
    def _accept_vm_requests(self):
        arrivals = self.rng3.poisson(self.config.arrival_rate)
        self.total_requests += arrivals
        placed_arrivals = min(arrivals, self.vm_placement[self.vm_placement ==  -2].size)
        to_accept = np.argwhere(self.vm_placement == -2).flatten()[:placed_arrivals]
        self.vm_placement[to_accept] = -1
        cpus = self.cpu_sequence[:to_accept.size]
        memories = self.memory_sequence[:to_accept.size]
        self.total_resource_requested += np.sum(cpus)
        self.cpu_sequence = self.cpu_sequence[to_accept.size:]
        self.memory_sequence = self.memory_sequence[to_accept.size:]
        self.vm_cpu[to_accept] = cpus
        self.vm_memory[to_accept] = memories

        # servicerates = truncnorm.rvs(a=1, b=1e10, loc=1, scale=self.config.service_rate, size=to_accept.size, random_state=None)
        self.vm_planned_runtime[to_accept] =  self.rng2.poisson(self.config.service_length, size=to_accept.size) + 1 
        self.vm_remaining_runtime[to_accept] = self.vm_planned_runtime[to_accept] # New request start counting
        self.dropped_requests += arrivals - placed_arrivals
        for i in to_accept: 
            self.vm_arrival_steps[i].append(self.timestep + 1) # Arrival at next step

    def _observation(self):
        return np.concatenate((self.vm_placement, self.vm_cpu, self.vm_memory, self.cpu, self.memory))