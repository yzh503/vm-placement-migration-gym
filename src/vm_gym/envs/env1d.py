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

        # [vm_placement, vm_remaining_runtime, vm_resource, pm_utilisation]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.config.vms * 3 + self.config.pms,1), dtype=np.float32) 
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
        return self.pm_utilisation[pm] + self.vm_resource[vm] > 1

    def _free_pm(self, pm, vm):
        self.pm_utilisation[pm] -= self.vm_resource[vm]
    
    def _place_vm(self, pm, vm):
        self.pm_utilisation[pm] += self.vm_resource[vm]

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
                elif move_to_pm > -2: # Allocate
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
        
        vms_arrived = np.count_nonzero(self.vm_placement > -2)
        waiting_ratio = np.count_nonzero(self.vm_placement == -1) / vms_arrived if vms_arrived > 0 else 0
        used_pm_ratio = np.count_nonzero(self.pm_utilisation > 0) / self.config.pms
        target_util_mean = np.sum(self.vm_resource[self.vm_placement > -2 + 1]) / self.config.pms
        sd = np.sqrt(self.config.var)

        if self.config.cap_target_util and target_util_mean > 1: 
            target_util_mean = 1.0

        if self.config.reward_function == 1: 
            reward = - (waiting_ratio + 1)/2 * self.config.alpha - (used_pm_ratio + 1)/2 * self.config.beta  # FF-321 PPO-333
        elif self.config.reward_function == 2:
            reward = - ((waiting_ratio + 1)/2) ** self.config.alpha * ((used_pm_ratio + 1)/2) ** self.config.beta
        elif self.config.reward_function == 3: 
            std = np.std(self.pm_utilisation) 
            current = Normal(np.mean(self.pm_utilisation), std if std > 0 else sd)
            target = Normal(0.8, sd)
            reward = - kl_divergence(current, target).item()
        elif self.config.reward_function == 4: 
            std = np.std(self.pm_utilisation) 
            current = Normal(np.mean(self.pm_utilisation), std if std > 0 else sd)
            target = Normal(target_util_mean, sd)
            if target_util_mean == 0:
                reward = 0.0
            else:
                reward = - 0.5 * kl_divergence(target,current).item() - 0.5 * kl_divergence(current, target).item()     # KL divergence between target utilisation and current utilisation
        elif self.config.reward_function == 5: # from true to approx
            std = np.std(self.pm_utilisation)
            current = Normal(np.mean(self.pm_utilisation), std if std > 0 else sd)
            target = Normal(target_util_mean, sd)
            if target_util_mean == 0:
                reward = 0.0
            else:
                reward = - kl_divergence(current, target).item()     # KL divergence between target utilisation and current utilisation
        elif self.config.reward_function == 6: # from approximator to true
            std = np.std(self.pm_utilisation) 
            current = Normal(np.mean(self.pm_utilisation), std if std > 0 else sd)
            target = Normal(target_util_mean, sd)
            if target_util_mean == 0:
                reward = 0.0
            else:
                reward = - kl_divergence(target,current).item()      # KL divergence between target utilisation and current utilisation
        elif self.config.reward_function == 7: # Only count running PMs, so that it performs better on lower load
            util = self.pm_utilisation[self.pm_utilisation > 0]
            if util.size > 0: 
                reward = np.mean(util)
            else:
                reward = 0.0
        else: 
            assert 1 == 2, 'No reward function is selected'

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
            "cpu": self.pm_utilisation.copy(),
            "memory": self.pm_utilisation.copy(),
            "target_cpu_mean": target_util_mean,
            "target_memory_mean": target_util_mean,
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
        self.vm_resource = np.zeros(self.config.vms) 
        self.pm_utilisation = np.zeros(self.config.pms)
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
            self.vm_sequence = np.tile(np.concatenate((np.repeat(0.15, 6 * self.config.eval_max_steps // 100), np.repeat(0.34, 8 * self.config.eval_max_steps // 100), np.repeat(0.51, 6 * self.config.eval_max_steps // 100))), max_steps // 10).tolist()
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

        return self._observation(), {}

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
        vm_running = np.argwhere(np.logical_and(self.vm_remaining_runtime > 0, self.vm_placement > -2))
        if vm_running.size > 0:
            self.vm_remaining_runtime[vm_running] -= 1  

        vm_waiting = np.argwhere(self.vm_placement == -1)
        self.vm_waiting_time[vm_waiting] += 1
            
        vm_to_terminate = np.argwhere(np.logical_and(self.vm_remaining_runtime == 0, self.vm_placement > -2)).flatten()

        if vm_to_terminate.size > 0:
            pms_to_free_up = self.vm_placement[vm_to_terminate]
            self.vm_placement[vm_to_terminate] = -2

            # Multiple VMs could be on the same PM, so use a loop to free up iteratively
            for vm, pm in zip(vm_to_terminate, pms_to_free_up): 
                self.pm_utilisation[pm] -= self.vm_resource[vm]

            self.vm_resource[vm_to_terminate] = 0
            self.vm_planned_runtime[vm_to_terminate] = 0
            self.vm_waiting_time[vm_to_terminate] = 0
            self.vm_remaining_runtime[vm_to_terminate] = 0
            self.vm_suspended[vm_to_terminate] = 0

            self.served_requests += vm_to_terminate.size

        self.pm_utilisation[self.pm_utilisation < 1e-7] = 0 # sometimes it will have a very small number
        
        #assert np.sum(self.vm_resource) - np.sum(self.vm_resource[self.vm_placement > EMPTY_SLOT]) < 1e-5
        #assert np.sum(self.vm_resource[self.vm_placement > -1]) - np.sum(self.pm_utilisation) < 1e-5, f'{np.sum(self.vm_resource[self.vm_placement > -1])} != {np.sum(self.pm_utilisation)}'
        
    def _accept_vm_requests(self):
        arrivals = self.rng3.poisson(self.config.arrival_rate)
        self.total_requests += arrivals
        placed_arrivals = min(arrivals, self.vm_placement[self.vm_placement ==  -2].size)
        to_accept = np.argwhere(self.vm_placement == -2).flatten()[:placed_arrivals]
        self.vm_placement[to_accept] = -1
        n_vms = self.vm_sequence[:to_accept.size]
        self.total_resource_requested += np.sum(n_vms)
        self.vm_sequence = self.vm_sequence[to_accept.size:]
        self.vm_resource[to_accept] = n_vms

        # servicerates = truncnorm.rvs(a=1, b=1e10, loc=1, scale=self.config.service_rate, size=to_accept.size, random_state=None)
        self.vm_planned_runtime[to_accept] =  self.rng2.poisson(self.config.service_length, size=to_accept.size) + 1 
        self.vm_remaining_runtime[to_accept] = self.vm_planned_runtime[to_accept] # New request start counting
        self.dropped_requests += arrivals - placed_arrivals
        for i in to_accept: 
            self.vm_arrival_steps[i].append(self.timestep + 1) # Arrival at next step

    def _observation(self):
        return np.concatenate((self.vm_placement, self.vm_resource, self.vm_remaining_runtime, self.pm_utilisation))