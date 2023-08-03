from typing import Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from vmenv.envs.config import Config

def kl_divergence(p_mean, p_cov, q_mean, q_cov):
    p_dim = p_mean.shape[0]
    det_p_cov = np.linalg.det(p_cov)
    det_q_cov = np.linalg.det(q_cov)
    q_cov_inv = np.linalg.inv(q_cov)
    trace_term = np.trace(np.dot(q_cov_inv, p_cov))
    diff = p_mean - q_mean
    m1 = np.dot(np.dot(diff.T, q_cov_inv), diff)
    m2 = np.trace(np.dot(q_cov_inv, p_cov))
    return 0.5 * (np.log(det_q_cov/det_p_cov) - p_dim + trace_term + m1 - m2)

class VmEnv(gym.Env):

    metadata = {'render_modes': ['ansi']}

    def __init__(self, config: Config):
        self.config = config
        self.eval_mode = False
        self.observation_space = spaces.Box(low=0, high=self.config.pms + 1, shape=(self.config.vms * 3 + self.config.pms * 2,)) 
        self.action_space = spaces.MultiDiscrete(np.full(self.config.vms , self.config.pms + 2))  # Every VM has (PMs or wait action or null action) actions
        self.WAIT_STATUS = self.config.pms
        self.NULL_STATUS = self.config.pms + 1
        self.reset(self.config.seed)

        print("Environment initialised with config: ", self.config)

    def validate(self, vm: int, current_pm: int, move_to_pm: int) -> bool:
        if current_pm == move_to_pm: # Null action
            return True
        if current_pm == self.WAIT_STATUS: # VM is waiting
            return move_to_pm < self.WAIT_STATUS and self._resource_valid(vm, move_to_pm)
        if current_pm < self.WAIT_STATUS: # VM is running 
            return move_to_pm == self.WAIT_STATUS
        return False

    # invalid is true
    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros([self.config.vms , self.config.pms + 2], dtype=bool)
        for vm, current_pm in enumerate(self.vm_placement):
            for move_to_pm in range(self.config.pms + 2):
                mask[vm, move_to_pm] = self.validate(vm, current_pm, move_to_pm)
        return mask

    def _resource_valid(self, vm, pm):
        return self.cpu[pm] + self.vm_cpu[vm] <= 1 and self.memory[pm] + self.vm_memory[vm] <= 1

    def _free_pm(self, pm, vm):
        self.cpu[pm] -= self.vm_cpu[vm]
        self.memory[pm] -= self.vm_memory[vm]
    
    def _place_vm(self, pm, vm):
        self.cpu[pm] += self.vm_cpu[vm]
        self.memory[pm] += self.vm_memory[vm]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = action.copy()
        actions_valid = np.zeros_like(action)
        for vm, move_to_pm in enumerate(action): 
            current_pm = self.vm_placement[vm]
            action_valid = self.validate(vm, current_pm, move_to_pm)
            actions_valid[vm] = int(action_valid)

            if action_valid: 
                self.vm_placement[vm] = move_to_pm
                if move_to_pm == current_pm:
                    pass # Unchanged
                elif move_to_pm == self.WAIT_STATUS:  # Free up PM
                    self._free_pm(current_pm, vm)
                    self.vm_suspended[vm] = 1
                    self.suspend_action += 1
                elif move_to_pm < self.WAIT_STATUS: # Allocate
                    self._place_vm(move_to_pm, vm)
                    self.vm_suspended[vm] = 0
                    self.place_action += 1
                else: 
                    raise ValueError("Invalid move_to_pm: " + str(move_to_pm))

        obs, reward, terminated, info = self._process_action()

        info = info | {
            "action": action,
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
    def n_actions(self) -> int:
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
        vms_existing = self.vm_placement <= self.WAIT_STATUS
        vms_arrived = np.count_nonzero(vms_existing)
        vms_waiting = np.count_nonzero(self.vm_placement == self.WAIT_STATUS)
        self.waiting_ratio = vms_waiting / vms_arrived if vms_arrived > 0 else 0
        self.target_cpu_mean = np.sum(self.vm_cpu[vms_existing]) / self.config.pms
        if self.config.cap_target_util and self.target_cpu_mean > 1: 
            self.target_cpu_mean = 1.0
        self.target_memory_mean = np.sum(self.vm_memory[vms_existing]) / self.config.pms
        if self.config.cap_target_util and self.target_memory_mean > 1: 
            self.target_memory_mean = 1.0
        
        if not vms_existing.any(): # No VMs running
            reward = 0.0
        elif self.config.reward_function == "kl": # KL divergence between from approximator to true
            current_cpu = np.mean(self.cpu)
            current_memory = np.mean(self.memory)
            current_mean = np.array([current_cpu, current_memory])
            cpu_vars = np.var(self.cpu)
            memory_vars = np.var(self.memory)
            if cpu_vars == 0:
                cpu_vars = 1e-6
            if memory_vars == 0:
                memory_vars = 1e-6
            current_cov = np.array([[cpu_vars, 0], [0, memory_vars]])

            
            target_mean = np.array([self.target_cpu_mean, self.target_memory_mean])
            target_cpu_var = np.var(self.vm_cpu[vms_existing]) 
            target_memory_var = np.var(self.vm_memory[vms_existing])
            if target_cpu_var == 0:
                target_cpu_var = 1e-6
            if target_memory_var == 0:
                target_memory_var = 1e-6
            target_cov = np.array([[target_cpu_var, 0], [0, target_memory_var]])

            if self.target_cpu_mean == 0 or self.target_memory_mean == 0:
                reward = 0.0
            else:
                reward = - kl_divergence(target_mean, target_cov, current_mean, current_cov)
        elif self.config.reward_function == "ut": 
            reward = self.config.beta * np.sum(self.cpu) + (1 - self.config.beta) * np.sum(self.memory)
        elif self.config.reward_function == "wr":
            reward = - self.waiting_ratio 
        else: 
            assert False, f'Function does not exist: {self.config.reward_function}'

        obs = self._get_obs()

        if self.eval_mode: 
            terminated = self.timestep >= self.config.eval_steps
        else:
            terminated = self.timestep >= self.config.training_steps

        if self.eval_mode:
            info = self._get_info()
        else: 
            info = {} # save time

        return obs, reward, terminated, info

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = self.config.seed
        self.rng1 = np.random.default_rng(seed)
        self.rng2 = np.random.default_rng(seed+1)
        self.rng3 = np.random.default_rng(seed+2)
        self.rng4 = np.random.default_rng(seed+3)

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is None:
            super().reset()
        else: 
            super().reset(seed=int(seed))
            self.seed(seed)
        # Observable
        self.vm_placement = np.full(self.config.vms, self.NULL_STATUS) #  0...P are PM indices. P is a VM request. P + 1 is an empty slot.
        self.vm_cpu = np.zeros(self.config.vms) 
        self.vm_memory = np.zeros(self.config.vms)
        self.cpu = np.zeros(self.config.pms)
        self.memory = np.zeros(self.config.pms)
        self.vm_remaining_runtime = np.zeros(self.config.vms, dtype=int)
        self.waiting_ratio = 0
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
        self.vm_suspended = np.zeros(self.config.vms, dtype=int)
        self.vm_arrival_steps = [[] for _ in range(self.config.vms)] # Make sure the inner arrays are not references to the same array
        self.target_mean = []
        self.total_cpu_requested = 0
        self.total_memory_requested = 0

        max_steps = max(self.config.training_steps, self.config.eval_steps)
        if self.config.sequence == 'uniform':
            self.vm_cpu_sequence = np.around(self.rng1.uniform(low=0.1, high=1, size=max_steps*2), decimals=2).tolist() # mean 0.55
            self.vm_memory_sequence = np.around(self.rng2.uniform(low=0.1, high=1, size=max_steps*2), decimals=2).tolist() # mean 0.55
        elif self.config.sequence == 'lowuniform':
            self.vm_cpu_sequence = np.around(self.rng1.uniform(low=0.1, high=0.65, size=max_steps*2), decimals=2).tolist() # mean 0.375
            self.vm_memory_sequence = np.around(self.rng2.uniform(low=0.1, high=0.65, size=max_steps*2), decimals=2).tolist() # mean 0.375
        elif self.config.sequence == 'highuniform':
            self.vm_cpu_sequence = np.around(self.rng1.uniform(low=0.25, high=1, size=max_steps*2), decimals=2).tolist() # mean 0.625
            self.vm_memory_sequence = np.around(self.rng2.uniform(low=0.25, high=1, size=max_steps*2), decimals=2).tolist() # mean 0.625

        # compute covariance matrix that includes vm_cpu_sequence and vm_memory_sequence
        matrix = np.stack((self.vm_cpu_sequence[:2000], self.vm_memory_sequence[:2000]), axis=-1)
        self.covariance_matrix = np.cov(matrix.T)

        # If requests drop, it will require a seq length longer than max_steps. This will only work when drop rate < 50%
        return self._get_obs(), self._get_info()

    def render(self, mode: str = 'ansi', close: bool = False):
        np.set_printoptions(linewidth=np.inf)
        print(f"Timestep: \t\t{self.timestep}")
        print(f"VM request: \t\t{np.count_nonzero(self.vm_placement == -1)}, dropped: {self.dropped_requests}")
        print(f"VM placement: \t\t{self.vm_placement}")
        print(f"VM suspended: \t\t{self.vm_suspended}")
        print(f"CPU (%): \t\t{np.array(self.cpu*100, dtype=int)} {np.round(np.sum(self.cpu), 3)}")
        print(f"Memory (%): \t\t{np.array(self.memory*100, dtype=int)} {np.round(np.sum(self.memory), 3)}")
        print(f"VM CPU (%): \t\t{np.array(self.vm_cpu*100, dtype=int)} {np.round(np.sum(self.vm_cpu), 3)}")
        print(f"VM Memory (%): \t\t{np.array(self.vm_memory*100, dtype=int)} {np.round(np.sum(self.vm_memory), 3)}")
        print(f"VM planned runtime: \t{self.vm_planned_runtime}")
        print(f"VM remaining runtime: \t{self.vm_remaining_runtime}")

    def close(self):
        pass

    def _run_vms(self):
        running = self.vm_placement < self.WAIT_STATUS
        continue_to_run = np.logical_and(self.vm_remaining_runtime > 0, running)
        self.vm_remaining_runtime[continue_to_run] -= 1
        to_terminate = np.logical_and(self.vm_remaining_runtime == 0, running)
        vm_to_terminate = np.flatnonzero(to_terminate)

        if vm_to_terminate.size > 0:
            pms_to_free_up = self.vm_placement[vm_to_terminate]
            self.vm_placement[vm_to_terminate] = self.NULL_STATUS

            for vm, pm in zip(vm_to_terminate, pms_to_free_up): 
                self.cpu[pm] -= self.vm_cpu[vm]
                self.memory[pm] -= self.vm_memory[vm]

            self.vm_cpu[vm_to_terminate] = 0
            self.vm_memory[vm_to_terminate] = 0
            self.vm_planned_runtime[vm_to_terminate] = 0
            self.vm_remaining_runtime[vm_to_terminate] = 0
            self.vm_suspended[vm_to_terminate] = 0

            self.served_requests += vm_to_terminate.size

        self.cpu[self.cpu < 1e-7] = 0 # precision problem 
        self.memory[self.memory < 1e-7] = 0 # precision problem

        
    def _accept_vm_requests(self):
        arrivals = self.rng3.poisson(self.config.arrival_rate)
        self.total_requests += arrivals
        null_vm_mask = self.vm_placement == self.NULL_STATUS
        placed_arrivals = min(arrivals, null_vm_mask.sum())
        to_accept = np.flatnonzero(null_vm_mask)[:placed_arrivals]
        self.vm_placement[to_accept] = self.config.pms

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
        return np.concatenate([self.vm_placement, self.vm_cpu, self.vm_memory, self.cpu, self.memory], dtype=np.float32)
    
    def _get_info(self):
        return {
            "waiting_ratio": self.waiting_ratio, 
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
            'rank': self._get_rank()
        }

    def _get_rank(self):
        M = np.zeros(shape=(self.config.vms, self.config.pms))
        for i, pm in enumerate(self.vm_placement):
            if pm < self.WAIT_STATUS:
                M[i, pm] = 1 
        return np.linalg.matrix_rank(M)