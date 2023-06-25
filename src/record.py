import numpy as np
import json
import src.utils as utils

class Record:

    def __init__(self, agent, env_config, training_config):
        self.agent = agent
        self.env_config = env_config if isinstance(env_config, dict) else vars(env_config)
        self.training_config = training_config if isinstance(training_config, dict) else vars(training_config)

        # Below records the info for each step in testing.
        self.cpu = list[float]()
        self.memory = list[float]()
        self.used_pm = list[int]()
        self.vm_placements = list[float]()
        self.waiting_ratio = list[float]()
        self.actions = list[int]()
        self.rewards = list[float]()
        self.dropped_requests = list[int]()
        self.total_requests = list[int]()
        self.vm_arrival_steps = list[int]()
        self.target_cpu_mean = list[int]()
        self.target_memory_mean = list[int]()
        self.served_requests = list[int]()
        self.total_cpu_requested = list[int]()
        self.total_memory_requested = list[int]()
        self.suspended = list[int]() # suspend actions
        self.placed = list[int]() # place actions

    @property
    def unique_vms_placement(self):
        # As a VM slot could have contained multiple VMs, we need to sepatate them. 
        unique_vms_placement = []
        vm_placements = np.transpose(np.array(self.vm_placements)) # row is vm, col is timestep
        for vm, vm_status in enumerate(vm_placements): 
            if len(self.vm_arrival_steps[vm]) == 0: 
                continue
    
            start = 0
            for end in self.vm_arrival_steps[vm][1:]:
                end -= 2 # Note that vm_placements starts from timestep 2
                spline = vm_status[start:end]
                unique_vms_placement.append(spline[spline > -2])
                start = end
            spline = vm_status[start:]
            assert spline[spline > -2].size != 0, spline[spline > -2]
            unique_vms_placement.append(spline[spline > -2])
        return unique_vms_placement 

    @property
    def total_running(self):
        total = 0
        for vm in self.unique_vms_placement:
            vm = np.array(vm)
            total += np.count_nonzero(vm > -1)
        return total

    @property
    def pending_rates(self):
        flat_pending_rates = []
        for status in self.unique_vms_placement:
            status = np.array(status)
            running_status = np.where(status > -1)[0]
            allocated_at = running_status[0] if running_status.size > 0 else None
            if allocated_at: 
                rate = np.around((allocated_at + 1.0) / len(status) , 3)
            else: 
                rate = 1.0
            flat_pending_rates.append(rate)
        return flat_pending_rates
    
    @property
    def slowdown_rates(self):
        flat_slowdown_rates = []
        for status in self.unique_vms_placement:
            status = np.array(status)
            running_status = np.where(status > -1)[0]
            allocated_at = running_status[0] if running_status.size > 0 else None
            if allocated_at:
                slowdown_steps = np.count_nonzero(status[allocated_at:] == -1)
                vm_life = len(status) - allocated_at - 1
                rate = 0 if vm_life == 0 else np.around(slowdown_steps / vm_life, 3)
                flat_slowdown_rates.append(rate)

        if len(flat_slowdown_rates) == 0: 
            flat_slowdown_rates = [0]
        return flat_slowdown_rates

    @property 
    def vm_lifetime(self):
        life = []
        for status in self.unique_vms_placement:
            status = np.array(status)
            running_status = np.where(status > -1)[0]
            allocated_at = running_status[0] if running_status.size > 0 else None
            if allocated_at:
                life.append(len(status) - allocated_at - 1)
            else: 
                life.append(0)
        return life

    @property
    def drop_rate(self):
        dropped = np.array(self.dropped_requests)
        total_requests = np.array(self.total_requests)
        return np.divide(dropped, total_requests, out=np.zeros(dropped.shape, dtype=float), where=total_requests!=0)

    @property
    def total_rewards(self):
        rewards = np.array(self.rewards)
        rewards[rewards < -1e7] = np.mean(rewards[rewards > -1e7])
        return np.round(np.sum(rewards), 3)
    
    def get_summary(self):
        return {
            'total rewards': self.total_rewards,
            'total served VMs': self.served_requests[-1],
            'total requests': self.total_requests[-1],
            'total cpu requested': np.round(self.total_cpu_requested, 3),
            'total memory requested': np.round(self.total_memory_requested, 3),
            'total suspend actions': self.suspended[-1],
            'total place actions': self.placed[-1],
            'average VM life': np.round(np.mean(self.vm_lifetime),3),
            'average pending': np.round(np.mean(self.pending_rates), 3), 
            'median pending': np.round(np.median(self.pending_rates), 3), 
            'max pending': np.round(np.max(self.pending_rates), 3) if len(self.pending_rates) > 0 else 0,
            'average slowdown': np.round(np.mean(self.slowdown_rates), 3), 
            'median slowdown': np.round(np.median(self.slowdown_rates), 3), 
            'max slowdown': np.round(np.max(self.slowdown_rates), 3),
            'drop rate': np.round(np.mean(self.drop_rate), 3),
            'cpu mean (second half)': np.round(np.mean(self.cpu[len(self.cpu)//2:]), 3),
            'cpu mean target (second half)': np.round(np.mean(self.target_cpu_mean[len(self.target_cpu_mean)//2:]), 3),
            'cpu std (second half)': np.round(np.std(self.cpu[len(self.cpu)//2:]), 3),
            'memory mean (second half)': np.round(np.mean(self.memory[len(self.memory)//2:]), 3),
            'memory mean target (second half)': np.round(np.mean(self.target_memory_mean[len(self.target_memory_mean)//2:]), 3),
            'memory std (second half)': np.round(np.std(self.memory[len(self.memory)//2:]), 3),
        }
    
    def save(self, path: str):
        self.summary = self.get_summary()
        utils.ensure_parent_dirs_exist(path)
        f = open(f'{path}', 'w')
        f.write(json.dumps(vars(self), cls=NpEncoder))
        f.close()

    @classmethod 
    def import_record(cls, agent: str, jsondict: dict):
        record = cls(agent, jsondict['env_config'], jsondict['training_config']) 

        record.cpu = jsondict['cpu']
        record.memory = jsondict['memory']
        record.vm_cpu = jsondict['vm_cpu']
        record.vm_memory = jsondict['vm_memory']
        if 'used_pm' in jsondict:
            record.used_pm = jsondict['used_pm']
        record.vm_placements = jsondict['vm_placements']
        record.waiting_ratio = jsondict['waiting_ratio']
        record.actions = jsondict['actions']
        record.rewards = jsondict['rewards']
        record.total_requests = jsondict['total_requests']
        record.dropped_requests = jsondict['dropped_requests']
        record.vm_arrival_steps = jsondict['vm_arrival_steps']
        record.target_cpu_mean = jsondict['target_cpu_mean']
        record.target_memory_mean = jsondict['target_memory_mean']
        record.served_requests = jsondict['served_requests']
        record.total_cpu_requested = jsondict['total_cpu_requested']
        record.total_memory_requested = jsondict['total_memory_requested']
        record.suspended = jsondict['suspended']
        if 'placed' in jsondict:
            record.placed = jsondict['placed']
        return record

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)