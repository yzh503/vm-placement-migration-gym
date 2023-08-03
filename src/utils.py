import numpy as np
import os, torch
from typing import Union

from vmenv.envs.config import Config

def override_dict(template: dict, overrider: dict) -> dict:
    template = template.copy()
    for c in overrider: 
        if c in template and overrider[c] is not None:
            template[c] = overrider[c]
    return template

def get_action_pair(action: int, pms: int):
    assert isinstance(action, int), action
    assert isinstance(pms, int)
    a = np.divmod(action, (pms + 1))
    return int(a[0]), int(a[1] - 1)

def get_action(v, p, pms):
    return (pms + 1) * v + (p + 1)

def check_dir(output: str):
    dir = '/'.join(output.split('/')[:-1])
    if len(dir) > 0 and not os.path.exists(dir): 
        os.makedirs(dir)

def ensure_parent_dirs_exist(file_path):
    parent_dir = os.path.dirname(file_path)
    print(parent_dir)
    if not os.path.exists(parent_dir):
        try: 
            os.makedirs(parent_dir)
        except Exception as e: 
            print(e)

def convert_obs_to_dict(config: Config, observation: Union[torch.Tensor, np.ndarray]) -> dict:
    if isinstance(observation, torch.Tensor):
        vm_placement = observation[:config.vms].to(int)
    else: 
        vm_placement = observation[:config.vms].astype(int)
    return dict(
        vm_placement=vm_placement,  
        vm_cpu=observation[config.vms:config.vms*2], 
        vm_memory=observation[config.vms*2:config.vms*3], 
        cpu=observation[config.vms*3:config.vms*3 + config.pms],
        memory=observation[config.vms*3 + config.pms:],
    )