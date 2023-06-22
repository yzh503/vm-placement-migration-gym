import numpy as np
import os

def override_dict(template: dict, overrider: dict) -> dict:
    template = template.copy()
    for c in overrider: 
        if c in template and overrider[c] is not None:
            template[c] = overrider[c]
    return template

def get_action_pair(action: int, p_num: int):
    assert isinstance(action, int), action
    assert isinstance(p_num, int)
    a = np.divmod(action, (p_num + 1))
    return int(a[0]), int(a[1] - 1)

def get_action(v, p, p_num):
    return (p_num + 1) * v + (p + 1)

def convert_obs_to_dict(v_num: int, observation: list) -> dict:
    return dict(
        vm_placement=[int(i) for i in observation[:v_num]], 
        vm_resource=observation[v_num:v_num*2], 
        vm_remaining_runtime=[int(i) for i in observation[v_num*2:v_num*3]], 
        cpu=observation[v_num*3:]
    )

def check_dir(output: str):
    dir = '/'.join(output.split('/')[:-1])
    if len(dir) > 0 and not os.path.exists(dir): 
        os.makedirs(dir)

def ensure_parent_dirs_exist(file_path):
    parent_dir = os.path.dirname(file_path)

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)