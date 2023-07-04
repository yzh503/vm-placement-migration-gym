import numpy as np
import os

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