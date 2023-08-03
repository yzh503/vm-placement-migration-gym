from dataclasses import dataclass

@dataclass
class Config(object):
    arrival_rate: float = 0.182 # 100% system load: pms / distribution expectation / service length 
    service_length: float = 100
    pms: int = 10
    vms: int = 30   
    var: float = 0.01 # std deviation of normal in KL divergence 
    training_steps: int = 500
    eval_steps: int = 100000
    seed: int = 0
    reward_function: str = "wr"
    sequence: str = "uniform"
    cap_target_util: bool = True
    beta: int = 0.5
