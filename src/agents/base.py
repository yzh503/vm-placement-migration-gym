from abc import abstractmethod
from src.record import Record
from dataclasses import dataclass, asdict
import numpy as np
from time import gmtime, strftime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.vm_gym.envs.env import VmEnv

@dataclass
class Config: 
    pass

class Base: 
    def __init__(self, name: str, env: VmEnv, config: Config):
        self.name = name
        self.env = env
        self.config = Config() if config is None else config
        self.record = Record(self.name, asdict(self.env.config), asdict(self.config)) 
        self.writer = None
        self.total_steps = 0

        self.rng = np.random.default_rng(self.env.config.seed)

        print("Agent initialised with config: ", self.config)


    def set_log(self, jobname, logdir):
        if logdir: 
            run_name = f"{strftime('%Y%m%d', gmtime())}-{self.name}-{jobname}"
            self.writer = SummaryWriter(f"{logdir}/{run_name}")

            self.writer.add_text(
                "Environment hyperparameters",
                "|param|value|\n|---|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.env.config).items()])),
            )
            self.writer.add_text(
                "Agent hyperparameters",
                "|param|value|\n|---|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.config).items()])),
            )

    @abstractmethod
    def learn(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def save_model(self, modelpath: str):
        raise NotImplementedError

    @abstractmethod
    def load_model(self, modelpath: str):
        raise NotImplementedError
    
    @abstractmethod
    def eval(self, model: bool = True):
        raise NotImplementedError
    
    def test(self, show: bool = False, output: str = None, debug: bool = False):
        
        self.env.eval()
        self.eval()
        obs, info = self.env.reset(seed=self.env.config.seed)
        done = False

        pbar = tqdm(total=self.env.config.eval_steps, leave=False)
        while not done: 
            if debug: 
                self.env.render()

            self.env.test = 0
            action = self.act(obs)
                
            obs, reward, done, truncated, info = self.env.step(action)
            if debug: 
                print('action: \t\t%s' % (action.flatten()))
                print('validity: \t\t%s' % (info['valid']))
                print('reward: \t\t%.2f' % (reward))
                print('')
            self.record_testing_step(reward, info)

            pbar.update(1)
        pbar.close()

        if debug: 
            self.env.render()
            print('')

        summary = self.record.get_summary()
        if self.writer:
            self.writer.add_text(
                "Test Summary",
                "|param|value|\n|---|-|\n%s" % ("\n".join(['|%s|%.2f|' % (k, v) for k, v in summary.items()])),
            )
            
        if show: 
            print(self.env.config)
            for k, v in summary.items():
                print('%s: %.2f' % (k, v))
            print('cpu: %s' % (info['cpu']))
            print('memory: %s' % (info['memory']))
    
        fig, axs = plt.subplots(2, figsize=(6, 2))
        im = axs[0].imshow(np.transpose(np.array(self.record.cpu)), cmap='pink', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        axs[0].set(yticks=np.arange(0, self.env.config.pms, dtype=int))
        axs[0].set(xlabel="Time step")
        axs[0].set(ylabel="PM #")
        cbar = plt.colorbar(im)
        cbar.set_label("CPU Utilisation")
        axs[1].plot(self.record.used_pm)
        plt.tight_layout()
        if debug:
            plt.savefig(f'data/{self.name}-util-timeline.png')
    
        if output: 
            self.record.save(output)
        
        plt.close()
    
        return self.record

    
    def end_log(self):
        if self.writer:
            self.writer.close()

    def record_testing_step(self, reward: float, info):
        self.record.cpu.append(info["cpu"])
        self.record.memory.append(info["memory"])
        self.record.used_pm.append(len(info["cpu"]) - np.count_nonzero(info["cpu"]))
        self.record.vm_placements.append(info["vm_placement"])
        self.record.waiting_ratio.append(info['waiting_ratio'])
        self.record.actions.append(info["action"])
        self.record.rewards.append(reward)
        self.record.dropped_requests.append(info["dropped_requests"])
        self.record.total_requests.append(info["total_requests"])
        self.record.vm_arrival_steps = info["vm_arrival_steps"]
        self.record.target_cpu_mean.append(info['target_cpu_mean'])
        self.record.target_memory_mean.append(info['target_memory_mean'])
        self.record.served_requests.append(int(info['served_requests']))
        self.record.total_cpu_requested = info['total_cpu_requested']
        self.record.total_memory_requested = info['total_memory_requested']
        self.record.suspended.append(info['suspend_actions'])
        self.record.placed.append(info['place_actions'])
        self.record.rank.append(info['rank'])