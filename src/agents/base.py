from abc import abstractmethod
from src.record import Record
import numpy as np
import torch
from typing import Union
from time import gmtime, strftime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm 

class Base: 
    def __init__(self, name, env, config):
        self.name = name
        self.env = env
        self.config = config
        self.record = Record(self.name, self.env.config, self.config) 
        self.writer = None
        self.total_steps = 0

        self.rng = np.random.default_rng(self.env.config.seed)

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
    def act(self, observation: Union[np.ndarray, torch.Tensor, list]) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def save_model(self, modelpath):
        raise NotImplementedError

    @abstractmethod
    def load_model(self, modelpath):
        raise NotImplementedError
    
    def test(self, show: bool = False, output: str = None, debug: bool = False):
        
        obs = self.env.reset()
        done = False

        pbar = tqdm(total=self.env.config.eval_steps, leave=False)
        while not done: 
            if debug: 
                self.env.render()

            self.env.test = 0
            action = self.act(obs)
            if debug: 
                print('action: \t\t%s' % ((action - 1).flatten().tolist()))
            obs, reward, done, info = self.env.step(action, eval_mode=True)
            if debug: 
                print('validity: \t\t%s' % (info['valid']))
                print('reward: \t\t%.2f' % (reward))
                print('')
            self.record_testing_step(self.env.timestep, obs, reward, info)

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
            print('pm utilisation: %s' % (info['pm_utilisation']))
    
        fig, axs = plt.subplots(2, figsize=(6, 2))
        im = axs[0].imshow(np.transpose(np.array(self.record.pm_utilisation)), cmap='pink', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        axs[0].set(yticks=np.arange(0, self.env.config.p_num, dtype=int))
        axs[0].set(xlabel="Time step")
        axs[0].set(ylabel="PM #")
        cbar = plt.colorbar(im)
        cbar.set_label("PM Utilisation")
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

    def record_testing_step(self, step: int, obs: np.ndarray, reward: float, info):
        self.record.pm_utilisation.append(info["pm_utilisation"])
        self.record.used_pm.append(len(info["pm_utilisation"]) - np.count_nonzero(info["pm_utilisation"]))
        self.record.vm_placements.append(info["vm_placement"])
        self.record.waiting_ratio.append(info['waiting_ratio'])
        self.record.actions.append(info["action"])
        self.record.rewards.append(reward)
        self.record.dropped_requests.append(info["dropped_requests"])
        self.record.total_requests.append(info["total_requests"])
        self.record.vm_arrival_steps = info["vm_arrival_steps"]
        self.record.target_util_mean.append(info['target_util_mean'])
        self.record.served_requests.append(int(info['served_requests']))
        self.record.total_resource_requested = info['total_resource_requested']
        self.record.suspended.append(info['suspend_actions'])
        self.record.placed.append(info['place_actions'])