import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import math
from vmenv.envs.env import VmEnv
from torch.distributions.categorical import Categorical
from src.agents.base import Base, Config
from dataclasses import dataclass
import gymnasium
from torch.utils.data import BatchSampler, SubsetRandomSampler, SequentialSampler
from torch.optim import lr_scheduler
import src.utils as utils
@dataclass
class PPOConfig(Config):
    episodes: int = 2000
    hidden_size: int = 256
    migration_ratio: float = 0.001
    masked: bool = True
    lr: float = 5e-5
    gamma: float = 0.99 # GAE parameter
    lamda: float = 0.98 # GAE parameter
    ent_coef: float = 0.01 # Entropy coefficient
    vf_coef: float = 0.5 # Value function coefficient
    vf_loss_clip: bool = True
    k_epochs: int = 4
    kl_max: float = 0.02
    eps_clip: float = 0.1
    max_grad_norm: float = 0.5 # Clip gradient
    batch_size: int = 100
    minibatch_size: int = 25    
    det: bool = False # Determinisitc action for evaludation
    network_arch: str = "separate"
    reward_scaling: bool = False
    training_progress_bar: bool = True
    device: str = "cpu"

class RunningMeanStd:
    # https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/69019cf9b1624db3871d4ed46e29389aadfdcb02/4.PPO-discrete/normalization.py
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

class RewardScaler:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def scale(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

class ObsNormalizer:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def normalize(self, x, update=True):
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x

def ortho_init(layer, scale=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain=scale)
    nn.init.constant_(layer.bias, 0)
    return layer

class Network(nn.Module): 
    def __init__(self, input_size: int, action_space: gymnasium.spaces.multi_discrete.MultiDiscrete, hidden_size: int, dtype: torch.dtype):
        super(Network, self).__init__()
        self.action_nvec = action_space.nvec
        self.critic = nn.Sequential(
            ortho_init(nn.Linear(input_size, hidden_size, dtype=dtype)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size, hidden_size, dtype=dtype)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size, 1, dtype=dtype), scale=1)
        )
        self.actor = nn.Sequential(
            ortho_init(nn.Linear(input_size, hidden_size, dtype=dtype)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size, hidden_size, dtype=dtype)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size, self.action_nvec.sum(), dtype=dtype), scale=0.01)
        )

    def get_value(self, obs):
        return self.critic(obs)
    
    # invalid_mask is a boolean tensor with the same shape as your action space, where True indicates an invalid action.
    def get_action(self, obs, action=None, invalid_mask=None):
        logits = self.actor(obs) # With batch calculation, logits could be inconsistent from non-batch calculation at scale of 1e-8 ~ 1e-4 due to limited precision. 
        if invalid_mask is not None:
            invalid_mask = invalid_mask.reshape(logits.shape)
            logits[invalid_mask] = -1e7
        split_logits = torch.split(logits, self.action_nvec.tolist(), dim=1)
        multi_dists = [Categorical(logits=logits) for logits in split_logits]
        if action is None: 
            action = torch.stack([dist.sample() for dist in multi_dists]).T
        logprob = torch.stack([dist.log_prob(a) for a, dist in zip(action.T, multi_dists)])
        entropy = torch.stack([dist.entropy() for dist in multi_dists])
        return action, logprob.sum(dim=0), entropy.sum(dim=0)
    
    def get_det_action(self, obs, action=None):
        logits = self.actor(obs)
        split_logits = torch.reshape(logits, (self.action_nvec.size, self.action_nvec[0]))
        return torch.argmax(split_logits, dim=1)

class PPOAgent(Base):
    def __init__(self, env: VmEnv, config: PPOConfig):
        super().__init__(type(self).__name__, env, config)
        self.init_model()
   
    def init_model(self):
        self.float_dtype = torch.float32
        self.obs_dim = self.env.observation_space.shape[0]
        self.model = Network(self.obs_dim, self.env.action_space, self.config.hidden_size, self.float_dtype) 
        self.model = torch.compile(self.model).to(self.config.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

    def eval(self, mode=True):
        if mode: 
            self.model.eval()
        else:
            self.model.train()

    def act(self, obs: np.ndarray) -> np.ndarray:
        invalid_mask = torch.tensor(self.env.get_invalid_action_mask(self.config.masked), dtype=bool, device=self.config.device)
        obs = torch.tensor(obs, device=self.config.device, dtype=self.float_dtype).unsqueeze(0) # a batch of size 1
        if self.config.det:
            action = self.model.get_det_action(obs)
        else:
            action, _, _ = self.model.get_action(obs, invalid_mask=invalid_mask)
        return action.flatten().cpu().numpy()

    def save_model(self, modelpath):
        if modelpath: 
            utils.check_dir(modelpath)
            torch.save(self.model.state_dict(), modelpath)

    def load_model(self, modelpath):
        self.model.load_state_dict(torch.load(modelpath))
        self.model.eval()

    def learn(self):
        ep_returns = np.zeros(self.config.episodes)
        pbar = tqdm(range(int(self.config.episodes)), disable=not bool(self.config.training_progress_bar), desc='Training')

        return_factor = int(self.config.episodes*0.01 if self.config.episodes >= 100 else 1)
        
        invalid_mask_batch = torch.zeros((self.config.batch_size,self.env.config.vms, self.env.action_dim), dtype=bool, device=self.config.device)
        action_batch = torch.zeros((self.config.batch_size,self.env.action_space.nvec.size), dtype=int, device=self.config.device)
        obs_batch = torch.zeros(self.config.batch_size, self.obs_dim, dtype=self.float_dtype, device=self.config.device)
        next_obs_batch = torch.zeros(self.config.batch_size, self.obs_dim, dtype=self.float_dtype, device=self.config.device)
        logprob_batch = torch.zeros(self.config.batch_size, dtype=torch.float32, device=self.config.device)
        rewards_batch = torch.zeros(self.config.batch_size, dtype=torch.float32, device=self.config.device)
        done_batch = torch.zeros(self.config.batch_size, dtype=int, device=self.config.device)
        i_batch = 0

        if self.config.reward_scaling:
            reward_scaler = RewardScaler(shape=1, gamma=self.config.gamma) # Reward scaling

        for i_episode in pbar:
            current_ep_reward = 0
            obs, _ = self.env.reset(seed=self.env.config.seed + i_episode)
            obs = torch.tensor(obs, device=self.config.device, dtype=self.float_dtype)
            done = False
            while not done:
                invalid_mask = torch.tensor(self.env.get_invalid_action_mask(self.config.masked), device=self.config.device)
                action, logprob, _ = self.model.get_action(obs.unsqueeze(0), invalid_mask=invalid_mask) # pass in a batch of size 1
                action = torch.flatten(action)
                next_obs, reward, done, _, _ = self.env.step(action.cpu().numpy())
                next_obs = torch.tensor(next_obs, device=self.config.device, dtype=self.float_dtype)
                reward_t = reward_scaler.scale(reward)[0] if self.config.reward_scaling else reward
                
                invalid_mask_batch[i_batch] = invalid_mask
                action_batch[i_batch] = action
                obs_batch[i_batch] = obs
                next_obs_batch[i_batch] = next_obs
                logprob_batch[i_batch] = logprob.item()
                rewards_batch[i_batch] = reward_t
                done_batch[i_batch] = done
                i_batch += 1

                if i_batch >= self.config.batch_size:
                    self.update(invalid_mask_batch, action_batch, obs_batch, next_obs_batch, logprob_batch, rewards_batch, done_batch)
                    i_batch = 0
                
                obs = next_obs
                self.total_steps += 1
                current_ep_reward += reward  # For logging

            ep_returns[i_episode] = current_ep_reward
            if self.writer: 
                self.writer.add_scalar('Training/ep_return', current_ep_reward, i_episode)

            if i_episode > return_factor: 
                pbar.set_description("Return %.2f" % np.median(ep_returns[i_episode-return_factor:i_episode]))
            

    def update(self, invalid_mask_batch, action_batch, obs_batch, next_obs_batch, logprob_batch, rewards_batch, done_batch):
        
        # GAE advantages         
        with torch.no_grad():      
            gae = 0     
            advantages = torch.zeros_like(rewards_batch)
            values_batch = torch.flatten(self.model.get_value(obs_batch))
            next_values = torch.flatten(self.model.get_value(next_obs_batch))
            deltas = rewards_batch + (1 - done_batch) * self.config.gamma * next_values - values_batch
            for i in reversed(range(len(deltas))):
                gae = deltas[i] + (1 - done_batch[i]) * self.config.gamma * self.config.lamda * gae 
                advantages[i] = gae
            
            returns = advantages + values_batch

        clipfracs = []

        for epoch in range(self.config.k_epochs):
            minibatches = BatchSampler(
                SubsetRandomSampler(range(self.config.batch_size)), 
                batch_size=self.config.minibatch_size, 
                drop_last=False)
            sequential_sampler = SequentialSampler(range(self.config.batch_size))
            minibatches = BatchSampler(sequential_sampler, batch_size=self.config.minibatch_size, drop_last=False)
            
            for bi, minibatch in enumerate(minibatches):
                adv_minibatch = advantages[minibatch]
                adv_minibatch = (adv_minibatch - adv_minibatch.mean()) / (adv_minibatch.std() + 1e-10) # Adv normalisation
                mask_minibatch = invalid_mask_batch[minibatch] 
                _, newlogprob, entropy = self.model.get_action(obs_batch[minibatch], action=action_batch[minibatch], invalid_mask=mask_minibatch)
                #assert newlogprob.shape == logprob_batch[minibatch].shape
                log_ratios = newlogprob - logprob_batch[minibatch] # KL divergence
                ratios = torch.exp(log_ratios)
                #assert bi != 0 or epoch != 0 or torch.all(torch.abs(ratios - 1.0) < 1e-4),  log_ratios # reconstruct logprobs in epoch 1 minibatch 1.
                if -log_ratios.mean() > self.config.kl_max:
                    break
                clipfracs.append(((ratios - 1.0).abs() > self.config.eps_clip).float().mean().item())
                
                surr = -ratios * adv_minibatch
                surr_clipped = -torch.clamp(ratios, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * adv_minibatch
                loss_clipped = torch.max(surr, surr_clipped).mean()

                newvalues = self.model.get_value(obs_batch[minibatch])
                loss_vf_unclipped = torch.square(newvalues - returns[minibatch])
                v_clipped = values_batch[minibatch] + torch.clamp(newvalues - values_batch[minibatch], -self.config.eps_clip, self.config.eps_clip)
                loss_vf_clipped = torch.square(v_clipped - returns[minibatch])
                loss_vf_max = torch.max(loss_vf_unclipped, loss_vf_clipped)

                if (self.config.vf_loss_clip):
                    loss_vf = 0.5 * loss_vf_max.mean() 
                else:
                    loss_vf = 0.5 * loss_vf_unclipped.mean() 
    
                loss = loss_clipped - self.config.ent_coef * entropy.mean() + self.config.vf_coef * loss_vf # maximise equation (9) from the original PPO paper

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                if epoch == self.config.k_epochs - 1 and bi == math.ceil(self.config.batch_size / self.config.minibatch_size) and self.writer: 
                    self.writer.add_scalar('Training/loss_clipped', loss_clipped.item(), self.total_steps)
                    self.writer.add_scalar('Training/loss_vf', loss_vf.item(), self.total_steps)
                    self.writer.add_scalar('Training/entropy', entropy.mean().item(), self.total_steps)
                    self.writer.add_scalar('Training/loss', loss.item(), self.total_steps)
                    self.writer.add_scalar('Training/kl', -log_ratios.mean().item(), self.total_steps)
                    self.writer.add_scalar('Training/clipfracs', np.mean(clipfracs), self.total_steps)
