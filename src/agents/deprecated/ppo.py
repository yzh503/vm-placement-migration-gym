from typing import Tuple
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from vmenv.envs.env import VmEnv
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import lr_scheduler
from src.agents.base import Base
from dataclasses import dataclass
import gymnasium
from torch.utils.data import BatchSampler, SubsetRandomSampler, SequentialSampler
@dataclass
class PPOConfig:
    migration_rate: float = 0.01
    episodes: int = 2000
    hidden_size: int = 256
    lr: float = 3e-5
    lr_lambda: float = 1
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

MIN_FLOAT = -1.0e7

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

class MlpShared(nn.Module): 
    def __init__(self, obs_dim, action_space, hidden_size):
        super(MlpShared, self).__init__()
        self.action_nvec = action_space.nvec
        self.shared_network = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.actor = ortho_init(nn.Linear(hidden_size, self.action_nvec.sum()), scale=0.01)
        self.critic = ortho_init(nn.Linear(hidden_size, 1), scale=1)

    def get_value(self, obs):
        return self.critic(self.shared_network(obs))

    def get_action(self, obs, action=None):
        logits = self.actor(self.shared_network(obs))
        split_logits = torch.split(logits, self.action_nvec.tolist(), dim=1)
        multi_dists = [Categorical(logits=logits) for logits in split_logits]
        if action is None: 
            action = torch.stack([dist.sample() for dist in multi_dists])
        else: 
            action = action.T
        logprob = torch.stack([dist.log_prob(a) for a, dist in zip(action, multi_dists)])
        entropy = torch.stack([dist.entropy() for dist in multi_dists])
        return action.T, logprob.sum(dim=0), entropy.sum(dim=0)
    
    def get_det_action(self, obs, action=None):
        logits = self.actor(self.shared_network(obs))
        split_logits = torch.split(logits, self.action_nvec.tolist(), dim=1)
        return torch.argmax(split_logits, dim=1)

class MlpSeparate(nn.Module): 
    def __init__(self, input_size: int, action_space: gymnasium.spaces.multi_discrete.MultiDiscrete, hidden_size: int):
        super(MlpSeparate, self).__init__()
        self.action_nvec = action_space.nvec
        self.critic = nn.Sequential(
            ortho_init(nn.Linear(input_size, hidden_size)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size, 1), scale=1)
        )
        self.actor = nn.Sequential(
            ortho_init(nn.Linear(input_size, hidden_size)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size, self.action_nvec.sum()), scale=0.01)
        )

    def get_value(self, obs):
        return self.critic(obs)
    
    # mask is a boolean tensor with the same shape as your action space, where True indicates an invalid action.
    def get_action(self, obs: torch.Tensor, action: torch.Tensor = None, mask : torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        split_logits = torch.split(logits, self.action_nvec.tolist(), dim=1)
        if mask is not None:
            split_logits = [logits.masked_fill(mask, float('-inf')) for logits in split_logits]
        multi_dists = [Categorical(logits=logits) for logits in split_logits]
        if action is None: 
            action = torch.stack([dist.sample() for dist in multi_dists])
        else: 
            action = action.T
        logprob = torch.stack([dist.log_prob(a) for a, dist in zip(action, multi_dists)])
        entropy = torch.stack([dist.entropy() for dist in multi_dists])
        return action.T, logprob.sum(dim=0), entropy.sum(dim=0)
    
    def get_det_action(self, obs, action=None):
        logits = self.actor(obs)
        split_logits = torch.reshape(logits, (self.action_nvec.size, self.action_nvec[0]))
        return torch.argmax(split_logits, dim=1)

class PPOAgent(Base):
    def __init__(self, env: VmEnv, config: PPOConfig):
        super().__init__(type(self).__name__, env, config)
        self.init_model()
   
    def init_model(self):
        self.obs_dim = self.env.observation_space.shape[0]
        if self.config.network_arch == 'shared':
            self.model = MlpShared(self.obs_dim, self.env.action_space, self.config.hidden_size)
        elif self.config.network_arch == 'separate':
            self.model = MlpSeparate(self.obs_dim, self.env.action_space, self.config.hidden_size) 
        else:
            assert self.config.network_arch not in ['shared', 'separate', 'continuous']
        self.model = torch.compile(self.model).to(self.config.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, eps=1e-5)


    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = torch.from_numpy(obs).float().to(self.config.device).unsqueeze(0) # a batch of size 1
        if self.config.det:
            action = self.model.get_det_action(obs)
        else:
            action, _, _ = self.model.get_action(obs)
        return action.flatten().cpu().numpy()

    def save_model(self, modelpath):
        if modelpath: 
            torch.save(self.model.state_dict(), modelpath)

    def load_model(self, modelpath):
        self.model.load_state_dict(torch.load(modelpath))
        self.model.eval()

    def learn(self):
        ep_returns = np.zeros(self.config.episodes)
        pbar = tqdm(range(int(self.config.episodes)), disable=not bool(self.config.training_progress_bar))
        return_factor = int(self.config.episodes*0.01 if self.config.episodes >= 100 else 1)

        action_batch = torch.zeros((self.config.batch_size,self.env.action_space.nvec.size), dtype=int, device=self.config.device)
        obs_batch = torch.zeros(self.config.batch_size, self.obs_dim, device=self.config.device)
        next_obs_batch = torch.zeros(self.config.batch_size, self.obs_dim, device=self.config.device)
        logprob_batch = torch.zeros(self.config.batch_size, device=self.config.device)
        rewards_batch = torch.zeros(self.config.batch_size, device=self.config.device)
        done_batch = torch.zeros(self.config.batch_size, dtype=int, device=self.config.device)
        i_batch = 0

    
        # obs_normalizer = ObsNormalizer(shape=self.obs_dim) # Observation normalization 
        if self.config.reward_scaling:
            reward_scaler = RewardScaler(shape=1, gamma=self.config.gamma) # Reward scaling

        scheduler = lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lambda epoch: self.config.lr_lambda) #StepLR(self.optimizer, step_size=self.config.episodes // 100, gamma=0.95)

        for i_episode in pbar:
            current_ep_reward = 0
            obs, _ = self.env.reset(seed=self.env.config.seed + i_episode)
            obs = torch.from_numpy(obs).float().to(self.config.device)
            done = False
            while not done:
                action, logprob, _ = self.model.get_action(obs.unsqueeze(0)) # pass in a batch of size 1
                action = torch.flatten(action)
                next_obs, reward, done, truncated, info = self.env.step(action.cpu().numpy())
                next_obs = torch.from_numpy(next_obs).float().to(self.config.device)
                reward_t = reward_scaler.scale(reward)[0] if self.config.reward_scaling else reward

                action_batch[i_batch] = action
                obs_batch[i_batch] = obs
                next_obs_batch[i_batch] = next_obs
                logprob_batch[i_batch] = logprob.item()
                rewards_batch[i_batch] = reward_t
                done_batch[i_batch] = done
                i_batch += 1

                if i_batch >= self.config.batch_size:
                    self.update(action_batch, obs_batch, next_obs_batch, logprob_batch, rewards_batch, done_batch)
                    scheduler.step()
                    i_batch = 0
                
                obs = next_obs
                self.total_steps += 1
                current_ep_reward += reward  # For logging

            ep_returns[i_episode] = current_ep_reward
            if self.writer: 
                self.writer.add_scalar('Training/ep_return', current_ep_reward, i_episode)
                self.writer.add_scalar('Training/lr', scheduler.get_last_lr()[0], i_episode)

            if i_episode > return_factor: 
                pbar.set_description("Return %.2f" % np.median(ep_returns[i_episode-return_factor:i_episode]))

    def update(self, action_batch, obs_batch, next_obs_batch, logprob_batch, rewards_batch, done_batch):

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
                _, newlogprob, entropy = self.model.get_action(obs_batch[minibatch], action_batch[minibatch])
                log_ratios = newlogprob - logprob_batch[minibatch] # KL divergence
                ratios = torch.exp(log_ratios)
                assert bi != 0 or epoch != 0 or torch.all(torch.abs(ratios - 1.0) < 5e-5),  log_ratios # newlogprob == logprob_batch in epoch 1 minibatch 1.
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

        if self.writer: 
            self.writer.add_scalar('Training/loss_clipped', loss_clipped.item(), self.total_steps)
            self.writer.add_scalar('Training/loss_vf', loss_vf.item(), self.total_steps)
            self.writer.add_scalar('Training/entropy', entropy.mean().item(), self.total_steps)
            self.writer.add_scalar('Training/loss', loss.item(), self.total_steps)
            self.writer.add_scalar('Training/kl', -log_ratios.mean().item(), self.total_steps)
            self.writer.add_scalar('Training/clipfracs', np.mean(clipfracs), self.total_steps)
