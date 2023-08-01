import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from vm_gym.envs.env2d import VmEnv
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from src.vm_gym.envs.preprocess import PreprocessEnv
from torch.optim import lr_scheduler
from src.agents.base import Base
from dataclasses import dataclass
from collections import deque
from torch.utils.data import SequentialSampler, BatchSampler

@dataclass
class ppolstmConfig:
    episodes: int
    hidden_size_1: int
    hidden_size_2: int
    lr: float
    lr_lambda: float
    gamma: float # GAE parameter
    lamda: float # GAE parameter
    ent_coef: float # Entropy coefficient
    vf_coef: float # Value function coefficient
    vf_loss_clip: bool
    k_epochs: int
    kl_max: float
    eps_clip: float
    max_grad_norm: float # Clip gradient
    batch_size: int
    minibatch_size: int
    reward_scaling: bool
    training_progress_bar: bool

    def __post_init__(self):
        self.episodes = int(self.episodes)
        self.hidden_size_1 = int(self.hidden_size_1)
        self.hidden_size_2 = int(self.hidden_size_2)
        self.lr = float(self.lr)
        self.lr_lambda = float(self.lr_lambda)
        self.gamma = float(self.gamma)
        self.lamda = float(self.lamda)
        self.ent_coef = float(self.ent_coef)
        self.vf_coef = float(self.vf_coef)
        self.vf_loss_clip = bool(self.vf_loss_clip)
        self.k_epochs = int(self.k_epochs)
        self.kl_max = float(self.kl_max)
        self.eps_clip = float(self.eps_clip)
        self.max_grad_norm = float(self.max_grad_norm)
        self.batch_size = int(self.batch_size)
        self.minibatch_size = int(self.minibatch_size)
        self.reward_scaling = bool(self.reward_scaling)
        self.training_progress_bar = bool(self.training_progress_bar)

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

def ortho_init(layer, scale=1.0):
    nn.init.orthogonal_(layer.weight, gain=scale)
    nn.init.constant_(layer.bias, 0)
    return layer

class Lstm(nn.Module): 
    def __init__(self, obs_dim, action_space, hidden_size_1, hidden_size_2):
        super(Lstm, self).__init__()
        self.action_nvec = action_space.nvec
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.layers = 2

        self.lstm_critic = nn.LSTM(input_size=obs_dim, hidden_size=hidden_size_1, num_layers=self.layers)
        self.critic = ortho_init(nn.Linear(hidden_size_2, 1), scale=1)

        self.lstm_actor = nn.LSTM(input_size=obs_dim, hidden_size=hidden_size_1, num_layers=self.layers)
        self.actor = ortho_init(nn.Linear(hidden_size_2, self.action_nvec.sum()), scale=0.01)

    def forward(self, obs, hidden_critic, hidden_actor):
        lstm_out_critic, hidden_critic = self.lstm_critic(obs.view(len(obs), 1, -1), hidden_critic)
        value = self.critic(lstm_out_critic.view(len(obs), -1))

        lstm_out_actor, hidden_actor = self.lstm_actor(obs.view(len(obs), 1, -1), hidden_actor)
        logits = self.actor(lstm_out_actor.view(len(obs), -1))
        
        return value, logits, hidden_critic, hidden_actor

    def init_hidden(self, actor_batch_size, critic_batch_size):
        weight = next(self.parameters()).data
        hidden_actor = (weight.new(self.layers, actor_batch_size, self.hidden_size_1).zero_(),
                weight.new(self.layers, actor_batch_size, self.hidden_size_1).zero_())
        hidden_critic = (weight.new(self.layers, critic_batch_size, self.hidden_size_1).zero_(),
                weight.new(self.layers, critic_batch_size, self.hidden_size_1).zero_())
        return hidden_actor, hidden_critic

    def get_value(self, obs, hidden_critic):
        value_out = []
        for i in range(obs.shape[0]):
            obs_i = obs[i].unsqueeze(0) if obs.shape[0] > 1 else obs
            lstm_out_critic, new_hidden_critic = self.lstm_critic(obs_i.view(1, obs_i.shape[0], obs_i.shape[1]), hidden_critic)
            value_out.append(self.critic(lstm_out_critic.view(obs_i.shape[0], -1)))
        values = torch.cat(value_out, dim=0)
        self.hidden_critic = (new_hidden_critic[0].clone(), new_hidden_critic[1].clone())
        return values, new_hidden_critic


    def get_action(self, obs, hidden_actor, action=None):
        logits_out = []
        for i in range(obs.shape[0]):
            obs_i = obs[i].unsqueeze(0) if obs.shape[0] > 1 else obs
            lstm_out_actor, new_hidden_actor = self.lstm_actor(obs_i.view(1, obs_i.shape[0], obs_i.shape[1]), hidden_actor)
            logits_out.append(self.actor(lstm_out_actor.view(obs_i.shape[0], -1)))
        logits = torch.cat(logits_out, dim=0)
        split_logits = torch.split(logits, self.action_nvec.tolist(), dim=1)
        multi_dists = [Categorical(logits=logits) for logits in split_logits]
        if action is None: 
            action = torch.stack([dist.sample() for dist in multi_dists])
        else: 
            action = action.T
        logprob = torch.stack([dist.log_prob(a) for a, dist in zip(action, multi_dists)])
        entropy = torch.stack([dist.entropy() for dist in multi_dists])
        return action.T, logprob.sum(dim=0), entropy.sum(dim=0), new_hidden_actor

class ppolstmAgent(Base):
    def __init__(self, env: VmEnv, config: ppolstmConfig):
        super().__init__(type(self).__name__, env, config)
        self.init_model()
   
    def init_model(self):
        self.env = PreprocessEnv(self.env)
        self.obs_dim = self.env.observation_space.shape[0]
        self.model = Lstm(self.obs_dim, self.env.action_space, self.config.hidden_size_1, self.config.hidden_size_2)
        self.model = torch.compile(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, eps=1e-5)
        self.hidden_actor = self.model.init_hidden(1, 1)[0]


    def act(self, obs):
        action, _, _, hidden_actor = self.model.get_action(obs, self.hidden_actor)
        self.hidden_actor = (hidden_actor[0].detach(), hidden_actor[1].detach())
        return action

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

        action_batch = torch.zeros((self.config.batch_size,self.env.action_space.nvec.size))
        obs_batch = torch.zeros(self.config.batch_size, self.obs_dim)
        next_obs_batch = torch.zeros(self.config.batch_size, self.obs_dim)
        logprob_batch = torch.zeros(self.config.batch_size)
        rewards_batch = torch.zeros(self.config.batch_size)
        done_batch = torch.zeros(self.config.batch_size)
        batch_head = 0
    
        # obs_normalizer = ObsNormalizer(shape=self.obs_dim) # Observation normalization 
        if self.config.reward_scaling:
            reward_scaler = RewardScaler(shape=1, gamma=self.config.gamma) # Reward scaling

        scheduler = lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lambda epoch: self.config.lr_lambda) #StepLR(self.optimizer, step_size=self.config.episodes // 100, gamma=0.95)

        for i_episode in pbar:
            current_ep_reward = 0
            obs, _ = self.env.reset(self.env.config.seed + i_episode)
            done = False

            hidden_actor, hidden_critic = self.model.init_hidden(actor_batch_size=1, critic_batch_size=1) 
            
            while not done:
                action, logprob, _, _ = self.model.get_action(obs, hidden_actor)

                next_obs, reward, done, truncated, info = self.env.step(action)
                self.total_steps += 1
                
                if self.config.reward_scaling:
                    reward_t = reward_scaler.scale(reward)[0]
                else: 
                    reward_t = reward

                action_batch[batch_head] = torch.flatten(action)
                obs_batch[batch_head] = torch.flatten(obs)
                next_obs_batch[batch_head] = torch.flatten(next_obs)
                logprob_batch[batch_head] = logprob.item()
                rewards_batch[batch_head] = reward_t
                done_batch[batch_head] = done
                batch_head += 1

                if batch_head >= self.config.batch_size:
                    hidden_actor, hidden_critic = self.update(action_batch, obs_batch, next_obs_batch, logprob_batch, rewards_batch, done_batch, hidden_actor, hidden_critic)
                    batch_head = 0
                    scheduler.step()
                
                obs = next_obs

                current_ep_reward += reward  # For logging

            ep_returns[i_episode] = current_ep_reward
            if self.writer: 
                self.writer.add_scalar('Training/ep_return', current_ep_reward, i_episode)
                self.writer.add_scalar('Training/lr', scheduler.get_last_lr()[0], i_episode)

            if i_episode > return_factor: 
                pbar.set_description("Return %.2f" % np.median(ep_returns[i_episode-return_factor:i_episode]))

    def update(self, action_batch, obs_batch, next_obs_batch, logprob_batch, rewards_batch, done_batch, initial_hidden_actor, initial_hidden_critic):
        done_batch = done_batch.int()

        # GAE advantages
        with torch.no_grad():
            gae = 0
            advantages = torch.zeros_like(rewards_batch)
            values_batch = torch.flatten(self.model.get_value(obs_batch, initial_hidden_critic)[0])
            next_values = torch.flatten(self.model.get_value(next_obs_batch, initial_hidden_critic)[0])
            deltas = rewards_batch + (1 - done_batch) * self.config.gamma * next_values - values_batch
            for i in reversed(range(len(deltas))):
                gae = deltas[i] + (1 - done_batch[i]) * self.config.gamma * self.config.lamda * gae
                advantages[i] = gae

            returns = advantages + values_batch

        clipfracs = []

        for epoch in range(self.config.k_epochs):
            sequential_sampler = SequentialSampler(range(self.config.batch_size))
            batch_sampler = BatchSampler(sequential_sampler, batch_size=self.config.minibatch_size, drop_last=False)

            for bi, minibatch in enumerate(batch_sampler):
                adv_minibatch = advantages[minibatch]
                adv_minibatch = (adv_minibatch - adv_minibatch.mean()) / (adv_minibatch.std() + 1e-10)  # Adv normalisation

                _, newlogprob, entropy, hidden_actor = self.model.get_action(obs_batch[minibatch], initial_hidden_actor,
                                                                            action_batch[minibatch])
                log_ratios = newlogprob - logprob_batch[minibatch]  # KL divergence
                ratios = torch.exp(log_ratios)
                assert bi != 0 or epoch != 0 or torch.all(torch.abs(ratios - 1.0) < 9e-4), str(
                    bi) + str(log_ratios)  # newlogprob == logprob_batch in epoch 1 minibatch 1
                if -log_ratios.mean() > self.config.kl_max:
                    break
                clipfracs.append(((ratios - 1.0).abs() > self.config.eps_clip).float().mean().item())

                surr = -ratios * adv_minibatch
                surr_clipped = -torch.clamp(ratios, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * adv_minibatch
                loss_clipped = torch.max(surr, surr_clipped).mean()

                newvalues, hidden_critic = self.model.get_value(obs_batch[minibatch], initial_hidden_critic)

                loss_vf_unclipped = torch.square(newvalues - returns[minibatch])
                v_clipped = values_batch[minibatch] + torch.clamp(newvalues - values_batch[minibatch], -self.config.eps_clip,
                                                                self.config.eps_clip)
                loss_vf_clipped = torch.square(v_clipped - returns[minibatch])
                loss_vf_max = torch.max(loss_vf_unclipped, loss_vf_clipped)

                if self.config.vf_loss_clip:
                    loss_vf = 0.5 * loss_vf_max.mean()
                else:
                    loss_vf = 0.5 * loss_vf_unclipped.mean()

                loss = loss_clipped - self.config.ent_coef * entropy.mean() + self.config.vf_coef * loss_vf

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward(retain_graph=True)  # Specify retain_graph=True
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

        # Update LSTM states with the final states after gradient descent
        hidden_actor = (hidden_actor[0].detach(), hidden_actor[1].detach())
        hidden_critic = (hidden_critic[0].detach(), hidden_critic[1].detach())

        if self.writer:
            self.writer.add_scalar('Training/loss_clipped', loss_clipped.item(), self.total_steps)
            self.writer.add_scalar('Training/loss_vf', loss_vf.item(), self.total_steps)
            self.writer.add_scalar('Training/entropy', entropy.mean().item(), self.total_steps)
            self.writer.add_scalar('Training/loss', loss.item(), self.total_steps)
            self.writer.add_scalar('Training/kl', -log_ratios.mean().item(), self.total_steps)
            self.writer.add_scalar('Training/clipfracs', np.mean(clipfracs), self.total_steps)

        return hidden_actor, hidden_critic

