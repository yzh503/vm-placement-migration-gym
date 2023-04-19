from dataclasses import dataclass
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from src.agents.base import Base
from src.envs.env import VmEnv
from torch.distributions.categorical import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
from src.envs.preprocess import PreprocessEnv
from torch.optim import lr_scheduler

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

class Model(nn.Module): 
    def __init__(self, obs_dim, action_space, hidden_size_1, hidden_size_2):
        super(Model, self).__init__()
        self.shared_network = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_size_1)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, hidden_size_2)),
            nn.Tanh(),
        )
        self.actor = ortho_init(nn.Linear(hidden_size_2, action_space.n), scale=0.01)
        self.critic = ortho_init(nn.Linear(hidden_size_2, 1), scale=1)

    def get_value(self, obs):
        return self.critic(self.shared_network(obs))

    def get_action(self, obs, action=None):
        probs = Categorical(logits=self.actor(self.shared_network(obs)))
        if action is None: 
            action = probs.sample()
        logprob = probs.log_prob(action)
        entropy = probs.entropy()
        return action, logprob, entropy
    
@dataclass
class PPOConfig:
    n_episodes: int
    hidden_size_1: int
    hidden_size_2: int
    lr: float
    lr_lambda: float
    gamma: float # GAE parameter
    lamda: float # GAE parameter
    ent_coef: float # Entropy coefficient
    vf_coef: float # Value function coefficient
    k_epochs: int
    kl_max: float
    eps_clip: float
    max_grad_norm: float # Clip gradient
    batch_size: int
    minibatch_size: int
    det_eval: bool
    network_arch: str
    reward_scaling: bool
    show_training_progress: bool

    def __post_init__(self):
        self.n_episodes = int(self.n_episodes)
        self.hidden_size_1 = int(self.hidden_size_1)
        self.hidden_size_2 = int(self.hidden_size_2)
        self.lr = float(self.lr)
        self.lr_lambda = float(self.lr_lambda)
        self.gamma = float(self.gamma)
        self.lamda = float(self.lamda)
        self.ent_coef = float(self.ent_coef)
        self.vf_coef = float(self.vf_coef)
        self.k_epochs = int(self.k_epochs)
        self.kl_max = float(self.kl_max)
        self.eps_clip = float(self.eps_clip)
        self.max_grad_norm = float(self.max_grad_norm)
        self.batch_size = int(self.batch_size)
        self.minibatch_size = int(self.minibatch_size)
        self.det_eval = bool(self.det_eval)
        self.network_arch = str(self.network_arch)
        self.reward_scaling = bool(self.reward_scaling)
        self.show_training_progress = bool(self.show_training_progress)


class PPOAgent(Base):
    def __init__(self, env: VmEnv=None, config: PPOConfig=None, subclassname: str=None):
        self.device =  torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "cpu") GPU is slower than CPU due to architecture.
        if subclassname is None: 
            super().__init__(type(self).__name__, env, config)
            self.init_model()
        else: 
            super().__init__(subclassname, env, config)

    def init_model(self):
        self.env = PreprocessEnv(self.env)
        self.obs_dim = self.env.observation_space.shape[0]
        self.model = Model(self.obs_dim, self.env.action_space, self.config.hidden_size_1, self.config.hidden_size_2) # separate network architecture
        self.model = torch.compile(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, eps=1e-5)
    
    def act(self, obs):
        if self.config.det_eval: 
            logit = self.model.actor(self.model.shared_network(obs))
            probs = torch.softmax(logit, dim=1)
            return torch.argmax(probs)
        else: 
            action, _, _ = self.model.get_action(obs)
            return action
    
    def save_model(self, modelpath):
        if modelpath: 
            torch.save(self.model.state_dict(), modelpath)

    def load_model(self, modelpath):
        self.model.load_state_dict(torch.load(modelpath))
        self.model.eval()

    def learn(self):
        ep_returns = np.zeros(self.config.n_episodes)
        pbar = tqdm(range(int(self.config.n_episodes)), disable=not bool(self.config.show_training_progress))
        return_factor = int(self.config.n_episodes*0.01 if self.config.n_episodes >= 100 else 1)

        action_batch = torch.zeros(self.config.batch_size)
        obs_batch = torch.zeros(self.config.batch_size, self.obs_dim)
        next_obs_batch = torch.zeros(self.config.batch_size, self.obs_dim)
        logprob_batch = torch.zeros(self.config.batch_size, dtype=torch.float64)
        rewards_batch = torch.zeros(self.config.batch_size)
        done_batch = torch.zeros(self.config.batch_size)
        batch_head = 0
    
        reward_scaler = RewardScaler(shape=1, gamma=self.config.gamma) # Reward scaling

        
        scheduler = lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lambda epoch: 0.9995)
        
        for i_episode in pbar:
            current_ep_reward = 0
            self.env.random_seed() # get different sequence
            obs = self.env.reset()
            # obs = obs_normalizer.normalize(obs).float()
            done = False
            while not done:
                action, logprob, _ = self.model.get_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                self.total_steps += 1

                reward_scaled = reward_scaler.scale(reward) # Reward scaling improves performance

                action_batch[batch_head] = action.item()
                obs_batch[batch_head] = torch.flatten(obs)
                next_obs_batch[batch_head] = torch.flatten(next_obs)
                logprob_batch[batch_head] = logprob.item()
                rewards_batch[batch_head] =  reward_scaled[0]
                done_batch[batch_head] = done
                batch_head += 1

                if batch_head >= self.config.batch_size:
                    self.update(action_batch, obs_batch, next_obs_batch, logprob_batch, rewards_batch, done_batch)
                    scheduler.step()
                    batch_head = 0

                obs = next_obs

                current_ep_reward += reward  # For logging

            ep_returns[i_episode] = current_ep_reward

            if self.writer: 
                self.writer.add_scalar('Training/ep_return', current_ep_reward, i_episode)
                self.writer.add_scalar('Training/lr', scheduler.get_last_lr()[0], i_episode)

            if i_episode > return_factor: 
                pbar.set_description("Return %.2f" % np.median(ep_returns[i_episode-return_factor:i_episode]))
    
    def update(self, action_batch, obs_batch, next_obs_batch, logprob_batch, rewards_batch, done_batch):

        done_batch = done_batch.int()

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


            for bi, minibatch in enumerate(minibatches):
                adv_minibatch = advantages[minibatch]
                adv_minibatch = (adv_minibatch - adv_minibatch.mean()) / (adv_minibatch.std() + 1e-8) # Adv normalisation

                _, newlogprob, entropy = self.model.get_action(obs_batch[minibatch], action_batch[minibatch])
                log_ratios = newlogprob - logprob_batch[minibatch] # KL divergence
                ratios = torch.exp(log_ratios)
                assert bi != 0 or epoch != 0 or torch.all(torch.abs(ratios - 1.0) < 2e-4), str(bi) + str(log_ratios) # newlogprob == logprob_batch in epoch 1 minibatch 1
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
                loss_vf = 0.5 * loss_vf_unclipped.mean() # unclipped should be better

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
