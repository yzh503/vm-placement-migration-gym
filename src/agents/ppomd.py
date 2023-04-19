import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from src.envs.env import VmEnv
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from src.envs.preprocess import PreprocessEnv
from src.agents.ppo import PPOConfig, RewardScaler, ortho_init, PPOAgent
from torch.optim import lr_scheduler

class MDModel(nn.Module): 
    def __init__(self, obs_dim, action_space, hidden_size_1, hidden_size_2):
        super(MDModel, self).__init__()
        self.action_nvec = action_space.nvec
        self.shared_network = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_size_1)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, hidden_size_2)),
            nn.Tanh(),
        )
        self.actor = ortho_init(nn.Linear(hidden_size_1, self.action_nvec.sum()), scale=0.01)
        self.critic = ortho_init(nn.Linear(hidden_size_1, 1), scale=1)

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
        return action.T, logprob.sum(dim=0, dtype=torch.float64), entropy.sum(dim=0, dtype=torch.float64)
    
    def get_det_action(self, obs, action=None):
        logits = self.actor(self.shared_network(obs))
        split_logits = torch.split(logits, self.action_nvec.tolist(), dim=1)
        return torch.argmax(split_logits, dim=1)



class MDSModel(nn.Module): 
    def __init__(self, obs_dim, action_space, hidden_size_1, hidden_size_2):
        super(MDSModel, self).__init__()
        self.action_nvec = action_space.nvec
        self.critic = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_size_1)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, hidden_size_2)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, 1), scale=1)
        )
        self.actor = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_size_1)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, hidden_size_2)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, self.action_nvec.sum()), scale=0.01)
        )

    def get_value(self, obs):
        return self.critic(obs)

    def get_action(self, obs, action=None):
        logits = self.actor(obs)
        split_logits = torch.split(logits, self.action_nvec.tolist(), dim=1)
        multi_dists = [Categorical(logits=logits) for logits in split_logits]
        if action is None: 
            action = torch.stack([dist.sample() for dist in multi_dists])
        else: 
            action = action.T
        logprob = torch.stack([dist.log_prob(a) for a, dist in zip(action, multi_dists)])
        entropy = torch.stack([dist.entropy() for dist in multi_dists])
        return action.T, logprob.sum(dim=0, dtype=torch.float64), entropy.sum(dim=0, dtype=torch.float64)
    
    def get_det_action(self, obs, action=None):
        logits = self.actor(obs)
        split_logits = torch.reshape(logits, (self.action_nvec.size, self.action_nvec[0]))
        return torch.argmax(split_logits, dim=1)

class MDSCModel(nn.Module): 
    def __init__(self, obs_dim, action_space, hidden_size_1, hidden_size_2):
        super(MDSCModel, self).__init__()
        self.action_nvec = action_space.nvec
        self.critic = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_size_1)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, hidden_size_2)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, 1), scale=1)
        )
        self.actor_means = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_size_1)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, hidden_size_2)),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_size_1, self.action_nvec.shape[0]), scale=0.01)
        )
        self.actor_logstds = nn.Parameter(torch.zeros(1, np.prod(self.action_nvec.shape[0])))

    def get_value(self, obs):
        return self.critic(obs)

    def get_action(self, obs, actions=None):
        action_means = self.actor_means(obs)
        action_logstds = self.actor_logstds.expand_as(action_means)
        action_stds = torch.exp(action_logstds)
        probs = Normal(action_means, action_stds)
        if actions is None:
            actions = probs.sample()

        actions[actions < 0] = 0
        actions[actions > 0] = self.action_nvec[0] - 1
        return actions.int(), probs.log_prob(actions).sum(dim=1, dtype=torch.float64), probs.entropy().sum(dim=1, dtype=torch.float64)

class PPOMDAgent(PPOAgent):
    def __init__(self, env: VmEnv, config: PPOConfig):
        super().__init__(env, config, type(self).__name__)
        self.init_model()
   
    def init_model(self):
        self.env = PreprocessEnv(self.env)
        self.obs_dim = self.env.observation_space.shape[0]
        if self.config.network_arch == 'shared':
            self.model = MDModel(self.obs_dim, self.env.action_space, self.config.hidden_size_1, self.config.hidden_size_2) 
        elif self.config.network_arch == 'separate':
            self.model = MDSModel(self.obs_dim, self.env.action_space, self.config.hidden_size_1, self.config.hidden_size_2) 
        elif self.config.network_arch == 'continuous':
            self.model = MDSCModel(self.obs_dim, self.env.action_space, self.config.hidden_size_1, self.config.hidden_size_2)
        else:
            assert self.config.network_arch not in ['shared', 'separate', 'continuous']
        self.model = torch.compile(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, eps=1e-5)


    def act(self, obs):
        if self.config.det_eval:
            action = self.model.get_det_action(obs)
            return action
        else:
            action, _, _ = self.model.get_action(obs)
            return action

    def learn(self):
        ep_returns = np.zeros(self.config.n_episodes)
        pbar = tqdm(range(int(self.config.n_episodes)), disable=not bool(self.config.show_training_progress))
        return_factor = int(self.config.n_episodes*0.01 if self.config.n_episodes >= 100 else 1)

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

        scheduler = lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lambda epoch: self.config.lr_lambda) #StepLR(self.optimizer, step_size=self.config.n_episodes // 100, gamma=0.95)

        for i_episode in pbar:
            current_ep_reward = 0
            self.env.random_seed() # get different sequence
            obs = self.env.reset()
            done = False
            while not done:
                action, logprob, _ = self.model.get_action(obs)

                next_obs, reward, done, _ = self.env.step(action)
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
                    self.update(action_batch, obs_batch, next_obs_batch, logprob_batch, rewards_batch, done_batch)
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
