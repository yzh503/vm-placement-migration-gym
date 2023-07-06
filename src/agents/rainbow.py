
# Part of the Rainbow DQN algorithm code is from https://github.com/Curt-Park/rainbow-is-all-you-need
from dataclasses import dataclass
import math
import random
from collections import deque
from typing import Deque, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from src.utils import convert_obs_to_dict
from src.segment_tree import MinSegmentTree, SumSegmentTree
from src.agents.base import Base
from src.vm_gym.envs.env import VmEnv

class ReplayBuffer:
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: list[int],
        size: int, 
        batch_size: int = 32, 
        n_step: int = 1, 
        gamma: float = 0.99
    ):
        self.obs_buf = torch.zeros([size, obs_dim], dtype=torch.float32)
        self.next_obs_buf = torch.zeros([size, obs_dim], dtype=torch.float32)
        self.acts_buf = torch.zeros([size, len(action_dim)], dtype=torch.long)
        self.rews_buf = torch.zeros([size], dtype=torch.float32)
        self.done_buf = torch.zeros(size, dtype=torch.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self, 
        obs: torch.Tensor, 
        act: torch.Tensor, 
        rew: float, 
        next_obs: torch.Tensor, 
        done: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs, :],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            # for N-step Learning
            indices=idxs,
        )
    
    def sample_batch_from_idxs(
        self, idxs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs, :],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )
    
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int,
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6,
        n_step: int = 1, 
        gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(
            obs_dim, action_dim, size, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)
        
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices, :]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    
        
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())
    
class Network(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        hidden_size: int,
        out_dim: list, 
        atom_size: int, 
        support: torch.Tensor
    ):
        super(Network, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_size), 
            nn.ReLU(),
        )
        
        self.advantage_hidden_layers = nn.ModuleList()
        self.advantage_layers = nn.ModuleList()
        self.value_hidden_layers = nn.ModuleList()
        self.value_layers = nn.ModuleList()
        
        for i in range(len(out_dim)):
            self.advantage_hidden_layers.append(NoisyLinear(hidden_size, hidden_size))
            self.advantage_layers.append(NoisyLinear(hidden_size, out_dim[i] * atom_size))
            self.value_hidden_layers.append(NoisyLinear(hidden_size, hidden_size))
            self.value_layers.append(NoisyLinear(hidden_size, atom_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dists = self.dist(x)
        q_values = []
        for i in range(len(self.out_dim)):
            dists[i] = dists[i] * self.support.unsqueeze(0)
            q_values.append(dists[i].sum(dim=2))
        return q_values

    
    def dist(self, x: torch.Tensor) -> List[torch.Tensor]:
        feature = self.feature_layer(x)
        dists = []
        for i in range(len(self.out_dim)):
            adv_hid = F.relu(self.advantage_hidden_layers[i](feature))
            val_hid = F.relu(self.value_hidden_layers[i](feature))
        
            advantage = self.advantage_layers[i](adv_hid).view(
                -1, self.out_dim[i], self.atom_size
            )
            value = self.value_layers[i](val_hid).view(-1, 1, self.atom_size)
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
            dist = F.softmax(q_atoms, dim=-1)
            dist = dist.clamp(min=1e-3)  # for avoiding nans

            dists.append(dist)
        
        return dists
    
    def reset_noise(self):
        """Reset all noisy layers."""
        for adv_hid_layer, adv_layer, val_hid_layer, val_layer in zip(
            self.advantage_hidden_layers, 
            self.advantage_layers, 
            self.value_hidden_layers, 
            self.value_layers
        ):
            adv_hid_layer.reset_noise()
            adv_layer.reset_noise()
            val_hid_layer.reset_noise()
            val_layer.reset_noise()

@dataclass
class RainbowConfig(object):
    episodes: int = 1000
    hidden_size: int = 73
    lr: float = 3e-5
    memory_size: int = 100000
    batch_size: int = 100
    target_update: int = 5
    gamma: float = 0.99
    alpha: float = 0.2
    beta: float = 0.5
    prior_eps: float = 1e-6
    v_min: float = 0.0
    v_max: float = 200.0
    atom_size: int = 51
    n_step: int = 3
    device: str = "cpu"
    show_training_progress: bool = True

class RainbowAgent(Base):

    def __init__(self, env: VmEnv, config: RainbowConfig):
        super().__init__(type(self).__name__, env, config)
        self.device = config.device
        
        obs_dim = self.env.observation_space.shape[0]
        # 1. select pm, 2. select action: [vms, pms + 1]
        action_dim = [len(self.env.action_space.nvec), self.env.action_space.nvec[0]]

        self.memory = PrioritizedReplayBuffer(obs_dim, action_dim, config.memory_size, config.batch_size, alpha=config.alpha)
        self.use_n_step = True if config.n_step > 1 else False
        if self.use_n_step:
            self.n_step = config.n_step
            self.memory_n = ReplayBuffer(
                obs_dim, action_dim, config.memory_size, config.batch_size, n_step=config.n_step, gamma=config.gamma
            )

        # Categorical DQN parameters
        self.support = torch.linspace(config.v_min, config.v_max, config.atom_size).to(self.device)

        self.dqn = Network(obs_dim, config.hidden_size, action_dim, config.atom_size, self.support).to(self.device)
        self.dqn = torch.compile(self.dqn)
        self.dqn_target = Network(obs_dim, config.hidden_size, action_dim, config.atom_size, self.support).to(self.device)
        self.dqn_target = torch.compile(self.dqn_target)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=config.lr)
        self.transition = list()
        self.is_test = False
    
    def load_model(self, modelpath):
        self.dqn.load_state_dict(torch.load(modelpath))
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
    
    def save_model(self, modelpath):
        if modelpath: 
            torch.save(self.dqn.state_dict(), modelpath)
            
    def learn(self):
        ep_returns = np.zeros(self.config.episodes)
        pbar = tqdm(range(int(self.config.episodes)), disable=not bool(self.config.show_training_progress))
        return_factor = int(self.config.episodes*0.01 if self.config.episodes >= 100 else 1)
        step = 0
        for i_episode in pbar:
            self.env.seed(self.env.config.seed + i_episode) # get different sequence
            current_ep_reward = 0
            previous_obs, info = self.env.reset()
            previous_obs = torch.from_numpy(previous_obs).float().to(self.device)
            done = False
            update_cnt = 0

            while not done:
                action = self._select_action(previous_obs)
                obs, reward, terminated, truncated, info = self.env.step(self._multi_discrete(previous_obs, action))
                obs = torch.from_numpy(obs).float().to(self.device)
                done = terminated or truncated

                fraction = min(i_episode / self.config.episodes, 1.0)
                self.config.beta = self.config.beta + fraction * (1.0 - self.config.beta)

                self.transition = [previous_obs, action, reward, obs, done]
                if self.use_n_step:
                    one_step_transition = self.memory_n.store(*self.transition)
                else:
                    one_step_transition = self.transition
                if one_step_transition:
                    self.memory.store(*one_step_transition)

                if len(self.memory) >= self.config.batch_size:
                    loss = self._optimize_model()
                    if self.writer and loss: 
                        self.writer.add_scalar('Training/loss', loss, step)
                    update_cnt += 1
                    if update_cnt % self.config.target_update == 0:
                        self._target_hard_update()

                previous_obs = obs
                current_ep_reward += reward  # For logging
                step += 1

            ep_returns[i_episode] = current_ep_reward

            if self.writer: 
                self.writer.add_scalar('Training/ep_return', current_ep_reward, i_episode)

            if i_episode > return_factor: 
                pbar.set_description("Return %.2f" % np.median(ep_returns[i_episode-return_factor:i_episode]))

        ep_returns_median = np.zeros_like(ep_returns)
        for r in range(ep_returns.size):
            ep_returns_median[r] = np.median(ep_returns[r-return_factor:r])


    def _multi_discrete(self, observation: torch.Tensor, action: list[torch.Tensor]) -> np.ndarray:
        obsdict = convert_obs_to_dict(observation.flatten().numpy()) 
        vm_placement = obsdict["vm_placement"]
        vm = action[0]
        vm_placement[vm] = action[1] - 1 # first action denotes waiting
        return np.array(vm_placement) + 1 # action space starts from 0

    def act(self, observation: np.ndarray) -> np.ndarray:
        selected_action = self._select_action(observation)
        return self._multi_discrete(observation, selected_action)

    def _select_action(self, observation: torch.Tensor) -> torch.Tensor:
        actions = self.dqn(observation)
        selected_action = torch.zeros(len(actions), dtype=int)
        for i, action in enumerate(actions):
            selected_action[i] = action.argmax(1)
        return selected_action
    
    def _optimize_model(self) -> torch.Tensor:
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.config.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.config.gamma)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.config.gamma ** self.config.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.config.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()
    
    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)
        
        # Categorical DQN algorithm
        delta_z = float(self.config.v_max - self.config.v_min) / (self.config.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state)
            next_action_list = torch.zeros(len(next_action), self.config.batch_size, dtype=int)
            for i, next_action in enumerate(next_action):
                next_action = next_action.argmax(1)
                next_action_list[i] = next_action
            next_dists = self.dqn_target.dist(next_state)
            proj_dists = []
            for i, next_dist in enumerate(next_dists):
                next_dist = next_dist[range(self.config.batch_size), next_action_list[i, :]]

                t_z = reward + (1 - done) * gamma * self.support
                t_z = t_z.clamp(min=self.config.v_min, max=self.config.v_max)
                b = (t_z - self.config.v_min) / delta_z
                l = b.floor().long()
                u = b.ceil().long()

                offset = (
                    torch.linspace(
                        0, (self.config.batch_size - 1) * self.config.atom_size, self.config.batch_size
                    ).long()
                    .unsqueeze(1)
                    .expand(self.config.batch_size, self.config.atom_size)
                    .to(self.device)
                )

                proj_dist = torch.zeros(next_dist.size(), device=self.device)
                proj_dist.view(-1).index_add_(
                    0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
                )
                proj_dist.view(-1).index_add_(
                    0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
                )

                proj_dists.append(proj_dist)

        dists = self.dqn.dist(state)
        elementwise_loss = 0
        for i, dist in enumerate(dists):
            log_p = torch.log(dist[range(self.config.batch_size), action[:, i]])
            elementwise_loss += -(proj_dists[i] * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())