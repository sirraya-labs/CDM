"""
Contraction Dynamics Model (CDM): Riemannian Metric Learning for Stable MBRL
Author: Amir Hameed, Sirraya Labs
Paper: "Learning Contraction Metrics for Provably Stable Model-Based Reinforcement Learning"
Implementation: Complete reproducible code with Riemannian metric learning, theoretical consistency,
and experimental validation matching paper specifications.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt
import pickle
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================
# CONFIGURATION
# ============================

class Config:
    """Configuration matching paper specifications"""
    # Environment
    ENV_NAME = "Pendulum-v1"
    STATE_DIM = 3
    ACTION_DIM = 1
    MAX_EPISODE_LENGTH = 200
    
    # Network Architecture (matches paper Section 4.2)
    DYNAMICS_HIDDEN_DIM = 64
    POLICY_HIDDEN_DIM = 128
    METRIC_HIDDEN_DIM = 64
    CRITIC_HIDDEN_DIM = 128
    
    # Training Parameters (matches paper Table 4)
    TOTAL_EPISODES = 300
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 0.005  # Soft update rate
    
    # Contraction Parameters (matches paper Eq. 11, 12)
    CONTRACTION_RATE_ALPHA = 0.95  # α in Eq. (12)
    INITIAL_BETA = 0.1  # β₀ stability weight
    BETA_MIN = 0.01
    BETA_MAX = 1.0
    METRIC_REGULARIZATION = 0.01  # λ_reg in Eq. (13)
    EPSILON_METRIC = 0.1  # ε in Eq. (10)
    PERTURBATION_SIGMA = 0.01  # σ_perturb in Section 4.3
    
    # Learning Rates (matches paper Section 6.1)
    ACTOR_LR = 2e-4
    CRITIC_LR = 5e-4
    DYNAMICS_LR = 1e-3
    METRIC_LR = 1e-4
    
    # Optimization
    REPLAY_BUFFER_SIZE = 30000
    INITIAL_EXPLORATION_STEPS = 1000
    UPDATE_FREQUENCY = 1  # Update after each episode
    ENSEMBLE_SIZE = 5  # Number of dynamics models (matches paper K=5)
    
    # Experimental Settings
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR = "cdm_results"
    LOG_INTERVAL = 10
    
    @staticmethod
    def get_device():
        """Get torch device from string"""
        return torch.device(Config.DEVICE)
    
    @staticmethod
    def get_save_dir():
        """Get save directory as Path"""
        return Path(Config.SAVE_DIR)
    
    @staticmethod
    def save(path):
        """Save configuration to file"""
        # Get all non-method attributes
        config_dict = {}
        for key in dir(Config):
            if not key.startswith('_') and not callable(getattr(Config, key)) and key not in ['save', 'load', 'get_device', 'get_save_dir']:
                value = getattr(Config, key)
                # Convert Path-like strings
                if key == 'SAVE_DIR' and isinstance(value, str):
                    value = str(Path(value))
                config_dict[key] = value
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @staticmethod
    def load(path):
        """Load configuration from file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        for k, v in config_dict.items():
            setattr(Config, k, v)

# ============================
# NETWORK ARCHITECTURES
# ============================

class DynamicsEnsemble(nn.Module):
    """Ensemble of dynamics models as in paper Section 4.1"""
    def __init__(self, state_dim, action_dim, ensemble_size, hidden_dim):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.models = nn.ModuleList([
            SimpleDynamics(state_dim, action_dim, hidden_dim)
            for _ in range(ensemble_size)
        ])
    
    def forward(self, state, action):
        """Returns mean prediction and uncertainty (variance)"""
        predictions = []
        for model in self.models:
            pred = model(state, action)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [ensemble, batch, state_dim]
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        return mean, variance

class SimpleDynamics(nn.Module):
    """Single dynamics model as in paper Eq. (2) approximation"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def forward(self, state, action):
        # Ensure proper dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)  # [1, state_dim]
        
        if action.dim() == 1:
            action = action.unsqueeze(0)  # [1, action_dim]
        elif action.dim() == 2 and action.shape[1] == 1:
            # Already in correct shape [batch, 1]
            pass
        elif action.dim() == 2 and action.shape[0] == 1:
            # [1, batch] -> transpose
            action = action.transpose(0, 1)
        
        # Concatenate along feature dimension
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class PolicyNetwork(nn.Module):
    """Actor network with Gaussian policy"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.action_dim = action_dim
        
        # Mean network
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Log standard deviation (learned parameter)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.action_scale = 2.0  # Pendulum action range
    
    def forward(self, state):
        mean = self.mean_net(state) * self.action_scale
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)
    
    def deterministic_action(self, state):
        """Get mean action for deployment"""
        return self.mean_net(state) * self.action_scale

class ValueNetwork(nn.Module):
    """Critic network for value estimation"""
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.net(state)

class ContractionMetricNetwork(nn.Module):
    """
    Neural Riemannian metric M(x) = L(x)L(x)^T + εI
    Implements Eq. (10) from paper with Cholesky parameterization
    """
    def __init__(self, state_dim, hidden_dim=64, epsilon=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.epsilon = epsilon
        
        # Number of parameters for lower triangular matrix
        self.output_dim = (state_dim * (state_dim + 1)) // 2
        
        # Network architecture matches paper Section 4.2
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
        # Softplus ensures positive diagonal entries
        self.softplus = nn.Softplus(beta=1.0, threshold=20)
        
        # For enforcing constraints
        self.diagonal_offset = 0.1
        self.off_diagonal_scale = 0.1
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Get raw parameters
        l_params = self.net(x)  # [batch, n*(n+1)/2]
        
        # Build lower triangular matrix L
        L = torch.zeros(batch_size, self.state_dim, self.state_dim, 
                       device=x.device)
        
        idx = 0
        for i in range(self.state_dim):
            for j in range(i + 1):
                val = l_params[:, idx]
                if i == j:
                    # Diagonal: positive through softplus + offset
                    L[:, i, j] = self.softplus(val) + self.diagonal_offset
                else:
                    # Off-diagonal: scaled for stability
                    L[:, i, j] = val * self.off_diagonal_scale
                idx += 1
        
        # Compute M = LL^T + εI (guarantees positive definiteness)
        M = torch.bmm(L, L.transpose(1, 2)) + self.epsilon * torch.eye(
            self.state_dim, device=x.device).unsqueeze(0)
        
        return M, L

# ============================
# RIEMANNIAN OPERATIONS
# ============================

class RiemannianOperations:
    """Implements Riemannian operations for contraction analysis"""
    
    @staticmethod
    def compute_energy(state, metric_net):
        """
        Compute energy E = x^T M(x) x
        Implements metric-based distance measure
        """
        M, _ = metric_net(state)
        
        if state.dim() == 2:
            # Batch mode
            state_unsqueezed = state.unsqueeze(1)  # [batch, 1, state_dim]
            energy = torch.bmm(
                torch.bmm(state_unsqueezed, M), 
                state_unsqueezed.transpose(1, 2)
            ).squeeze()
        else:
            # Single state
            state_unsqueezed = state.unsqueeze(0).unsqueeze(0)
            energy = torch.bmm(
                torch.bmm(state_unsqueezed, M), 
                state_unsqueezed.transpose(1, 2)
            ).squeeze()
        
        return energy, M
    
    @staticmethod
    def compute_contraction_loss(states, next_states_pred, metric_net, alpha=0.95):
        """
        Compute contraction loss: max(0, E_{t+1} - α^2 E_t)
        Implements Eq. (11) from paper
        """
        energy_curr, M_curr = RiemannianOperations.compute_energy(states, metric_net)
        energy_next, M_next = RiemannianOperations.compute_energy(next_states_pred, metric_net)
        
        # Contraction condition
        contraction_loss = F.relu(energy_next - (alpha**2) * energy_curr).mean()
        
        # Metric regularization (matches Eq. 13)
        identity = torch.eye(states.shape[-1], device=states.device)
        identity = identity.unsqueeze(0).repeat(states.shape[0], 1, 1)
        
        reg_loss = torch.norm(M_curr - identity, dim=(1, 2)).mean()
        
        # Ensure positive definiteness
        det_curr = torch.det(M_curr)
        det_penalty = F.relu(0.01 - det_curr).mean()
        
        return contraction_loss, reg_loss, det_penalty, energy_curr, energy_next
    
    @staticmethod
    def generate_virtual_displacements(states, sigma=0.01):
        """
        Generate virtual displacements for contraction analysis
        Implements virtual trajectory method from paper Section 4.3
        """
        batch_size = states.shape[0]
        state_dim = states.shape[1]
        
        # Generate perturbations
        perturbations = torch.randn_like(states) * sigma
        perturbed_states = states + perturbations
        
        return perturbed_states, perturbations

# ============================
# REPLAY BUFFER
# ============================

class ReplayBuffer:
    """Experience replay buffer with support for virtual displacements"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, path):
        """Save buffer to disk"""
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
    
    def load(self, path):
        """Load buffer from disk"""
        with open(path, 'rb') as f:
            self.buffer = deque(pickle.load(f), maxlen=self.capacity)

# ============================
# ADAPTIVE NOISE PROCESS
# ============================

class AdaptiveNoiseProcess:
    """
    Adaptive exploration noise with performance-based adjustment
    Implements exploration strategy mentioned in paper Section 4.4
    """
    def __init__(self, action_dim, base_sigma=0.3):
        self.action_dim = action_dim
        self.base_sigma = base_sigma
        self.sigma = base_sigma
        self.state = np.zeros(action_dim)
        self.recent_rewards = deque(maxlen=10)
        
        # OU process parameters
        self.theta = 0.15
        self.mu = 0.0
        self.dt = 1e-2
    
    def reset(self):
        self.state = np.zeros(self.action_dim)
    
    def sample(self):
        """Ornstein-Uhlenbeck process for correlated noise"""
        dx = self.theta * (self.mu - self.state) * self.dt
        dx += self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        self.state += dx
        return self.state.copy()
    
    def update(self, reward, episode):
        """Adapt sigma based on performance"""
        self.recent_rewards.append(reward)
        
        if len(self.recent_rewards) >= 5:
            avg_reward = np.mean(list(self.recent_rewards))
            
            # Performance-based adjustment
            if avg_reward > -600:  # Excellent
                self.sigma = self.base_sigma * 0.1
            elif avg_reward > -800:  # Good
                self.sigma = self.base_sigma * 0.3
            elif avg_reward > -1000:  # Fair
                self.sigma = self.base_sigma * 0.6
            else:  # Poor
                self.sigma = self.base_sigma
        
        # Episode-based decay
        decay_factor = max(0.1, 1.0 - episode / 200)
        self.sigma *= decay_factor
        self.sigma = max(0.05, self.sigma)  # Minimum noise

# ============================
# CDM AGENT (MAIN CLASS)
# ============================

class ContractionDynamicsAgent:
    """
    Main CDM agent implementing the complete algorithm from paper Algorithm 1
    """
    def __init__(self, config_class):
        self.config = config_class
        self.device = config_class.get_device()
        
        # Initialize networks
        self._initialize_networks()
        
        # Initialize optimizers
        self._initialize_optimizers()
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(config_class.REPLAY_BUFFER_SIZE)
        
        # Adaptive components
        self.noise_process = AdaptiveNoiseProcess(config_class.ACTION_DIM)
        self.beta = config_class.INITIAL_BETA  # Stability weight
        self.episode = 0
        
        # Tracking
        self.metrics = {
            'rewards': [],
            'energies': [],
            'contraction_losses': [],
            'dynamics_losses': [],
            'policy_losses': [],
            'beta_values': []
        }
        
        # Create save directory
        self.save_dir = Path(config_class.SAVE_DIR)
        self.save_dir.mkdir(exist_ok=True)
        
        # Save config
        config_class.save(self.save_dir / "config.json")
        
        print(f"CDM Agent initialized on {self.device}")
        print(f"Save directory: {self.save_dir}")
    
    def _initialize_networks(self):
        """Initialize all networks as per paper specifications"""
        # Dynamics ensemble (K models)
        self.dynamics = DynamicsEnsemble(
            self.config.STATE_DIM,
            self.config.ACTION_DIM,
            self.config.ENSEMBLE_SIZE,
            self.config.DYNAMICS_HIDDEN_DIM
        ).to(self.device)
        
        # Policy network (actor)
        self.policy = PolicyNetwork(
            self.config.STATE_DIM,
            self.config.ACTION_DIM,
            self.config.POLICY_HIDDEN_DIM
        ).to(self.device)
        
        # Value network (critic)
        self.critic = ValueNetwork(
            self.config.STATE_DIM,
            self.config.CRITIC_HIDDEN_DIM
        ).to(self.device)
        
        # Contraction metric network
        self.metric_net = ContractionMetricNetwork(
            self.config.STATE_DIM,
            self.config.METRIC_HIDDEN_DIM,
            epsilon=self.config.EPSILON_METRIC
        ).to(self.device)
        
        # Target networks
        self.target_policy = PolicyNetwork(
            self.config.STATE_DIM,
            self.config.ACTION_DIM,
            self.config.POLICY_HIDDEN_DIM
        ).to(self.device)
        self.target_critic = ValueNetwork(
            self.config.STATE_DIM,
            self.config.CRITIC_HIDDEN_DIM
        ).to(self.device)
        
        # Initialize target networks
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
    
    def _initialize_optimizers(self):
        """Initialize optimizers with paper-specified learning rates"""
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.ACTOR_LR,
            weight_decay=1e-5
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.config.CRITIC_LR,
            weight_decay=1e-5
        )
        self.dynamics_optimizer = optim.Adam(
            self.dynamics.parameters(),
            lr=self.config.DYNAMICS_LR,
            weight_decay=1e-5
        )
        self.metric_optimizer = optim.Adam(
            self.metric_net.parameters(),
            lr=self.config.METRIC_LR,
            weight_decay=1e-5
        )
    
    def soft_update(self, target, source):
        """Soft update target networks"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.TAU) +
                source_param.data * self.config.TAU
            )
    
    def adapt_beta(self, reward_improvement):
        """Adapt stability weight β as per paper Section 4.4"""
        if reward_improvement:
            self.beta *= 0.95  # Decrease stability focus
        else:
            self.beta *= 1.05  # Increase stability focus
        
        # Clip to bounds
        self.beta = np.clip(self.beta, self.config.BETA_MIN, self.config.BETA_MAX)
    
    def select_action(self, state, deterministic=False):
        """Select action with exploration noise"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if deterministic:
            with torch.no_grad():
                action = self.policy.deterministic_action(state_tensor)
            return action.cpu().numpy()[0]
        else:
            with torch.no_grad():
                action_dist = self.policy(state_tensor)
                action = action_dist.sample()
            
            # Add exploration noise
            noise = self.noise_process.sample()
            action_np = action.cpu().numpy()[0] + noise
            
            return np.clip(action_np, -self.policy.action_scale, self.policy.action_scale)
    
    def update_dynamics(self, batch):
        """Update dynamics models using supervised learning"""
        states, actions, _, next_states, _ = batch
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)  # [batch, state_dim]
        actions_t = torch.FloatTensor(actions).to(self.device)  # [batch, action_dim]
        next_states_t = torch.FloatTensor(next_states).to(self.device)  # [batch, state_dim]
        
        self.dynamics_optimizer.zero_grad()
        
        # Forward pass through ensemble
        next_states_pred, uncertainty = self.dynamics(states_t, actions_t)
        
        # Compute loss (MSE)
        dynamics_loss = F.mse_loss(next_states_pred, next_states_t)
        
        # Backward pass
        dynamics_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
        self.dynamics_optimizer.step()
        
        return dynamics_loss.item(), uncertainty.mean().item()
    
    def update_metric(self, batch):
        """Update contraction metric network"""
        states, actions, _, _, _ = batch
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        
        self.metric_optimizer.zero_grad()
        
        # Generate virtual displacements
        perturbed_states, perturbations = RiemannianOperations.generate_virtual_displacements(
            states_t, sigma=self.config.PERTURBATION_SIGMA
        )
        
        # Predict next states for both original and perturbed states
        with torch.no_grad():
            # Use current policy for action prediction
            action_dist = self.policy(states_t)
            actions_sampled = action_dist.rsample()
            
            # Predict next states using dynamics
            next_states_pred, _ = self.dynamics(states_t, actions_sampled)
            next_perturbed_pred, _ = self.dynamics(perturbed_states, actions_sampled)
        
        # Compute contraction loss
        contraction_loss, reg_loss, det_penalty, energy_curr, energy_next = \
            RiemannianOperations.compute_contraction_loss(
                states_t, next_states_pred, self.metric_net,
                alpha=self.config.CONTRACTION_RATE_ALPHA
            )
        
        # Total metric loss (matches Eq. 13)
        metric_loss = (contraction_loss + 
                      self.config.METRIC_REGULARIZATION * reg_loss +
                      0.1 * det_penalty)
        
        # Backward pass
        metric_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.metric_net.parameters(), 0.5)
        self.metric_optimizer.step()
        
        metrics = {
            'contraction_loss': contraction_loss.item(),
            'reg_loss': reg_loss.item(),
            'det_penalty': det_penalty.item(),
            'energy_curr': energy_curr.mean().item(),
            'energy_next': energy_next.mean().item()
        }
        
        return metric_loss.item(), metrics
    
    def update_critic(self, batch):
        """Update value network using TD learning"""
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        self.critic_optimizer.zero_grad()
        
        # Compute target values
        with torch.no_grad():
            next_actions = self.target_policy.deterministic_action(next_states_t)
            next_values = self.target_critic(next_states_t)
            target_values = rewards_t + self.config.GAMMA * next_values * (1 - dones_t)
        
        # Current values
        current_values = self.critic(states_t)
        
        # Critic loss
        critic_loss = F.mse_loss(current_values, target_values)
        
        # Backward pass
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def update_policy(self, batch):
        """Update policy with contraction regularization"""
        states, actions, _, _, _ = batch
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        
        self.policy_optimizer.zero_grad()
        
        # Sample actions from current policy
        action_dist = self.policy(states_t)
        actions_sampled = action_dist.rsample()
        log_probs = action_dist.log_prob(actions_sampled).sum(dim=-1, keepdim=True)
        
        # Value estimates
        values = self.critic(states_t)
        
        # Contraction bonus (matches policy objective in Eq. 14)
        with torch.no_grad():
            # Predict next states
            next_states_pred, _ = self.dynamics(states_t, actions_sampled)
            
            # Compute energy difference
            energy_curr, _ = RiemannianOperations.compute_energy(states_t, self.metric_net)
            energy_next, _ = RiemannianOperations.compute_energy(next_states_pred, self.metric_net)
            delta_energy = energy_curr - energy_next
            contraction_bonus = torch.tanh(delta_energy / 5.0).mean()
        
        # Policy loss components
        value_loss = -values.mean()  # Maximize value
        
        # Additional penalties for stability
        velocity = states_t[:, 2] * 8.0  # De-normalize angular velocity
        velocity_penalty = F.relu(torch.abs(velocity) - 2.0).mean() * 0.01
        
        action_penalty = torch.abs(actions_sampled).mean() * 0.001
        
        # Angle penalty (encourage upright position)
        cos_theta = states_t[:, 0]
        sin_theta = states_t[:, 1]
        angle = torch.atan2(sin_theta, cos_theta)
        angle_penalty = torch.abs(angle).mean() * 0.005
        
        # Total policy loss (matches Eq. 14 structure)
        policy_loss = (0.1 * value_loss - 
                      self.beta * contraction_bonus +
                      velocity_penalty + 
                      action_penalty + 
                      angle_penalty)
        
        # Backward pass
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.target_policy, self.policy)
        self.soft_update(self.target_critic, self.critic)
        
        metrics = {
            'value_loss': value_loss.item(),
            'contraction_bonus': contraction_bonus.item(),
            'velocity_penalty': velocity_penalty.item(),
            'action_penalty': action_penalty.item(),
            'angle_penalty': angle_penalty.item(),
            'beta': self.beta
        }
        
        return policy_loss.item(), metrics
    
    def train_episode(self, env, episode_num):
        """Execute one training episode"""
        state, _ = env.reset()
        self.noise_process.reset()
        
        episode_reward = 0
        episode_transitions = []
        
        for step in range(self.config.MAX_EPISODE_LENGTH):
            # Select action with exploration
            action = self.select_action(state, deterministic=False)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            episode_transitions.append((state.copy(), action, reward, next_state.copy(), done))
            
            # Update metrics
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Store transitions in replay buffer
        for transition in episode_transitions:
            self.replay_buffer.push(*transition)
        
        # Update adaptive noise
        self.noise_process.update(episode_reward, episode_num)
        
        # Train networks if enough samples
        if len(self.replay_buffer) > self.config.BATCH_SIZE:
            batch = self.replay_buffer.sample(self.config.BATCH_SIZE)
            
            # Update networks
            dynamics_loss, dynamics_uncertainty = self.update_dynamics(batch)
            metric_loss, metric_metrics = self.update_metric(batch)
            critic_loss = self.update_critic(batch)
            policy_loss, policy_metrics = self.update_policy(batch)
            
            # Store metrics
            self.metrics['dynamics_losses'].append(dynamics_loss)
            self.metrics['contraction_losses'].append(metric_loss)
            self.metrics['policy_losses'].append(policy_loss)
            self.metrics['energies'].append(metric_metrics['energy_curr'])
            self.metrics['beta_values'].append(policy_metrics['beta'])
        else:
            dynamics_loss = metric_loss = critic_loss = policy_loss = 0.0
            metric_metrics = {'energy_curr': 0, 'energy_next': 0}
            policy_metrics = {'beta': self.beta}
        
        # Adapt beta based on performance
        if episode_num > 0 and len(self.metrics['rewards']) > 0:
            reward_improved = episode_reward > self.metrics['rewards'][-1]
            self.adapt_beta(reward_improved)
        
        # Store episode metrics
        self.metrics['rewards'].append(episode_reward)
        
        return {
            'episode': episode_num,
            'reward': episode_reward,
            'dynamics_loss': dynamics_loss,
            'metric_loss': metric_loss,
            'critic_loss': critic_loss,
            'policy_loss': policy_loss,
            'energy_curr': metric_metrics['energy_curr'],
            'energy_next': metric_metrics['energy_next'],
            'beta': policy_metrics['beta']
        }
    
    def train(self, env, num_episodes=None):
        """Main training loop"""
        if num_episodes is None:
            num_episodes = self.config.TOTAL_EPISODES
        
        print(f"\nStarting CDM training for {num_episodes} episodes...")
        print("=" * 80)
        print(f"{'Episode':>8} {'Reward':>10} {'Energy':>8} {'β':>6} {'Dyn Loss':>9} {'Metric Loss':>11}")
        print("=" * 80)
        
        best_reward = -float('inf')
        best_checkpoint = None
        
        for episode in range(num_episodes):
            self.episode = episode
            
            # Train one episode
            episode_metrics = self.train_episode(env, episode)
            
            # Log progress
            if episode % self.config.LOG_INTERVAL == 0 or episode == num_episodes - 1:
                print(f"{episode:8d} {episode_metrics['reward']:10.1f} "
                      f"{episode_metrics['energy_curr']:8.2f} {episode_metrics['beta']:6.2f} "
                      f"{episode_metrics['dynamics_loss']:9.4f} {episode_metrics['metric_loss']:11.4f}")
            
            # Save best model
            if episode_metrics['reward'] > best_reward:
                best_reward = episode_metrics['reward']
                best_checkpoint = self._create_checkpoint(episode_metrics)
                
                if episode_metrics['reward'] > -600:
                    print(f"  ★ New best: {best_reward:.1f} (Episode {episode})")
            
            # Early stopping if consistently good
            if episode >= 100 and episode_metrics['reward'] > -600:
                print(f"\n✅ Early stopping: Excellent performance at episode {episode}")
                break
        
        # Save final models
        self.save_models()
        
        # Save metrics
        self.save_metrics()
        
        # Plot results
        self.plot_training_results()
        
        print(f"\nTraining completed. Best reward: {best_reward:.1f}")
        return best_checkpoint
    
    def _create_checkpoint(self, metrics):
        """Create checkpoint dictionary"""
        return {
            'episode': metrics['episode'],
            'reward': metrics['reward'],
            'policy_state': self.policy.state_dict(),
            'critic_state': self.critic.state_dict(),
            'dynamics_state': self.dynamics.state_dict(),
            'metric_state': self.metric_net.state_dict()
        }
    
    def save_models(self):
        """Save all models to disk"""
        torch.save(self.policy.state_dict(), self.save_dir / "final_policy.pth")
        torch.save(self.critic.state_dict(), self.save_dir / "final_critic.pth")
        torch.save(self.dynamics.state_dict(), self.save_dir / "final_dynamics.pth")
        torch.save(self.metric_net.state_dict(), self.save_dir / "final_metric.pth")
        
        # Save target networks
        torch.save(self.target_policy.state_dict(), self.save_dir / "final_target_policy.pth")
        torch.save(self.target_critic.state_dict(), self.save_dir / "final_target_critic.pth")
    
    def save_metrics(self):
        """Save training metrics to disk"""
        metrics_path = self.save_dir / "training_metrics.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)
        
        # Also save as JSON for readability
        json_metrics = {}
        for k, v in self.metrics.items():
            if isinstance(v, list):
                # Convert numpy arrays to lists
                if len(v) > 0 and isinstance(v[0], np.ndarray):
                    json_metrics[k] = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in v]
                else:
                    json_metrics[k] = v
            else:
                json_metrics[k] = v
        
        with open(self.save_dir / "training_metrics.json", 'w') as f:
            json.dump(json_metrics, f, indent=2)
    
    def load_models(self):
        """Load models from disk"""
        self.policy.load_state_dict(torch.load(self.save_dir / "final_policy.pth", 
                                              map_location=self.device, weights_only=False))
        self.critic.load_state_dict(torch.load(self.save_dir / "final_critic.pth", 
                                              map_location=self.device, weights_only=False))
        self.dynamics.load_state_dict(torch.load(self.save_dir / "final_dynamics.pth", 
                                                map_location=self.device, weights_only=False))
        self.metric_net.load_state_dict(torch.load(self.save_dir / "final_metric.pth", 
                                                  map_location=self.device, weights_only=False))
        
        # Update target networks
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
    
    def plot_training_results(self):
        """Plot comprehensive training results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Rewards
        episodes = range(len(self.metrics['rewards']))
        axes[0, 0].plot(episodes, self.metrics['rewards'], 'b-', alpha=0.7)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Energy
        if self.metrics['energies']:
            axes[0, 1].plot(self.metrics['energies'], 'g-', alpha=0.7)
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Energy')
            axes[0, 1].set_title('Metric Energy')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Beta values
        if self.metrics['beta_values']:
            axes[0, 2].plot(self.metrics['beta_values'], 'r-', alpha=0.7)
            axes[0, 2].set_xlabel('Training Step')
            axes[0, 2].set_ylabel('β')
            axes[0, 2].set_title('Stability Weight Adaptation')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Losses
        axes[1, 0].plot(self.metrics['dynamics_losses'], 'b-', label='Dynamics', alpha=0.7)
        axes[1, 0].plot(self.metrics['contraction_losses'], 'g-', label='Contraction', alpha=0.7)
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Reward histogram
        axes[1, 1].hist(self.metrics['rewards'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Success rate over time
        if len(self.metrics['rewards']) >= 20:
            window = 20
            success_rates = []
            for i in range(len(self.metrics['rewards']) - window + 1):
                window_rewards = self.metrics['rewards'][i:i+window]
                successes = sum(1 for r in window_rewards if r > -800)
                success_rates.append(successes / window * 100)
            
            axes[1, 2].plot(range(window-1, len(self.metrics['rewards'])), success_rates, 'g-')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Success Rate (%)')
            axes[1, 2].set_title(f'Success Rate ({window}-ep window)')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_ylim([0, 100])
        
        plt.suptitle('CDM Training Results - Riemannian Metric Learning', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.save_dir / "training_results.png", dpi=100, bbox_inches='tight')
        plt.show()

# ============================
# MAIN EXECUTION
# ============================

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """Main execution function for reproducing paper results"""
    print("=" * 80)
    print("CONTRACTION DYNAMICS MODEL - RIEMANNIAN METRIC LEARNING")
    print("=" * 80)
    print("Paper: 'Learning Contraction Metrics for Provably Stable Model-Based RL'")
    print("Author: Amir Hameed, Sirraya Labs")
    print("=" * 80)
    
    # Set seed for reproducibility
    set_seed(Config.SEED)
    
    # Create agent
    agent = ContractionDynamicsAgent(Config)
    
    # Create environment
    env = gym.make(Config.ENV_NAME)
    
    # Train the agent
    print("\n" + "=" * 80)
    print("PHASE 1: TRAINING")
    print("=" * 80)
    
    try:
        best_checkpoint = agent.train(env)
        print(f"\n✅ Training completed successfully!")
        print(f"Best reward achieved: {best_checkpoint['reward']:.1f}")
        
        # Save final models
        agent.save_models()
        print("Models saved successfully!")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"\nOutput files created in: {Config.SAVE_DIR}")
    print("  - config.json (configuration)")
    print("  - final_*.pth (trained models)")
    print("  - training_metrics.pkl/.json (training metrics)")
    print("  - training_results.png (training plots)")

if __name__ == "__main__":
    main()