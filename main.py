"""
Contraction Dynamics Model (CDM): Riemannian Metric Learning for Stable MBRL
Author: Amir Hameed, Sirraya Labs
Paper: "Learning Contraction Metrics for Provably Stable Model-Based RL"

ROBUST IMPLEMENTATION: Complete reproducible code with enhanced stability, 
performance optimization, and comprehensive testing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque, defaultdict
import random
import matplotlib.pyplot as plt
import pickle
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from typing import Tuple, Dict, List, Optional, Any
import time
from dataclasses import dataclass, asdict

# ============================
# ENHANCED CONFIGURATION
# ============================

@dataclass
class Config:
    """Enhanced configuration with robust defaults"""
    # Environment
    ENV_NAME: str = "Pendulum-v1"
    STATE_DIM: int = 3
    ACTION_DIM: int = 1
    MAX_EPISODE_LENGTH: int = 200
    
    # Enhanced Network Architecture
    DYNAMICS_HIDDEN_DIM: int = 128
    POLICY_HIDDEN_DIM: int = 256
    METRIC_HIDDEN_DIM: int = 128
    CRITIC_HIDDEN_DIM: int = 256
    
    # Training Parameters (optimized)
    TOTAL_EPISODES: int = 200
    BATCH_SIZE: int = 256
    GAMMA: float = 0.99
    TAU: float = 0.005
    
    # Enhanced Contraction Parameters
    CONTRACTION_RATE_ALPHA: float = 0.85  # Less strict for better learning
    INITIAL_BETA: float = 0.3  # Higher initial stability focus
    BETA_MIN: float = 0.05
    BETA_MAX: float = 2.0
    METRIC_REGULARIZATION: float = 0.001  # Reduced for less constraint
    EPSILON_METRIC: float = 0.05  # Smaller epsilon
    PERTURBATION_SIGMA: float = 0.02  # Slightly larger for better exploration
    
    # Optimized Learning Rates
    ACTOR_LR: float = 3e-4
    CRITIC_LR: float = 3e-4
    DYNAMICS_LR: float = 1e-3
    METRIC_LR: float = 5e-5  # Slower learning for stability
    
    # Enhanced Optimization
    REPLAY_BUFFER_SIZE: int = 100000
    INITIAL_EXPLORATION_STEPS: int = 5000
    UPDATE_FREQUENCY: int = 1
    ENSEMBLE_SIZE: int = 7  # Larger ensemble for better uncertainty
    
    # Learning Schedule
    LEARNING_START: int = 1000
    UPDATE_EVERY: int = 50
    GRADIENT_STEPS: int = 40
    TARGET_UPDATE_INTERVAL: int = 1
    
    # Adaptive Parameters
    BETA_DECAY: float = 0.995
    BETA_INCREASE: float = 1.02
    NOISE_DECAY: float = 0.999
    MIN_NOISE: float = 0.1
    
    # Experimental Settings
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR: str = "cdm_robust_results"
    LOG_INTERVAL: int = 5
    EVAL_INTERVAL: int = 20
    EVAL_EPISODES: int = 5
    PLOT_RESULTS: bool = True
    
    # Early Stopping
    EARLY_STOP_REWARD: float = -300.0
    EARLY_STOP_PATIENCE: int = 50
    
    # Normalization
    REWARD_SCALE: float = 0.1
    STATE_NORMALIZATION: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        assert 0 < self.CONTRACTION_RATE_ALPHA < 1
        assert self.BATCH_SIZE <= self.REPLAY_BUFFER_SIZE
        assert self.ENSEMBLE_SIZE >= 3
    
    def save(self, path: Path):
        """Save configuration to file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        """Load configuration from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

# ============================
# ENHANCED NETWORK ARCHITECTURES
# ============================

class EnhancedDynamics(nn.Module):
    """Enhanced dynamics model with residual connections"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Enhanced architecture with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.output = nn.Linear(hidden_dim, state_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Ensure proper dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        x = torch.cat([state, action], dim=-1)
        x = self.encoder(x)
        
        # Residual block 1
        residual = x
        x = self.res_block1(x)
        x = F.relu(x + residual)
        
        # Residual block 2
        residual = x
        x = self.res_block2(x)
        x = F.relu(x + residual)
        
        return self.output(x)

class DynamicsEnsemble(nn.Module):
    """Enhanced ensemble with uncertainty quantification"""
    def __init__(self, state_dim: int, action_dim: int, 
                 ensemble_size: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.ensemble_size = ensemble_size
        
        # Create ensemble of enhanced dynamics models
        self.models = nn.ModuleList([
            EnhancedDynamics(state_dim, action_dim, hidden_dim)
            for _ in range(ensemble_size)
        ])
        
        # Learnable weights for each model
        self.model_weights = nn.Parameter(torch.ones(ensemble_size) / ensemble_size)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = []
        for model in self.models:
            pred = model(state, action)
            predictions.append(pred.unsqueeze(0))
        
        predictions = torch.cat(predictions, dim=0)  # [ensemble, batch, state_dim]
        
        # Weighted mean using learned weights
        weights = F.softmax(self.model_weights, dim=0)
        weighted_predictions = predictions * weights.view(-1, 1, 1)
        mean = weighted_predictions.sum(dim=0)
        
        # Compute uncertainty (aleatoric + epistemic)
        variance = torch.var(predictions, dim=0, unbiased=True)
        epistemic = variance.mean(dim=-1, keepdim=True)
        
        return mean, epistemic
    
    def sample_model(self) -> nn.Module:
        """Sample a random model from ensemble for exploration"""
        return random.choice(self.models)

class EnhancedPolicyNetwork(nn.Module):
    """Enhanced policy network with adaptive exploration"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        
        # Enhanced mean network
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Learnable log standard deviation with initialization
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)
        self.action_scale = 2.0
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean_net(state) * self.action_scale
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)
    
    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get deterministic action for deployment"""
        with torch.no_grad():
            return self.mean_net(state) * self.action_scale
    
    def sample_with_entropy(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action with entropy"""
        dist = self(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

class EnhancedValueNetwork(nn.Module):
    """Enhanced critic network with double Q-learning"""
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1 Network
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 Network
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self.q1_net(state)
        q2 = self.q2_net(state)
        return q1, q2
    
    def q_min(self, state: torch.Tensor) -> torch.Tensor:
        """Minimum Q-value for conservative updates"""
        q1, q2 = self(state)
        return torch.min(q1, q2)

class RobustContractionMetric(nn.Module):
    """
    Robust Riemannian metric with enhanced numerical stability
    M(x) = L(x)L(x)^T + εI
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128, epsilon: float = 0.05):
        super().__init__()
        self.state_dim = state_dim
        self.epsilon = epsilon
        
        # Number of parameters for lower triangular matrix
        self.output_dim = (state_dim * (state_dim + 1)) // 2
        
        # Enhanced network architecture
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
        # Softplus for diagonal entries
        self.softplus = nn.Softplus(beta=1.0, threshold=20)
        
        # For numerical stability
        self.diagonal_offset = 0.01
        self.off_diagonal_scale = 0.1
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.1)  # Small gain for stability
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        
        # Get raw parameters
        l_params = self.net(x)
        
        # Build lower triangular matrix L with enhanced stability
        L = torch.zeros(batch_size, self.state_dim, self.state_dim, 
                       device=x.device, dtype=x.dtype)
        
        idx = 0
        for i in range(self.state_dim):
            for j in range(i + 1):
                val = l_params[:, idx]
                if i == j:
                    # Diagonal: positive with lower bound
                    L[:, i, j] = self.softplus(val) + self.diagonal_offset
                else:
                    # Off-diagonal: bounded for stability
                    L[:, i, j] = torch.tanh(val) * self.off_diagonal_scale
                idx += 1
        
        # Compute M = LL^T + εI with numerical safeguards
        LLT = torch.bmm(L, L.transpose(1, 2))
        identity = torch.eye(self.state_dim, device=x.device, dtype=x.dtype)
        identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
        
        M = LLT + self.epsilon * identity
        
        # Ensure positive definiteness via Cholesky decomposition
        try:
            torch.linalg.cholesky(M)  # Just check, don't store
        except RuntimeError:
            # Add damping if not positive definite
            M = M + 0.1 * identity
        
        return M, L
    
    def compute_metrics(self, M: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute additional metrics for monitoring"""
        batch_size = M.shape[0]
        metrics = {}
        
        # Eigenvalues for condition number
        eigenvalues = torch.linalg.eigvalsh(M)  # Real symmetric
        
        # Extract min and max eigenvalues across batch
        min_eigenvalues = eigenvalues.min(dim=1).values  # [batch]
        max_eigenvalues = eigenvalues.max(dim=1).values  # [batch]
        
        metrics['min_eigenvalue'] = min_eigenvalues.mean()
        metrics['max_eigenvalue'] = max_eigenvalues.mean()
        metrics['condition_number'] = (max_eigenvalues / torch.clamp(min_eigenvalues, min=1e-6)).mean()
        
        # Determinant
        metrics['det'] = torch.det(M).mean()
        
        return metrics

# ============================
# ENHANCED RIEMANNIAN OPERATIONS
# ============================

class EnhancedRiemannianOperations:
    """Enhanced Riemannian operations with numerical stability"""
    
    @staticmethod
    def compute_energy(state: torch.Tensor, metric_net: RobustContractionMetric) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute energy E = x^T M(x) x with numerical stability
        """
        M, _ = metric_net(state)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        if state.dim() == 2:
            # Batch mode: x^T M x
            state_expanded = state.unsqueeze(1)  # [batch, 1, state_dim]
            energy = torch.bmm(torch.bmm(state_expanded, M), 
                             state_expanded.transpose(1, 2)).squeeze()
        else:
            # Single sample
            state_expanded = state.unsqueeze(0).unsqueeze(0)
            energy = torch.bmm(torch.bmm(state_expanded, M), 
                             state_expanded.transpose(1, 2)).squeeze()
        
        # Clip to avoid numerical issues
        energy = torch.clamp(energy, min=1e-6, max=1e6)
        
        return energy, M
    
    @staticmethod
    def compute_contraction_loss(
        states: torch.Tensor, 
        next_states: torch.Tensor,
        metric_net: RobustContractionMetric,
        alpha: float = 0.85,
        beta: float = 1.0  # Temperature parameter
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Enhanced contraction loss with smooth penalty
        """
        energy_curr, M_curr = EnhancedRiemannianOperations.compute_energy(states, metric_net)
        energy_next, M_next = EnhancedRiemannianOperations.compute_energy(next_states, metric_net)
        
        # Smooth contraction loss (softplus instead of ReLU)
        energy_diff = energy_next - (alpha**2) * energy_curr
        contraction_loss = F.softplus(beta * energy_diff).mean() / beta
        
        # Symmetry and positive definiteness losses
        symmetry_loss = torch.norm(M_curr - M_curr.transpose(1, 2), dim=(1, 2)).mean()
        
        # Metric near-identity regularization
        identity = torch.eye(states.shape[-1], device=states.device)
        identity = identity.unsqueeze(0).expand_as(M_curr)
        identity_loss = torch.norm(M_curr - identity, dim=(1, 2)).mean()
        
        # REMOVED gradient penalty for now - it's causing dimension issues
        
        smoothness_loss = torch.tensor(0.0, device=states.device)
        
        metrics = {
            'energy_curr': energy_curr.mean(),
            'energy_next': energy_next.mean(),
            'energy_diff': energy_diff.mean(),
            'symmetry_loss': symmetry_loss,
            'identity_loss': identity_loss,
            'smoothness_loss': smoothness_loss
        }
        
        total_loss = (contraction_loss + 
                     0.01 * symmetry_loss + 
                     0.001 * identity_loss)
        
        return total_loss, metrics
    
    @staticmethod
    def generate_displacements(
        states: torch.Tensor, 
        num_displacements: int = 3,
        sigma_min: float = 0.01,
        sigma_max: float = 0.1
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate multiple virtual displacements at different scales
        """
        perturbed_states_list = []
        perturbations_list = []
        
        for i in range(num_displacements):
            # Multi-scale perturbations
            sigma = sigma_min + (sigma_max - sigma_min) * (i / max(num_displacements - 1, 1))
            perturbations = torch.randn_like(states) * sigma
            perturbed_states = states + perturbations
            
            perturbed_states_list.append(perturbed_states)
            perturbations_list.append(perturbations)
        
        return perturbed_states_list, perturbations_list

# ============================
# ENHANCED REPLAY BUFFER
# ============================

class PrioritizedReplayBuffer:
    """Prioritized experience replay with proportional prioritization"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # For sampling efficiency
        self._max_priority = 1.0
    
    def push(self, state: np.ndarray, action: np.ndarray, 
             reward: float, next_state: np.ndarray, done: bool):
        """Add experience with max priority"""
        experience = (state, action, reward, next_state, done)
        
        if self.size < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # New experiences get max priority
        self.priorities[self.position] = self._max_priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with priorities"""
        if self.size < batch_size:
            return None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32)
        )
    
    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, errors):
            # Add small constant to avoid zero priority
            priority = (abs(error) + 1e-5) ** self.alpha
            self.priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)
    
    def __len__(self):
        return self.size
    
    def save(self, path: Path):
        """Save buffer to disk"""
        data = {
            'buffer': self.buffer,
            'priorities': self.priorities[:self.size],
            'position': self.position,
            'size': self.size,
            'max_priority': self._max_priority
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: Path):
        """Load buffer from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.buffer = data['buffer']
        self.priorities[:len(data['priorities'])] = data['priorities']
        self.position = data['position']
        self.size = data['size']
        self._max_priority = data['max_priority']

# ============================
# ENHANCED EXPLORATION STRATEGY
# ============================

class AdaptiveExploration:
    """Enhanced adaptive exploration with multiple strategies"""
    
    def __init__(self, action_dim: int, config: Config):
        self.action_dim = action_dim
        self.config = config
        
        # Multiple exploration strategies
        self.ou_noise = self._init_ou_noise()
        self.gaussian_noise = self._init_gaussian_noise()
        self.parameter_noise = self._init_parameter_noise()
        
        # Adaptive parameters
        self.sigma = 0.3
        self.theta = 0.15
        self.state = np.zeros(action_dim)
        
        # Strategy selection
        self.strategies = ['ou', 'gaussian', 'parameter']
        self.strategy_weights = np.ones(3) / 3
        self.strategy_success = np.ones(3)
        
        # Tracking
        self.recent_rewards = deque(maxlen=50)
        self.exploration_rate = 1.0
    
    def _init_ou_noise(self):
        """Ornstein-Uhlenbeck process for temporally correlated noise"""
        return {
            'theta': 0.15,
            'mu': 0.0,
            'sigma': 0.3,
            'dt': 1e-2,
            'state': np.zeros(self.action_dim)
        }
    
    def _init_gaussian_noise(self):
        """Simple Gaussian noise"""
        return {
            'sigma': 0.3,
            'decay': 0.999
        }
    
    def _init_parameter_noise(self):
        """Parameter noise for deep exploration"""
        return {
            'scale': 0.1,
            'adaptation_rate': 1.01
        }
    
    def sample(self, strategy: str = None) -> np.ndarray:
        """Sample noise from selected strategy"""
        if strategy is None:
            # Select strategy based on success rates
            probs = self.strategy_weights * self.strategy_success
            probs = probs / probs.sum()
            strategy_idx = np.random.choice(len(self.strategies), p=probs)
            strategy = self.strategies[strategy_idx]
        
        if strategy == 'ou':
            # Ornstein-Uhlenbeck noise
            ou = self.ou_noise
            dx = ou['theta'] * (ou['mu'] - ou['state']) * ou['dt']
            dx += ou['sigma'] * np.sqrt(ou['dt']) * np.random.randn(self.action_dim)
            ou['state'] += dx
            noise = ou['state'].copy()
            
        elif strategy == 'gaussian':
            # Gaussian noise with decay
            gaussian = self.gaussian_noise
            noise = np.random.randn(self.action_dim) * gaussian['sigma']
            gaussian['sigma'] *= gaussian['decay']
            gaussian['sigma'] = max(gaussian['sigma'], 0.05)
            
        elif strategy == 'parameter':
            # Parameter noise (scaled by exploration rate)
            noise = np.random.randn(self.action_dim) * self.parameter_noise['scale']
            noise *= self.exploration_rate
            
        else:
            noise = np.zeros(self.action_dim)
        
        # Scale by current exploration rate
        noise *= self.exploration_rate
        
        return noise
    
    def update(self, episode_reward: float, episode: int):
        """Update exploration parameters based on performance"""
        self.recent_rewards.append(episode_reward)
        
        if len(self.recent_rewards) >= 10:
            # Update strategy weights based on performance
            recent_avg = np.mean(list(self.recent_rewards))
            
            # Adjust exploration rate
            if recent_avg > -500:  # Good performance
                self.exploration_rate *= 0.98
            elif recent_avg > -800:  # Moderate performance
                self.exploration_rate *= 0.995
            else:  # Poor performance
                self.exploration_rate = min(1.0, self.exploration_rate * 1.02)
            
            # Keep exploration rate in reasonable bounds
            self.exploration_rate = max(0.1, min(1.0, self.exploration_rate))
            
            # Adjust parameter noise scale
            if recent_avg < -1000:
                self.parameter_noise['scale'] = min(0.5, self.parameter_noise['scale'] * 1.05)
            else:
                self.parameter_noise['scale'] *= 0.99
        
        # Episode-based decay
        self.exploration_rate *= self.config.NOISE_DECAY
        self.exploration_rate = max(self.config.MIN_NOISE, self.exploration_rate)
    
    def reset(self):
        """Reset noise processes"""
        self.ou_noise['state'] = np.zeros(self.action_dim)
        self.state = np.zeros(self.action_dim)

# ============================
# ENHANCED CDM AGENT
# ============================

class RobustContractionDynamicsAgent:
    """
    Robust CDM agent with enhanced stability and performance
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize networks
        self._initialize_networks()
        
        # Initialize optimizers with learning rate scheduling
        self._initialize_optimizers()
        
        # Initialize replay buffer with prioritization
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.REPLAY_BUFFER_SIZE,
            alpha=0.6,
            beta=0.4
        )
        
        # Enhanced exploration
        self.exploration = AdaptiveExploration(config.ACTION_DIM, config)
        
        # Adaptive stability weight
        self.beta = config.INITIAL_BETA
        self.beta_history = []
        
        # Normalization
        self.state_mean = np.zeros(config.STATE_DIM)
        self.state_std = np.ones(config.STATE_DIM)
        self.reward_scale = config.REWARD_SCALE
        
        # Tracking and metrics
        self.metrics = {
            'episode_rewards': [],
            'eval_rewards': [],
            'dynamics_losses': [],
            'metric_losses': [],
            'critic_losses': [],
            'actor_losses': [],
            'energies': [],
            'betas': [],
            'exploration_rates': [],
            'grad_norms': defaultdict(list)
        }
        
        # Create save directory
        self.save_dir = Path(config.SAVE_DIR)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save config
        config.save(self.save_dir / "config.json")
        
        print(f"Robust CDM Agent initialized on {self.device}")
        print(f"Save directory: {self.save_dir}")
        print(f"Network parameters:")
        print(f"  Dynamics: {sum(p.numel() for p in self.dynamics.parameters()):,}")
        print(f"  Policy: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"  Critic: {sum(p.numel() for p in self.critic.parameters()):,}")
        print(f"  Metric: {sum(p.numel() for p in self.metric_net.parameters()):,}")
    
    def _initialize_networks(self):
        """Initialize all enhanced networks"""
        # Dynamics ensemble
        self.dynamics = DynamicsEnsemble(
            self.config.STATE_DIM,
            self.config.ACTION_DIM,
            self.config.ENSEMBLE_SIZE,
            self.config.DYNAMICS_HIDDEN_DIM
        ).to(self.device)
        
        # Policy network
        self.policy = EnhancedPolicyNetwork(
            self.config.STATE_DIM,
            self.config.ACTION_DIM,
            self.config.POLICY_HIDDEN_DIM
        ).to(self.device)
        
        # Critic network (double Q)
        self.critic = EnhancedValueNetwork(
            self.config.STATE_DIM,
            self.config.CRITIC_HIDDEN_DIM
        ).to(self.device)
        self.target_critic = EnhancedValueNetwork(
            self.config.STATE_DIM,
            self.config.CRITIC_HIDDEN_DIM
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Contraction metric network
        self.metric_net = RobustContractionMetric(
            self.config.STATE_DIM,
            self.config.METRIC_HIDDEN_DIM,
            epsilon=self.config.EPSILON_METRIC
        ).to(self.device)
        
        # Target networks
        self.target_policy = EnhancedPolicyNetwork(
            self.config.STATE_DIM,
            self.config.ACTION_DIM,
            self.config.POLICY_HIDDEN_DIM
        ).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
    
    def _initialize_optimizers(self):
        """Initialize optimizers with weight decay"""
        self.dynamics_optimizer = optim.AdamW(
            self.dynamics.parameters(),
            lr=self.config.DYNAMICS_LR,
            weight_decay=1e-4
        )
        
        self.policy_optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=self.config.ACTOR_LR,
            weight_decay=1e-4
        )
        
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(),
            lr=self.config.CRITIC_LR,
            weight_decay=1e-4
        )
        
        self.metric_optimizer = optim.AdamW(
            self.metric_net.parameters(),
            lr=self.config.METRIC_LR,
            weight_decay=1e-4
        )
        
        # Learning rate schedulers
        self.dynamics_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.dynamics_optimizer,
            T_max=self.config.TOTAL_EPISODES
        )
        
        self.policy_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer,
            T_max=self.config.TOTAL_EPISODES
        )
    
    def soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target networks"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.TAU) +
                source_param.data * self.config.TAU
            )
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state if enabled"""
        if self.config.STATE_NORMALIZATION:
            return (state - self.state_mean) / (self.state_std + 1e-8)
        return state
    
    def update_normalization(self, states: np.ndarray):
        """Update normalization statistics"""
        if self.config.STATE_NORMALIZATION:
            self.state_mean = 0.9 * self.state_mean + 0.1 * states.mean(axis=0)
            self.state_std = 0.9 * self.state_std + 0.1 * states.std(axis=0)
            self.state_std = np.maximum(self.state_std, 1e-8)
    
    def adapt_beta(self, reward_improvement: bool):
        """Adapt stability weight β"""
        if reward_improvement:
            # Decrease stability focus when improving
            self.beta *= self.config.BETA_DECAY
        else:
            # Increase stability focus when not improving
            self.beta *= self.config.BETA_INCREASE
        
        # Clip to bounds
        self.beta = np.clip(self.beta, self.config.BETA_MIN, self.config.BETA_MAX)
        self.beta_history.append(self.beta)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False,
                     use_exploration: bool = True) -> np.ndarray:
        """Select action with enhanced exploration"""
        state_normalized = self.normalize_state(state)
        state_tensor = torch.FloatTensor(state_normalized).unsqueeze(0).to(self.device)
        
        if deterministic or not use_exploration:
            with torch.no_grad():
                action = self.policy.deterministic_action(state_tensor)
            return action.cpu().numpy()[0]
        else:
            with torch.no_grad():
                action_dist = self.policy(state_tensor)
                action = action_dist.sample()
                action_np = action.cpu().numpy()[0]
            
            # Add exploration noise
            if use_exploration:
                noise = self.exploration.sample()
                action_np += noise
            
            # Clip to valid range
            action_np = np.clip(action_np, -self.policy.action_scale, self.policy.action_scale)
            
            return action_np
    
    def update_dynamics(self, batch: Tuple, step: int) -> Tuple[float, Dict]:
        """Update dynamics model with uncertainty weighting"""
        states, actions, _, next_states, _, indices, weights = batch
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)
        
        self.dynamics_optimizer.zero_grad()
        
        # Forward pass
        next_states_pred, uncertainty = self.dynamics(states_t, actions_t)
        
        # Weighted MSE loss
        mse_loss = F.mse_loss(next_states_pred, next_states_t, reduction='none')
        weighted_loss = (mse_loss * weights_t.unsqueeze(1)).mean()
        
        # Uncertainty regularization
        uncertainty_loss = uncertainty.mean() * 0.01
        
        # Total loss
        dynamics_loss = weighted_loss + uncertainty_loss
        
        # Backward pass with gradient clipping
        dynamics_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
        
        self.dynamics_optimizer.step()
        
        # Update priorities based on prediction error
        with torch.no_grad():
            errors = mse_loss.mean(dim=1).cpu().numpy()
            self.replay_buffer.update_priorities(indices, errors)
        
        metrics = {
            'dynamics_loss': dynamics_loss.item(),
            'uncertainty': uncertainty.mean().item(),
            'grad_norm': grad_norm.item()
        }
        
        return dynamics_loss.item(), metrics
    
    def update_metric(self, batch: Tuple, step: int) -> Tuple[float, Dict]:
        """Update contraction metric with enhanced stability"""
        states, actions, _, _, _, _, weights = batch
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)
        
        self.metric_optimizer.zero_grad()
        
        # Generate multiple virtual displacements
        perturbed_states_list, _ = EnhancedRiemannianOperations.generate_displacements(
            states_t, num_displacements=3
        )
        
        # Predict next states using current policy
        with torch.no_grad():
            action_dist = self.policy(states_t)
            actions_sampled = action_dist.rsample()
            next_states_pred, _ = self.dynamics(states_t, actions_sampled)
            
            # Also predict for perturbed states
            next_perturbed_preds = []
            for perturbed_states in perturbed_states_list:
                next_pred, _ = self.dynamics(perturbed_states, actions_sampled)
                next_perturbed_preds.append(next_pred)
        
        # Compute contraction loss with temperature scheduling
        temperature = max(0.1, 1.0 - step / 10000)  # Anneal temperature
        metric_loss, metric_metrics = EnhancedRiemannianOperations.compute_contraction_loss(
            states_t, next_states_pred, self.metric_net,
            alpha=self.config.CONTRACTION_RATE_ALPHA,
            beta=temperature
        )
        
        # Weight the loss
        weighted_loss = metric_loss * weights_t.mean()
        
        # Additional consistency loss for perturbed trajectories
        consistency_loss = 0.0
        for next_perturbed in next_perturbed_preds:
            _, metrics_perturbed = EnhancedRiemannianOperations.compute_contraction_loss(
                states_t, next_perturbed, self.metric_net,
                alpha=self.config.CONTRACTION_RATE_ALPHA
            )
            consistency_loss += metrics_perturbed['energy_diff'].abs().mean()
        
        consistency_loss = consistency_loss / len(next_perturbed_preds) * 0.1
        
        # Total loss
        total_loss = weighted_loss + consistency_loss
        
        # Backward pass
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.metric_net.parameters(), 0.5)
        
        self.metric_optimizer.step()
        
        # Additional metrics from metric network
        with torch.no_grad():
            M, _ = self.metric_net(states_t)
            metric_net_metrics = self.metric_net.compute_metrics(M)
        
        metrics = {
            'metric_loss': metric_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'energy_curr': metric_metrics['energy_curr'].item(),
            'energy_next': metric_metrics['energy_next'].item(),
            'energy_diff': metric_metrics['energy_diff'].item(),
            'grad_norm': grad_norm.item(),
        }
        
        # Add metric network metrics safely
        for k, v in metric_net_metrics.items():
            if torch.is_tensor(v):
                if v.numel() == 1:  # Scalar tensor
                    metrics[k] = v.item()
                else:
                    # For batched metrics, take mean
                    metrics[k] = v.mean().item()
            else:
                metrics[k] = v
        
        return total_loss.item(), metrics
    
    def update_critic(self, batch: Tuple, step: int) -> Tuple[float, Dict]:
        """Update critic with double Q-learning and conservative penalty"""
        states, actions, rewards, next_states, dones, _, weights = batch
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) * self.reward_scale
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        self.critic_optimizer.zero_grad()
        
        with torch.no_grad():
            # Target actions from target policy
            next_action_dist = self.target_policy(next_states_t)
            next_actions = next_action_dist.rsample()
            
            # Target Q-values (double Q-learning)
            target_q1, target_q2 = self.target_critic(next_states_t)
            target_q = torch.min(target_q1, target_q2)
            
            # Conservative penalty (CQL-like)
            conservative_penalty = target_q.mean() * 0.1
            
            # TD target
            target_values = rewards_t + self.config.GAMMA * (1 - dones_t) * target_q
            target_values = target_values - conservative_penalty
        
        # Current Q-values
        current_q1, current_q2 = self.critic(states_t)
        
        # TD errors
        td_error1 = F.mse_loss(current_q1, target_values, reduction='none')
        td_error2 = F.mse_loss(current_q2, target_values, reduction='none')
        
        # Weighted losses
        critic_loss = (td_error1 * weights_t).mean() + (td_error2 * weights_t).mean()
        
        # Backward pass
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        
        self.critic_optimizer.step()
        
        # Update priorities based on TD error
        with torch.no_grad():
            td_errors = (td_error1 + td_error2).squeeze().cpu().numpy() / 2
            self.replay_buffer.update_priorities(batch[5], td_errors)
        
        metrics = {
            'critic_loss': critic_loss.item(),
            'q_values': current_q1.mean().item(),
            'td_error': td_errors.mean(),
            'grad_norm': grad_norm.item()
        }
        
        return critic_loss.item(), metrics
    
    def update_policy(self, batch: Tuple, step: int) -> Tuple[float, Dict]:
        """Update policy with contraction regularization and entropy bonus"""
        states, _, _, _, _, _, weights = batch
        
        states_t = torch.FloatTensor(states).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)
        
        self.policy_optimizer.zero_grad()
        
        # Sample actions with entropy
        actions, log_probs = self.policy.sample_with_entropy(states_t)
        
        # Q-values for sampled actions
        q_values = self.critic.q_min(states_t)
        
        # Contraction bonus
        with torch.no_grad():
            next_states_pred, _ = self.dynamics(states_t, actions)
            energy_curr, _ = EnhancedRiemannianOperations.compute_energy(states_t, self.metric_net)
            energy_next, _ = EnhancedRiemannianOperations.compute_energy(next_states_pred, self.metric_net)
            delta_energy = energy_curr - energy_next
            contraction_bonus = torch.tanh(delta_energy / 5.0).mean()
        
        # Entropy bonus for exploration
        entropy_bonus = -0.2 * log_probs.mean()  # Encourage exploration
        
        # Value loss (maximize Q)
        value_loss = -q_values.mean()
        
        # Additional stability penalties
        velocity = states_t[:, 2] * 8.0
        velocity_penalty = F.relu(torch.abs(velocity) - 3.0).mean() * 0.01
        
        action_penalty = torch.abs(actions).mean() * 0.001
        
        # Total policy loss
        policy_loss = (
            0.5 * value_loss -
            self.beta * contraction_bonus +
            entropy_bonus +
            velocity_penalty +
            action_penalty
        )
        
        # Weight the loss
        weighted_loss = policy_loss * weights_t.mean()
        
        # Backward pass
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        
        self.policy_optimizer.step()
        
        # Soft update target networks
        if step % self.config.TARGET_UPDATE_INTERVAL == 0:
            self.soft_update(self.target_policy, self.policy)
            self.soft_update(self.target_critic, self.critic)
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'contraction_bonus': contraction_bonus.item(),
            'entropy_bonus': entropy_bonus.item(),
            'velocity_penalty': velocity_penalty.item(),
            'action_penalty': action_penalty.item(),
            'beta': self.beta,
            'grad_norm': grad_norm.item()
        }
        
        return weighted_loss.item(), metrics
    
    def train_step(self, step: int):
        """Perform one training step on a batch"""
        if len(self.replay_buffer) < self.config.BATCH_SIZE:
            return None
        
        batch = self.replay_buffer.sample(self.config.BATCH_SIZE)
        if batch is None:
            return None
        
        # Update dynamics
        dynamics_loss, dynamics_metrics = self.update_dynamics(batch, step)
        self.metrics['dynamics_losses'].append(dynamics_loss)
        self.metrics['grad_norms']['dynamics'].append(dynamics_metrics['grad_norm'])
        
        # Update metric
        metric_loss, metric_metrics = self.update_metric(batch, step)
        self.metrics['metric_losses'].append(metric_loss)
        self.metrics['energies'].append(metric_metrics['energy_curr'])
        self.metrics['grad_norms']['metric'].append(metric_metrics['grad_norm'])
        
        # Update critic
        critic_loss, critic_metrics = self.update_critic(batch, step)
        self.metrics['critic_losses'].append(critic_loss)
        self.metrics['grad_norms']['critic'].append(critic_metrics['grad_norm'])
        
        # Update policy
        policy_loss, policy_metrics = self.update_policy(batch, step)
        self.metrics['actor_losses'].append(policy_loss)
        self.metrics['grad_norms']['policy'].append(policy_metrics['grad_norm'])
        
        # Update learning rates
        if step % 1000 == 0:
            self.dynamics_scheduler.step()
            self.policy_scheduler.step()
        
        return {
            'dynamics': (dynamics_loss, dynamics_metrics),
            'metric': (metric_loss, metric_metrics),
            'critic': (critic_loss, critic_metrics),
            'policy': (policy_loss, policy_metrics),
            'step': step
        }
    
    def train_episode(self, env: gym.Env, episode_num: int) -> Dict:
        """Execute one training episode with enhanced exploration"""
        state, _ = env.reset()
        self.exploration.reset()
        
        episode_reward = 0
        episode_steps = 0
        episode_transitions = []
        
        for step in range(self.config.MAX_EPISODE_LENGTH):
            # Select action with exploration
            use_exploration = (episode_num < self.config.TOTAL_EPISODES * 0.8)  # Phase out exploration
            action = self.select_action(state, deterministic=False, use_exploration=use_exploration)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            episode_transitions.append((state.copy(), action, reward, next_state.copy(), done))
            
            # Update metrics
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
        
        # Store transitions in replay buffer
        for transition in episode_transitions:
            self.replay_buffer.push(*transition)
        
        # Update normalization statistics
        states = np.array([t[0] for t in episode_transitions])
        self.update_normalization(states)
        
        # Update exploration strategy
        self.exploration.update(episode_reward, episode_num)
        
        # Perform multiple training steps
        training_metrics = []
        if len(self.replay_buffer) > self.config.LEARNING_START:
            num_steps = min(self.config.GRADIENT_STEPS, len(self.replay_buffer) // self.config.BATCH_SIZE)
            for i in range(num_steps):
                step_num = episode_num * self.config.GRADIENT_STEPS + i
                metrics = self.train_step(step_num)
                if metrics:
                    training_metrics.append(metrics)
        
        # Adapt beta based on performance
        if episode_num > 0 and self.metrics['episode_rewards']:
            last_reward = self.metrics['episode_rewards'][-1]
            reward_improved = episode_reward > last_reward
            self.adapt_beta(reward_improved)
        
        # Store episode metrics
        self.metrics['episode_rewards'].append(episode_reward)
        self.metrics['betas'].append(self.beta)
        self.metrics['exploration_rates'].append(self.exploration.exploration_rate)
        
        # Aggregate training metrics
        agg_metrics = {}
        if training_metrics:
            for key in ['dynamics', 'metric', 'critic', 'policy']:
                losses = [m[key][0] for m in training_metrics]
                if losses:
                    agg_metrics[f'{key}_loss'] = np.mean(losses)
        
        return {
            'episode': episode_num,
            'reward': episode_reward,
            'steps': episode_steps,
            **agg_metrics
        }
    
    def evaluate(self, env: gym.Env, num_episodes: int = 5) -> float:
        """Evaluate agent performance without exploration"""
        total_reward = 0
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for _ in range(self.config.MAX_EPISODE_LENGTH):
                action = self.select_action(state, deterministic=True, use_exploration=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                episode_reward += reward
                state = next_state
                
                if terminated or truncated:
                    break
            
            total_reward += episode_reward
        
        avg_reward = total_reward / num_episodes
        self.metrics['eval_rewards'].append(avg_reward)
        
        return avg_reward
    
    def train(self, env: gym.Env, eval_env: gym.Env = None) -> Dict:
        """Main training loop with evaluation"""
        if eval_env is None:
            eval_env = gym.make(self.config.ENV_NAME)
        
        print(f"\nStarting robust CDM training for {self.config.TOTAL_EPISODES} episodes...")
        print("=" * 100)
        print(f"{'Episode':>8} {'Reward':>10} {'Eval':>10} {'β':>6} {'Expl':>6} {'Dyn Loss':>9} {'Metric Loss':>11} {'Critic Loss':>11}")
        print("=" * 100)
        
        best_eval_reward = -float('inf')
        patience_counter = 0
        
        for episode in range(self.config.TOTAL_EPISODES):
            # Train one episode
            episode_start = time.time()
            episode_metrics = self.train_episode(env, episode)
            episode_time = time.time() - episode_start
            
            # Evaluate periodically
            eval_reward = None
            if episode % self.config.EVAL_INTERVAL == 0:
                eval_reward = self.evaluate(eval_env, self.config.EVAL_EPISODES)
                
                # Check for early stopping
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    patience_counter = 0
                    self.save_models(prefix="best")
                    print(f"  ★ New best: {best_eval_reward:.1f} (Episode {episode})")
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.EARLY_STOP_PATIENCE and eval_reward > self.config.EARLY_STOP_REWARD:
                    print(f"\n✅ Early stopping: Consistent performance at episode {episode}")
                    break
            
            # Log progress
            if episode % self.config.LOG_INTERVAL == 0 or episode == self.config.TOTAL_EPISODES - 1:
                eval_str = f"{eval_reward:10.1f}" if eval_reward is not None else "      N/A"
                print(f"{episode:8d} {episode_metrics['reward']:10.1f} {eval_str} "
                      f"{self.beta:6.2f} {self.exploration.exploration_rate:6.2f} "
                      f"{episode_metrics.get('dynamics_loss', 0):9.4f} "
                      f"{episode_metrics.get('metric_loss', 0):11.4f} "
                      f"{episode_metrics.get('critic_loss', 0):11.4f}")
        
        # Final evaluation
        final_eval = self.evaluate(eval_env, 10)
        print(f"\nFinal evaluation reward: {final_eval:.1f}")
        print(f"Best evaluation reward: {best_eval_reward:.1f}")
        
        # Save final models
        self.save_models(prefix="final")
        self.save_metrics()
        self.plot_training_results()
        
        return {
            'best_eval_reward': best_eval_reward,
            'final_eval_reward': final_eval,
            'total_episodes': len(self.metrics['episode_rewards'])
        }
    
    def save_models(self, prefix: str = ""):
        """Save all models to disk"""
        torch.save(self.policy.state_dict(), self.save_dir / f"{prefix}_policy.pth")
        torch.save(self.critic.state_dict(), self.save_dir / f"{prefix}_critic.pth")
        torch.save(self.dynamics.state_dict(), self.save_dir / f"{prefix}_dynamics.pth")
        torch.save(self.metric_net.state_dict(), self.save_dir / f"{prefix}_metric.pth")
        torch.save(self.target_policy.state_dict(), self.save_dir / f"{prefix}_target_policy.pth")
        torch.save(self.target_critic.state_dict(), self.save_dir / f"{prefix}_target_critic.pth")
    
    def load_models(self, prefix: str = ""):
        """Load models from disk"""
        self.policy.load_state_dict(torch.load(self.save_dir / f"{prefix}_policy.pth", 
                                              map_location=self.device))
        self.critic.load_state_dict(torch.load(self.save_dir / f"{prefix}_critic.pth", 
                                              map_location=self.device))
        self.dynamics.load_state_dict(torch.load(self.save_dir / f"{prefix}_dynamics.pth", 
                                                map_location=self.device))
        self.metric_net.load_state_dict(torch.load(self.save_dir / f"{prefix}_metric.pth", 
                                                  map_location=self.device))
        
        # Update target networks
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
    
    def save_metrics(self):
        """Save training metrics to disk"""
        # Convert metrics to serializable format
        serializable_metrics = {}
        for k, v in self.metrics.items():
            if isinstance(v, list):
                # Convert any tensors to lists
                serializable_metrics[k] = [x.item() if torch.is_tensor(x) else x for x in v]
            else:
                serializable_metrics[k] = v
        
        # Save as both pickle and JSON
        with open(self.save_dir / "training_metrics.pkl", 'wb') as f:
            pickle.dump(serializable_metrics, f)
        
        with open(self.save_dir / "training_metrics.json", 'w') as f:
            json.dump(serializable_metrics, f, indent=2, default=str)
    
    def plot_training_results(self):
        """Plot comprehensive training results with robust error handling"""
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            
            # Flatten axes for easier iteration
            ax_flat = axes.flatten()
            
            # Episode indices for training rewards
            if self.metrics.get('episode_rewards'):
                episodes = range(len(self.metrics['episode_rewards']))
                
                # 1. Training and Evaluation Rewards - FIXED
                ax = axes[0, 0]
                ax.plot(episodes, self.metrics['episode_rewards'], 'b-', alpha=0.7, label='Training')
                
                # Fix eval plotting with proper alignment
                if self.metrics.get('eval_rewards') and len(self.metrics['eval_rewards']) > 0:
                    # Ensure eval data aligns with eval intervals
                    eval_x = np.arange(0, len(episodes), self.config.EVAL_INTERVAL)
                    # Trim eval_x to match eval_rewards length
                    eval_x = eval_x[:len(self.metrics['eval_rewards'])]
                    # Ensure they have same length
                    min_len = min(len(eval_x), len(self.metrics['eval_rewards']))
                    if min_len > 0:
                        ax.plot(eval_x[:min_len], self.metrics['eval_rewards'][:min_len], 
                            'r-', alpha=0.9, label='Evaluation', linewidth=2)
                
                ax.set_xlabel('Episode')
                ax.set_ylabel('Reward')
                ax.set_title('Training Progress')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 2. Loss Curves - FIXED
            ax = axes[0, 1]
            loss_data_exists = False
            
            if self.metrics.get('dynamics_losses') and len(self.metrics['dynamics_losses']) > 0:
                loss_steps = range(len(self.metrics['dynamics_losses']))
                ax.plot(loss_steps, self.metrics['dynamics_losses'], 'b-', label='Dynamics', alpha=0.7)
                loss_data_exists = True
                
            if self.metrics.get('metric_losses') and len(self.metrics['metric_losses']) > 0:
                min_len = min(len(loss_steps), len(self.metrics['metric_losses'])) if 'loss_steps' in locals() else len(self.metrics['metric_losses'])
                if min_len > 0:
                    ax.plot(range(min_len), self.metrics['metric_losses'][:min_len], 
                        'g-', label='Metric', alpha=0.7)
                    loss_data_exists = True
                    
            if self.metrics.get('critic_losses') and len(self.metrics['critic_losses']) > 0:
                min_len = min(len(loss_steps), len(self.metrics['critic_losses'])) if 'loss_steps' in locals() else len(self.metrics['critic_losses'])
                if min_len > 0:
                    ax.plot(range(min_len), self.metrics['critic_losses'][:min_len], 
                        'r-', label='Critic', alpha=0.7)
                    loss_data_exists = True
                    
            if self.metrics.get('actor_losses') and len(self.metrics['actor_losses']) > 0:
                min_len = min(len(loss_steps), len(self.metrics['actor_losses'])) if 'loss_steps' in locals() else len(self.metrics['actor_losses'])
                if min_len > 0:
                    ax.plot(range(min_len), self.metrics['actor_losses'][:min_len], 
                        'm-', label='Actor', alpha=0.7)
                    loss_data_exists = True
            
            if loss_data_exists:
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Loss')
                ax.set_title('Training Losses')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
            else:
                ax.text(0.5, 0.5, 'No loss data available', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Training Losses')
            
            # 3. Beta Adaptation - FIXED
            ax = axes[0, 2]
            if self.metrics.get('betas') and len(self.metrics['betas']) > 0:
                # Ensure x and y have same length
                x_range = range(len(self.metrics['betas']))
                ax.plot(x_range, self.metrics['betas'], 'r-', alpha=0.7)
                ax.set_xlabel('Episode' if len(x_range) <= len(episodes) else 'Training Step')
                ax.set_ylabel('β')
                ax.set_title('Stability Weight Adaptation')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No beta data available', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Stability Weight Adaptation')
            
            # 4. Exploration Rate - FIXED
            ax = axes[1, 0]
            if self.metrics.get('exploration_rates') and len(self.metrics['exploration_rates']) > 0:
                # Ensure x and y have same length
                x_range = range(len(self.metrics['exploration_rates']))
                ax.plot(x_range, self.metrics['exploration_rates'], 'g-', alpha=0.7)
                ax.set_xlabel('Episode' if len(x_range) <= len(episodes) else 'Training Step')
                ax.set_ylabel('Exploration Rate')
                ax.set_title('Exploration Schedule')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No exploration data available', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Exploration Schedule')
            
            # 5. Energy Levels - FIXED
            ax = axes[1, 1]
            if self.metrics.get('energies') and len(self.metrics['energies']) > 0:
                # Ensure x and y have same length
                x_range = range(len(self.metrics['energies']))
                ax.plot(x_range, self.metrics['energies'], 'purple', alpha=0.7)
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Energy')
                ax.set_title('Metric Energy')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No energy data available', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Metric Energy')
            
            # 6. Reward Distribution - FIXED
            ax = axes[1, 2]
            if self.metrics.get('episode_rewards') and len(self.metrics['episode_rewards']) > 0:
                ax.hist(self.metrics['episode_rewards'], bins=30, alpha=0.7, 
                    color='blue', edgecolor='black')
                ax.set_xlabel('Reward')
                ax.set_ylabel('Frequency')
                ax.set_title('Reward Distribution')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No reward data available', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Reward Distribution')
            
            # 7. Moving Average Rewards - FIXED
            ax = axes[2, 0]
            if self.metrics.get('episode_rewards') and len(self.metrics['episode_rewards']) >= 20:
                window = min(20, len(self.metrics['episode_rewards']))
                moving_avg = np.convolve(self.metrics['episode_rewards'], 
                                        np.ones(window)/window, mode='valid')
                # Ensure x and y have same length
                x_range = range(window-1, len(self.metrics['episode_rewards']))
                min_len = min(len(x_range), len(moving_avg))
                if min_len > 0:
                    ax.plot(x_range[:min_len], moving_avg[:min_len], 'orange', linewidth=2)
                    ax.set_xlabel('Episode')
                    ax.set_ylabel(f'Reward ({window}-ep MA)')
                    ax.set_title('Moving Average Performance')
                    ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Insufficient data for moving average', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Moving Average Performance')
            
            # 8. Success Rate - FIXED
            ax = axes[2, 1]
            if self.metrics.get('episode_rewards') and len(self.metrics['episode_rewards']) >= 20:
                window = min(20, len(self.metrics['episode_rewards']))
                success_rates = []
                
                # Calculate success rates
                for i in range(len(self.metrics['episode_rewards']) - window + 1):
                    window_rewards = self.metrics['episode_rewards'][i:i+window]
                    successes = sum(1 for r in window_rewards if r > -500)
                    success_rates.append(successes / window * 100)
                
                # Ensure x and y have same length
                if success_rates:
                    x_range = range(window-1, len(self.metrics['episode_rewards']))
                    min_len = min(len(x_range), len(success_rates))
                    if min_len > 0:
                        ax.plot(x_range[:min_len], success_rates[:min_len], 'g-', linewidth=2)
                        ax.set_xlabel('Episode')
                        ax.set_ylabel('Success Rate (%)')
                        ax.set_title(f'Success Rate (> -500)')
                        ax.grid(True, alpha=0.3)
                        ax.set_ylim([0, 100])
            else:
                ax.text(0.5, 0.5, 'Insufficient data for success rate', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Success Rate')
            
            # 9. Beta vs Reward - FIXED
            ax = axes[2, 2]
            if (self.metrics.get('betas') and self.metrics.get('episode_rewards') and
                len(self.metrics['betas']) > 0 and len(self.metrics['episode_rewards']) > 0):
                # Ensure same length
                min_len = min(len(self.metrics['betas']), len(self.metrics['episode_rewards']))
                if min_len > 0:
                    scatter = ax.scatter(self.metrics['betas'][:min_len], 
                                    self.metrics['episode_rewards'][:min_len],
                                    c=range(min_len),
                                    cmap='viridis', alpha=0.6, s=20)
                    ax.set_xlabel('β (Stability Weight)')
                    ax.set_ylabel('Reward')
                    ax.set_title('Stability-Performance Tradeoff')
                    ax.grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=ax, label='Episode')
            else:
                ax.text(0.5, 0.5, 'Insufficient data for scatter plot', 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Stability-Performance Tradeoff')
            
            # Hide any unused axes
            for i in range(9):
                if not axes.flatten()[i].has_data():
                    axes.flatten()[i].axis('off')
            
            plt.suptitle(f'Robust CDM: Training Results (Final Reward: {self.metrics.get("episode_rewards", [-1])[-1]:.1f})', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save the plot
            save_path = Path(self.save_dir) / "training_results.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✓ Training results saved to {save_path}")
            
            # Show plot if enabled
            if self.config.PLOT_RESULTS:
                plt.show()
            else:
                plt.close(fig)
                
        except Exception as e:
            print(f"✗ Error plotting training results: {e}")
            import traceback
            traceback.print_exc()
            
            # Create simple fallback plot
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                if self.metrics.get('episode_rewards'):
                    ax.plot(range(len(self.metrics['episode_rewards'])), 
                        self.metrics['episode_rewards'])
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Reward')
                    ax.set_title('Basic Training Progress')
                    ax.grid(True)
                    
                    save_path = Path(self.save_dir) / "training_results_fallback.png"
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"✓ Fallback plot saved to {save_path}")
                    
                    if self.config.PLOT_RESULTS:
                        plt.show()
                    else:
                        plt.close()
            except:
                print("✗ Could not create fallback plot either")
# ============================
# TESTING AND VALIDATION
# ============================

def run_tests():
    """Run comprehensive tests on the implementation"""
    print("Running comprehensive tests...")
    
    # Test 1: Configuration validation
    config = Config()
    config.save(Path("test_config.json"))
    loaded_config = Config.load(Path("test_config.json"))
    # Compare configurations
    assert config.ENV_NAME == loaded_config.ENV_NAME, "Configuration save/load failed"
    print("✓ Configuration test passed")
    
    # Test 2: Network initialization
    device = torch.device("cpu")
    
    # Test dynamics ensemble
    dynamics = DynamicsEnsemble(3, 1, 5, 64).to(device)
    test_state = torch.randn(10, 3)
    test_action = torch.randn(10, 1)
    mean, unc = dynamics(test_state, test_action)
    assert mean.shape == (10, 3), f"Dynamics output shape: {mean.shape}"
    print("✓ Dynamics ensemble test passed")
    
    # Test policy network
    policy = EnhancedPolicyNetwork(3, 1, 64).to(device)
    dist = policy(test_state)
    action = dist.sample()
    assert action.shape == (10, 1), f"Policy output shape: {action.shape}"
    print("✓ Policy network test passed")
    
    # Test critic network
    critic = EnhancedValueNetwork(3, 64).to(device)
    q1, q2 = critic(test_state)
    assert q1.shape == (10, 1), f"Critic Q1 shape: {q1.shape}"
    assert q2.shape == (10, 1), f"Critic Q2 shape: {q2.shape}"
    print("✓ Critic network test passed")
    
    # Test metric network
    metric = RobustContractionMetric(3, 64).to(device)
    M, L = metric(test_state)
    assert M.shape == (10, 3, 3), f"Metric shape: {M.shape}"
    assert L.shape == (10, 3, 3), f"Cholesky shape: {L.shape}"
    print("✓ Metric network test passed")
    
    # Test Riemannian operations
    energy, _ = EnhancedRiemannianOperations.compute_energy(test_state, metric)
    assert energy.shape == (10,), f"Energy shape: {energy.shape}"
    print("✓ Riemannian operations test passed")
    
    # Test replay buffer
    buffer = PrioritizedReplayBuffer(100)
    for i in range(50):
        buffer.push(np.random.randn(3), np.random.randn(1), 
                   np.random.randn(), np.random.randn(3), False)
    batch = buffer.sample(32)
    assert batch is not None, "Replay buffer sampling failed"
    print("✓ Replay buffer test passed")
    
    print("\nAll tests passed successfully!")
    
    # Clean up
    if Path("test_config.json").exists():
        Path("test_config.json").unlink()

# ============================
# MAIN EXECUTION
# ============================

def set_seed(seed: int):
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
    """Main execution with robust training"""
    print("=" * 100)
    print("ROBUST CONTRACTION DYNAMICS MODEL - RIEMANNIAN METRIC LEARNING")
    print("=" * 100)
    print("Paper: 'Learning Contraction Metrics for Provably Stable Model-Based RL'")
    print("Author: Amir Hameed, Sirraya Labs")
    print("=" * 100)
    
    # Run tests first
    try:
        run_tests()
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Set seed for reproducibility
    set_seed(Config.SEED)
    
    # Create agent with robust configuration
    config = Config()
    agent = RobustContractionDynamicsAgent(config)
    
    # Create environments
    train_env = gym.make(config.ENV_NAME)
    eval_env = gym.make(config.ENV_NAME)
    
    # Train the agent
    print("\n" + "=" * 100)
    print("PHASE 1: ROBUST TRAINING")
    print("=" * 100)
    
    try:
        start_time = time.time()
        results = agent.train(train_env, eval_env)
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed successfully!")
        print(f"Training time: {training_time:.1f} seconds")
        print(f"Best evaluation reward: {results['best_eval_reward']:.1f}")
        print(f"Final evaluation reward: {results['final_eval_reward']:.1f}")
        print(f"Total episodes: {results['total_episodes']}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    finally:
        # Close environments
        train_env.close()
        eval_env.close()
    
    print("\n" + "=" * 100)
    print("EXPERIMENT COMPLETED")
    print("=" * 100)
    print(f"\nOutput files created in: {config.SAVE_DIR}")
    print("  - config.json (configuration)")
    print("  - best_*.pth (best performing models)")
    print("  - final_*.pth (final trained models)")
    print("  - training_metrics.pkl/.json (comprehensive metrics)")
    print("  - training_results.png (detailed plots)")

if __name__ == "__main__":
    main()