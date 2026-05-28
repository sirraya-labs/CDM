"""
Contraction Dynamics Model (CDM): Riemannian Metric Learning for Stable MBRL
Author: Amir Hameed, Sirraya Labs
Paper: "Learning Contraction Metrics for Provably Stable Model-Based RL"

ENHANCED IMPLEMENTATION: With adaptive contraction rates, curriculum learning,
attention-based metrics, safety constraints, and meta-learning capabilities.
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
from typing import Tuple, Dict, List, Optional, Any, Union
import time
from dataclasses import dataclass, asdict, field

# ============================
# ENHANCED CONFIGURATION
# ============================

@dataclass
class CurriculumStage:
    """Curriculum learning stage configuration"""
    name: str = "stability_focus"
    beta: float = 2.0
    duration: int = 50
    exploration_scale: float = 1.0
    noise_scale: float = 1.0

@dataclass
class Config:
    """Enhanced configuration with robust defaults and curriculum learning"""
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
    ATTENTION_HEADS: int = 3  # For attention-based metric
    
    # Training Parameters (optimized)
    TOTAL_EPISODES: int = 200
    BATCH_SIZE: int = 256
    GAMMA: float = 0.99
    TAU: float = 0.005
    
    # Enhanced Contraction Parameters
    CONTRACTION_RATE_ALPHA: float = 0.85
    CONTRACTION_RATE_MIN: float = 0.7
    CONTRACTION_RATE_MAX: float = 0.95
    INITIAL_BETA: float = 0.3
    BETA_MIN: float = 0.05
    BETA_MAX: float = 2.0
    METRIC_REGULARIZATION: float = 0.001
    EPSILON_METRIC: float = 0.05
    PERTURBATION_SIGMA: float = 0.02
    TARGET_CONDITION_NUMBER: float = 100.0
    
    # Optimized Learning Rates
    ACTOR_LR: float = 3e-4
    CRITIC_LR: float = 3e-4
    DYNAMICS_LR: float = 1e-3
    METRIC_LR: float = 5e-5
    
    # Enhanced Optimization
    REPLAY_BUFFER_SIZE: int = 100000
    INITIAL_EXPLORATION_STEPS: int = 5000
    UPDATE_FREQUENCY: int = 1
    ENSEMBLE_SIZE: int = 7
    
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
    
    # Curriculum Learning
    USE_CURRICULUM: bool = True
    CURRICULUM_STAGES: List[CurriculumStage] = field(default_factory=lambda: [
        CurriculumStage("stability_focus", beta=2.0, duration=50, exploration_scale=1.0, noise_scale=1.0),
        CurriculumStage("performance_focus", beta=1.0, duration=100, exploration_scale=0.8, noise_scale=0.7),
        CurriculumStage("fine_tuning", beta=0.3, duration=50, exploration_scale=0.5, noise_scale=0.3)
    ])
    
    # Meta-Learning
    USE_META_LEARNING: bool = True
    META_LEARNING_WINDOW: int = 20
    
    # Safety Constraints
    USE_SAFETY_CONSTRAINTS: bool = True
    SAFETY_MARGIN_THRESHOLD: float = 0.1
    
    # Attention Mechanism
    USE_ATTENTION_METRIC: bool = True
    
    # Geodesic Regularization
    USE_GEODESIC_REGULARIZATION: bool = True
    GEODESIC_WEIGHT: float = 0.01
    
    # Experimental Settings
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR: str = "cdm_enhanced_results"
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
    
    # Performance Optimization
    USE_JIT: bool = False  # JIT compilation (experimental)
    USE_AMP: bool = False  # Automatic Mixed Precision
    NUM_PARALLEL_ENVS: int = 1  # Parallel environments
    
    def __post_init__(self):
        """Validate configuration"""
        assert 0 < self.CONTRACTION_RATE_ALPHA < 1
        assert self.BATCH_SIZE <= self.REPLAY_BUFFER_SIZE
        assert self.ENSEMBLE_SIZE >= 3
        if self.USE_ATTENTION_METRIC:
            assert self.ATTENTION_HEADS <= self.STATE_DIM
    
    def save(self, path: Path):
        """Save configuration to file"""
        with open(path, 'w') as f:
            config_dict = asdict(self)
            # Convert CurriculumStage objects to dicts
            config_dict['CURRICULUM_STAGES'] = [
                asdict(stage) for stage in self.CURRICULUM_STAGES
            ]
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        """Load configuration from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert curriculum stages back to objects
        if 'CURRICULUM_STAGES' in data:
            data['CURRICULUM_STAGES'] = [
                CurriculumStage(**stage) for stage in data['CURRICULUM_STAGES']
            ]
        
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

class AttentionBasedMetric(nn.Module):
    """
    Metric network with attention mechanism for state-dependent importance
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128, 
                 num_heads: int = 3, epsilon: float = 0.05):
        super().__init__()
        self.state_dim = state_dim
        self.epsilon = epsilon
        
        # Number of parameters for lower triangular matrix
        self.output_dim = (state_dim * (state_dim + 1)) // 2
        
        # Ensure num_heads is valid
        self.num_heads = min(num_heads, state_dim)
        
        # Self-attention for state features (only if state_dim >= num_heads)
        if state_dim >= self.num_heads:
            self.attention = nn.MultiheadAttention(
                embed_dim=state_dim,
                num_heads=self.num_heads,
                batch_first=True
            )
            self.use_attention = True
        else:
            self.use_attention = False
        
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
        
        # Learnable temperature for attention
        if self.use_attention:
            self.temperature = nn.Parameter(torch.ones(1))
        
        # Softplus for diagonal entries
        self.softplus = nn.Softplus(beta=1.0, threshold=20)
        
        # For numerical stability
        self.diagonal_offset = 0.01
        self.off_diagonal_scale = 0.1
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.1)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        
        # Apply self-attention if enabled
        if self.use_attention:
            x_expanded = x.unsqueeze(1)  # Add sequence dimension [batch, 1, state_dim]
            attended, _ = self.attention(x_expanded, x_expanded, x_expanded)
            x_enhanced = (x + attended.squeeze(1)) * self.temperature
        else:
            x_enhanced = x
        
        # Get raw parameters
        l_params = self.net(x_enhanced)
        
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
            torch.linalg.cholesky(M)
        except RuntimeError:
            # Add damping if not positive definite
            M = M + 0.1 * identity
        
        return M, L
    
    def compute_metrics(self, M: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute additional metrics for monitoring"""
        batch_size = M.shape[0]
        metrics = {}
        
        # Eigenvalues for condition number
        eigenvalues = torch.linalg.eigvalsh(M)
        
        # Extract min and max eigenvalues across batch
        min_eigenvalues = eigenvalues.min(dim=1).values
        max_eigenvalues = eigenvalues.max(dim=1).values
        
        metrics['min_eigenvalue'] = min_eigenvalues.mean()
        metrics['max_eigenvalue'] = max_eigenvalues.mean()
        metrics['condition_number'] = (max_eigenvalues / torch.clamp(min_eigenvalues, min=1e-6)).mean()
        
        # Determinant
        metrics['det'] = torch.det(M).mean()
        
        return metrics

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
            nn.init.orthogonal_(module.weight, gain=0.1)
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
            torch.linalg.cholesky(M)
        except RuntimeError:
            # Add damping if not positive definite
            M = M + 0.1 * identity
        
        return M, L
    
    def compute_metrics(self, M: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute additional metrics for monitoring"""
        batch_size = M.shape[0]
        metrics = {}
        
        # Eigenvalues for condition number
        eigenvalues = torch.linalg.eigvalsh(M)
        
        # Extract min and max eigenvalues across batch
        min_eigenvalues = eigenvalues.min(dim=1).values
        max_eigenvalues = eigenvalues.max(dim=1).values
        
        metrics['min_eigenvalue'] = min_eigenvalues.mean()
        metrics['max_eigenvalue'] = max_eigenvalues.mean()
        metrics['condition_number'] = (max_eigenvalues / torch.clamp(min_eigenvalues, min=1e-6)).mean()
        
        # Determinant
        metrics['det'] = torch.det(M).mean()
        
        return metrics

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

# ============================
# ENHANCED RIEMANNIAN OPERATIONS
# ============================

class EnhancedRiemannianOperations:
    """Enhanced Riemannian operations with numerical stability and geodesic regularization"""
    
    @staticmethod
    def compute_energy(state: torch.Tensor, metric_net: Union[RobustContractionMetric, AttentionBasedMetric]) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def condition_metric(M: torch.Tensor, target_condition_number: float = 100.0) -> torch.Tensor:
        """
        Ensure metric matrix has bounded condition number
        """
        batch_size = M.shape[0]
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(M)
        
        # Clamp eigenvalues to bound condition number
        max_eigenvalues = eigenvalues.max(dim=1, keepdim=True).values
        min_eigenvalues = max_eigenvalues / target_condition_number
        
        eigenvalues = torch.clamp(eigenvalues, 
                                min=min_eigenvalues,
                                max=max_eigenvalues)
        
        # Reconstruct matrix with bounded condition number
        M_conditioned = torch.bmm(
            eigenvectors,
            torch.bmm(torch.diag_embed(eigenvalues), 
                     eigenvectors.transpose(1, 2))
        )
        
        return M_conditioned
    
    @staticmethod
    def compute_geodesic_regularization(states: torch.Tensor, 
                                       metric_net: Union[RobustContractionMetric, AttentionBasedMetric]) -> torch.Tensor:
        """
        Additional regularization to ensure smooth metric variation
        along geodesic paths in state space
        """
        num_interp = 5
        directions = torch.randn_like(states)
        directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-8)
        
        interp_points = []
        for t in torch.linspace(0, 1, num_interp, device=states.device):
            interp_points.append(states + t * directions * 0.1)
        
        interp_tensor = torch.cat(interp_points, dim=0)
        
        # Compute metric variation along path
        M_interp, _ = metric_net(interp_tensor)
        
        # Penalize rapid changes in metric
        metric_diffs = []
        for i in range(num_interp - 1):
            diff = M_interp[i::num_interp] - M_interp[(i+1)::num_interp]
            metric_diffs.append(torch.norm(diff, dim=(1,2)))
        
        geodesic_loss = torch.cat(metric_diffs).mean()
        return geodesic_loss
    
    @staticmethod
    def compute_contraction_loss(
        states: torch.Tensor, 
        next_states: torch.Tensor,
        metric_net: Union[RobustContractionMetric, AttentionBasedMetric],
        alpha: float = 0.85,
        beta: float = 1.0,
        use_geodesic_reg: bool = False,
        geodesic_weight: float = 0.01
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Enhanced contraction loss with smooth penalty and geodesic regularization
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
        
        # Geodesic regularization
        geodesic_loss = torch.tensor(0.0, device=states.device)
        if use_geodesic_reg:
            geodesic_loss = EnhancedRiemannianOperations.compute_geodesic_regularization(
                states, metric_net
            )
        
        metrics = {
            'energy_curr': energy_curr.mean(),
            'energy_next': energy_next.mean(),
            'energy_diff': energy_diff.mean(),
            'symmetry_loss': symmetry_loss,
            'identity_loss': identity_loss,
            'geodesic_loss': geodesic_loss
        }
        
        total_loss = (contraction_loss + 
                     0.01 * symmetry_loss + 
                     0.001 * identity_loss +
                     geodesic_weight * geodesic_loss)
        
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
    
    @staticmethod
    def compute_safety_margin(state: torch.Tensor, 
                             metric_net: Union[RobustContractionMetric, AttentionBasedMetric]) -> torch.Tensor:
        """
        Compute safety margin using contraction metric
        """
        M, _ = metric_net(state)
        
        # For Pendulum-v1: safe region when angle is within ±π/2
        theta = state[:, 0]  # Angle
        safe_region = torch.cos(theta)  # Positive when within safe bounds
        
        # Compute distance in metric
        state_expanded = state.unsqueeze(1)
        distance = torch.sqrt(
            torch.abs(torch.bmm(state_expanded, 
                              torch.bmm(M, state_expanded.transpose(1, 2))))
        ).squeeze()
        
        # Safety margin combines geometric distance and region
        safety_margin = distance * torch.clamp(safe_region, min=0)
        
        return safety_margin

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
    
    def sample(self, batch_size: int) -> Optional[Tuple]:
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
# CURRICULUM LEARNING SCHEDULER
# ============================

class CurriculumScheduler:
    """Progressive difficulty scheduler for stability learning"""
    
    def __init__(self, config: Config):
        self.config = config
        self.current_stage = 0
        self.stages = config.CURRICULUM_STAGES
        self.episodes_in_stage = 0
        self.enabled = config.USE_CURRICULUM
    
    def get_parameters(self, episode: int) -> Dict:
        """Get curriculum parameters for current stage"""
        if not self.enabled:
            return {
                'beta': self.config.INITIAL_BETA,
                'exploration_scale': 1.0,
                'noise_scale': 1.0
            }
        
        if self.episodes_in_stage >= self.stages[self.current_stage].duration:
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                self.episodes_in_stage = 0
                print(f"  🎓 Advancing to curriculum stage: {self.stages[self.current_stage].name}")
        
        self.episodes_in_stage += 1
        
        stage = self.stages[self.current_stage]
        
        # Smooth interpolation between stages
        if self.current_stage < len(self.stages) - 1:
            next_stage = self.stages[self.current_stage + 1]
            progress = self.episodes_in_stage / max(stage.duration, 1)
            
            beta = stage.beta + (next_stage.beta - stage.beta) * progress
            exploration = stage.exploration_scale + (next_stage.exploration_scale - stage.exploration_scale) * progress
            noise = stage.noise_scale + (next_stage.noise_scale - stage.noise_scale) * progress
        else:
            beta = stage.beta
            exploration = stage.exploration_scale
            noise = stage.noise_scale
        
        return {
            'beta': beta,
            'exploration_scale': exploration,
            'noise_scale': noise
        }

# ============================
# META-LEARNING CONTROLLER
# ============================

class MetaLearningController:
    """Meta-controller for adaptive hyperparameter tuning"""
    
    def __init__(self, config: Config):
        self.config = config
        self.enabled = config.USE_META_LEARNING
        self.performance_history = deque(maxlen=config.META_LEARNING_WINDOW)
        self.energy_history = deque(maxlen=config.META_LEARNING_WINDOW)
        self.contraction_rate_history = deque(maxlen=config.META_LEARNING_WINDOW)
    
    def update(self, reward: float, energy: float):
        """Update meta-learning with new data"""
        self.performance_history.append(reward)
        self.energy_history.append(energy)
    
    def suggest_contraction_rate(self) -> float:
        """Suggest new contraction rate based on performance"""
        if not self.enabled or len(self.energy_history) < 10:
            return self.config.CONTRACTION_RATE_ALPHA
        
        # Compute energy trend
        energy_array = np.array(list(self.energy_history))
        if len(energy_array) >= 2:
            energy_trend = np.polyfit(range(len(energy_array)), energy_array, 1)[0]
        else:
            energy_trend = 0
        
        alpha = self.config.CONTRACTION_RATE_ALPHA
        
        if energy_trend > 0:  # Energy increasing (instability)
            # Tighten contraction
            alpha *= 0.98
        else:  # Energy decreasing (stabilizing)
            # Relax contraction for better performance
            alpha *= 1.01
        
        # Clip to valid range
        alpha = np.clip(alpha, self.config.CONTRACTION_RATE_MIN, self.config.CONTRACTION_RATE_MAX)
        
        self.contraction_rate_history.append(alpha)
        self.config.CONTRACTION_RATE_ALPHA = alpha
        
        return alpha
    
    def suggest_hyperparameters(self, current_metrics: Dict) -> Dict:
        """Suggest new hyperparameters based on performance"""
        if not self.enabled or len(self.performance_history) < 5:
            return {}
        
        adjustments = {}
        
        # Compute performance trend
        reward_array = np.array(list(self.performance_history))
        if len(reward_array) >= 2:
            reward_trend = np.polyfit(range(len(reward_array)), reward_array, 1)[0]
        else:
            reward_trend = 0
        
        # Adjust learning rates based on loss trends
        if current_metrics.get('critic_loss', 0) > 1.0:
            adjustments['CRITIC_LR'] = max(1e-5, self.config.CRITIC_LR * 0.9)
        
        # Adjust contraction rate
        if reward_trend < 0:  # Performance declining
            self.suggest_contraction_rate()
        
        return adjustments

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
    
    def sample(self, strategy: str = None, scale: float = 1.0) -> np.ndarray:
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
        
        # Scale by current exploration rate and curriculum scale
        noise *= self.exploration_rate * scale
        
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

class EnhancedContractionDynamicsAgent:
    """
    Enhanced CDM agent with curriculum learning, meta-learning,
    attention-based metrics, and safety constraints
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
        
        # Curriculum learning
        self.curriculum = CurriculumScheduler(config)
        
        # Meta-learning controller
        self.meta_controller = MetaLearningController(config)
        
        # Adaptive stability weight
        self.beta = config.INITIAL_BETA
        self.beta_history = []
        
        # Adaptive contraction rate
        self.contraction_alpha = config.CONTRACTION_RATE_ALPHA
        self.contraction_alpha_history = []
        
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
            'contraction_alphas': [],
            'exploration_rates': [],
            'safety_margins': [],
            'grad_norms': defaultdict(list),
            'curriculum_stages': [],
            'geodesic_losses': []
        }
        
        # Create save directory
        self.save_dir = Path(config.SAVE_DIR)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save config
        config.save(self.save_dir / "config.json")
        
        print(f"Enhanced CDM Agent initialized on {self.device}")
        print(f"Save directory: {self.save_dir}")
        print(f"Features enabled:")
        print(f"  - Curriculum Learning: {config.USE_CURRICULUM}")
        print(f"  - Meta-Learning: {config.USE_META_LEARNING}")
        print(f"  - Attention Metric: {config.USE_ATTENTION_METRIC}")
        print(f"  - Safety Constraints: {config.USE_SAFETY_CONSTRAINTS}")
        print(f"  - Geodesic Regularization: {config.USE_GEODESIC_REGULARIZATION}")
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
        
        # Contraction metric network (attention-based or standard)
        if self.config.USE_ATTENTION_METRIC:
            self.metric_net = AttentionBasedMetric(
                self.config.STATE_DIM,
                self.config.METRIC_HIDDEN_DIM,
                num_heads=self.config.ATTENTION_HEADS,
                epsilon=self.config.EPSILON_METRIC
            ).to(self.device)
        else:
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
                     use_exploration: bool = True, curriculum_scale: float = 1.0) -> np.ndarray:
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
            
            # Add exploration noise with curriculum scaling
            if use_exploration:
                noise = self.exploration.sample(scale=curriculum_scale)
                action_np += noise
            
            # Clip to valid range
            action_np = np.clip(action_np, -self.policy.action_scale, self.policy.action_scale)
            
            return action_np
    
    def compute_safety_margin(self, state: torch.Tensor) -> torch.Tensor:
        """Compute safety margin for current state"""
        if not self.config.USE_SAFETY_CONSTRAINTS:
            return torch.zeros(state.shape[0], device=state.device)
        
        with torch.no_grad():
            safety_margin = EnhancedRiemannianOperations.compute_safety_margin(
                state, self.metric_net
            )
        
        return safety_margin
    
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
        """Update contraction metric with enhanced stability and geodesic regularization"""
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
        
        # Compute contraction loss with temperature scheduling and geodesic regularization
        temperature = max(0.1, 1.0 - step / 10000)  # Anneal temperature
        metric_loss, metric_metrics = EnhancedRiemannianOperations.compute_contraction_loss(
            states_t, next_states_pred, self.metric_net,
            alpha=self.contraction_alpha,
            beta=temperature,
            use_geodesic_reg=self.config.USE_GEODESIC_REGULARIZATION,
            geodesic_weight=self.config.GEODESIC_WEIGHT
        )
        
        # Weight the loss
        weighted_loss = metric_loss * weights_t.mean()
        
        # Additional consistency loss for perturbed trajectories
        consistency_loss = 0.0
        for next_perturbed in next_perturbed_preds:
            _, metrics_perturbed = EnhancedRiemannianOperations.compute_contraction_loss(
                states_t, next_perturbed, self.metric_net,
                alpha=self.contraction_alpha
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
            
            # Compute safety margins if enabled
            if self.config.USE_SAFETY_CONSTRAINTS:
                safety_margin = EnhancedRiemannianOperations.compute_safety_margin(
                    states_t, self.metric_net
                )
                metric_net_metrics['safety_margin'] = safety_margin.mean()
        
        metrics = {
            'metric_loss': metric_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'energy_curr': metric_metrics['energy_curr'].item(),
            'energy_next': metric_metrics['energy_next'].item(),
            'energy_diff': metric_metrics['energy_diff'].item(),
            'geodesic_loss': metric_metrics.get('geodesic_loss', torch.tensor(0.0)).item(),
            'grad_norm': grad_norm.item(),
        }
        
        # Add metric network metrics safely
        for k, v in metric_net_metrics.items():
            if torch.is_tensor(v):
                if v.numel() == 1:
                    metrics[k] = v.item()
                else:
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
        """Update policy with contraction regularization, entropy bonus, and safety constraints"""
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
        
        # Safety penalty
        safety_penalty = torch.tensor(0.0, device=self.device)
        if self.config.USE_SAFETY_CONSTRAINTS:
            safety_margin = self.compute_safety_margin(states_t)
            safety_penalty = F.relu(
                self.config.SAFETY_MARGIN_THRESHOLD - safety_margin
            ).mean() * 0.1
        
        # Entropy bonus for exploration
        entropy_bonus = -0.2 * log_probs.mean()
        
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
            action_penalty +
            safety_penalty
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
            'safety_penalty': safety_penalty.item(),
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
        if 'geodesic_loss' in metric_metrics:
            self.metrics['geodesic_losses'].append(metric_metrics['geodesic_loss'])
        if 'safety_margin' in metric_metrics:
            self.metrics['safety_margins'].append(metric_metrics['safety_margin'])
        
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
        """Execute one training episode with enhanced exploration and curriculum"""
        state, _ = env.reset()
        self.exploration.reset()
        
        # Get curriculum parameters
        curriculum_params = self.curriculum.get_parameters(episode_num)
        
        episode_reward = 0
        episode_steps = 0
        episode_transitions = []
        episode_safety_margins = []
        
        for step in range(self.config.MAX_EPISODE_LENGTH):
            # Select action with exploration and curriculum scaling
            use_exploration = (episode_num < self.config.TOTAL_EPISODES * 0.8)
            action = self.select_action(
                state, 
                deterministic=False, 
                use_exploration=use_exploration,
                curriculum_scale=curriculum_params['noise_scale']
            )
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Compute safety margin if enabled
            if self.config.USE_SAFETY_CONSTRAINTS:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    safety_margin = self.compute_safety_margin(state_t)
                    episode_safety_margins.append(safety_margin.item())
            
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
        
        # Update meta-learning controller
        if self.metrics['energies']:
            avg_energy = np.mean(self.metrics['energies'][-100:]) if len(self.metrics['energies']) >= 100 else np.mean(self.metrics['energies'])
            self.meta_controller.update(episode_reward, avg_energy)
            
            # Update contraction rate based on meta-learning
            self.contraction_alpha = self.meta_controller.suggest_contraction_rate()
            self.contraction_alpha_history.append(self.contraction_alpha)
        
        # Perform multiple training steps
        training_metrics = []
        if len(self.replay_buffer) > self.config.LEARNING_START:
            num_steps = min(self.config.GRADIENT_STEPS, len(self.replay_buffer) // self.config.BATCH_SIZE)
            for i in range(num_steps):
                step_num = episode_num * self.config.GRADIENT_STEPS + i
                metrics = self.train_step(step_num)
                if metrics:
                    training_metrics.append(metrics)
        
        # Adapt beta based on performance and curriculum
        if episode_num > 0 and self.metrics['episode_rewards']:
            last_reward = self.metrics['episode_rewards'][-1]
            reward_improved = episode_reward > last_reward
            self.adapt_beta(reward_improved)
        
        # Apply curriculum beta
        self.beta = curriculum_params['beta']
        
        # Store episode metrics
        self.metrics['episode_rewards'].append(episode_reward)
        self.metrics['betas'].append(self.beta)
        self.metrics['exploration_rates'].append(self.exploration.exploration_rate)
        self.metrics['curriculum_stages'].append(self.curriculum.current_stage)
        
        if episode_safety_margins:
            self.metrics['safety_margins'].append(np.mean(episode_safety_margins))
        
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
            'curriculum_stage': self.curriculum.current_stage,
            'contraction_alpha': self.contraction_alpha,
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
        """Main training loop with evaluation and curriculum learning"""
        if eval_env is None:
            eval_env = gym.make(self.config.ENV_NAME)
        
        print(f"\nStarting enhanced CDM training for {self.config.TOTAL_EPISODES} episodes...")
        print("Features: Curriculum Learning + Meta-Learning + Attention Metrics + Safety Constraints")
        print("=" * 120)
        print(f"{'Episode':>8} {'Reward':>10} {'Eval':>10} {'β':>6} {'α':>6} {'Stage':>6} {'Dyn Loss':>9} {'Metric Loss':>11} {'Critic Loss':>11}")
        print("=" * 120)
        
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
                stage_names = ["stability", "perf", "tuning"]
                stage_name = stage_names[min(episode_metrics['curriculum_stage'], len(stage_names)-1)]
                print(f"{episode:8d} {episode_metrics['reward']:10.1f} {eval_str} "
                      f"{self.beta:6.2f} {self.contraction_alpha:6.2f} {stage_name:>6} "
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
            elif isinstance(v, defaultdict):
                serializable_metrics[k] = dict(v)
            else:
                serializable_metrics[k] = v
        
        # Save as both pickle and JSON
        with open(self.save_dir / "training_metrics.pkl", 'wb') as f:
            pickle.dump(serializable_metrics, f)
        
        with open(self.save_dir / "training_metrics.json", 'w') as f:
            json.dump(serializable_metrics, f, indent=2, default=str)
    
    def plot_training_results(self):
        """Plot comprehensive training results with enhanced visualization"""
        try:
            # Create figure with subplots for enhanced metrics
            fig = plt.figure(figsize=(24, 20))
            
            # Create grid for subplots
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
            
            # Episode indices
            episodes = range(len(self.metrics['episode_rewards']))
            
            # 1. Training and Evaluation Rewards
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(episodes, self.metrics['episode_rewards'], 'b-', alpha=0.7, label='Training')
            if self.metrics.get('eval_rewards') and len(self.metrics['eval_rewards']) > 0:
                eval_x = np.arange(0, len(episodes), self.config.EVAL_INTERVAL)
                eval_x = eval_x[:len(self.metrics['eval_rewards'])]
                min_len = min(len(eval_x), len(self.metrics['eval_rewards']))
                if min_len > 0:
                    ax.plot(eval_x[:min_len], self.metrics['eval_rewards'][:min_len], 
                           'r-', alpha=0.9, label='Evaluation', linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Training Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. Curriculum Stages
            ax = fig.add_subplot(gs[0, 1])
            if self.metrics.get('curriculum_stages'):
                ax.plot(episodes, self.metrics['curriculum_stages'], 'g-', alpha=0.7)
                ax.set_xlabel('Episode')
                ax.set_ylabel('Curriculum Stage')
                ax.set_title('Curriculum Learning Progress')
                ax.set_yticks([0, 1, 2])
                ax.set_yticklabels(['Stability', 'Performance', 'Fine-tuning'])
                ax.grid(True, alpha=0.3)
            
            # 3. Beta and Alpha Adaptation
            ax = fig.add_subplot(gs[0, 2])
            if self.metrics.get('betas') and len(self.metrics['betas']) > 0:
                ax.plot(episodes[:len(self.metrics['betas'])], self.metrics['betas'], 
                       'r-', alpha=0.7, label='β (Stability)')
            if self.contraction_alpha_history:
                ax.plot(episodes[:len(self.contraction_alpha_history)], 
                       self.contraction_alpha_history, 
                       'b-', alpha=0.7, label='α (Contraction)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Parameter Value')
            ax.set_title('Adaptive Parameters')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. Safety Margins
            ax = fig.add_subplot(gs[0, 3])
            if self.metrics.get('safety_margins') and len(self.metrics['safety_margins']) > 0:
                safety_episodes = episodes[:len(self.metrics['safety_margins'])]
                ax.plot(safety_episodes, self.metrics['safety_margins'], 'm-', alpha=0.7)
                ax.axhline(y=self.config.SAFETY_MARGIN_THRESHOLD, color='r', 
                          linestyle='--', label='Threshold')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Safety Margin')
                ax.set_title('Safety Metrics')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 5. Loss Curves
            ax = fig.add_subplot(gs[1, :2])
            if self.metrics.get('dynamics_losses') and len(self.metrics['dynamics_losses']) > 0:
                loss_steps = range(len(self.metrics['dynamics_losses']))
                ax.plot(loss_steps, self.metrics['dynamics_losses'], 'b-', label='Dynamics', alpha=0.7)
            if self.metrics.get('metric_losses') and len(self.metrics['metric_losses']) > 0:
                ax.plot(range(len(self.metrics['metric_losses'])), 
                       self.metrics['metric_losses'], 'g-', label='Metric', alpha=0.7)
            if self.metrics.get('critic_losses') and len(self.metrics['critic_losses']) > 0:
                ax.plot(range(len(self.metrics['critic_losses'])), 
                       self.metrics['critic_losses'], 'r-', label='Critic', alpha=0.7)
            if self.metrics.get('actor_losses') and len(self.metrics['actor_losses']) > 0:
                ax.plot(range(len(self.metrics['actor_losses'])), 
                       self.metrics['actor_losses'], 'm-', label='Actor', alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training Losses')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            # 6. Energy and Geodesic Losses
            ax = fig.add_subplot(gs[1, 2:])
            if self.metrics.get('energies') and len(self.metrics['energies']) > 0:
                ax.plot(range(len(self.metrics['energies'])), self.metrics['energies'], 
                       'purple', alpha=0.7, label='Energy')
            if self.metrics.get('geodesic_losses') and len(self.metrics['geodesic_losses']) > 0:
                ax_twin = ax.twinx()
                ax_twin.plot(range(len(self.metrics['geodesic_losses'])), 
                            self.metrics['geodesic_losses'], 'orange', alpha=0.7, label='Geodesic')
                ax_twin.set_ylabel('Geodesic Loss', color='orange')
                ax_twin.tick_params(axis='y', labelcolor='orange')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Energy', color='purple')
            ax.set_title('Metric Energy & Geodesic Regularization')
            ax.grid(True, alpha=0.3)
            
            # 7. Exploration Rate
            ax = fig.add_subplot(gs[2, 0])
            if self.metrics.get('exploration_rates') and len(self.metrics['exploration_rates']) > 0:
                ax.plot(episodes[:len(self.metrics['exploration_rates'])], 
                       self.metrics['exploration_rates'], 'g-', alpha=0.7)
                ax.set_xlabel('Episode')
                ax.set_ylabel('Exploration Rate')
                ax.set_title('Exploration Schedule')
                ax.grid(True, alpha=0.3)
            
            # 8. Reward Distribution
            ax = fig.add_subplot(gs[2, 1])
            if self.metrics.get('episode_rewards') and len(self.metrics['episode_rewards']) > 0:
                ax.hist(self.metrics['episode_rewards'], bins=30, alpha=0.7, 
                       color='blue', edgecolor='black')
                ax.set_xlabel('Reward')
                ax.set_ylabel('Frequency')
                ax.set_title('Reward Distribution')
                ax.grid(True, alpha=0.3)
            
            # 9. Moving Average
            ax = fig.add_subplot(gs[2, 2])
            if self.metrics.get('episode_rewards') and len(self.metrics['episode_rewards']) >= 20:
                window = min(20, len(self.metrics['episode_rewards']))
                moving_avg = np.convolve(self.metrics['episode_rewards'], 
                                        np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.metrics['episode_rewards'])), 
                       moving_avg, 'orange', linewidth=2)
                ax.set_xlabel('Episode')
                ax.set_ylabel(f'Reward ({window}-ep MA)')
                ax.set_title('Moving Average Performance')
                ax.grid(True, alpha=0.3)
            
            # 10. Success Rate
            ax = fig.add_subplot(gs[2, 3])
            if self.metrics.get('episode_rewards') and len(self.metrics['episode_rewards']) >= 20:
                window = min(20, len(self.metrics['episode_rewards']))
                success_rates = []
                for i in range(len(self.metrics['episode_rewards']) - window + 1):
                    window_rewards = self.metrics['episode_rewards'][i:i+window]
                    successes = sum(1 for r in window_rewards if r > -500)
                    success_rates.append(successes / window * 100)
                if success_rates:
                    ax.plot(range(window-1, len(self.metrics['episode_rewards'])), 
                           success_rates, 'g-', linewidth=2)
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Success Rate (%)')
                    ax.set_title(f'Success Rate (> -500)')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim([0, 100])
            
            # 11. Beta vs Reward
            ax = fig.add_subplot(gs[3, 0])
            if (self.metrics.get('betas') and self.metrics.get('episode_rewards') and
                len(self.metrics['betas']) > 0 and len(self.metrics['episode_rewards']) > 0):
                min_len = min(len(self.metrics['betas']), len(self.metrics['episode_rewards']))
                if min_len > 0:
                    scatter = ax.scatter(self.metrics['betas'][:min_len], 
                                       self.metrics['episode_rewards'][:min_len],
                                       c=range(min_len), cmap='viridis', alpha=0.6, s=20)
                    ax.set_xlabel('β (Stability Weight)')
                    ax.set_ylabel('Reward')
                    ax.set_title('Stability-Performance Tradeoff')
                    ax.grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=ax, label='Episode')
            
            # 12. Contraction Alpha vs Reward
            ax = fig.add_subplot(gs[3, 1])
            if (self.contraction_alpha_history and self.metrics.get('episode_rewards') and
                len(self.contraction_alpha_history) > 0 and len(self.metrics['episode_rewards']) > 0):
                min_len = min(len(self.contraction_alpha_history), len(self.metrics['episode_rewards']))
                if min_len > 0:
                    scatter = ax.scatter(self.contraction_alpha_history[:min_len], 
                                       self.metrics['episode_rewards'][:min_len],
                                       c=range(min_len), cmap='plasma', alpha=0.6, s=20)
                    ax.set_xlabel('α (Contraction Rate)')
                    ax.set_ylabel('Reward')
                    ax.set_title('Contraction-Performance Tradeoff')
                    ax.grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=ax, label='Episode')
            
            # 13. Gradient Norms
            ax = fig.add_subplot(gs[3, 2])
            if self.metrics.get('grad_norms'):
                for key in ['dynamics', 'metric', 'critic', 'policy']:
                    if key in self.metrics['grad_norms'] and self.metrics['grad_norms'][key]:
                        ax.plot(range(len(self.metrics['grad_norms'][key])), 
                               self.metrics['grad_norms'][key], alpha=0.7, label=key.capitalize())
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Gradient Norm')
                ax.set_title('Gradient Norms Over Time')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
            
            # 14. Summary Statistics
            ax = fig.add_subplot(gs[3, 3])
            ax.axis('off')
            
            # Create summary text
            if self.metrics.get('episode_rewards') and len(self.metrics['episode_rewards']) > 0:
                rewards = self.metrics['episode_rewards']
                summary_text = (
                    f"Training Summary\n"
                    f"{'='*30}\n"
                    f"Episodes: {len(rewards)}\n"
                    f"Final Reward: {rewards[-1]:.1f}\n"
                    f"Best Reward: {max(rewards):.1f}\n"
                    f"Mean Reward: {np.mean(rewards):.1f}\n"
                    f"Std Reward: {np.std(rewards):.1f}\n\n"
                    f"Best Eval: {max(self.metrics.get('eval_rewards', [0])):.1f}\n"
                    f"Final β: {self.beta:.3f}\n"
                    f"Final α: {self.contraction_alpha:.3f}"
                )
                
                # Add early episode stats
                if len(rewards) >= 50:
                    early_mean = np.mean(rewards[:50])
                    late_mean = np.mean(rewards[-50:])
                    improvement = ((late_mean - early_mean) / abs(early_mean)) * 100
                    summary_text += f"\n\nImprovement: {improvement:.1f}%"
                
                ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='center',
                       fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.suptitle(f'Enhanced CDM: Training Results\n'
                        f'(Features: Curriculum + Meta-Learning + Attention + Safety)\n'
                        f'Final Reward: {self.metrics.get("episode_rewards", [-1])[-1]:.1f}', 
                        fontsize=14, fontweight='bold')
            
            # Save the plot
            save_path = Path(self.save_dir) / "training_results.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✓ Enhanced training results saved to {save_path}")
            
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
                fig, ax = plt.subplots(figsize=(12, 8))
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

def run_enhanced_tests():
    """Run comprehensive tests on the enhanced implementation"""
    print("Running enhanced comprehensive tests...")
    
    # Test 1: Configuration with enhanced features
    config = Config()
    config.USE_CURRICULUM = True
    config.USE_META_LEARNING = True
    config.USE_ATTENTION_METRIC = True
    config.USE_SAFETY_CONSTRAINTS = True
    config.USE_GEODESIC_REGULARIZATION = True
    
    config.save(Path("test_config.json"))
    loaded_config = Config.load(Path("test_config.json"))
    assert config.ENV_NAME == loaded_config.ENV_NAME, "Configuration save/load failed"
    print("✓ Enhanced configuration test passed")
    
    # Test 2: Network initialization with attention metric
    device = torch.device("cpu")
    
    # Test attention-based metric
    metric = AttentionBasedMetric(3, 64, num_heads=3).to(device)
    test_state = torch.randn(10, 3)
    M, L = metric(test_state)
    assert M.shape == (10, 3, 3), f"Attention metric shape: {M.shape}"
    print("✓ Attention-based metric test passed")
    
    # Test 3: Geodesic regularization
    geodesic_loss = EnhancedRiemannianOperations.compute_geodesic_regularization(
        test_state, metric
    )
    assert geodesic_loss.ndim == 0, "Geodesic loss should be scalar"
    print("✓ Geodesic regularization test passed")
    
    # Test 4: Safety margin computation
    safety_margin = EnhancedRiemannianOperations.compute_safety_margin(
        test_state, metric
    )
    assert safety_margin.shape == (10,), f"Safety margin shape: {safety_margin.shape}"
    print("✓ Safety margin test passed")
    
    # Test 5: Metric conditioning
    M_conditioned = EnhancedRiemannianOperations.condition_metric(M)
    assert M_conditioned.shape == M.shape, "Conditioned metric shape mismatch"
    
    # Check condition number
    eigenvalues = torch.linalg.eigvalsh(M_conditioned)
    condition_number = eigenvalues.max() / eigenvalues.min()
    assert condition_number < 200, f"Condition number too high: {condition_number}"
    print("✓ Metric conditioning test passed")
    
    # Test 6: Curriculum scheduler
    curriculum = CurriculumScheduler(config)
    params = curriculum.get_parameters(0)
    assert 'beta' in params, "Curriculum missing beta"
    assert 'exploration_scale' in params, "Curriculum missing exploration_scale"
    print("✓ Curriculum scheduler test passed")
    
    # Test 7: Meta-learning controller
    meta = MetaLearningController(config)
    meta.update(-500, 1.0)
    alpha = meta.suggest_contraction_rate()
    assert 0 < alpha < 1, f"Invalid contraction rate: {alpha}"
    print("✓ Meta-learning controller test passed")
    
    # Test 8: Full agent initialization
    agent = EnhancedContractionDynamicsAgent(config)
    assert agent.metric_net is not None, "Agent metric network not initialized"
    print("✓ Full agent initialization test passed")
    
    print("\n✅ All enhanced tests passed successfully!")
    
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
    """Main execution with enhanced training"""
    print("=" * 120)
    print("ENHANCED CONTRACTION DYNAMICS MODEL - RIEMANNIAN METRIC LEARNING")
    print("=" * 120)
    print("Paper: 'Learning Contraction Metrics for Provably Stable Model-Based RL'")
    print("Author: Amir Hameed, Sirraya Labs")
    print("=" * 120)
    print("\nEnhanced Features:")
    print("  ✓ Curriculum Learning for progressive stability training")
    print("  ✓ Meta-Learning for adaptive hyperparameter tuning")
    print("  ✓ Attention-Based Metric for state-dependent importance")
    print("  ✓ Safety Constraints for bounded exploration")
    print("  ✓ Geodesic Regularization for smooth metric learning")
    print("=" * 120)
    
    # Run tests first
    try:
        run_enhanced_tests()
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Set seed for reproducibility
    set_seed(Config.SEED)
    
    # Create enhanced configuration
    config = Config()
    config.USE_CURRICULUM = True
    config.USE_META_LEARNING = True
    config.USE_ATTENTION_METRIC = True
    config.USE_SAFETY_CONSTRAINTS = True
    config.USE_GEODESIC_REGULARIZATION = True
    
    # Create agent with enhanced configuration
    agent = EnhancedContractionDynamicsAgent(config)
    
    # Create environments
    train_env = gym.make(config.ENV_NAME)
    eval_env = gym.make(config.ENV_NAME)
    
    # Train the agent
    print("\n" + "=" * 120)
    print("PHASE 1: ENHANCED TRAINING WITH CURRICULUM AND META-LEARNING")
    print("=" * 120)
    
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
    
    print("\n" + "=" * 120)
    print("EXPERIMENT COMPLETED")
    print("=" * 120)
    print(f"\nOutput files created in: {config.SAVE_DIR}")
    print("  - config.json (enhanced configuration)")
    print("  - best_*.pth (best performing models)")
    print("  - final_*.pth (final trained models)")
    print("  - training_metrics.pkl/.json (comprehensive metrics)")
    print("  - training_results.png (enhanced visualization)")

if __name__ == "__main__":
    main()
