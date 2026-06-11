"""
CR-MBRL: Contraction-Regularized Model-Based Reinforcement Learning
Investigating whether contraction-inspired geometric regularization improves
empirical robustness in model-based RL.

Author: Amir Hameed, Sirraya Labs
Status: Research Prototype - Phase 2 (Benchmark Ready)
Version: 4.0 - Multi-Environment Benchmark Support

Key features:
- Multiple energy formulations (absolute, relative, masked) for different task types
- Real-transition anchored metric updates (80/20 split with PER filtering)
- RobustnessEvaluator with Gaussian and adversarial perturbation protocols
- Adaptive identity regularization based on metric condition number
- Statistical significance testing utilities
- Multi-environment benchmark support with automatic configuration
- Proper error handling for optional dependencies (MuJoCo)
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
from scipy import stats
import argparse

# ============================
# CONFIGURATION
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
    """Configuration for CR-MBRL research prototype."""
    
    # Environment
    ENV_NAME: str = "Pendulum-v1"
    STATE_DIM: int = 3
    ACTION_DIM: int = 1
    MAX_EPISODE_LENGTH: int = 200
    
    # Energy formulation
    ENERGY_TYPE: str = "absolute"  # "absolute", "relative", "masked"
    TASK_MASK: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    
    # Architecture
    DYNAMICS_HIDDEN_DIM: int = 128
    POLICY_HIDDEN_DIM: int = 256
    METRIC_HIDDEN_DIM: int = 128
    CRITIC_HIDDEN_DIM: int = 256
    ATTENTION_HEADS: int = 3
    
    # Training
    TOTAL_EPISODES: int = 200
    BATCH_SIZE: int = 256
    GAMMA: float = 0.99
    TAU: float = 0.005
    
    # Contraction regularization
    CONTRACTION_RATE_ALPHA: float = 0.85
    CONTRACTION_RATE_MIN: float = 0.7
    CONTRACTION_RATE_MAX: float = 0.95
    INITIAL_BETA: float = 0.3
    BETA_MIN: float = 0.05
    BETA_MAX: float = 2.0
    METRIC_REGULARIZATION: float = 0.001
    EPSILON_METRIC: float = 0.05
    TARGET_CONDITION_NUMBER: float = 100.0
    
    # Metric update anchoring (real vs imagined transitions)
    REAL_TRANSITION_WEIGHT: float = 0.8
    IMAGINED_TRANSITION_WEIGHT: float = 0.2
    PER_METRIC_FILTERING: bool = True
    
    # Adaptive identity regularization
    IDENTITY_REG_BASE: float = 0.001
    IDENTITY_REG_MAX: float = 0.05
    CONDITION_NUMBER_THRESHOLD: float = 500.0
    
    # Learning rates
    ACTOR_LR: float = 3e-4
    CRITIC_LR: float = 3e-4
    DYNAMICS_LR: float = 1e-3
    METRIC_LR: float = 5e-5
    
    # Optimization
    REPLAY_BUFFER_SIZE: int = 100000
    INITIAL_EXPLORATION_STEPS: int = 5000
    UPDATE_FREQUENCY: int = 1
    ENSEMBLE_SIZE: int = 7
    LEARNING_START: int = 1000
    UPDATE_EVERY: int = 50
    GRADIENT_STEPS: int = 40
    TARGET_UPDATE_INTERVAL: int = 1
    
    # Adaptive parameters
    BETA_DECAY: float = 0.995
    BETA_INCREASE: float = 1.02
    NOISE_DECAY: float = 0.999
    MIN_NOISE: float = 0.1
    
    # Robustness evaluation
    PERTURBATION_SCALES: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.5])
    PERTURBATION_TIMESTEPS: List[int] = field(default_factory=lambda: [50, 100])
    USE_ADVERSARIAL_PERTURBATIONS: bool = True
    
    # Curriculum
    USE_CURRICULUM: bool = True
    CURRICULUM_STAGES: List[CurriculumStage] = field(default_factory=lambda: [
        CurriculumStage("stability_focus", beta=2.0, duration=50, exploration_scale=1.0, noise_scale=1.0),
        CurriculumStage("performance_focus", beta=1.0, duration=100, exploration_scale=0.8, noise_scale=0.7),
        CurriculumStage("fine_tuning", beta=0.3, duration=50, exploration_scale=0.5, noise_scale=0.3)
    ])
    
    # Meta-learning
    USE_META_LEARNING: bool = True
    META_LEARNING_WINDOW: int = 20
    
    # Safety
    USE_SAFETY_CONSTRAINTS: bool = True
    SAFETY_MARGIN_THRESHOLD: float = 0.1
    
    # Attention
    USE_ATTENTION_METRIC: bool = True
    
    # Geodesic regularization
    USE_GEODESIC_REGULARIZATION: bool = True
    GEODESIC_WEIGHT: float = 0.01
    
    # Experiment
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR: str = "crmbrl_results"
    LOG_INTERVAL: int = 5
    EVAL_INTERVAL: int = 20
    EVAL_EPISODES: int = 5
    PLOT_RESULTS: bool = True
    ROBUSTNESS_EVAL_INTERVAL: int = 10
    
    # Early stopping
    EARLY_STOP_REWARD: float = -300.0
    EARLY_STOP_PATIENCE: int = 50
    
    # Normalization
    REWARD_SCALE: float = 0.1
    STATE_NORMALIZATION: bool = True
    
    # Performance
    USE_JIT: bool = False
    USE_AMP: bool = False
    NUM_PARALLEL_ENVS: int = 1
    
    def __post_init__(self):
        assert 0 < self.CONTRACTION_RATE_ALPHA < 1
        assert self.BATCH_SIZE <= self.REPLAY_BUFFER_SIZE
        assert self.ENSEMBLE_SIZE >= 3
        assert self.REAL_TRANSITION_WEIGHT + self.IMAGINED_TRANSITION_WEIGHT == 1.0
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            config_dict = asdict(self)
            config_dict['CURRICULUM_STAGES'] = [
                asdict(stage) for stage in self.CURRICULUM_STAGES
            ]
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        with open(path, 'r') as f:
            data = json.load(f)
        if 'CURRICULUM_STAGES' in data:
            data['CURRICULUM_STAGES'] = [
                CurriculumStage(**stage) for stage in data['CURRICULUM_STAGES']
            ]
        return cls(**data)
    
    @classmethod
    def from_env(cls, env_name: str, **overrides) -> 'Config':
        """
        Create configuration automatically from environment name.
        Uses predefined environment configurations for optimal defaults.
        
        Args:
            env_name: Gymnasium environment name
            **overrides: Additional configuration overrides
            
        Returns:
            Config instance with environment-specific defaults
        """
        if env_name not in ENV_CONFIGS:
            available = list(ENV_CONFIGS.keys())
            raise ValueError(
                f"Unknown environment: {env_name}. "
                f"Available environments: {available}"
            )
        
        env_config = ENV_CONFIGS[env_name]
        
        # Create base config with environment-specific defaults
        config = cls(
            ENV_NAME=env_name,
            STATE_DIM=env_config["state_dim"],
            ACTION_DIM=env_config["action_dim"],
            MAX_EPISODE_LENGTH=env_config["max_episode_length"],
            ENERGY_TYPE=env_config["energy_type"],
            TASK_MASK=env_config["task_mask"],
            REWARD_SCALE=env_config["reward_scale"],
            TOTAL_EPISODES=env_config["total_episodes"],
            INITIAL_BETA=env_config["initial_beta"],
            EARLY_STOP_REWARD=env_config["early_stop_reward"],
        )
        
        # Apply architecture adjustments for high-dimensional environments
        if config.STATE_DIM > 100:
            config.DYNAMICS_HIDDEN_DIM = 256
            config.POLICY_HIDDEN_DIM = 512
            config.METRIC_HIDDEN_DIM = 256
            config.CRITIC_HIDDEN_DIM = 512
            config.BATCH_SIZE = 128  # Reduce batch size for memory
        
        # Apply any additional overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config


# ============================
# ENVIRONMENT REGISTRY
# ============================

# Predefined configurations for various continuous control environments
# Each entry specifies optimal defaults based on task characteristics
ENV_CONFIGS = {
    # Classic control environments
    "Pendulum-v1": {
        "state_dim": 3,           # cos(theta), sin(theta), theta_dot
        "action_dim": 1,          # Torque
        "max_episode_length": 200,
        "energy_type": "absolute", # Origin is target (pendulum upright)
        "task_mask": [1.0, 1.0, 1.0],
        "reward_scale": 0.1,      # Scale rewards for better gradient flow
        "total_episodes": 200,
        "initial_beta": 0.3,      # Moderate contraction regularization
        "early_stop_reward": -300.0
    },
    
    "MountainCarContinuous-v0": {
        "state_dim": 2,           # Position, velocity
        "action_dim": 1,          # Acceleration
        "max_episode_length": 999,
        "energy_type": "absolute",
        "task_mask": [1.0, 1.0],
        "reward_scale": 0.01,
        "total_episodes": 300,
        "initial_beta": 0.5,      # Higher regularization for sparse rewards
        "early_stop_reward": 90.0
    },
    
    "LunarLanderContinuous-v3": {
        "state_dim": 8,           # Position, velocity, angle, angular velocity, leg contacts
        "action_dim": 2,          # Main engine, side engines
        "max_episode_length": 1000,
        "energy_type": "relative", # Target is landing pad
        "task_mask": [1.0] * 8,
        "reward_scale": 0.01,
        "total_episodes": 500,
        "initial_beta": 0.2,
        "early_stop_reward": 200.0
    },
    
    "BipedalWalker-v3": {
        "state_dim": 24,          # Hull angle, velocity, joint angles, contacts, LIDAR
        "action_dim": 4,          # Hip and knee torques
        "max_episode_length": 1600,
        "energy_type": "relative", # Forward locomotion target
        "task_mask": [1.0] * 24,
        "reward_scale": 0.01,
        "total_episodes": 1000,
        "initial_beta": 0.1,
        "early_stop_reward": 300.0
    },
    
    # MuJoCo locomotion environments (require mujoco installation)
    "HalfCheetah-v4": {
        "state_dim": 17,          # Joint positions, velocities
        "action_dim": 6,          # Joint torques
        "max_episode_length": 1000,
        "energy_type": "relative", # Forward running target
        "task_mask": [1.0] * 17,
        "reward_scale": 0.01,
        "total_episodes": 500,
        "initial_beta": 0.1,
        "early_stop_reward": 10000.0
    },
    
    "Hopper-v4": {
        "state_dim": 11,          # Joint positions, velocities, contact
        "action_dim": 3,          # Hip, knee, ankle torques
        "max_episode_length": 1000,
        "energy_type": "relative", # Hopping target
        "task_mask": [1.0] * 11,
        "reward_scale": 0.01,
        "total_episodes": 500,
        "initial_beta": 0.1,
        "early_stop_reward": 3000.0
    },
    
    "Walker2d-v4": {
        "state_dim": 17,          # Joint positions, velocities
        "action_dim": 6,          # Joint torques
        "max_episode_length": 1000,
        "energy_type": "relative", # Walking target
        "task_mask": [1.0] * 17,
        "reward_scale": 0.01,
        "total_episodes": 500,
        "initial_beta": 0.1,
        "early_stop_reward": 4000.0
    },
    
    "Ant-v4": {
        "state_dim": 27,          # Joint positions, velocities, contact
        "action_dim": 8,          # Joint torques
        "max_episode_length": 1000,
        "energy_type": "relative", # Quadrupedal locomotion
        "task_mask": [1.0] * 27,
        "reward_scale": 0.01,
        "total_episodes": 500,
        "initial_beta": 0.05,     # Lower regularization for complex dynamics
        "early_stop_reward": 6000.0
    },
    
    "Humanoid-v4": {
        "state_dim": 376,         # Full body joint positions, velocities
        "action_dim": 17,         # Joint torques
        "max_episode_length": 1000,
        "energy_type": "relative", # Bipedal locomotion
        "task_mask": [1.0] * 376,
        "reward_scale": 0.001,    # Very small reward scale for high-dimensional
        "total_episodes": 1000,
        "initial_beta": 0.05,
        "early_stop_reward": 5000.0
    },
    
    "Swimmer-v4": {
        "state_dim": 8,           # Joint positions, velocities
        "action_dim": 2,          # Joint torques
        "max_episode_length": 1000,
        "energy_type": "relative", # Swimming target
        "task_mask": [1.0] * 8,
        "reward_scale": 0.01,
        "total_episodes": 300,
        "initial_beta": 0.2,
        "early_stop_reward": 300.0
    },
    
    "InvertedPendulum-v4": {
        "state_dim": 4,           # Cart position, velocity, pole angle, angular velocity
        "action_dim": 1,          # Cart force
        "max_episode_length": 1000,
        "energy_type": "absolute", # Origin is target (pole upright)
        "task_mask": [1.0] * 4,
        "reward_scale": 0.1,
        "total_episodes": 200,
        "initial_beta": 0.3,
        "early_stop_reward": 1000.0
    },
    
    "InvertedDoublePendulum-v4": {
        "state_dim": 11,          # Cart position, velocity, pole angles, angular velocities
        "action_dim": 1,          # Cart force
        "max_episode_length": 1000,
        "energy_type": "absolute", # Origin is target
        "task_mask": [1.0] * 11,
        "reward_scale": 0.01,
        "total_episodes": 300,
        "initial_beta": 0.2,
        "early_stop_reward": 9000.0
    },
    
    "Reacher-v4": {
        "state_dim": 11,          # Arm joint angles, velocities, target position
        "action_dim": 2,          # Joint torques
        "max_episode_length": 50,
        "energy_type": "relative", # Target reaching
        "task_mask": [1.0] * 11,
        "reward_scale": 0.1,
        "total_episodes": 200,
        "initial_beta": 0.2,
        "early_stop_reward": -5.0
    }
}


# ============================
# UTILITY FUNCTIONS
# ============================

def set_seed(seed: int):
    """Set random seeds for reproducibility across all libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_env(env_name: str, render_mode: str = None) -> gym.Env:
    """
    Create a Gymnasium environment with proper error handling.
    
    Args:
        env_name: Name of the environment to create
        render_mode: Optional render mode ('human', 'rgb_array')
        
    Returns:
        Gymnasium environment instance
        
    Raises:
        gym.error.DependencyNotInstalled: If required dependencies are missing
    """
    try:
        env = gym.make(env_name, render_mode=render_mode)
        return env
    except gym.error.DependencyNotInstalled as e:
        print(f"\nEnvironment '{env_name}' requires additional dependencies:")
        print(f"  {e}")
        print(f"\n  For MuJoCo: pip install gymnasium[mujoco]")
        print(f"  For Box2D: pip install gymnasium[box2d]")
        raise
    except Exception as e:
        print(f"\nFailed to create environment '{env_name}': {e}")
        raise

def get_available_envs() -> List[str]:
    """
    Get list of environments that can be successfully loaded.
    
    Returns:
        List of environment names that are available on this system
    """
    available = []
    for env_name in ENV_CONFIGS.keys():
        try:
            env = gym.make(env_name)
            env.close()
            available.append(env_name)
        except:
            pass
    return available

def check_mujoco_available() -> bool:
    """
    Check if MuJoCo environments are available.
    
    Returns:
        True if MuJoCo can be loaded, False otherwise
    """
    try:
        env = gym.make("HalfCheetah-v4")
        env.close()
        return True
    except:
        return False

def list_available_envs():
    """Print a formatted list of all available environments with their properties."""
    print("\n" + "="*80)
    print("AVAILABLE ENVIRONMENTS FOR CR-MBRL")
    print("="*80)
    
    classic = []
    mujoco = []
    
    for env_name in ENV_CONFIGS.keys():
        try:
            env = gym.make(env_name)
            env.close()
            config = ENV_CONFIGS[env_name]
            if env_name.endswith("-v4") or env_name.endswith("-v3"):
                if env_name not in ["LunarLanderContinuous-v3", "BipedalWalker-v3"]:
                    mujoco.append((env_name, config))
                else:
                    classic.append((env_name, config))
            else:
                classic.append((env_name, config))
        except:
            pass
    
    if classic:
        print("\nClassic Control / Box2D Environments:")
        print("-" * 50)
        for env_name, config in classic:
            print(f"  {env_name:<35} | State: {config['state_dim']:>4} | Action: {config['action_dim']:>2} | "
                  f"Energy: {config['energy_type']:<10} | Episodes: {config['total_episodes']:>5}")
    
    if mujoco:
        print("\nMuJoCo Environments (requires 'pip install gymnasium[mujoco]'):")
        print("-" * 50)
        for env_name, config in mujoco:
            print(f"  {env_name:<35} | State: {config['state_dim']:>4} | Action: {config['action_dim']:>2} | "
                  f"Energy: {config['energy_type']:<10} | Episodes: {config['total_episodes']:>5}")
    
    missing = [env for env in ENV_CONFIGS if env not in dict(classic + mujoco)]
    if missing:
        print(f"\nUnavailable environments: {missing}")
    
    print("\n" + "="*80)

def compute_significance(results_baseline: List[float], 
                         results_crmbrl: List[float], 
                         alpha: float = 0.05) -> Dict:
    """
    Statistical significance testing for CR-MBRL vs baseline.
    
    Performs Welch's t-test, Cohen's d effect size, and bootstrap confidence intervals.
    
    Args:
        results_baseline: List of baseline method results
        results_crmbrl: List of CR-MBRL results
        alpha: Significance level
        
    Returns:
        Dictionary with test statistics and significance indicators
    """
    
    # Welch's t-test (does not assume equal variance)
    t_stat, p_value = stats.ttest_ind(results_crmbrl, results_baseline, equal_var=False)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(results_baseline)**2 + np.std(results_crmbrl)**2) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(results_crmbrl) - np.mean(results_baseline)) / pooled_std
    else:
        cohens_d = 0.0
    
    # Bootstrap confidence interval for the mean difference
    n_bootstrap = 10000
    differences = []
    for _ in range(n_bootstrap):
        sample_baseline = np.random.choice(results_baseline, size=len(results_baseline), replace=True)
        sample_crmbrl = np.random.choice(results_crmbrl, size=len(results_crmbrl), replace=True)
        differences.append(np.mean(sample_crmbrl) - np.mean(sample_baseline))
    
    ci_lower = np.percentile(differences, 2.5)
    ci_upper = np.percentile(differences, 97.5)
    
    return {
        'p_value': p_value,
        'significant': p_value < alpha,
        'cohens_d': cohens_d,
        'ci_95': (ci_lower, ci_upper),
        'mean_difference': np.mean(differences),
        't_statistic': t_stat
    }


# ============================
# NETWORK ARCHITECTURES
# ============================

class EnhancedDynamics(nn.Module):
    """
    Enhanced dynamics model with residual connections and layer normalization.
    
    Predicts state transitions: s_{t+1} - s_t = f(s_t, a_t)
    Uses orthogonal initialization for stable training.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
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
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with orthogonal initialization for stable gradients."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict state residual given current state and action."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        x = torch.cat([state, action], dim=-1)
        x = self.encoder(x)
        
        residual = x
        x = self.res_block1(x)
        x = F.relu(x + residual)
        
        residual = x
        x = self.res_block2(x)
        x = F.relu(x + residual)
        
        return self.output(x)


class DynamicsEnsemble(nn.Module):
    """
    Ensemble of dynamics models with uncertainty quantification.
    
    Maintains multiple independent dynamics models and computes
    epistemic uncertainty as variance across ensemble predictions.
    """
    def __init__(self, state_dim: int, action_dim: int, 
                 ensemble_size: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.ensemble_size = ensemble_size
        
        self.models = nn.ModuleList([
            EnhancedDynamics(state_dim, action_dim, hidden_dim)
            for _ in range(ensemble_size)
        ])
        
        self.model_weights = nn.Parameter(torch.ones(ensemble_size) / ensemble_size)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Returns:
            Tuple of (weighted mean prediction, epistemic uncertainty)
        """
        predictions = []
        for model in self.models:
            pred = model(state, action)
            predictions.append(pred.unsqueeze(0))
        
        predictions = torch.cat(predictions, dim=0)
        weights = F.softmax(self.model_weights, dim=0)
        weighted_predictions = predictions * weights.view(-1, 1, 1)
        mean = weighted_predictions.sum(dim=0)
        
        # Epistemic uncertainty as variance across ensemble
        variance = torch.var(predictions, dim=0, unbiased=True)
        epistemic = variance.mean(dim=-1, keepdim=True)
        
        return mean, epistemic
    
    def sample_model(self) -> nn.Module:
        """Sample a single model from the ensemble for model-based rollouts."""
        return random.choice(self.models)


class AttentionBasedMetric(nn.Module):
    """
    Contraction metric network with attention mechanism.
    
    Learns a state-dependent Riemannian metric M(x) = L(x)L(x)^T + eps*I
    with multi-head attention for identifying important state dimensions.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128, 
                 num_heads: int = 3, epsilon: float = 0.05):
        super().__init__()
        self.state_dim = state_dim
        self.epsilon = epsilon
        
        # Output dimension for lower triangular Cholesky factor
        self.output_dim = (state_dim * (state_dim + 1)) // 2
        self.num_heads = min(num_heads, state_dim)
        
        # Attention mechanism for state-dependent importance weighting
        if state_dim >= self.num_heads:
            self.attention = nn.MultiheadAttention(
                embed_dim=state_dim,
                num_heads=self.num_heads,
                batch_first=True
            )
            self.use_attention = True
        else:
            self.use_attention = False
        
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
        
        if self.use_attention:
            self.temperature = nn.Parameter(torch.ones(1))
        
        self.softplus = nn.Softplus(beta=1.0, threshold=20)
        self.diagonal_offset = 0.01
        self.off_diagonal_scale = 0.1
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for stable metric learning."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.1)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute state-dependent metric matrix.
        
        Args:
            x: State tensor [batch_size, state_dim]
            
        Returns:
            Tuple of (metric matrix M, Cholesky factor L)
        """
        batch_size = x.shape[0]
        
        # Apply attention for state importance
        if self.use_attention:
            x_expanded = x.unsqueeze(1)
            attended, _ = self.attention(x_expanded, x_expanded, x_expanded)
            x_enhanced = (x + attended.squeeze(1)) * self.temperature
        else:
            x_enhanced = x
        
        l_params = self.net(x_enhanced)
        
        # Construct lower triangular matrix L
        L = torch.zeros(batch_size, self.state_dim, self.state_dim, 
                       device=x.device, dtype=x.dtype)
        
        idx = 0
        for i in range(self.state_dim):
            for j in range(i + 1):
                val = l_params[:, idx]
                if i == j:
                    L[:, i, j] = self.softplus(val) + self.diagonal_offset
                else:
                    L[:, i, j] = torch.tanh(val) * self.off_diagonal_scale
                idx += 1
        
        # M = LL^T + eps*I ensures positive definiteness
        LLT = torch.bmm(L, L.transpose(1, 2))
        identity = torch.eye(self.state_dim, device=x.device, dtype=x.dtype)
        identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
        
        M = LLT + self.epsilon * identity
        
        # Numerical safety check
        try:
            torch.linalg.cholesky(M)
        except RuntimeError:
            M = M + 0.1 * identity
        
        return M, L
    
    def compute_metrics(self, M: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute diagnostic metrics for the learned metric."""
        batch_size = M.shape[0]
        metrics = {}
        
        eigenvalues = torch.linalg.eigvalsh(M)
        min_eigenvalues = eigenvalues.min(dim=1).values
        max_eigenvalues = eigenvalues.max(dim=1).values
        
        metrics['min_eigenvalue'] = min_eigenvalues.mean()
        metrics['max_eigenvalue'] = max_eigenvalues.mean()
        metrics['condition_number'] = (max_eigenvalues / torch.clamp(min_eigenvalues, min=1e-6)).mean()
        metrics['det'] = torch.det(M).mean()
        
        return metrics


class RobustContractionMetric(nn.Module):
    """
    Standard contraction metric network without attention.
    
    Learns M(x) = L(x)L(x)^T + eps*I where L is lower triangular.
    More computationally efficient than attention-based variant.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128, epsilon: float = 0.05):
        super().__init__()
        self.state_dim = state_dim
        self.epsilon = epsilon
        
        self.output_dim = (state_dim * (state_dim + 1)) // 2
        
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
        
        self.softplus = nn.Softplus(beta=1.0, threshold=20)
        self.diagonal_offset = 0.01
        self.off_diagonal_scale = 0.1
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.1)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        l_params = self.net(x)
        
        L = torch.zeros(batch_size, self.state_dim, self.state_dim, 
                       device=x.device, dtype=x.dtype)
        
        idx = 0
        for i in range(self.state_dim):
            for j in range(i + 1):
                val = l_params[:, idx]
                if i == j:
                    L[:, i, j] = self.softplus(val) + self.diagonal_offset
                else:
                    L[:, i, j] = torch.tanh(val) * self.off_diagonal_scale
                idx += 1
        
        LLT = torch.bmm(L, L.transpose(1, 2))
        identity = torch.eye(self.state_dim, device=x.device, dtype=x.dtype)
        identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
        
        M = LLT + self.epsilon * identity
        
        try:
            torch.linalg.cholesky(M)
        except RuntimeError:
            M = M + 0.1 * identity
        
        return M, L
    
    def compute_metrics(self, M: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = M.shape[0]
        metrics = {}
        
        eigenvalues = torch.linalg.eigvalsh(M)
        min_eigenvalues = eigenvalues.min(dim=1).values
        max_eigenvalues = eigenvalues.max(dim=1).values
        
        metrics['min_eigenvalue'] = min_eigenvalues.mean()
        metrics['max_eigenvalue'] = max_eigenvalues.mean()
        metrics['condition_number'] = (max_eigenvalues / torch.clamp(min_eigenvalues, min=1e-6)).mean()
        metrics['det'] = torch.det(M).mean()
        
        return metrics


class EnhancedPolicyNetwork(nn.Module):
    """
    Gaussian policy network for continuous action spaces.
    
    Outputs mean action with learnable state-independent standard deviation.
    Uses tanh squashing to bound actions to [-action_scale, action_scale].
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        
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
        
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)
        self.action_scale = 2.0
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.distributions.Normal:
        """Return action distribution given state."""
        mean = self.mean_net(state) * self.action_scale
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)
    
    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """Return deterministic action (mean of distribution)."""
        with torch.no_grad():
            return self.mean_net(state) * self.action_scale
    
    def sample_with_entropy(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return log probability for entropy regularization."""
        dist = self(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob


class EnhancedValueNetwork(nn.Module):
    """
    Twin Q-network for double Q-learning to reduce overestimation bias.
    
    Uses two independent Q-value estimators and takes the minimum
    for conservative value estimates.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
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
        """Return both Q-value estimates."""
        q1 = self.q1_net(state)
        q2 = self.q2_net(state)
        return q1, q2
    
    def q_min(self, state: torch.Tensor) -> torch.Tensor:
        """Return minimum Q-value for conservative estimation."""
        q1, q2 = self(state)
        return torch.min(q1, q2)


# ============================
# ENHANCED RIEMANNIAN OPERATIONS
# ============================

class EnhancedRiemannianOperations:
    """
    Riemannian geometry operations for contraction analysis.
    
    Implements various energy function formulations and contraction
    condition verification using learned state-dependent metrics.
    """
    
    @staticmethod
    def compute_energy_absolute(state: torch.Tensor, 
                                metric_net: Union[RobustContractionMetric, AttentionBasedMetric]
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute energy E(x) = x^T M(x) x (assumes origin is target).
        
        Used for tasks where the goal is to reach/stabilize at zero state,
        such as pendulum swing-up or cartpole balancing.
        """
        M, _ = metric_net(state)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state_expanded = state.unsqueeze(1)
        energy = torch.bmm(torch.bmm(state_expanded, M), 
                          state_expanded.transpose(1, 2)).squeeze()
        
        energy = torch.clamp(energy, min=1e-6, max=1e6)
        return energy, M
    
    @staticmethod
    def compute_energy_relative(state: torch.Tensor, 
                                target_state: torch.Tensor,
                                metric_net: Union[RobustContractionMetric, AttentionBasedMetric]
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute energy E(x) = (x - x_target)^T M(x) (x - x_target).
        
        Used for locomotion tasks where we track a moving target
        or maintain a specific configuration relative to a reference.
        """
        delta = state - target_state
        M, _ = metric_net(delta)
        
        if delta.dim() == 1:
            delta = delta.unsqueeze(0)
        
        delta_expanded = delta.unsqueeze(1)
        energy = torch.bmm(torch.bmm(delta_expanded, M), 
                          delta_expanded.transpose(1, 2)).squeeze()
        
        energy = torch.clamp(energy, min=1e-6, max=1e6)
        return energy, M
    
    @staticmethod
    def compute_energy_masked(state: torch.Tensor, 
                              mask: torch.Tensor,
                              metric_net: Union[RobustContractionMetric, AttentionBasedMetric]
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute energy E(x) = (mask * x)^T M(mask * x) (mask * x).
        
        Used when we only care about contraction in certain state dimensions,
        e.g., position but not velocity, or specific joint angles.
        """
        masked_state = state * mask
        
        M, _ = metric_net(masked_state)
        
        if masked_state.dim() == 1:
            masked_state = masked_state.unsqueeze(0)
        
        state_expanded = masked_state.unsqueeze(1)
        energy = torch.bmm(torch.bmm(state_expanded, M), 
                          state_expanded.transpose(1, 2)).squeeze()
        
        energy = torch.clamp(energy, min=1e-6, max=1e6)
        return energy, M
    
    @staticmethod
    def compute_energy(state: torch.Tensor,
                      metric_net: Union[RobustContractionMetric, AttentionBasedMetric],
                      energy_type: str = "absolute",
                      target_state: Optional[torch.Tensor] = None,
                      mask: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch to appropriate energy computation based on type.
        
        Args:
            state: Current state tensor
            metric_net: Learned metric network
            energy_type: "absolute", "relative", or "masked"
            target_state: Target state for relative energy (optional)
            mask: State dimension mask for masked energy (optional)
            
        Returns:
            Tuple of (energy values, metric matrices)
        """
        if energy_type == "relative" and target_state is not None:
            return EnhancedRiemannianOperations.compute_energy_relative(
                state, target_state, metric_net
            )
        elif energy_type == "masked" and mask is not None:
            return EnhancedRiemannianOperations.compute_energy_masked(
                state, mask, metric_net
            )
        else:
            return EnhancedRiemannianOperations.compute_energy_absolute(
                state, metric_net
            )
    
    @staticmethod
    def condition_metric(M: torch.Tensor, target_condition_number: float = 100.0) -> torch.Tensor:
        """
        Ensure metric matrix has bounded condition number.
        
        Prevents numerical instability by clamping eigenvalues
        to maintain a reasonable condition number.
        """
        batch_size = M.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(M)
        
        max_eigenvalues = eigenvalues.max(dim=1, keepdim=True).values
        min_eigenvalues = max_eigenvalues / target_condition_number
        
        eigenvalues = torch.clamp(eigenvalues, min=min_eigenvalues, max=max_eigenvalues)
        
        M_conditioned = torch.bmm(
            eigenvectors,
            torch.bmm(torch.diag_embed(eigenvalues), eigenvectors.transpose(1, 2))
        )
        
        return M_conditioned
    
    @staticmethod
    def compute_geodesic_regularization(
        states: torch.Tensor, 
        metric_net: Union[RobustContractionMetric, AttentionBasedMetric]
    ) -> torch.Tensor:
        """
        Smoothness constraint on metric variation along geodesic paths.
        
        Penalizes rapid changes in the learned metric to ensure
        the contraction property holds across the state space.
        """
        num_interp = 5
        directions = torch.randn_like(states)
        directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-8)
        
        interp_points = []
        for t in torch.linspace(0, 1, num_interp, device=states.device):
            interp_points.append(states + t * directions * 0.1)
        
        interp_tensor = torch.cat(interp_points, dim=0)
        M_interp, _ = metric_net(interp_tensor)
        
        metric_diffs = []
        for i in range(num_interp - 1):
            diff = M_interp[i::num_interp] - M_interp[(i+1)::num_interp]
            metric_diffs.append(torch.norm(diff, dim=(1, 2)))
        
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
        geodesic_weight: float = 0.01,
        energy_type: str = "absolute",
        target_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        adaptive_identity_weight: float = 0.001
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute contraction loss with adaptive identity regularization.
        
        The contraction condition requires: E(x_{t+1}) <= alpha^2 * E(x_t)
        where alpha < 1 is the contraction rate. The loss penalizes violations
        using a softplus surrogate.
        
        Args:
            states: Current state batch
            next_states: Next state batch
            metric_net: Learned metric network
            alpha: Contraction rate (0 < alpha < 1)
            beta: Temperature for softplus sharpness
            use_geodesic_reg: Whether to use geodesic regularization
            geodesic_weight: Weight for geodesic smoothness loss
            energy_type: Type of energy formulation
            target_state: Target state for relative energy
            mask: Dimension mask for masked energy
            adaptive_identity_weight: Weight for identity regularization
            
        Returns:
            Tuple of (total loss, dictionary of component metrics)
        """
        
        energy_curr, M_curr = EnhancedRiemannianOperations.compute_energy(
            states, metric_net, energy_type, target_state, mask
        )
        energy_next, M_next = EnhancedRiemannianOperations.compute_energy(
            next_states, metric_net, energy_type, target_state, mask
        )
        
        # Softplus surrogate for contraction violation: softplus(beta * (E_next - alpha^2 * E_curr))
        energy_diff = energy_next - (alpha**2) * energy_curr
        contraction_loss = F.softplus(beta * energy_diff).mean() / beta
        
        # Symmetry loss to ensure M is symmetric
        symmetry_loss = torch.norm(M_curr - M_curr.transpose(1, 2), dim=(1, 2)).mean()
        
        # Adaptive identity regularization based on condition number
        eigenvalues = torch.linalg.eigvalsh(M_curr)
        condition_numbers = eigenvalues.max(dim=1).values / torch.clamp(
            eigenvalues.min(dim=1).values, min=1e-6
        )
        
        # Scale identity regularization when condition number is high
        condition_ratio = condition_numbers / 100.0
        scaled_identity_weight = adaptive_identity_weight * torch.clamp(condition_ratio, min=1.0, max=50.0)
        
        identity = torch.eye(states.shape[-1], device=states.device)
        identity = identity.unsqueeze(0).expand_as(M_curr)
        identity_loss = (scaled_identity_weight.unsqueeze(1).unsqueeze(2) * 
                        torch.norm(M_curr - identity, dim=(1, 2))).mean()
        
        # Geodesic regularization for metric smoothness
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
            'geodesic_loss': geodesic_loss,
            'condition_number': condition_numbers.mean(),
            'scaled_identity_weight': scaled_identity_weight.mean()
        }
        
        total_loss = (contraction_loss + 
                     0.01 * symmetry_loss + 
                     identity_loss +
                     geodesic_weight * geodesic_loss)
        
        return total_loss, metrics
    
    @staticmethod
    def compute_safety_margin(
        state: torch.Tensor, 
        metric_net: Union[RobustContractionMetric, AttentionBasedMetric]
    ) -> torch.Tensor:
        """
        Compute safety margin using contraction metric.
        
        Estimates distance to unsafe regions using the learned
        Riemannian metric, providing a measure of safety.
        """
        M, _ = metric_net(state)
        
        state_expanded = state.unsqueeze(1)
        distance = torch.sqrt(
            torch.abs(torch.bmm(state_expanded, 
                              torch.bmm(M, state_expanded.transpose(1, 2))))
        ).squeeze()
        
        return distance


# ============================
# REPLAY BUFFER WITH PER FILTERING
# ============================

class PrioritizedReplayBuffer:
    """
    Prioritized experience replay with dual priority queues.
    
    Maintains separate priorities for:
    1. TD-error based sampling (for value learning)
    2. Contraction-error based sampling (for metric learning)
    
    This allows the metric to focus on transitions where contraction
    violations are most severe.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling correction
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.contraction_priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self._max_priority = 1.0
        self._max_contraction_priority = 1.0
    
    def push(self, state: np.ndarray, action: np.ndarray, 
             reward: float, next_state: np.ndarray, done: bool):
        """
        Add experience with maximum priority to ensure it gets sampled
        at least once.
        """
        experience = (state, action, reward, next_state, done)
        
        if self.size < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = self._max_priority ** self.alpha
        self.contraction_priorities[self.position] = self._max_contraction_priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, use_contraction_priority: bool = False) -> Optional[Tuple]:
        """
        Sample batch with priorities.
        
        Args:
            batch_size: Number of transitions to sample
            use_contraction_priority: If True, use contraction-error priorities;
                                     if False, use TD-error priorities
        """
        if self.size < batch_size:
            return None
        
        if use_contraction_priority:
            priorities = self.contraction_priorities[:self.size]
        else:
            priorities = self.priorities[:self.size]
        
        probs = priorities / priorities.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
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
        """
        Update TD-error based priorities.
        
        Higher TD-error transitions get higher priority for value learning.
        """
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            self.priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)
    
    def update_contraction_priorities(self, indices: np.ndarray, contraction_errors: np.ndarray):
        """
        Update contraction-error based priorities.
        
        Higher contraction violations get higher priority for metric learning.
        """
        for idx, error in zip(indices, contraction_errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            self.contraction_priorities[idx] = priority
            self._max_contraction_priority = max(self._max_contraction_priority, priority)
    
    def __len__(self):
        return self.size


# ============================
# CURRICULUM SCHEDULER
# ============================

class CurriculumScheduler:
    """
    Progressive difficulty scheduler for stability learning.
    
    Transitions through stages:
    1. Stability focus: High contraction regularization
    2. Performance focus: Balanced reward and stability
    3. Fine tuning: Low regularization, high performance
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.current_stage = 0
        self.stages = config.CURRICULUM_STAGES
        self.episodes_in_stage = 0
        self.enabled = config.USE_CURRICULUM
    
    def get_parameters(self, episode: int) -> Dict:
        """
        Get curriculum parameters for current episode.
        
        Smoothly interpolates between stages for continuous adaptation.
        """
        if not self.enabled:
            return {
                'beta': self.config.INITIAL_BETA,
                'exploration_scale': 1.0,
                'noise_scale': 1.0
            }
        
        # Move to next stage if current stage duration exceeded
        if self.episodes_in_stage >= self.stages[self.current_stage].duration:
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                self.episodes_in_stage = 0
        
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
    """
    Meta-controller for adaptive hyperparameter tuning.
    
    Adjusts contraction rate based on energy trends:
    - If energy is increasing, tighten contraction (lower alpha)
    - If energy is decreasing, relax contraction (raise alpha)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.enabled = config.USE_META_LEARNING
        self.performance_history = deque(maxlen=config.META_LEARNING_WINDOW)
        self.energy_history = deque(maxlen=config.META_LEARNING_WINDOW)
        self.contraction_rate_history = deque(maxlen=config.META_LEARNING_WINDOW)
    
    def update(self, reward: float, energy: float):
        """Update history with latest metrics."""
        self.performance_history.append(reward)
        self.energy_history.append(energy)
    
    def suggest_contraction_rate(self) -> float:
        """
        Suggest contraction rate based on energy trends.
        
        Returns:
            Adjusted contraction rate alpha
        """
        if not self.enabled or len(self.energy_history) < 10:
            return self.config.CONTRACTION_RATE_ALPHA
        
        energy_array = np.array(list(self.energy_history))
        energy_trend = np.polyfit(range(len(energy_array)), energy_array, 1)[0] if len(energy_array) >= 2 else 0
        
        alpha = self.config.CONTRACTION_RATE_ALPHA
        
        if energy_trend > 0:
            # Energy increasing - tighten contraction
            alpha *= 0.98
        else:
            # Energy decreasing - relax contraction
            alpha *= 1.01
        
        alpha = np.clip(alpha, self.config.CONTRACTION_RATE_MIN, self.config.CONTRACTION_RATE_MAX)
        self.contraction_rate_history.append(alpha)
        self.config.CONTRACTION_RATE_ALPHA = alpha
        
        return alpha


# ============================
# ADAPTIVE EXPLORATION
# ============================

class AdaptiveExploration:
    """
    Multi-strategy adaptive exploration for continuous action spaces.
    
    Implements three exploration strategies:
    1. Ornstein-Uhlenbeck process (temporally correlated noise)
    2. Gaussian noise (uncorrelated exploration)
    3. Parameter space noise (state-dependent exploration)
    
    Adapts exploration rate based on recent performance.
    """
    
    def __init__(self, action_dim: int, config: Config):
        self.action_dim = action_dim
        self.config = config
        
        self.ou_noise = self._init_ou_noise()
        self.gaussian_noise = self._init_gaussian_noise()
        self.parameter_noise = self._init_parameter_noise()
        
        self.strategies = ['ou', 'gaussian', 'parameter']
        self.strategy_weights = np.ones(3) / 3
        self.strategy_success = np.ones(3)
        
        self.recent_rewards = deque(maxlen=50)
        self.exploration_rate = 1.0
    
    def _init_ou_noise(self):
        """Initialize Ornstein-Uhlenbeck noise parameters."""
        return {
            'theta': 0.15, 'mu': 0.0, 'sigma': 0.3,
            'dt': 1e-2, 'state': np.zeros(self.action_dim)
        }
    
    def _init_gaussian_noise(self):
        """Initialize Gaussian noise parameters."""
        return {'sigma': 0.3, 'decay': 0.999}
    
    def _init_parameter_noise(self):
        """Initialize parameter space noise parameters."""
        return {'scale': 0.1, 'adaptation_rate': 1.01}
    
    def sample(self, strategy: str = None, scale: float = 1.0) -> np.ndarray:
        """
        Sample exploration noise.
        
        Args:
            strategy: Specific strategy or None for adaptive selection
            scale: Additional noise scaling factor
            
        Returns:
            Noise array to add to actions
        """
        if strategy is None:
            probs = self.strategy_weights * self.strategy_success
            probs = probs / probs.sum()
            strategy_idx = np.random.choice(len(self.strategies), p=probs)
            strategy = self.strategies[strategy_idx]
        
        if strategy == 'ou':
            ou = self.ou_noise
            dx = ou['theta'] * (ou['mu'] - ou['state']) * ou['dt']
            dx += ou['sigma'] * np.sqrt(ou['dt']) * np.random.randn(self.action_dim)
            ou['state'] += dx
            noise = ou['state'].copy()
        elif strategy == 'gaussian':
            gaussian = self.gaussian_noise
            noise = np.random.randn(self.action_dim) * gaussian['sigma']
            gaussian['sigma'] *= gaussian['decay']
            gaussian['sigma'] = max(gaussian['sigma'], 0.05)
        elif strategy == 'parameter':
            noise = np.random.randn(self.action_dim) * self.parameter_noise['scale']
            noise *= self.exploration_rate
        else:
            noise = np.zeros(self.action_dim)
        
        noise *= self.exploration_rate * scale
        return noise
    
    def update(self, episode_reward: float, episode: int):
        """
        Update exploration parameters based on performance.
        
        Reduces exploration when performing well, increases when struggling.
        """
        self.recent_rewards.append(episode_reward)
        
        if len(self.recent_rewards) >= 10:
            recent_avg = np.mean(list(self.recent_rewards))
            
            if recent_avg > -500:
                self.exploration_rate *= 0.98
            elif recent_avg > -800:
                self.exploration_rate *= 0.995
            else:
                self.exploration_rate = min(1.0, self.exploration_rate * 1.02)
            
            self.exploration_rate = max(0.1, min(1.0, self.exploration_rate))
            
            if recent_avg < -1000:
                self.parameter_noise['scale'] = min(0.5, self.parameter_noise['scale'] * 1.05)
            else:
                self.parameter_noise['scale'] *= 0.99
        
        self.exploration_rate *= self.config.NOISE_DECAY
        self.exploration_rate = max(self.config.MIN_NOISE, self.exploration_rate)
    
    def reset(self):
        """Reset Ornstein-Uhlenbeck state for new episode."""
        self.ou_noise['state'] = np.zeros(self.action_dim)


# ============================
# ROBUSTNESS EVALUATOR
# ============================

class RobustnessEvaluator:
    """
    Standardized perturbation testing for robustness evaluation.
    
    Tests policy robustness against:
    1. Gaussian perturbations: Random state disturbances
    2. Adversarial perturbations: Worst-case disturbances in energy-maximizing direction
    """
    
    def __init__(self, 
                 perturbation_scales: List[float] = None,
                 perturbation_timesteps: List[int] = None,
                 use_adversarial: bool = True):
        self.perturbation_scales = perturbation_scales or [0.01, 0.05, 0.1, 0.2, 0.5]
        self.perturbation_timesteps = perturbation_timesteps or [50, 100]
        self.use_adversarial = use_adversarial
    
    def compute_adversarial_perturbation(self, state: torch.Tensor, 
                                         metric_net: nn.Module,
                                         sigma: float) -> np.ndarray:
        """
        Compute worst-case perturbation that maximizes energy.
        
        Uses the gradient of the learned energy function to find
        the direction that most violates contraction.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        state_tensor.requires_grad_(True)
        
        energy, _ = EnhancedRiemannianOperations.compute_energy_absolute(
            state_tensor, metric_net
        )
        
        energy.backward()
        gradient = state_tensor.grad.data.cpu().numpy()[0]
        
        # Normalize and scale to specified magnitude
        grad_norm = np.linalg.norm(gradient) + 1e-8
        adversarial_direction = gradient / grad_norm
        
        return sigma * np.sign(adversarial_direction)
    
    def evaluate(self, agent, env: gym.Env, num_episodes: int = 10) -> Dict:
        """
        Evaluate robustness across perturbation scales.
        
        Returns:
            Dictionary of results for each perturbation type and scale
        """
        results = {}
        
        for sigma in self.perturbation_scales:
            gaussian_results = self._evaluate_gaussian(agent, env, sigma, num_episodes)
            results[f'gaussian_sigma_{sigma}'] = gaussian_results
            
            if self.use_adversarial:
                adversarial_results = self._evaluate_adversarial(agent, env, sigma, num_episodes)
                results[f'adversarial_sigma_{sigma}'] = adversarial_results
        
        return results
    
    def _evaluate_gaussian(self, agent, env: gym.Env, sigma: float, 
                          num_episodes: int) -> Dict:
        """Evaluate with Gaussian perturbations at specified timesteps."""
        episode_rewards = []
        recovery_times = []
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            perturbation_applied = False
            recovery_steps = 0
            
            for step in range(env.spec.max_episode_steps if hasattr(env.spec, 'max_episode_steps') else 200):
                if step in self.perturbation_timesteps and not perturbation_applied:
                    state += np.random.randn(*state.shape) * sigma
                    perturbation_applied = True
                
                action = agent.select_action(state, deterministic=True, use_exploration=False)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                if perturbation_applied:
                    recovery_steps += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            recovery_times.append(recovery_steps)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_recovery_steps': np.mean(recovery_times),
            'rewards': episode_rewards
        }
    
    def _evaluate_adversarial(self, agent, env: gym.Env, sigma: float,
                             num_episodes: int) -> Dict:
        """Evaluate with adversarial perturbations using metric gradient."""
        episode_rewards = []
        recovery_times = []
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            perturbation_applied = False
            recovery_steps = 0
            
            for step in range(env.spec.max_episode_steps if hasattr(env.spec, 'max_episode_steps') else 200):
                if step in self.perturbation_timesteps and not perturbation_applied:
                    adv_perturbation = self.compute_adversarial_perturbation(
                        state, agent.metric_net, sigma
                    )
                    state += adv_perturbation
                    perturbation_applied = True
                
                action = agent.select_action(state, deterministic=True, use_exploration=False)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                if perturbation_applied:
                    recovery_steps += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            recovery_times.append(recovery_steps)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_recovery_steps': np.mean(recovery_times),
            'rewards': episode_rewards
        }


# ============================
# CR-MBRL AGENT
# ============================

class EnhancedContractionDynamicsAgent:
    """
    CR-MBRL Agent with contraction-regularized model-based RL.
    
    Key features:
    - Energy formulation options (absolute, relative, masked) for different tasks
    - Real-transition anchored metric updates (80/20 split with PER filtering)
    - Adaptive identity regularization based on metric condition number
    - Robustness evaluation with Gaussian and adversarial perturbations
    - Curriculum learning for progressive stability training
    - Meta-learning for adaptive contraction rate adjustment
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Task-specific components
        self.task_mask = torch.tensor(config.TASK_MASK, device=self.device)
        self.target_state = torch.zeros(config.STATE_DIM, device=self.device)
        
        # Adaptive regularization
        self.current_identity_weight = config.IDENTITY_REG_BASE
        
        self._initialize_networks()
        self._initialize_optimizers()
        
        # Experience replay
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.REPLAY_BUFFER_SIZE, alpha=0.6, beta=0.4
        )
        
        # Training components
        self.exploration = AdaptiveExploration(config.ACTION_DIM, config)
        self.curriculum = CurriculumScheduler(config)
        self.meta_controller = MetaLearningController(config)
        self.robustness_evaluator = RobustnessEvaluator(
            perturbation_scales=config.PERTURBATION_SCALES,
            perturbation_timesteps=config.PERTURBATION_TIMESTEPS,
            use_adversarial=config.USE_ADVERSARIAL_PERTURBATIONS
        )
        
        # Adaptive parameters
        self.beta = config.INITIAL_BETA
        self.contraction_alpha = config.CONTRACTION_RATE_ALPHA
        
        # State normalization
        self.state_mean = np.zeros(config.STATE_DIM)
        self.state_std = np.ones(config.STATE_DIM)
        
        # Metrics tracking
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
            'condition_numbers': [],
            'identity_weights': [],
            'robustness_results': [],
            'grad_norms': defaultdict(list),
            'curriculum_stages': [],
            'geodesic_losses': []
        }
        
        # Results directory
        self.save_dir = Path(config.SAVE_DIR)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        config.save(self.save_dir / "config.json")
        
        print(f"CR-MBRL Agent initialized on {self.device}")
        print(f"  Environment: {config.ENV_NAME}")
        print(f"  State dim: {config.STATE_DIM}, Action dim: {config.ACTION_DIM}")
        print(f"  Energy formulation: {config.ENERGY_TYPE}")
        print(f"  Real transition weight: {config.REAL_TRANSITION_WEIGHT}")
        print(f"  Adaptive identity reg: enabled (base={config.IDENTITY_REG_BASE})")
    
    def _initialize_networks(self):
        """Initialize all neural network components."""
        self.dynamics = DynamicsEnsemble(
            self.config.STATE_DIM, self.config.ACTION_DIM,
            self.config.ENSEMBLE_SIZE, self.config.DYNAMICS_HIDDEN_DIM
        ).to(self.device)
        
        self.policy = EnhancedPolicyNetwork(
            self.config.STATE_DIM, self.config.ACTION_DIM, self.config.POLICY_HIDDEN_DIM
        ).to(self.device)
        
        self.critic = EnhancedValueNetwork(
            self.config.STATE_DIM, self.config.CRITIC_HIDDEN_DIM
        ).to(self.device)
        self.target_critic = EnhancedValueNetwork(
            self.config.STATE_DIM, self.config.CRITIC_HIDDEN_DIM
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        if self.config.USE_ATTENTION_METRIC:
            self.metric_net = AttentionBasedMetric(
                self.config.STATE_DIM, self.config.METRIC_HIDDEN_DIM,
                num_heads=self.config.ATTENTION_HEADS, epsilon=self.config.EPSILON_METRIC
            ).to(self.device)
        else:
            self.metric_net = RobustContractionMetric(
                self.config.STATE_DIM, self.config.METRIC_HIDDEN_DIM,
                epsilon=self.config.EPSILON_METRIC
            ).to(self.device)
        
        self.target_policy = EnhancedPolicyNetwork(
            self.config.STATE_DIM, self.config.ACTION_DIM, self.config.POLICY_HIDDEN_DIM
        ).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
    
    def _initialize_optimizers(self):
        """Initialize optimizers with learning rate schedulers."""
        self.dynamics_optimizer = optim.AdamW(
            self.dynamics.parameters(), lr=self.config.DYNAMICS_LR, weight_decay=1e-4
        )
        self.policy_optimizer = optim.AdamW(
            self.policy.parameters(), lr=self.config.ACTOR_LR, weight_decay=1e-4
        )
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(), lr=self.config.CRITIC_LR, weight_decay=1e-4
        )
        self.metric_optimizer = optim.AdamW(
            self.metric_net.parameters(), lr=self.config.METRIC_LR, weight_decay=1e-4
        )
        
        self.dynamics_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.dynamics_optimizer, T_max=self.config.TOTAL_EPISODES
        )
        self.policy_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer, T_max=self.config.TOTAL_EPISODES
        )
    
    def soft_update(self, target: nn.Module, source: nn.Module):
        """Polyak averaging for target networks."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.TAU) + source_param.data * self.config.TAU
            )
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using running statistics."""
        if self.config.STATE_NORMALIZATION:
            return (state - self.state_mean) / (self.state_std + 1e-8)
        return state
    
    def update_normalization(self, states: np.ndarray):
        """Update running statistics for state normalization."""
        if self.config.STATE_NORMALIZATION:
            self.state_mean = 0.9 * self.state_mean + 0.1 * states.mean(axis=0)
            self.state_std = 0.9 * self.state_std + 0.1 * states.std(axis=0)
            self.state_std = np.maximum(self.state_std, 1e-8)
    
    def adapt_beta(self, reward_improvement: bool):
        """Adapt contraction regularization strength based on progress."""
        if reward_improvement:
            self.beta *= self.config.BETA_DECAY
        else:
            self.beta *= self.config.BETA_INCREASE
        self.beta = np.clip(self.beta, self.config.BETA_MIN, self.config.BETA_MAX)
    
    def update_identity_weight(self, condition_number: float):
        """Adapt identity regularization based on metric condition number."""
        if condition_number > self.config.CONDITION_NUMBER_THRESHOLD:
            self.current_identity_weight = min(
                self.current_identity_weight * 1.5,
                self.config.IDENTITY_REG_MAX
            )
        else:
            self.current_identity_weight = max(
                self.current_identity_weight * 0.95,
                self.config.IDENTITY_REG_BASE
            )
    
    def select_action(self, state: np.ndarray, deterministic: bool = False,
                     use_exploration: bool = True, curriculum_scale: float = 1.0) -> np.ndarray:
        """
        Select action given current state.
        
        Args:
            state: Current state observation
            deterministic: If True, use mean action without noise
            use_exploration: If True, add exploration noise
            curriculum_scale: Additional noise scaling from curriculum
            
        Returns:
            Action array
        """
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
            
            if use_exploration:
                noise = self.exploration.sample(scale=curriculum_scale)
                action_np += noise
            
            action_np = np.clip(action_np, -self.policy.action_scale, self.policy.action_scale)
            return action_np
    
    def update_dynamics(self, batch: Tuple, step: int) -> Tuple[float, Dict]:
        """Update dynamics model ensemble."""
        states, actions, _, next_states, _, indices, weights = batch
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)
        
        self.dynamics_optimizer.zero_grad()
        
        next_states_pred, uncertainty = self.dynamics(states_t, actions_t)
        
        mse_loss = F.mse_loss(next_states_pred, next_states_t, reduction='none')
        weighted_loss = (mse_loss * weights_t.unsqueeze(1)).mean()
        uncertainty_loss = uncertainty.mean() * 0.01
        
        dynamics_loss = weighted_loss + uncertainty_loss
        
        dynamics_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
        self.dynamics_optimizer.step()
        
        with torch.no_grad():
            errors = mse_loss.mean(dim=1).cpu().numpy()
            self.replay_buffer.update_priorities(indices, errors)
        
        return dynamics_loss.item(), {
            'dynamics_loss': dynamics_loss.item(),
            'uncertainty': uncertainty.mean().item(),
            'grad_norm': grad_norm.item()
        }
    
    def update_metric(self, batch: Tuple, step: int) -> Tuple[float, Dict]:
        """
        Update contraction metric with real-transition anchoring.
        
        Uses 80% real transitions and 20% model-imagined transitions
        for robust metric learning that doesn't exploit model errors.
        """
        states, actions, _, next_states_real, _, indices, weights = batch
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        next_states_real_t = torch.FloatTensor(next_states_real).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)
        
        self.metric_optimizer.zero_grad()
        
        # Primary loss on real transitions (80% weight)
        metric_loss_real, metrics_real = EnhancedRiemannianOperations.compute_contraction_loss(
            states_t, next_states_real_t, self.metric_net,
            alpha=self.contraction_alpha,
            beta=max(0.1, 1.0 - step / 10000),
            use_geodesic_reg=self.config.USE_GEODESIC_REGULARIZATION,
            geodesic_weight=self.config.GEODESIC_WEIGHT,
            energy_type=self.config.ENERGY_TYPE,
            target_state=self.target_state,
            mask=self.task_mask,
            adaptive_identity_weight=self.current_identity_weight
        )
        
        # Secondary loss on imagined transitions (20% weight)
        with torch.no_grad():
            action_dist = self.policy(states_t)
            actions_sampled = action_dist.rsample()
            next_states_imagined, _ = self.dynamics(states_t, actions_sampled)
        
        metric_loss_imagined, metrics_imagined = EnhancedRiemannianOperations.compute_contraction_loss(
            states_t, next_states_imagined, self.metric_net,
            alpha=self.contraction_alpha,
            beta=max(0.1, 1.0 - step / 10000),
            energy_type=self.config.ENERGY_TYPE,
            target_state=self.target_state,
            mask=self.task_mask,
            adaptive_identity_weight=self.current_identity_weight
        )
        
        # Weighted combination
        total_loss = (self.config.REAL_TRANSITION_WEIGHT * metric_loss_real +
                     self.config.IMAGINED_TRANSITION_WEIGHT * metric_loss_imagined)
        
        # Update contraction priorities for PER
        with torch.no_grad():
            contraction_errors = metrics_real['energy_diff'].abs().cpu().numpy()
            if len(contraction_errors.shape) == 0:
                contraction_errors = np.array([contraction_errors.item()] * len(indices))
            self.replay_buffer.update_contraction_priorities(indices, contraction_errors)
        
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.metric_net.parameters(), 0.5)
        self.metric_optimizer.step()
        
        # Update adaptive identity weight
        condition_number = metrics_real['condition_number'].item()
        self.update_identity_weight(condition_number)
        
        # Compute additional metric diagnostics
        with torch.no_grad():
            M, _ = self.metric_net(states_t)
            metric_net_metrics = self.metric_net.compute_metrics(M)
        
        metrics = {
            'metric_loss_real': metric_loss_real.item(),
            'metric_loss_imagined': metric_loss_imagined.item(),
            'metric_loss_total': total_loss.item(),
            'energy_curr': metrics_real['energy_curr'].item(),
            'energy_next': metrics_real['energy_next'].item(),
            'energy_diff': metrics_real['energy_diff'].item(),
            'condition_number': condition_number,
            'identity_weight': self.current_identity_weight,
            'grad_norm': grad_norm.item(),
        }
        
        for k, v in metric_net_metrics.items():
            if torch.is_tensor(v):
                metrics[k] = v.item() if v.numel() == 1 else v.mean().item()
        
        return total_loss.item(), metrics
    
    def update_critic(self, batch: Tuple, step: int) -> Tuple[float, Dict]:
        """Update twin Q-networks with double Q-learning."""
        states, actions, rewards, next_states, dones, _, weights = batch
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) * self.config.REWARD_SCALE
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        self.critic_optimizer.zero_grad()
        
        with torch.no_grad():
            next_action_dist = self.target_policy(next_states_t)
            next_actions = next_action_dist.rsample()
            target_q1, target_q2 = self.target_critic(next_states_t)
            target_q = torch.min(target_q1, target_q2)
            conservative_penalty = target_q.mean() * 0.1
            target_values = rewards_t + self.config.GAMMA * (1 - dones_t) * target_q - conservative_penalty
        
        current_q1, current_q2 = self.critic(states_t)
        
        td_error1 = F.mse_loss(current_q1, target_values, reduction='none')
        td_error2 = F.mse_loss(current_q2, target_values, reduction='none')
        critic_loss = (td_error1 * weights_t).mean() + (td_error2 * weights_t).mean()
        
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        with torch.no_grad():
            td_errors = (td_error1 + td_error2).squeeze().cpu().numpy() / 2
            self.replay_buffer.update_priorities(batch[5], td_errors)
        
        return critic_loss.item(), {
            'critic_loss': critic_loss.item(),
            'q_values': current_q1.mean().item(),
            'td_error': td_errors.mean(),
            'grad_norm': grad_norm.item()
        }
    
    def update_policy(self, batch: Tuple, step: int) -> Tuple[float, Dict]:
        """Update policy with contraction bonus and safety constraints."""
        states, _, _, _, _, _, weights = batch
        
        states_t = torch.FloatTensor(states).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)
        
        self.policy_optimizer.zero_grad()
        
        actions, log_probs = self.policy.sample_with_entropy(states_t)
        q_values = self.critic.q_min(states_t)
        
        with torch.no_grad():
            next_states_pred, _ = self.dynamics(states_t, actions)
            energy_curr, _ = EnhancedRiemannianOperations.compute_energy(
                states_t, self.metric_net, self.config.ENERGY_TYPE,
                self.target_state, self.task_mask
            )
            energy_next, _ = EnhancedRiemannianOperations.compute_energy(
                next_states_pred, self.metric_net, self.config.ENERGY_TYPE,
                self.target_state, self.task_mask
            )
            delta_energy = energy_curr - energy_next
            contraction_bonus = torch.tanh(delta_energy / 5.0).mean()
        
        safety_penalty = torch.tensor(0.0, device=self.device)
        if self.config.USE_SAFETY_CONSTRAINTS:
            safety_margin = EnhancedRiemannianOperations.compute_safety_margin(
                states_t, self.metric_net
            )
            safety_penalty = F.relu(self.config.SAFETY_MARGIN_THRESHOLD - safety_margin).mean() * 0.1
        
        entropy_bonus = -0.2 * log_probs.mean()
        value_loss = -q_values.mean()
        
        # Additional penalties for smooth control
        velocity = states_t[:, 2] * 8.0
        velocity_penalty = F.relu(torch.abs(velocity) - 3.0).mean() * 0.01
        action_penalty = torch.abs(actions).mean() * 0.001
        
        policy_loss = (0.5 * value_loss - self.beta * contraction_bonus + 
                      entropy_bonus + velocity_penalty + action_penalty + safety_penalty)
        
        weighted_loss = policy_loss * weights_t.mean()
        
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()
        
        if step % self.config.TARGET_UPDATE_INTERVAL == 0:
            self.soft_update(self.target_policy, self.policy)
            self.soft_update(self.target_critic, self.critic)
        
        return weighted_loss.item(), {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'contraction_bonus': contraction_bonus.item(),
            'entropy_bonus': entropy_bonus.item(),
            'safety_penalty': safety_penalty.item(),
            'beta': self.beta,
            'grad_norm': grad_norm.item()
        }
    
    def train_step(self, step: int):
        """Perform one training step updating all components."""
        if len(self.replay_buffer) < self.config.BATCH_SIZE:
            return None
        
        batch = self.replay_buffer.sample(self.config.BATCH_SIZE)
        if batch is None:
            return None
        
        dynamics_loss, dynamics_metrics = self.update_dynamics(batch, step)
        self.metrics['dynamics_losses'].append(dynamics_loss)
        self.metrics['grad_norms']['dynamics'].append(dynamics_metrics['grad_norm'])
        
        metric_loss, metric_metrics = self.update_metric(batch, step)
        self.metrics['metric_losses'].append(metric_loss)
        self.metrics['energies'].append(metric_metrics['energy_curr'])
        self.metrics['condition_numbers'].append(metric_metrics['condition_number'])
        self.metrics['identity_weights'].append(metric_metrics['identity_weight'])
        self.metrics['grad_norms']['metric'].append(metric_metrics['grad_norm'])
        
        critic_loss, critic_metrics = self.update_critic(batch, step)
        self.metrics['critic_losses'].append(critic_loss)
        self.metrics['grad_norms']['critic'].append(critic_metrics['grad_norm'])
        
        policy_loss, policy_metrics = self.update_policy(batch, step)
        self.metrics['actor_losses'].append(policy_loss)
        self.metrics['grad_norms']['policy'].append(policy_metrics['grad_norm'])
        
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
        """Execute one training episode."""
        state, _ = env.reset()
        self.exploration.reset()
        
        curriculum_params = self.curriculum.get_parameters(episode_num)
        
        episode_reward = 0
        episode_steps = 0
        episode_transitions = []
        
        for step in range(self.config.MAX_EPISODE_LENGTH):
            use_exploration = (episode_num < self.config.TOTAL_EPISODES * 0.8)
            action = self.select_action(
                state, deterministic=False, use_exploration=use_exploration,
                curriculum_scale=curriculum_params['noise_scale']
            )
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_transitions.append((state.copy(), action, reward, next_state.copy(), done))
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done:
                break
        
        # Store transitions in replay buffer
        for transition in episode_transitions:
            self.replay_buffer.push(*transition)
        
        # Update normalization
        states = np.array([t[0] for t in episode_transitions])
        self.update_normalization(states)
        self.exploration.update(episode_reward, episode_num)
        
        # Meta-learning updates
        if self.metrics['energies']:
            avg_energy = np.mean(self.metrics['energies'][-100:]) if len(self.metrics['energies']) >= 100 else np.mean(self.metrics['energies'])
            self.meta_controller.update(episode_reward, avg_energy)
            self.contraction_alpha = self.meta_controller.suggest_contraction_rate()
        
        # Perform training steps
        training_metrics = []
        if len(self.replay_buffer) > self.config.LEARNING_START:
            num_steps = min(self.config.GRADIENT_STEPS, len(self.replay_buffer) // self.config.BATCH_SIZE)
            for i in range(num_steps):
                step_num = episode_num * self.config.GRADIENT_STEPS + i
                metrics = self.train_step(step_num)
                if metrics:
                    training_metrics.append(metrics)
        
        # Adaptive beta update
        if episode_num > 0 and self.metrics['episode_rewards']:
            last_reward = self.metrics['episode_rewards'][-1]
            reward_improved = episode_reward > last_reward
            self.adapt_beta(reward_improved)
        
        self.beta = curriculum_params['beta']
        
        # Log metrics
        self.metrics['episode_rewards'].append(episode_reward)
        self.metrics['betas'].append(self.beta)
        self.metrics['exploration_rates'].append(self.exploration.exploration_rate)
        self.metrics['curriculum_stages'].append(self.curriculum.current_stage)
        
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
        """Evaluate agent performance without exploration."""
        total_reward = 0
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            for _ in range(self.config.MAX_EPISODE_LENGTH):
                action = self.select_action(state, deterministic=True, use_exploration=False)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            total_reward += episode_reward
        avg_reward = total_reward / num_episodes
        self.metrics['eval_rewards'].append(avg_reward)
        return avg_reward
    
    def evaluate_robustness(self, env: gym.Env) -> Dict:
        """Run robustness evaluation with perturbations."""
        results = self.robustness_evaluator.evaluate(self, env, num_episodes=5)
        self.metrics['robustness_results'].append(results)
        return results
    
    def train(self, env: gym.Env, eval_env: gym.Env = None) -> Dict:
        """Main training loop."""
        if eval_env is None:
            eval_env = gym.make(self.config.ENV_NAME)
        
        print(f"\nStarting CR-MBRL training for {self.config.TOTAL_EPISODES} episodes...")
        print(f"Energy formulation: {self.config.ENERGY_TYPE}")
        print("=" * 100)
        
        best_eval_reward = -float('inf')
        patience_counter = 0
        
        for episode in range(self.config.TOTAL_EPISODES):
            episode_metrics = self.train_episode(env, episode)
            
            eval_reward = None
            if episode % self.config.EVAL_INTERVAL == 0:
                eval_reward = self.evaluate(eval_env, self.config.EVAL_EPISODES)
                
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    patience_counter = 0
                    self.save_models(prefix="best")
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.EARLY_STOP_PATIENCE and eval_reward > self.config.EARLY_STOP_REWARD:
                    print(f"\nEarly stopping at episode {episode}")
                    break
            
            # Periodic robustness evaluation
            if episode % self.config.ROBUSTNESS_EVAL_INTERVAL == 0:
                _ = self.evaluate_robustness(eval_env)
            
            if episode % self.config.LOG_INTERVAL == 0 or episode == self.config.TOTAL_EPISODES - 1:
                cond_num = self.metrics['condition_numbers'][-1] if self.metrics['condition_numbers'] else 0
                print(f"Episode {episode:4d} | Reward: {episode_metrics['reward']:8.1f} | "
                      f"Beta: {self.beta:.3f} | Alpha: {self.contraction_alpha:.3f} | "
                      f"Cond: {cond_num:.1f}")
        
        final_eval = self.evaluate(eval_env, 10)
        print(f"\nFinal evaluation reward: {final_eval:.1f}")
        
        self.save_models(prefix="final")
        self.save_metrics()
        self.plot_training_results()
        
        return {
            'best_eval_reward': best_eval_reward,
            'final_eval_reward': final_eval,
            'total_episodes': len(self.metrics['episode_rewards'])
        }
    
    def save_models(self, prefix: str = ""):
        """Save all model weights."""
        torch.save(self.policy.state_dict(), self.save_dir / f"{prefix}_policy.pth")
        torch.save(self.critic.state_dict(), self.save_dir / f"{prefix}_critic.pth")
        torch.save(self.dynamics.state_dict(), self.save_dir / f"{prefix}_dynamics.pth")
        torch.save(self.metric_net.state_dict(), self.save_dir / f"{prefix}_metric.pth")
        torch.save(self.target_policy.state_dict(), self.save_dir / f"{prefix}_target_policy.pth")
        torch.save(self.target_critic.state_dict(), self.save_dir / f"{prefix}_target_critic.pth")
    
    def load_models(self, prefix: str = ""):
        """Load saved model weights."""
        self.policy.load_state_dict(torch.load(self.save_dir / f"{prefix}_policy.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(self.save_dir / f"{prefix}_critic.pth", map_location=self.device))
        self.dynamics.load_state_dict(torch.load(self.save_dir / f"{prefix}_dynamics.pth", map_location=self.device))
        self.metric_net.load_state_dict(torch.load(self.save_dir / f"{prefix}_metric.pth", map_location=self.device))
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
    
    def save_metrics(self):
        """Save training metrics to disk."""
        serializable_metrics = {}
        for k, v in self.metrics.items():
            if isinstance(v, list):
                serializable_metrics[k] = [x.item() if torch.is_tensor(x) else x for x in v]
            elif isinstance(v, defaultdict):
                serializable_metrics[k] = dict(v)
            else:
                serializable_metrics[k] = v
        
        with open(self.save_dir / "training_metrics.pkl", 'wb') as f:
            pickle.dump(serializable_metrics, f)
        with open(self.save_dir / "training_metrics.json", 'w') as f:
            json.dump(serializable_metrics, f, indent=2, default=str)
    
    def plot_training_results(self):
        """Generate training visualization plots."""
        try:
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            episodes = range(len(self.metrics['episode_rewards']))
            
            # Training rewards
            ax = axes[0, 0]
            ax.plot(episodes, self.metrics['episode_rewards'], 'b-', alpha=0.7)
            if self.metrics.get('eval_rewards'):
                eval_x = np.arange(0, len(episodes), self.config.EVAL_INTERVAL)[:len(self.metrics['eval_rewards'])]
                ax.plot(eval_x, self.metrics['eval_rewards'], 'r-', linewidth=2)
            ax.set_title('Training Progress')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
            
            # Condition number monitoring
            ax = axes[0, 1]
            if self.metrics.get('condition_numbers'):
                ax.plot(range(len(self.metrics['condition_numbers'])), 
                       self.metrics['condition_numbers'], 'orange', alpha=0.7)
                ax.axhline(y=self.config.CONDITION_NUMBER_THRESHOLD, color='r', 
                          linestyle='--', label='Threshold')
                ax.set_title('Metric Condition Number')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Beta and alpha
            ax = axes[0, 2]
            if self.metrics.get('betas'):
                ax.plot(episodes, self.metrics['betas'], 'r-', alpha=0.7, label='Beta')
            ax2 = ax.twinx()
            if self.metrics.get('contraction_alphas'):
                ax2.plot(episodes[:len(self.metrics['contraction_alphas'])], 
                        self.metrics['contraction_alphas'], 'b-', alpha=0.7, label='Alpha')
            ax.set_title('Adaptive Parameters')
            ax.grid(True, alpha=0.3)
            
            # Losses
            ax = axes[1, 0]
            if self.metrics.get('dynamics_losses'):
                ax.plot(range(len(self.metrics['dynamics_losses'])), 
                       self.metrics['dynamics_losses'], 'b-', alpha=0.5, label='Dynamics')
            if self.metrics.get('metric_losses'):
                ax.plot(range(len(self.metrics['metric_losses'])), 
                       self.metrics['metric_losses'], 'g-', alpha=0.5, label='Metric')
            if self.metrics.get('critic_losses'):
                ax.plot(range(len(self.metrics['critic_losses'])), 
                       self.metrics['critic_losses'], 'r-', alpha=0.5, label='Critic')
            ax.set_title('Training Losses')
            ax.legend()
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Identity weight
            ax = axes[1, 1]
            if self.metrics.get('identity_weights'):
                ax.plot(range(len(self.metrics['identity_weights'])), 
                       self.metrics['identity_weights'], 'purple', alpha=0.7)
                ax.set_title('Adaptive Identity Regularization')
                ax.grid(True, alpha=0.3)
            
            # Energy
            ax = axes[1, 2]
            if self.metrics.get('energies'):
                ax.plot(range(len(self.metrics['energies'])), 
                       self.metrics['energies'], 'green', alpha=0.7)
                ax.set_title('Contraction Energy')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = self.save_dir / "training_results.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            if self.config.PLOT_RESULTS:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Plotting error: {e}")
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(len(self.metrics['episode_rewards'])), 
                       self.metrics['episode_rewards'])
                ax.set_xlabel('Episode')
                ax.set_ylabel('Reward')
                ax.grid(True)
                plt.savefig(self.save_dir / "training_results_fallback.png", dpi=150)
                if self.config.PLOT_RESULTS:
                    plt.show()
                else:
                    plt.close()
            except:
                pass


# ============================
# BENCHMARK RUNNER
# ============================

class BenchmarkRunner:
    """
    Run CR-MBRL across multiple environments and random seeds.
    
    Handles:
    - Sequential execution of environment/seed combinations
    - Proper error isolation (failure in one run doesn't affect others)
    - Automatic result aggregation and statistical reporting
    - Intermediate result saving for crash recovery
    """
    
    def __init__(self, env_names: List[str] = None, seeds: List[int] = None, 
                 use_gpu: bool = True, save_dir: str = "benchmark_results"):
        self.env_names = env_names or get_available_envs()
        self.seeds = seeds or [42, 123, 456, 789, 1024]
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.results = defaultdict(list)
        
        print(f"\n{'='*80}")
        print(f"CR-MBRL BENCHMARK RUNNER")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Environments: {len(self.env_names)}")
        print(f"Seeds per env: {len(self.seeds)}")
        print(f"Total runs: {len(self.env_names) * len(self.seeds)}")
        print(f"{'='*80}\n")
    
    def run_single(self, env_name: str, seed: int) -> Dict:
        """
        Run a single experiment.
        
        Args:
            env_name: Environment to run
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with experiment results
        """
        run_name = f"{env_name}_seed{seed}"
        print(f"\n{'='*60}")
        print(f"Running: {run_name}")
        print(f"{'='*60}")
        
        try:
            set_seed(seed)
            
            config = Config.from_env(env_name)
            config.SEED = seed
            config.DEVICE = self.device
            config.SAVE_DIR = str(self.save_dir / run_name)
            
            agent = EnhancedContractionDynamicsAgent(config)
            
            train_env = make_env(env_name)
            eval_env = make_env(env_name)
            
            try:
                start_time = time.time()
                results = agent.train(train_env, eval_env)
                training_time = time.time() - start_time
                
                robustness_results = agent.evaluate_robustness(eval_env)
                
                return {
                    'env_name': env_name,
                    'seed': seed,
                    'status': 'success',
                    'best_eval_reward': results['best_eval_reward'],
                    'final_eval_reward': results['final_eval_reward'],
                    'training_time': training_time,
                    'robustness_results': robustness_results,
                    'metrics': agent.metrics
                }
                
            finally:
                train_env.close()
                eval_env.close()
                
        except Exception as e:
            print(f"\nFailed on {run_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'env_name': env_name,
                'seed': seed,
                'status': 'failed',
                'error': str(e)
            }
    
    def run_benchmark(self, env_names: List[str] = None):
        """Run full benchmark across environments and seeds."""
        envs_to_run = env_names or self.env_names
        
        for env_name in envs_to_run:
            env_results = []
            
            for seed in self.seeds:
                result = self.run_single(env_name, seed)
                env_results.append(result)
                
                if result['status'] == 'success':
                    print(f"  Completed {env_name} seed={seed}: "
                          f"Best={result['best_eval_reward']:.1f}, "
                          f"Final={result['final_eval_reward']:.1f}")
            
            self.results[env_name] = env_results
            self._save_results(env_name)
        
        self._generate_report()
    
    def _save_results(self, env_name: str):
        """Save intermediate results for crash recovery."""
        save_path = self.save_dir / f"results_{env_name}.json"
        
        serializable = []
        for r in self.results[env_name]:
            if r['status'] == 'success':
                serializable.append({
                    'env_name': r['env_name'],
                    'seed': r['seed'],
                    'status': r['status'],
                    'best_eval_reward': float(r['best_eval_reward']),
                    'final_eval_reward': float(r['final_eval_reward']),
                    'training_time': float(r['training_time']),
                    'episode_rewards': [float(x) if not isinstance(x, (int, float)) else x 
                                       for x in r['metrics']['episode_rewards'][-100:]],
                })
            else:
                serializable.append({
                    'env_name': r['env_name'],
                    'seed': r['seed'],
                    'status': r['status'],
                    'error': r['error']
                })
        
        with open(save_path, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    def _generate_report(self):
        """Generate comprehensive benchmark report with statistics."""
        print(f"\n{'='*80}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'='*80}")
        
        report = {}
        
        for env_name, env_results in self.results.items():
            successful = [r for r in env_results if r['status'] == 'success']
            failed = [r for r in env_results if r['status'] != 'success']
            
            if successful:
                best_rewards = [r['best_eval_reward'] for r in successful]
                final_rewards = [r['final_eval_reward'] for r in successful]
                times = [r['training_time'] for r in successful]
                
                report[env_name] = {
                    'num_seeds': len(successful),
                    'best_reward_mean': np.mean(best_rewards),
                    'best_reward_std': np.std(best_rewards),
                    'best_reward_max': np.max(best_rewards),
                    'final_reward_mean': np.mean(final_rewards),
                    'final_reward_std': np.std(final_rewards),
                    'avg_training_time': np.mean(times),
                    'failed_runs': len(failed)
                }
                
                print(f"\n{env_name}:")
                print(f"  Successful runs: {len(successful)}/{len(env_results)}")
                print(f"  Best reward: {np.mean(best_rewards):.1f} +/- {np.std(best_rewards):.1f}")
                print(f"  Final reward: {np.mean(final_rewards):.1f} +/- {np.std(final_rewards):.1f}")
                print(f"  Avg training time: {np.mean(times):.1f}s")
            else:
                print(f"\n{env_name}: All runs failed")
        
        with open(self.save_dir / "benchmark_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self._plot_benchmark_results(report)
    
    def _plot_benchmark_results(self, report: Dict):
        """Create visual comparison of benchmark results."""
        try:
            envs = list(report.keys())
            best_means = [report[e]['best_reward_mean'] for e in envs]
            best_stds = [report[e]['best_reward_std'] for e in envs]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(envs))
            
            bars = ax.bar(x, best_means, yerr=best_stds, capsize=5, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(envs, rotation=45, ha='right')
            ax.set_ylabel('Best Evaluation Reward')
            ax.set_title('CR-MBRL Benchmark Results')
            ax.grid(True, alpha=0.3)
            
            for bar, mean in zip(bars, best_means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{mean:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.save_dir / "benchmark_results.png", dpi=150, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not create benchmark plot: {e}")


# ============================
# TESTING
# ============================

def run_tests():
    """Run comprehensive unit tests for core components."""
    print("Running CR-MBRL tests...")
    
    config = Config()
    device = torch.device("cpu")
    
    # Test energy formulations
    metric = RobustContractionMetric(3, 64).to(device)
    test_state = torch.randn(10, 3)
    
    energy_abs, _ = EnhancedRiemannianOperations.compute_energy_absolute(test_state, metric)
    assert energy_abs.shape == (10,), f"Absolute energy shape: {energy_abs.shape}"
    print("  Absolute energy: OK")
    
    target = torch.zeros(3)
    energy_rel, _ = EnhancedRiemannianOperations.compute_energy_relative(test_state, target, metric)
    assert energy_rel.shape == (10,), f"Relative energy shape: {energy_rel.shape}"
    print("  Relative energy: OK")
    
    mask = torch.tensor([1.0, 1.0, 0.0])
    energy_mask, _ = EnhancedRiemannianOperations.compute_energy_masked(test_state, mask, metric)
    assert energy_mask.shape == (10,), f"Masked energy shape: {energy_mask.shape}"
    print("  Masked energy: OK")
    
    # Test metric properties
    M, _ = metric(test_state)
    condition_numbers = torch.linalg.eigvalsh(M).max(dim=1).values / torch.clamp(
        torch.linalg.eigvalsh(M).min(dim=1).values, min=1e-6
    )
    print(f"  Condition numbers: {condition_numbers.mean():.1f}")
    
    # Test contraction loss
    next_state = test_state + torch.randn_like(test_state) * 0.1
    loss, loss_metrics = EnhancedRiemannianOperations.compute_contraction_loss(
        test_state, next_state, metric,
        adaptive_identity_weight=0.001
    )
    print(f"  Contraction loss: {loss.item():.4f}")
    print(f"  Scaled identity weight: {loss_metrics['scaled_identity_weight'].item():.4f}")
    
    # Test robustness evaluator
    config_agent = Config()
    agent = EnhancedContractionDynamicsAgent(config_agent)
    evaluator = RobustnessEvaluator(use_adversarial=True)
    
    state_np = np.random.randn(3)
    adv_pert = evaluator.compute_adversarial_perturbation(state_np, agent.metric_net, 0.1)
    assert adv_pert.shape == (3,), f"Adversarial perturbation shape: {adv_pert.shape}"
    print(f"  Adversarial perturbation: OK (norm={np.linalg.norm(adv_pert):.3f})")
    
    # Test statistical testing
    baseline = np.random.randn(20) * 10 + 100
    treatment = np.random.randn(20) * 8 + 115
    sig_results = compute_significance(baseline, treatment)
    print(f"  Significance test: p={sig_results['p_value']:.3f}, d={sig_results['cohens_d']:.2f}")
    
    print("\nAll tests passed successfully.\n")


# ============================
# MAIN ENTRY POINT
# ============================

def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="CR-MBRL: Contraction-Regularized Model-Based Reinforcement Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on Pendulum (default)
  python crmbrl.py
  
  # Train on a specific MuJoCo environment
  python crmbrl.py --env HalfCheetah-v4
  
  # Quick test with fewer episodes
  python crmbrl.py --env Hopper-v4 --quick
  
  # Run full benchmark on all available environments
  python crmbrl.py --benchmark
  
  # Run benchmark on specific environments with custom seeds
  python crmbrl.py --benchmark --envs Pendulum-v1 Hopper-v4 Walker2d-v4 --seeds 1 2 3
  
  # List all available environments
  python crmbrl.py --list-envs
  
  # Force CPU usage
  python crmbrl.py --env Ant-v4 --cpu
        """
    )
    
    parser.add_argument('--env', type=str, default=None,
                       help='Single environment to run (default: Pendulum-v1)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run full benchmark across multiple environments')
    parser.add_argument('--list-envs', action='store_true',
                       help='List available environments and exit')
    parser.add_argument('--envs', type=str, nargs='+', default=None,
                       help='Specific environments for benchmark mode')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='Random seeds to use (default: 5 seeds)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage even if GPU is available')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with reduced number of episodes')
    
    args = parser.parse_args()
    
    # List environments and exit
    if args.list_envs:
        list_available_envs()
        return
    
    # Check MuJoCo availability
    has_mujoco = check_mujoco_available()
    if not has_mujoco:
        print("\nNote: MuJoCo environments are not available.")
        print("To install: pip install gymnasium[mujoco]")
        print("Classic control environments will still work.\n")
    
    # Print header
    print("=" * 80)
    print("CR-MBRL: Contraction-Regularized Model-Based Reinforcement Learning")
    print("=" * 80)
    
    # Run tests first
    try:
        run_tests()
    except Exception as e:
        print(f"Warning: Tests failed: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with training anyway...\n")
    
    # Benchmark mode
    if args.benchmark:
        envs_to_run = args.envs if args.envs else get_available_envs()
        seeds = args.seeds if args.seeds else [42, 123, 456, 789, 1024]
        
        if not envs_to_run:
            print("No environments available. Exiting.")
            return
        
        print(f"\nRunning benchmark on {len(envs_to_run)} environment(s) with {len(seeds)} seed(s) each")
        print(f"Environments: {envs_to_run}")
        print(f"Seeds: {seeds}")
        
        runner = BenchmarkRunner(
            env_names=envs_to_run,
            seeds=seeds,
            use_gpu=not args.cpu,
            save_dir="benchmark_results"
        )
        
        runner.run_benchmark()
        return
    
    # Single environment mode
    env_name = args.env if args.env else "Pendulum-v1"
    
    if env_name not in ENV_CONFIGS:
        print(f"Error: Unknown environment '{env_name}'")
        print(f"Available environments: {list(ENV_CONFIGS.keys())}")
        return
    
    # Create configuration
    config = Config.from_env(env_name)
    config.SAVE_DIR = f"results_{env_name.replace('-', '_')}"
    
    if args.quick:
        print("\nQuick mode: Reduced number of episodes")
        config.TOTAL_EPISODES = min(50, config.TOTAL_EPISODES)
        config.EVAL_INTERVAL = 10
    
    # Feature flags
    config.USE_CURRICULUM = True
    config.USE_META_LEARNING = True
    config.USE_ATTENTION_METRIC = True
    config.USE_GEODESIC_REGULARIZATION = True
    
    if args.cpu:
        config.DEVICE = "cpu"
    
    print(f"\nConfiguration:")
    print(f"  Environment: {config.ENV_NAME}")
    print(f"  State dim: {config.STATE_DIM}, Action dim: {config.ACTION_DIM}")
    print(f"  Max episode length: {config.MAX_EPISODE_LENGTH}")
    print(f"  Energy type: {config.ENERGY_TYPE}")
    print(f"  Total episodes: {config.TOTAL_EPISODES}")
    print(f"  Device: {config.DEVICE}")
    
    # Create agent
    agent = EnhancedContractionDynamicsAgent(config)
    
    # Create environments
    train_env = make_env(config.ENV_NAME)
    eval_env = make_env(config.ENV_NAME)
    
    try:
        start_time = time.time()
        results = agent.train(train_env, eval_env)
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.1f}s")
        print(f"Best evaluation reward: {results['best_eval_reward']:.1f}")
        print(f"Final evaluation reward: {results['final_eval_reward']:.1f}")
        
        # Run robustness evaluation
        print("\nRunning robustness evaluation...")
        robustness_results = agent.evaluate_robustness(eval_env)
        for key, val in robustness_results.items():
            print(f"  {key}: mean_reward={val['mean_reward']:.1f} +/- {val['std_reward']:.1f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        train_env.close()
        eval_env.close()
    
    print(f"\nResults saved to: {config.SAVE_DIR}")


if __name__ == "__main__":
    main()