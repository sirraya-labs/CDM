"""
AUTOMATED ABLATION STUDY FRAMEWORK
Author: Amir Hameed, Sirraya Labs
Paper: "Learning Contraction Metrics for Provably Stable Model-Based RL"

This script provides:
1. Comprehensive ablation studies with statistical significance
2. Automatic figure generation for publication
3. Detailed metric tracking (condition numbers, eigenvalues, etc.)
4. Dashboard integration for real-time monitoring
5. Reproducible results with exact seeds
"""

import numpy as np
import torch
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
import warnings
warnings.filterwarnings('ignore')

# Import your existing code (adjust based on your actual module structure)
from main import Config, RobustContractionDynamicsAgent

# ============================
# ABLATION STUDY CONFIGURATION
# ============================

@dataclass
class AblationConfig:
    """Configuration for automated ablation studies"""
    
    # Base configuration (your default settings)
    base_config: Config = field(default_factory=Config)
    
    # Ablation variants to test
    ablation_variants: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        # Full CDM (baseline for comparison)
        'full_cdm': {
            'name': 'Full CDM',
            'description': 'Complete implementation with all features',
            'config_modifications': {}
        },
        
        # Core component ablations
        'no_contraction': {
            'name': 'No Contraction',
            'description': 'β=0, no stability regularization',
            'config_modifications': {
                'INITIAL_BETA': 0.0,
                'BETA_MIN': 0.0,
                'BETA_MAX': 0.0
            }
        },
        
        'fixed_metric': {
            'name': 'Fixed Metric',
            'description': 'M(x) = I (identity metric)',
            'config_modifications': {},
            'special_case': 'fixed_metric'  # Special handling needed
        },
        
        'no_cholesky': {
            'name': 'No Cholesky',
            'description': 'Direct metric prediction without Cholesky',
            'config_modifications': {},
            'special_case': 'no_cholesky'
        },
        
        # Architectural ablations
        'single_dynamics': {
            'name': 'Single Dynamics',
            'description': 'No ensemble (K=1)',
            'config_modifications': {
                'ENSEMBLE_SIZE': 1
            }
        },
        
        'no_ensemble': {
            'name': 'No Ensemble Weighting',
            'description': 'Equal ensemble weights',
            'config_modifications': {},
            'special_case': 'no_ensemble_weights'
        },
        
        # Learning parameter ablations
        'no_adaptive_beta': {
            'name': 'Fixed β',
            'description': 'No adaptive β adjustment',
            'config_modifications': {
                'BETA_DECAY': 1.0,
                'BETA_INCREASE': 1.0
            }
        },
        
        'no_metric_reg': {
            'name': 'No Metric Reg',
            'description': 'No metric regularization',
            'config_modifications': {
                'METRIC_REGULARIZATION': 0.0
            }
        },
        
        # Contraction rate variations
        'alpha_0_9': {
            'name': 'α=0.9',
            'description': 'Faster contraction',
            'config_modifications': {
                'CONTRACTION_RATE_ALPHA': 0.9
            }
        },
        
        'alpha_0_97': {
            'name': 'α=0.97',
            'description': 'Slower contraction',
            'config_modifications': {
                'CONTRACTION_RATE_ALPHA': 0.97
            }
        },
        
        # Exploration variations
        'no_exploration': {
            'name': 'No Exploration',
            'description': 'Deterministic policy only',
            'config_modifications': {
                'MIN_NOISE': 0.0,
                'NOISE_DECAY': 1.0
            }
        },
        
        # Activation function variations
        'relu_activation': {
            'name': 'ReLU Activation',
            'description': 'ReLU instead of softplus for Cholesky',
            'config_modifications': {},
            'special_case': 'relu_activation'
        },
        
        'exp_activation': {
            'name': 'Exp Activation',
            'description': 'Exp instead of softplus for Cholesky',
            'config_modifications': {},
            'special_case': 'exp_activation'
        }
    })
    
    # Experimental settings
    num_seeds: int = 5
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1337])
    num_episodes: int = 300  # Shorter for ablation studies
    eval_interval: int = 20
    num_eval_episodes: int = 3
    
    # Output settings
    save_dir: str = "ablation_studies"
    generate_plots: bool = True
    save_metrics: bool = True
    wandb_enabled: bool = False  # Set to True if using Weights & Biases
    
    # Statistical analysis
    confidence_level: float = 0.95
    pairwise_comparisons: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.num_seeds <= len(self.seeds), "Not enough seeds provided"
        Path(self.save_dir).mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        Path(f"{self.save_dir}/logs").mkdir(exist_ok=True)
        Path(f"{self.save_dir}/figures").mkdir(exist_ok=True)
        Path(f"{self.save_dir}/models").mkdir(exist_ok=True)
        Path(f"{self.save_dir}/tables").mkdir(exist_ok=True)
        
        # Save configuration
        self._save_config()
    
    def _save_config(self):
        """Save ablation configuration to file"""
        config_dict = asdict(self)
        # Remove base_config (too large)
        config_dict.pop('base_config')
        
        with open(f"{self.save_dir}/ablation_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

# ============================
# METRIC TRACKER
# ============================

class ComprehensiveMetricTracker:
    """Tracks comprehensive metrics during ablation studies"""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.metrics = {}
        
    def initialize_trial(self, variant_name: str, seed: int):
        """Initialize tracking for a specific trial"""
        trial_id = f"{variant_name}_seed{seed}"
        self.metrics[trial_id] = {
            'episode_rewards': [],
            'eval_rewards': [],
            'dynamics_losses': [],
            'metric_losses': [],
            'critic_losses': [],
            'actor_losses': [],
            'energies': [],
            'betas': [],
            'exploration_rates': [],
            
            # Metric network specific metrics
            'condition_numbers': [],
            'min_eigenvalues': [],
            'max_eigenvalues': [],
            'metric_determinants': [],
            
            # Stability metrics
            'contraction_violations': [],  # Percentage of states violating contraction
            'energy_decay_rates': [],  # Actual vs theoretical decay
            
            # Computational metrics
            'training_times': [],
            'memory_usage': [],
            'gradient_norms': {
                'dynamics': [],
                'metric': [],
                'policy': [],
                'critic': []
            }
        }
        
        return trial_id
    
    def update(self, trial_id: str, step_metrics: Dict[str, Any], 
               episode_metrics: Dict[str, Any] = None):
        """Update metrics for a trial"""
        # Step metrics (from training loop)
        if step_metrics:
            for key in ['dynamics_losses', 'metric_losses', 'critic_losses', 
                       'actor_losses', 'energies']:
                if key in step_metrics and step_metrics[key]:
                    self.metrics[trial_id][key].append(np.mean(step_metrics[key]))
            
            # Gradient norms
            if 'grad_norms' in step_metrics:
                for network, norms in step_metrics['grad_norms'].items():
                    self.metrics[trial_id]['gradient_norms'][network].extend(norms)
        
        # Episode metrics (from agent.metrics)
        if episode_metrics:
            for key in ['episode_rewards', 'eval_rewards', 'betas', 
                       'exploration_rates']:
                if key in episode_metrics:
                    self.metrics[trial_id][key].extend(episode_metrics[key])
    
    def update_metric_network_stats(self, trial_id: str, agent):
        """Update metric network specific statistics"""
        if hasattr(agent, 'metric_net') and hasattr(agent.metric_net, 'compute_metrics'):
            # Sample states from replay buffer
            if len(agent.replay_buffer) > 0:
                sample_size = min(100, len(agent.replay_buffer))
                indices = np.random.choice(len(agent.replay_buffer), sample_size, replace=False)
                
                condition_numbers = []
                min_eigs = []
                max_eigs = []
                determinants = []
                
                for idx in indices:
                    state = agent.replay_buffer.buffer[idx][0]
                    state_t = torch.FloatTensor(agent.normalize_state(state)).unsqueeze(0).to(agent.device)
                    
                    with torch.no_grad():
                        M, _ = agent.metric_net(state_t)
                        eigvals = torch.linalg.eigvalsh(M[0])
                        min_eig = eigvals.min().item()
                        max_eig = eigvals.max().item()
                        
                        condition_numbers.append(max_eig / min_eig if min_eig > 1e-6 else 1e6)
                        min_eigs.append(min_eig)
                        max_eigs.append(max_eig)
                        determinants.append(torch.det(M[0]).item())
                
                self.metrics[trial_id]['condition_numbers'].append(np.mean(condition_numbers))
                self.metrics[trial_id]['min_eigenvalues'].append(np.mean(min_eigs))
                self.metrics[trial_id]['max_eigenvalues'].append(np.mean(max_eigs))
                self.metrics[trial_id]['metric_determinants'].append(np.mean(determinants))
    
    def calculate_contraction_violation(self, trial_id: str, agent, num_samples: int = 100):
        """Calculate percentage of states violating contraction condition"""
        if len(agent.replay_buffer) < num_samples:
            return 0.0
        
        violations = 0
        indices = np.random.choice(len(agent.replay_buffer), num_samples, replace=False)
        
        for idx in indices:
            state = agent.replay_buffer.buffer[idx][0]
            state_t = torch.FloatTensor(agent.normalize_state(state)).unsqueeze(0).to(agent.device)
            
            with torch.no_grad():
                # Get energy at current state
                energy_curr, _ = agent.metric_net.compute_energy(state_t)
                
                # Predict next state
                action_dist = agent.policy(state_t)
                action = action_dist.rsample()
                next_state_pred, _ = agent.dynamics(state_t, action)
                
                # Get energy at next state
                energy_next, _ = agent.metric_net.compute_energy(next_state_pred)
                
                # Check contraction violation
                alpha = agent.config.CONTRACTION_RATE_ALPHA
                if energy_next > (alpha ** 2) * energy_curr + 1e-6:
                    violations += 1
        
        violation_rate = violations / num_samples
        self.metrics[trial_id]['contraction_violations'].append(violation_rate)
        return violation_rate
    
    def get_summary_stats(self, trial_id: str) -> Dict[str, Any]:
        """Get summary statistics for a trial"""
        if trial_id not in self.metrics:
            return {}
        
        trial_data = self.metrics[trial_id]
        summary = {}
        
        # Episode rewards statistics
        if trial_data['episode_rewards']:
            rewards = np.array(trial_data['episode_rewards'])
            summary['final_reward'] = float(rewards[-1])
            summary['best_reward'] = float(rewards.max())
            summary['avg_reward'] = float(rewards.mean())
            summary['std_reward'] = float(rewards.std())
            summary['reward_cv'] = float(rewards.std() / (abs(rewards.mean()) + 1e-6))  # Coefficient of variation
        
        # Evaluation rewards
        if trial_data['eval_rewards']:
            eval_rewards = np.array(trial_data['eval_rewards'])
            summary['final_eval'] = float(eval_rewards[-1])
            summary['best_eval'] = float(eval_rewards.max())
        
        # Learning curves
        if trial_data['episode_rewards']:
            # Calculate convergence speed (episodes to reach 90% of best)
            rewards = np.array(trial_data['episode_rewards'])
            target = 0.9 * summary.get('best_reward', rewards[-1])
            convergence_episode = np.argmax(rewards >= target) if np.any(rewards >= target) else len(rewards)
            summary['convergence_episode'] = int(convergence_episode)
        
        # Metric network statistics
        if trial_data['condition_numbers']:
            cond_nums = np.array(trial_data['condition_numbers'])
            summary['avg_condition'] = float(cond_nums.mean())
            summary['max_condition'] = float(cond_nums.max())
            summary['condition_stability'] = float(cond_nums.std() / (cond_nums.mean() + 1e-6))
        
        if trial_data['min_eigenvalues']:
            min_eigs = np.array(trial_data['min_eigenvalues'])
            summary['min_eig_avg'] = float(min_eigs.mean())
            summary['min_eig_min'] = float(min_eigs.min())
        
        if trial_data['contraction_violations']:
            violations = np.array(trial_data['contraction_violations'])
            summary['avg_violation'] = float(violations.mean())
            summary['max_violation'] = float(violations.max())
        
        # Training stability
        if trial_data['dynamics_losses']:
            dynamics_losses = np.array(trial_data['dynamics_losses'])
            summary['dynamics_loss_final'] = float(dynamics_losses[-1])
        
        if trial_data['metric_losses']:
            metric_losses = np.array(trial_data['metric_losses'])
            summary['metric_loss_final'] = float(metric_losses[-1])
        
        # Energy decay
        if trial_data['energies']:
            energies = np.array(trial_data['energies'])
            if len(energies) > 10:
                # Calculate empirical decay rate
                decay_rates = energies[1:] / energies[:-1]
                summary['empirical_alpha'] = float(np.sqrt(decay_rates.mean()))
                summary['alpha_error'] = float(abs(summary['empirical_alpha'] - agent.config.CONTRACTION_RATE_ALPHA))
        
        return summary

# ============================
# SPECIAL CASE HANDLERS
# ============================

class SpecialCaseHandler:
    """Handles special cases for ablation variants"""
    
    @staticmethod
    def create_fixed_metric_agent(config: Config):
        """Create agent with fixed identity metric"""
        from robust_cdm import RobustContractionDynamicsAgent
        
        class FixedMetricAgent(RobustContractionDynamicsAgent):
            def _initialize_networks(self):
                """Override to create fixed identity metric"""
                super()._initialize_networks()
                
                # Replace metric network with identity
                class IdentityMetric(torch.nn.Module):
                    def __init__(self, state_dim):
                        super().__init__()
                        self.state_dim = state_dim
                    
                    def forward(self, x):
                        batch_size = x.shape[0]
                        identity = torch.eye(self.state_dim, device=x.device, dtype=x.dtype)
                        return identity.unsqueeze(0).repeat(batch_size, 1, 1), None
                    
                    def compute_metrics(self, M):
                        return {'condition_number': 1.0, 'min_eigenvalue': 1.0, 'max_eigenvalue': 1.0}
                
                self.metric_net = IdentityMetric(config.STATE_DIM).to(self.device)
                # Don't optimize fixed metric
                self.metric_optimizer = None
        
        return FixedMetricAgent(config)
    
    @staticmethod
    def create_no_cholesky_agent(config: Config):
        """Create agent without Cholesky parameterization"""
        from robust_cdm import RobustContractionDynamicsAgent, RobustContractionMetric
        
        class NoCholeskyMetric(RobustContractionMetric):
            """Metric without Cholesky parameterization"""
            def __init__(self, state_dim: int, hidden_dim: int = 128, epsilon: float = 0.05):
                super().__init__(state_dim, hidden_dim, epsilon)
                # Override to output full matrix
                self.output_dim = state_dim * state_dim
                self.net[-1] = torch.nn.Linear(hidden_dim, self.output_dim)
                
                # Remove Cholesky constraints
                self.softplus = None
                self.diagonal_offset = 0
                self.off_diagonal_scale = 1.0
            
            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                batch_size = x.shape[0]
                
                # Get raw parameters
                m_params = self.net(x)
                
                # Reshape to matrix
                M = m_params.view(batch_size, self.state_dim, self.state_dim)
                
                # Make symmetric
                M = (M + M.transpose(1, 2)) / 2
                
                # Ensure positive definiteness (simple method)
                identity = torch.eye(self.state_dim, device=x.device, dtype=x.dtype)
                identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
                M = M + self.epsilon * identity
                
                # No L matrix
                return M, None
        
        class NoCholeskyAgent(RobustContractionDynamicsAgent):
            def _initialize_networks(self):
                """Override to use non-Cholesky metric"""
                super()._initialize_networks()
                self.metric_net = NoCholeskyMetric(
                    self.config.STATE_DIM,
                    self.config.METRIC_HIDDEN_DIM,
                    self.config.EPSILON_METRIC
                ).to(self.device)
        
        return NoCholeskyAgent(config)
    
    @staticmethod
    def create_relu_activation_agent(config: Config):
        """Create agent with ReLU activation instead of softplus"""
        from robust_cdm import RobustContractionDynamicsAgent, RobustContractionMetric
        
        class ReLUMetric(RobustContractionMetric):
            def __init__(self, state_dim: int, hidden_dim: int = 128, epsilon: float = 0.05):
                super().__init__(state_dim, hidden_dim, epsilon)
                # Replace softplus with ReLU
                self.softplus = torch.nn.ReLU()
                self.diagonal_offset = 0.1  # Larger offset for ReLU
        
        class ReLUAgent(RobustContractionDynamicsAgent):
            def _initialize_networks(self):
                super()._initialize_networks()
                self.metric_net = ReLUMetric(
                    self.config.STATE_DIM,
                    self.config.METRIC_HIDDEN_DIM,
                    self.config.EPSILON_METRIC
                ).to(self.device)
        
        return ReLUAgent(config)
    
    @staticmethod
    def create_exp_activation_agent(config: Config):
        """Create agent with exponential activation"""
        from robust_cdm import RobustContractionDynamicsAgent, RobustContractionMetric
        
        class ExpMetric(RobustContractionMetric):
            def __init__(self, state_dim: int, hidden_dim: int = 128, epsilon: float = 0.05):
                super().__init__(state_dim, hidden_dim, epsilon)
                # Custom forward to use exp
                def exp_activation(x):
                    return torch.exp(x) + 0.01
                self.softplus = exp_activation
        
        class ExpAgent(RobustContractionDynamicsAgent):
            def _initialize_networks(self):
                super()._initialize_networks()
                self.metric_net = ExpMetric(
                    self.config.STATE_DIM,
                    self.config.METRIC_HIDDEN_DIM,
                    self.config.EPSILON_METRIC
                ).to(self.device)
        
        return ExpAgent(config)
    
    @staticmethod
    def create_no_ensemble_weights_agent(config: Config):
        """Create agent without learned ensemble weights"""
        from robust_cdm import RobustContractionDynamicsAgent, DynamicsEnsemble
        
        class FixedEnsemble(DynamicsEnsemble):
            def __init__(self, state_dim: int, action_dim: int, 
                         ensemble_size: int = 7, hidden_dim: int = 128):
                super().__init__(state_dim, action_dim, ensemble_size, hidden_dim)
                # Remove learnable weights
                del self.model_weights
            
            def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                predictions = []
                for model in self.models:
                    pred = model(state, action)
                    predictions.append(pred.unsqueeze(0))
                
                predictions = torch.cat(predictions, dim=0)
                
                # Equal weighting
                mean = predictions.mean(dim=0)
                variance = torch.var(predictions, dim=0, unbiased=True)
                epistemic = variance.mean(dim=-1, keepdim=True)
                
                return mean, epistemic
        
        class FixedEnsembleAgent(RobustContractionDynamicsAgent):
            def _initialize_networks(self):
                """Override to use fixed ensemble"""
                self.dynamics = FixedEnsemble(
                    self.config.STATE_DIM,
                    self.config.ACTION_DIM,
                    self.config.ENSEMBLE_SIZE,
                    self.config.DYNAMICS_HIDDEN_DIM
                ).to(self.device)
                
                # Initialize other networks normally
                super(RobustContractionDynamicsAgent, self)._initialize_networks()
        
        return FixedEnsembleAgent(config)

# ============================
# ABLATION RUNNER
# ============================

class AutomatedAblationRunner:
    """Main class for running automated ablation studies"""
    
    def __init__(self, ablation_config: AblationConfig):
        self.config = ablation_config
        self.metric_tracker = ComprehensiveMetricTracker(ablation_config)
        self.results = {}
        self.summary_stats = {}
        
        # Statistics for significance testing
        self.statistical_tests = {}
        
        # Create results DataFrame
        self.results_df = None
        
    def run_single_trial(self, variant_name: str, variant_config: Dict[str, Any], 
                        seed: int, trial_num: int) -> Dict[str, Any]:
        """Run a single trial for a specific variant"""
        print(f"\n{'='*80}")
        print(f"TRIAL {trial_num}: {variant_name} (Seed: {seed})")
        print('='*80)
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create configuration
        trial_config = self._create_trial_config(variant_config)
        trial_config.SEED = seed
        trial_config.TOTAL_EPISODES = self.config.num_episodes
        trial_config.EVAL_INTERVAL = self.config.eval_interval
        trial_config.EVAL_EPISODES = self.config.num_eval_episodes
        
        # Special save directory for this trial
        trial_config.SAVE_DIR = f"{self.config.save_dir}/models/{variant_name}_seed{seed}"
        
        # Create agent (handle special cases)
        start_time = time.time()
        
        if 'special_case' in variant_config:
            agent = self._create_special_case_agent(variant_config['special_case'], trial_config)
        else:
            agent = RobustContractionDynamicsAgent(trial_config)
        
        # Create environments
        env = gym.make(trial_config.ENV_NAME)
        eval_env = gym.make(trial_config.ENV_NAME)
        
        # Initialize metric tracking
        trial_id = self.metric_tracker.initialize_trial(variant_name, seed)
        
        # Run training
        try:
            training_results = agent.train(env, eval_env)
            
            # Update metrics
            self.metric_tracker.update(trial_id, {}, agent.metrics)
            
            # Calculate additional metrics
            self.metric_tracker.update_metric_network_stats(trial_id, agent)
            self.metric_tracker.calculate_contraction_violation(trial_id, agent)
            
            # Get summary statistics
            summary = self.metric_tracker.get_summary_stats(trial_id)
            summary['training_time'] = time.time() - start_time
            summary['success'] = True
            
            print(f"✓ Trial completed successfully in {summary['training_time']:.1f}s")
            print(f"  Final reward: {summary.get('final_reward', 'N/A'):.1f}")
            print(f"  Best reward: {summary.get('best_reward', 'N/A'):.1f}")
            
        except Exception as e:
            print(f"✗ Trial failed with error: {e}")
            import traceback
            traceback.print_exc()
            summary = {
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time
            }
        
        finally:
            env.close()
            eval_env.close()
        
        return summary
    
    def _create_trial_config(self, variant_config: Dict[str, Any]) -> Config:
        """Create configuration for a specific variant"""
        # Start with base config
        config_dict = asdict(self.config.base_config)
        
        # Apply modifications
        if 'config_modifications' in variant_config:
            for key, value in variant_config['config_modifications'].items():
                if key in config_dict:
                    config_dict[key] = value
        
        # Create Config object
        return Config(**config_dict)
    
    def _create_special_case_agent(self, special_case: str, config: Config):
        """Create agent for special cases"""
        handler = SpecialCaseHandler()
        
        if special_case == 'fixed_metric':
            return handler.create_fixed_metric_agent(config)
        elif special_case == 'no_cholesky':
            return handler.create_no_cholesky_agent(config)
        elif special_case == 'relu_activation':
            return handler.create_relu_activation_agent(config)
        elif special_case == 'exp_activation':
            return handler.create_exp_activation_agent(config)
        elif special_case == 'no_ensemble_weights':
            return handler.create_no_ensemble_weights_agent(config)
        else:
            raise ValueError(f"Unknown special case: {special_case}")
    
    def run_all_ablations(self):
        """Run all ablation studies"""
        print("\n" + "="*80)
        print("STARTING AUTOMATED ABLATION STUDIES")
        print("="*80)
        print(f"Total variants: {len(self.config.ablation_variants)}")
        print(f"Seeds per variant: {self.config.num_seeds}")
        print(f"Total trials: {len(self.config.ablation_variants) * self.config.num_seeds}")
        print("="*80)
        
        overall_start_time = time.time()
        trial_count = 0
        
        for variant_key, variant_config in self.config.ablation_variants.items():
            variant_name = variant_config['name']
            print(f"\n\n{'#'*60}")
            print(f"RUNNING VARIANT: {variant_name}")
            print(f"{'#'*60}")
            print(f"Description: {variant_config.get('description', 'No description')}")
            
            variant_results = []
            
            for seed_idx in range(self.config.num_seeds):
                seed = self.config.seeds[seed_idx]
                trial_count += 1
                
                print(f"\n--- Trial {trial_count}/{len(self.config.ablation_variants) * self.config.num_seeds} ---")
                
                trial_result = self.run_single_trial(variant_name, variant_config, seed, trial_count)
                trial_result['variant'] = variant_name
                trial_result['variant_key'] = variant_key
                trial_result['seed'] = seed
                
                variant_results.append(trial_result)
                
                # Save intermediate results
                if trial_count % 5 == 0:
                    self._save_intermediate_results()
            
            # Store results for this variant
            self.results[variant_name] = variant_results
        
        total_time = time.time() - overall_start_time
        print(f"\n{'='*80}")
        print(f"ALL ABLATION STUDIES COMPLETED")
        print(f"Total time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
        print(f"Average per trial: {total_time/trial_count:.1f}s")
        print("="*80)
        
        # Generate comprehensive analysis
        self._generate_analysis()
        
        return self.results
    
    def _save_intermediate_results(self):
        """Save intermediate results to disk"""
        save_path = Path(f"{self.config.save_dir}/logs/intermediate_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        
        data = {
            'results': self.results,
            'metrics': self.metric_tracker.metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Intermediate results saved to {save_path}")
    
    def _generate_analysis(self):
        """Generate comprehensive analysis of results"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE ANALYSIS")
        print("="*80)
        
        # 1. Create summary DataFrame
        self._create_summary_dataframe()
        
        # 2. Perform statistical tests
        self._perform_statistical_tests()
        
        # 3. Generate publication-quality tables
        self._generate_publication_tables()
        
        # 4. Generate plots
        if self.config.generate_plots:
            self._generate_all_plots()
        
        # 5. Save final results
        self._save_final_results()
        
        print("\n✓ Analysis complete!")
    
    def _create_summary_dataframe(self):
        """Create summary DataFrame from results"""
        rows = []
        
        for variant_name, variant_results in self.results.items():
            successful_trials = [r for r in variant_results if r.get('success', False)]
            
            if not successful_trials:
                continue
            
            # Extract metrics
            final_rewards = [t.get('final_reward', 0) for t in successful_trials]
            best_rewards = [t.get('best_reward', 0) for t in successful_trials]
            avg_rewards = [t.get('avg_reward', 0) for t in successful_trials]
            convergence_eps = [t.get('convergence_episode', self.config.num_episodes) 
                              for t in successful_trials]
            condition_numbers = [t.get('avg_condition', 1.0) for t in successful_trials]
            violation_rates = [t.get('avg_violation', 0.0) for t in successful_trials]
            training_times = [t.get('training_time', 0) for t in successful_trials]
            
            # Calculate statistics
            row = {
                'Variant': variant_name,
                'N': len(successful_trials),
                'Final Reward': f"{np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}",
                'Best Reward': f"{np.mean(best_rewards):.1f} ± {np.std(best_rewards):.1f}",
                'Avg Reward': f"{np.mean(avg_rewards):.1f} ± {np.std(avg_rewards):.1f}",
                'Convergence Episode': f"{np.mean(convergence_eps):.0f} ± {np.std(convergence_eps):.0f}",
                'Condition Number': f"{np.mean(condition_numbers):.2f} ± {np.std(condition_numbers):.2f}",
                'Contraction Violation (%)': f"{np.mean(violation_rates)*100:.1f} ± {np.std(violation_rates)*100:.1f}",
                'Training Time (s)': f"{np.mean(training_times):.0f} ± {np.std(training_times):.0f}",
                'Success Rate': f"{len(successful_trials)/len(variant_results)*100:.0f}%"
            }
            
            # Store raw values for statistical tests
            self.summary_stats[variant_name] = {
                'final_rewards': final_rewards,
                'best_rewards': best_rewards,
                'convergence_episodes': convergence_eps,
                'condition_numbers': condition_numbers,
                'violation_rates': violation_rates
            }
            
            rows.append(row)
        
        self.results_df = pd.DataFrame(rows)
        
        # Sort by final reward (descending)
        if not self.results_df.empty:
            # Extract mean final reward for sorting
            self.results_df['_sort_key'] = self.results_df['Final Reward'].apply(
                lambda x: float(x.split('±')[0])
            )
            self.results_df = self.results_df.sort_values('_sort_key', ascending=False)
            self.results_df = self.results_df.drop('_sort_key', axis=1)
    
    def _perform_statistical_tests(self):
        """Perform statistical significance tests"""
        from scipy import stats
        
        print("\nPerforming statistical significance tests...")
        
        variants = list(self.summary_stats.keys())
        
        if 'Full CDM' in self.summary_stats and len(variants) > 1:
            baseline = 'Full CDM'
            baseline_rewards = self.summary_stats[baseline]['final_rewards']
            
            for variant in variants:
                if variant == baseline:
                    continue
                
                variant_rewards = self.summary_stats.get(variant, {}).get('final_rewards', [])
                
                if len(variant_rewards) >= 3 and len(baseline_rewards) >= 3:
                    # Independent t-test
                    t_stat, p_value = stats.ttest_ind(baseline_rewards, variant_rewards, 
                                                    equal_var=False)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((np.std(baseline_rewards, ddof=1)**2 + 
                                         np.std(variant_rewards, ddof=1)**2) / 2)
                    cohens_d = (np.mean(baseline_rewards) - np.mean(variant_rewards)) / pooled_std
                    
                    self.statistical_tests[f"{baseline}_vs_{variant}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant_0_05': p_value < 0.05,
                        'significant_0_01': p_value < 0.01
                    }
                    
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    print(f"  {baseline} vs {variant}: p = {p_value:.4f} {significance} (d = {cohens_d:.2f})")
    
    def _generate_publication_tables(self):
        """Generate publication-ready LaTeX tables"""
        print("\nGenerating publication tables...")
        
        # Main results table
        if self.results_df is not None and not self.results_df.empty:
            latex_table = self.results_df.to_latex(index=False, 
                                                  caption='Ablation Study Results',
                                                  label='tab:ablation_results',
                                                  column_format='l' + 'c' * (len(self.results_df.columns) - 1))
            
            table_path = Path(f"{self.config.save_dir}/tables/ablation_results.tex")
            with open(table_path, 'w') as f:
                f.write(latex_table)
            
            print(f"✓ LaTeX table saved to {table_path}")
            
            # Also save as CSV
            csv_path = Path(f"{self.config.save_dir}/tables/ablation_results.csv")
            self.results_df.to_csv(csv_path, index=False)
            print(f"✓ CSV table saved to {csv_path}")
        
        # Statistical significance table
        if self.statistical_tests:
            stats_rows = []
            for test_name, test_result in self.statistical_tests.items():
                stats_rows.append({
                    'Comparison': test_name,
                    't-statistic': f"{test_result['t_statistic']:.2f}",
                    'p-value': f"{test_result['p_value']:.4f}",
                    "Cohen's d": f"{test_result['cohens_d']:.2f}",
                    'Significant (α=0.05)': 'Yes' if test_result['significant_0_05'] else 'No',
                    'Significant (α=0.01)': 'Yes' if test_result['significant_0_01'] else 'No'
                })
            
            stats_df = pd.DataFrame(stats_rows)
            stats_latex = stats_df.to_latex(index=False, 
                                          caption='Statistical Significance Tests',
                                          label='tab:statistical_tests')
            
            stats_path = Path(f"{self.config.save_dir}/tables/statistical_tests.tex")
            with open(stats_path, 'w') as f:
                f.write(stats_latex)
            
            print(f"✓ Statistical tests table saved to {stats_path}")
    
    def _generate_all_plots(self):
        """Generate all analysis plots"""
        print("\nGenerating analysis plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Performance comparison plot
        self._plot_performance_comparison()
        
        # 2. Learning curves
        self._plot_learning_curves()
        
        # 3. Metric stability analysis
        self._plot_metric_stability()
        
        # 4. Convergence analysis
        self._plot_convergence_analysis()
        
        # 5. Statistical significance visualization
        self._plot_statistical_significance()
        
        print("✓ All plots generated")
    
    def _plot_performance_comparison(self):
        """Plot performance comparison across variants"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Ablation Study: Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        variants = []
        final_rewards = []
        best_rewards = []
        convergence_eps = []
        condition_numbers = []
        violation_rates = []
        
        for variant_name, stats in self.summary_stats.items():
            variants.append(variant_name)
            final_rewards.append(stats['final_rewards'])
            best_rewards.append(stats['best_rewards'])
            convergence_eps.append(stats['convergence_episodes'])
            condition_numbers.append(stats['condition_numbers'])
            violation_rates.append(stats['violation_rates'])
        
        # 1. Final rewards (box plot)
        ax = axes[0, 0]
        bp = ax.boxplot(final_rewards, labels=variants, patch_artist=True)
        
        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Final Episode Reward')
        ax.set_title('Final Performance Distribution')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 2. Best rewards (bar plot with error bars)
        ax = axes[0, 1]
        means = [np.mean(r) for r in best_rewards]
        stds = [np.std(r) for r in best_rewards]
        
        x_pos = np.arange(len(variants))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='lightgreen')
        
        # Highlight the best performing variant
        best_idx = np.argmax(means)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_alpha(0.9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variants, rotation=45)
        ax.set_ylabel('Best Episode Reward')
        ax.set_title('Best Performance Achieved')
        ax.grid(True, alpha=0.3)
        
        # 3. Convergence speed
        ax = axes[0, 2]
        means = [np.mean(c) for c in convergence_eps]
        stds = [np.std(c) for c in convergence_eps]
        
        x_pos = np.arange(len(variants))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='lightcoral')
        
        # Lower is better (faster convergence)
        best_idx = np.argmin(means)
        bars[best_idx].set_color('lightgreen')
        bars[best_idx].set_alpha(0.9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variants, rotation=45)
        ax.set_ylabel('Convergence Episode')
        ax.set_title('Learning Speed (Lower is Better)')
        ax.grid(True, alpha=0.3)
        
        # 4. Condition numbers (stability metric)
        ax = axes[1, 0]
        bp = ax.boxplot(condition_numbers, labels=variants, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('plum')
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Condition Number (log scale)')
        ax.set_title('Metric Condition Number Distribution')
        ax.tick_params(axis='x', rotation=45)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 5. Contraction violations
        ax = axes[1, 1]
        means = [np.mean(v)*100 for v in violation_rates]
        stds = [np.std(v)*100 for v in violation_rates]
        
        x_pos = np.arange(len(variants))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='orange')
        
        # Lower is better (fewer violations)
        best_idx = np.argmin(means)
        bars[best_idx].set_color('lightgreen')
        bars[best_idx].set_alpha(0.9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variants, rotation=45)
        ax.set_ylabel('Contraction Violations (%)')
        ax.set_title('Stability Violation Rate')
        ax.grid(True, alpha=0.3)
        
        # 6. Performance vs Stability trade-off
        ax = axes[1, 2]
        
        # Calculate average performance and stability
        perf_scores = [np.mean(r) for r in final_rewards]
        stability_scores = [1 - np.mean(v) for v in violation_rates]  # Higher is more stable
        
        # Normalize for visualization
        perf_norm = (perf_scores - np.min(perf_scores)) / (np.max(perf_scores) - np.min(perf_scores) + 1e-6)
        stability_norm = (stability_scores - np.min(stability_scores)) / (np.max(stability_scores) - np.min(stability_scores) + 1e-6)
        
        scatter = ax.scatter(perf_norm, stability_norm, s=200, alpha=0.6)
        
        # Add labels
        for i, variant in enumerate(variants):
            ax.annotate(variant, (perf_norm[i], stability_norm[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Normalized Performance (Higher is Better)')
        ax.set_ylabel('Normalized Stability (Higher is Better)')
        ax.set_title('Performance-Stability Trade-off')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/figures/performance_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_learning_curves(self):
        """Plot learning curves for all variants"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ablation Study: Learning Curves Analysis', fontsize=16, fontweight='bold')
        
        # Select key variants for clarity
        key_variants = ['Full CDM', 'No Contraction', 'Fixed Metric', 'No Cholesky']
        if len(self.config.ablation_variants) > 4:
            key_variants.append(list(self.config.ablation_variants.keys())[4])
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(key_variants)))
        
        # 1. Episode rewards
        ax = axes[0, 0]
        for i, variant_name in enumerate(key_variants):
            if variant_name in self.metric_tracker.metrics:
                # Get all trials for this variant
                trial_keys = [k for k in self.metric_tracker.metrics.keys() 
                            if k.startswith(f"{variant_name}_seed")]
                
                if trial_keys:
                    # Find maximum episode length
                    max_episodes = 0
                    all_rewards = []
                    
                    for trial_key in trial_keys:
                        rewards = self.metric_tracker.metrics[trial_key].get('episode_rewards', [])
                        if len(rewards) > max_episodes:
                            max_episodes = len(rewards)
                        all_rewards.append(rewards)
                    
                    # Pad rewards to same length
                    padded_rewards = []
                    for rewards in all_rewards:
                        if len(rewards) < max_episodes:
                            padded = list(rewards) + [rewards[-1]] * (max_episodes - len(rewards))
                        else:
                            padded = rewards[:max_episodes]
                        padded_rewards.append(padded)
                    
                    if padded_rewards:
                        mean_rewards = np.mean(padded_rewards, axis=0)
                        std_rewards = np.std(padded_rewards, axis=0)
                        
                        episodes = np.arange(len(mean_rewards))
                        ax.plot(episodes, mean_rewards, label=variant_name, 
                               color=colors[i], linewidth=2)
                        ax.fill_between(episodes, 
                                       mean_rewards - std_rewards,
                                       mean_rewards + std_rewards,
                                       alpha=0.2, color=colors[i])
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Learning Progress')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # 2. Moving average (window=20)
        ax = axes[0, 1]
        window = 20
        
        for i, variant_name in enumerate(key_variants):
            if variant_name in [k.split('_seed')[0] for k in self.metric_tracker.metrics.keys()]:
                trial_keys = [k for k in self.metric_tracker.metrics.keys() 
                            if k.startswith(f"{variant_name}_seed")]
                
                if trial_keys:
                    all_ma = []
                    max_length = 0
                    
                    for trial_key in trial_keys:
                        rewards = self.metric_tracker.metrics[trial_key].get('episode_rewards', [])
                        if len(rewards) >= window:
                            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
                            if len(ma) > max_length:
                                max_length = len(ma)
                            all_ma.append(ma)
                    
                    # Pad moving averages
                    padded_ma = []
                    for ma in all_ma:
                        if len(ma) < max_length:
                            padded = list(ma) + [ma[-1]] * (max_length - len(ma))
                        else:
                            padded = ma[:max_length]
                        padded_ma.append(padded)
                    
                    if padded_ma:
                        mean_ma = np.mean(padded_ma, axis=0)
                        std_ma = np.std(padded_ma, axis=0)
                        
                        episodes = np.arange(window-1, window-1 + len(mean_ma))
                        ax.plot(episodes, mean_ma, label=variant_name, 
                               color=colors[i], linewidth=2)
                        ax.fill_between(episodes, 
                                       mean_ma - std_ma,
                                       mean_ma + std_ma,
                                       alpha=0.2, color=colors[i])
        
        ax.set_xlabel('Episode')
        ax.set_ylabel(f'Reward ({window}-ep Moving Avg)')
        ax.set_title('Smoothed Learning Progress')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # 3. Contraction energy
        ax = axes[1, 0]
        
        for i, variant_name in enumerate(key_variants):
            if variant_name in [k.split('_seed')[0] for k in self.metric_tracker.metrics.keys()]:
                trial_keys = [k for k in self.metric_tracker.metrics.keys() 
                            if k.startswith(f"{variant_name}_seed")]
                
                if trial_keys:
                    all_energies = []
                    
                    for trial_key in trial_keys:
                        energies = self.metric_tracker.metrics[trial_key].get('energies', [])
                        if energies:
                            all_energies.append(energies[:500])  # First 500 steps
                    
                    if all_energies:
                        # Find minimum length
                        min_length = min([len(e) for e in all_energies])
                        trimmed_energies = [e[:min_length] for e in all_energies]
                        
                        mean_energy = np.mean(trimmed_energies, axis=0)
                        std_energy = np.std(trimmed_energies, axis=0)
                        
                        steps = np.arange(len(mean_energy))
                        ax.plot(steps, mean_energy, label=variant_name, 
                               color=colors[i], linewidth=1, alpha=0.8)
                        ax.fill_between(steps, 
                                       mean_energy - std_energy,
                                       mean_energy + std_energy,
                                       alpha=0.1, color=colors[i])
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Contraction Energy')
        ax.set_title('Metric Energy Evolution')
        ax.set_yscale('log')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 4. Beta adaptation
        ax = axes[1, 1]
        
        for i, variant_name in enumerate(key_variants):
            if variant_name == 'No Contraction':
                continue  # Skip no contraction variant
            
            if variant_name in [k.split('_seed')[0] for k in self.metric_tracker.metrics.keys()]:
                trial_keys = [k for k in self.metric_tracker.metrics.keys() 
                            if k.startswith(f"{variant_name}_seed")]
                
                if trial_keys:
                    all_betas = []
                    max_episodes = 0
                    
                    for trial_key in trial_keys:
                        betas = self.metric_tracker.metrics[trial_key].get('betas', [])
                        if betas:
                            all_betas.append(betas)
                            if len(betas) > max_episodes:
                                max_episodes = len(betas)
                    
                    if all_betas:
                        # Pad betas
                        padded_betas = []
                        for betas in all_betas:
                            if len(betas) < max_episodes:
                                padded = list(betas) + [betas[-1]] * (max_episodes - len(betas))
                            else:
                                padded = betas[:max_episodes]
                            padded_betas.append(padded)
                        
                        mean_betas = np.mean(padded_betas, axis=0)
                        std_betas = np.std(padded_betas, axis=0)
                        
                        episodes = np.arange(len(mean_betas))
                        ax.plot(episodes, mean_betas, label=variant_name, 
                               color=colors[i], linewidth=2)
                        ax.fill_between(episodes, 
                                       mean_betas - std_betas,
                                       mean_betas + std_betas,
                                       alpha=0.2, color=colors[i])
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('β (Stability Weight)')
        ax.set_title('Adaptive Stability-Exploration Balance')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/figures/learning_curves.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_stability(self):
        """Plot metric stability analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ablation Study: Metric Stability Analysis', fontsize=16, fontweight='bold')
        
        # Select key variants
        key_variants = ['Full CDM', 'No Cholesky', 'Fixed Metric', 'No Metric Reg']
        colors = plt.cm.Set2(np.linspace(0, 1, len(key_variants)))
        
        # 1. Condition number over time
        ax = axes[0, 0]
        
        for i, variant_name in enumerate(key_variants):
            if variant_name in [k.split('_seed')[0] for k in self.metric_tracker.metrics.keys()]:
                trial_keys = [k for k in self.metric_tracker.metrics.keys() 
                            if k.startswith(f"{variant_name}_seed")]
                
                if trial_keys:
                    all_conditions = []
                    max_steps = 0
                    
                    for trial_key in trial_keys:
                        conditions = self.metric_tracker.metrics[trial_key].get('condition_numbers', [])
                        if conditions:
                            all_conditions.append(conditions[:100])  # First 100 measurements
                            if len(conditions) > max_steps:
                                max_steps = len(conditions)
                    
                    if all_conditions:
                        min_length = min([len(c) for c in all_conditions])
                        trimmed = [c[:min_length] for c in all_conditions]
                        
                        mean_cond = np.mean(trimmed, axis=0)
                        std_cond = np.std(trimmed, axis=0)
                        
                        steps = np.arange(len(mean_cond))
                        ax.plot(steps, mean_cond, label=variant_name, 
                               color=colors[i], linewidth=2)
                        ax.fill_between(steps, 
                                       mean_cond - std_cond,
                                       mean_cond + std_cond,
                                       alpha=0.2, color=colors[i])
        
        ax.set_xlabel('Measurement Step')
        ax.set_ylabel('Condition Number')
        ax.set_title('Metric Condition Number Evolution')
        ax.set_yscale('log')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 2. Minimum eigenvalue distribution
        ax = axes[0, 1]
        
        data_to_plot = []
        labels = []
        
        for i, variant_name in enumerate(key_variants):
            if variant_name in self.summary_stats:
                min_eigs = self.summary_stats[variant_name].get('min_eigenvalues', [])
                if min_eigs:
                    data_to_plot.append(min_eigs)
                    labels.append(variant_name)
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Minimum Eigenvalue')
            ax.set_title('Metric Positive Definiteness')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 3. Contraction violation rate
        ax = axes[1, 0]
        
        data_to_plot = []
        labels = []
        
        for i, variant_name in enumerate(key_variants):
            if variant_name in self.summary_stats:
                violations = self.summary_stats[variant_name].get('violation_rates', [])
                if violations:
                    data_to_plot.append([v*100 for v in violations])  # Convert to percentage
                    labels.append(variant_name)
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Contraction Violations (%)')
            ax.set_title('Stability Guarantee Violations')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 4. Performance vs Condition number
        ax = axes[1, 1]
        
        perf_scores = []
        cond_scores = []
        variant_labels = []
        
        for variant_name in key_variants:
            if variant_name in self.summary_stats:
                final_rewards = self.summary_stats[variant_name].get('final_rewards', [])
                conditions = self.summary_stats[variant_name].get('condition_numbers', [])
                
                if final_rewards and conditions:
                    perf_scores.append(np.mean(final_rewards))
                    cond_scores.append(np.mean(conditions))
                    variant_labels.append(variant_name)
        
        if perf_scores and cond_scores:
            scatter = ax.scatter(perf_scores, cond_scores, s=150, alpha=0.6, c=colors[:len(perf_scores)])
            
            # Add labels
            for i, label in enumerate(variant_labels):
                ax.annotate(label, (perf_scores[i], cond_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax.set_xlabel('Final Reward (Higher is Better)')
            ax.set_ylabel('Condition Number (Lower is Better)')
            ax.set_title('Performance vs Metric Stability Trade-off')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/figures/metric_stability.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_convergence_analysis(self):
        """Plot convergence analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Ablation Study: Convergence Analysis', fontsize=16, fontweight='bold')
        
        key_variants = ['Full CDM', 'No Contraction', 'Fixed Metric', 'α=0.9', 'α=0.97']
        colors = plt.cm.tab10(np.linspace(0, 1, len(key_variants)))
        
        # 1. Cumulative reward over time
        ax = axes[0, 0]
        
        for i, variant_name in enumerate(key_variants):
            if variant_name in [k.split('_seed')[0] for k in self.metric_tracker.metrics.keys()]:
                trial_keys = [k for k in self.metric_tracker.metrics.keys() 
                            if k.startswith(f"{variant_name}_seed")]
                
                if trial_keys:
                    all_cumulative = []
                    max_episodes = 0
                    
                    for trial_key in trial_keys:
                        rewards = self.metric_tracker.metrics[trial_key].get('episode_rewards', [])
                        if rewards:
                            cumulative = np.cumsum(rewards)
                            all_cumulative.append(cumulative)
                            if len(cumulative) > max_episodes:
                                max_episodes = len(cumulative)
                    
                    if all_cumulative:
                        # Pad to same length
                        padded = []
                        for cumulative in all_cumulative:
                            if len(cumulative) < max_episodes:
                                padded.append(list(cumulative) + [cumulative[-1]] * (max_episodes - len(cumulative)))
                            else:
                                padded.append(cumulative[:max_episodes])
                        
                        mean_cumulative = np.mean(padded, axis=0)
                        std_cumulative = np.std(padded, axis=0)
                        
                        episodes = np.arange(len(mean_cumulative))
                        ax.plot(episodes, mean_cumulative, label=variant_name, 
                               color=colors[i], linewidth=2)
                        ax.fill_between(episodes, 
                                       mean_cumulative - std_cumulative,
                                       mean_cumulative + std_cumulative,
                                       alpha=0.2, color=colors[i])
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Cumulative Learning Progress')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # 2. Success rate over time
        ax = axes[0, 1]
        threshold = -500  # Define success threshold
        
        for i, variant_name in enumerate(key_variants):
            if variant_name in [k.split('_seed')[0] for k in self.metric_tracker.metrics.keys()]:
                trial_keys = [k for k in self.metric_tracker.metrics.keys() 
                            if k.startswith(f"{variant_name}_seed")]
                
                if trial_keys:
                    all_success_rates = []
                    max_episodes = 0
                    
                    for trial_key in trial_keys:
                        rewards = self.metric_tracker.metrics[trial_key].get('episode_rewards', [])
                        if rewards:
                            # Calculate rolling success rate
                            window = 20
                            success_rates = []
                            for j in range(len(rewards) - window + 1):
                                window_rewards = rewards[j:j+window]
                                successes = sum(1 for r in window_rewards if r > threshold)
                                success_rates.append(successes / window * 100)
                            
                            if success_rates:
                                all_success_rates.append(success_rates)
                                if len(success_rates) > max_episodes:
                                    max_episodes = len(success_rates)
                    
                    if all_success_rates:
                        # Pad to same length
                        padded = []
                        for rates in all_success_rates:
                            if len(rates) < max_episodes:
                                padded.append(list(rates) + [rates[-1]] * (max_episodes - len(rates)))
                            else:
                                padded.append(rates[:max_episodes])
                        
                        mean_success = np.mean(padded, axis=0)
                        std_success = np.std(padded, axis=0)
                        
                        episodes = np.arange(window-1, window-1 + len(mean_success))
                        ax.plot(episodes, mean_success, label=variant_name, 
                               color=colors[i], linewidth=2)
                        ax.fill_between(episodes, 
                                       mean_success - std_success,
                                       mean_success + std_success,
                                       alpha=0.2, color=colors[i])
        
        ax.set_xlabel('Episode')
        ax.set_ylabel(f'Success Rate (%) > {threshold}')
        ax.set_title('Learning Reliability')
        ax.legend(loc='lower right')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)
        
        # 3. Convergence speed distribution
        ax = axes[1, 0]
        
        data_to_plot = []
        labels = []
        
        for variant_name in key_variants:
            if variant_name in self.summary_stats:
                convergence = self.summary_stats[variant_name].get('convergence_episodes', [])
                if convergence:
                    data_to_plot.append(convergence)
                    labels.append(variant_name)
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Convergence Episode (Lower is Faster)')
            ax.set_title('Convergence Speed Distribution')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # 4. Final performance vs Convergence speed
        ax = axes[1, 1]
        
        perf_scores = []
        conv_scores = []
        variant_labels = []
        
        for variant_name in key_variants:
            if variant_name in self.summary_stats:
                final_rewards = self.summary_stats[variant_name].get('final_rewards', [])
                convergence = self.summary_stats[variant_name].get('convergence_episodes', [])
                
                if final_rewards and convergence:
                    perf_scores.append(np.mean(final_rewards))
                    conv_scores.append(np.mean(convergence))
                    variant_labels.append(variant_name)
        
        if perf_scores and conv_scores:
            scatter = ax.scatter(conv_scores, perf_scores, s=150, alpha=0.6, c=colors[:len(perf_scores)])
            
            # Add labels
            for i, label in enumerate(variant_labels):
                ax.annotate(label, (conv_scores[i], perf_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax.set_xlabel('Convergence Episode (Lower is Faster)')
            ax.set_ylabel('Final Reward (Higher is Better)')
            ax.set_title('Speed vs Performance Trade-off')
            ax.grid(True, alpha=0.3)
            
            # Add ideal quadrant indicators
            ax.axhline(y=np.median(perf_scores), color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=np.median(conv_scores), color='gray', linestyle='--', alpha=0.5)
            
            # Label quadrants
            ax.text(0.05, 0.95, 'Fast & Good', transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            ax.text(0.65, 0.95, 'Slow & Good', transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
            ax.text(0.05, 0.05, 'Fast & Poor', transform=ax.transAxes, fontsize=10,
                   verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax.text(0.65, 0.05, 'Slow & Poor', transform=ax.transAxes, fontsize=10,
                   verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/figures/convergence_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_significance(self):
        """Plot statistical significance results"""
        if not self.statistical_tests:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Statistical Significance Analysis', fontsize=16, fontweight='bold')
        
        # 1. P-values visualization
        ax = axes[0]
        
        comparisons = []
        p_values = []
        significant = []
        
        for test_name, test_result in self.statistical_tests.items():
            # Extract variant names
            variants = test_name.replace('Full CDM_vs_', '')
            comparisons.append(variants)
            p_values.append(test_result['p_value'])
            significant.append(test_result['significant_0_05'])
        
        # Sort by p-value
        sorted_idx = np.argsort(p_values)
        comparisons = [comparisons[i] for i in sorted_idx]
        p_values = [p_values[i] for i in sorted_idx]
        significant = [significant[i] for i in sorted_idx]
        
        # Create bar plot
        x_pos = np.arange(len(comparisons))
        colors = ['lightcoral' if sig else 'lightblue' for sig in significant]
        
        bars = ax.barh(x_pos, p_values, color=colors, alpha=0.7)
        
        # Add significance markers
        for i, (p, sig) in enumerate(zip(p_values, significant)):
            if sig:
                ax.text(p, i, ' *', va='center', fontsize=12, fontweight='bold')
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(comparisons)
        ax.set_xlabel('p-value (log scale)')
        ax.set_title('Statistical Significance vs Full CDM')
        ax.set_xscale('log')
        ax.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label='α=0.05')
        ax.axvline(x=0.01, color='darkred', linestyle='--', alpha=0.5, label='α=0.01')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # 2. Effect sizes (Cohen's d)
        ax = axes[1]
        
        comparisons = []
        effect_sizes = []
        magnitudes = []
        
        for test_name, test_result in self.statistical_tests.items():
            variants = test_name.replace('Full CDM_vs_', '')
            comparisons.append(variants)
            d = test_result['cohens_d']
            effect_sizes.append(d)
            
            # Classify effect size
            if abs(d) < 0.2:
                magnitudes.append('Negligible')
            elif abs(d) < 0.5:
                magnitudes.append('Small')
            elif abs(d) < 0.8:
                magnitudes.append('Medium')
            else:
                magnitudes.append('Large')
        
        # Sort by effect size
        sorted_idx = np.argsort(effect_sizes)[::-1]  # Descending
        comparisons = [comparisons[i] for i in sorted_idx]
        effect_sizes = [effect_sizes[i] for i in sorted_idx]
        magnitudes = [magnitudes[i] for i in sorted_idx]
        
        # Color by magnitude
        color_map = {
            'Negligible': 'lightgray',
            'Small': 'lightblue',
            'Medium': 'lightgreen',
            'Large': 'gold'
        }
        colors = [color_map[m] for m in magnitudes]
        
        x_pos = np.arange(len(comparisons))
        bars = ax.barh(x_pos, effect_sizes, color=colors, alpha=0.7)
        
        # Add magnitude labels
        for i, (effect, mag) in enumerate(zip(effect_sizes, magnitudes)):
            ax.text(effect/2 if effect > 0 else effect*2, i, mag, 
                   va='center', ha='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(comparisons)
        ax.set_xlabel("Cohen's d (Effect Size)")
        ax.set_title('Effect Size of Performance Differences')
        
        # Add effect size reference lines
        ax.axvline(x=0.2, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=0.8, color='gray', linestyle=':', alpha=0.5)
        
        ax.text(0.1, 0.95, 'Negligible', transform=ax.transAxes, fontsize=8, alpha=0.7)
        ax.text(0.35, 0.95, 'Small', transform=ax.transAxes, fontsize=8, alpha=0.7)
        ax.text(0.65, 0.95, 'Medium', transform=ax.transAxes, fontsize=8, alpha=0.7)
        ax.text(0.9, 0.95, 'Large', transform=ax.transAxes, fontsize=8, alpha=0.7)
        
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/figures/statistical_significance.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_final_results(self):
        """Save final results to disk"""
        print("\nSaving final results...")
        
        # Save complete results dictionary
        final_results = {
            'results': self.results,
            'summary_stats': self.summary_stats,
            'statistical_tests': self.statistical_tests,
            'metrics': self.metric_tracker.metrics,
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat(),
            'git_hash': self._get_git_hash()  # For reproducibility
        }
        
        # Save as pickle
        pickle_path = Path(f"{self.config.save_dir}/final_results.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(final_results, f)
        
        # Save as JSON (partial, excluding large arrays)
        json_results = {
            'summary_stats': {
                k: {sk: np.mean(sv) if isinstance(sv, list) else sv 
                    for sk, sv in v.items()} 
                for k, v in self.summary_stats.items()
            },
            'statistical_tests': self.statistical_tests,
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat()
        }
        
        json_path = Path(f"{self.config.save_dir}/final_results.json")
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save results DataFrame
        if self.results_df is not None:
            csv_path = Path(f"{self.config.save_dir}/tables/final_summary.csv")
            self.results_df.to_csv(csv_path, index=False)
            
            # Also save as Markdown for easy reading
            md_path = Path(f"{self.config.save_dir}/tables/final_summary.md")
            with open(md_path, 'w') as f:
                f.write("# Ablation Study Results\n\n")
                f.write(self.results_df.to_markdown(index=False))
        
        print(f"✓ Final results saved to {self.config.save_dir}/")
        print(f"  - final_results.pkl (complete data)")
        print(f"  - final_results.json (summary)")
        print(f"  - tables/final_summary.csv (CSV)")
        print(f"  - tables/final_summary.md (Markdown)")
    
    def _get_git_hash(self):
        """Get current git hash for reproducibility"""
        try:
            import subprocess
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        except:
            return "unknown"

# ============================
# MAIN EXECUTION
# ============================

def main():
    """Main function to run automated ablation studies"""
    print("="*100)
    print("AUTOMATED ABLATION STUDY FRAMEWORK")
    print("="*100)
    print("Paper: 'Learning Contraction Metrics for Provably Stable Model-Based RL'")
    print("Author: Amir Hameed, Sirraya Labs")
    print("="*100)
    
    # Create ablation configuration
    ablation_config = AblationConfig(
        num_seeds=3,  # Start with 3 seeds for quick testing, increase to 5-10 for final
        num_episodes=200,  # Shorter for ablation studies
        save_dir=f"ablation_studies_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Create and run ablation runner
    runner = AutomatedAblationRunner(ablation_config)
    
    try:
        results = runner.run_all_ablations()
        
        # Print summary
        print("\n" + "="*100)
        print("ABLATION STUDY SUMMARY")
        print("="*100)
        
        if runner.results_df is not None:
            print("\nPerformance Summary:")
            print(runner.results_df.to_string(index=False))
        
        print("\n" + "="*100)
        print("STUDY COMPLETE")
        print("="*100)
        print(f"\nResults saved to: {ablation_config.save_dir}/")
        print("\nGenerated files:")
        print("  - figures/ (publication-quality plots)")
        print("  - tables/ (LaTeX and CSV tables)")
        print("  - logs/ (intermediate results)")
        print("  - models/ (trained models)")
        print("  - final_results.pkl (complete dataset)")
        
    except Exception as e:
        print(f"\n❌ Ablation study failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()