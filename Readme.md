# Contraction-Regularized Model-Based RL (CR-MBRL)

**A Contraction-Inspired Stability Framework for Model-Based Reinforcement Learning**  
*Empirically Evaluating the Impact of Learned Riemannian Metrics on Policy Robustness*

[Python 3.8+](https://www.python.org/downloads/) | [PyTorch 2.0+](https://pytorch.org/) | [License: MIT](https://opensource.org/licenses/MIT)

---

## Overview

CR-MBRL is a research prototype that explores whether contraction-inspired regularization can improve the robustness and sample efficiency of model-based reinforcement learning. The framework learns a state-dependent Riemannian metric alongside a dynamics model and control policy, using contraction-like energy shaping as a stability-inducing regularizer.

**Primary Research Question:** Does adding contraction-inspired regularization to standard MBRL improve empirical robustness to perturbations and reduce policy variance?

Developed by **Amir Hameed** at **Sirraya Labs**, this is an active research project at the hypothesis-testing phase.

---

## Research Disclaimer

This implementation provides **contraction-inspired regularization**, not mathematically proven stability guarantees. The contraction metric serves as a soft constraint that empirically encourages stable behavior, but the system does not formally certify contraction.

**What this code implements:**
- Dynamics model using an ensemble of neural networks with uncertainty quantification
- State-dependent Riemannian metric that penalizes energy growth along trajectories
- Metric-based energy regularization in the policy optimization objective
- Adaptive balancing of task reward with stability regularization
- Comprehensive logging and visualization of training dynamics

**What this code does not implement:**
- Formal stability proofs or certificates
- Mathematical contraction guarantees
- Comparisons against established RL baselines (SAC, TD3, MBPO, etc.)
- Demonstrated superiority over existing methods

---

## Implemented Features

### Core Components

**Ensemble Dynamics Model:** 7-member ensemble with learned weights and uncertainty estimation for robust next-state prediction. Each member uses residual connections and layer normalization for training stability.

**Riemannian Metric Network:** Learns state-dependent positive-definite metric M(x) = L(x)L(x)^T + epsilon*I using Cholesky decomposition for guaranteed positive-definiteness. Supports both standard and attention-based architectures.

**Contraction-Inspired Energy Regularization:** Penalizes energy growth E(x_{t+1}) - alpha^2 * E(x_t) using softplus loss, where energy is computed as E(x) = x^T M(x) x.

**Stochastic Policy with Adaptive Exploration:** Gaussian policy with learnable standard deviation and multiple exploration strategies (OU noise, Gaussian, parameter noise). Strategy selection adapts based on historical success rates.

**Prioritized Experience Replay:** TD-error based sampling with importance sampling weights. New experiences receive maximum priority to ensure recent data utilization.

### Enhanced Features

**Curriculum Learning:** Progressive training stages that automatically transition from stability-focused to performance-focused optimization. Three stages with smooth interpolation: stability focus, performance focus, and fine-tuning.

**Meta-Learning Controller:** Adaptive contraction rate based on energy trend analysis. Increases contraction when energy rises (potential instability), relaxes when energy decreases.

**Attention-Based Metric Network:** Self-attention mechanism over state dimensions for learning which state components are most important for stability. Includes learnable temperature parameter for attention scaling.

**Geodesic Regularization:** Smoothness constraint that penalizes rapid changes in the metric along random directions in state space. Encourages the metric to vary smoothly, preventing discontinuous stability boundaries.

**Safety Margin Monitoring:** Distance-to-boundary estimation using the learned metric geometry. Computes safety margin as distance in the metric space to unsafe regions, though this is purely observational without enforcement.

**Metric Conditioning:** Eigendecomposition-based conditioning to bound the condition number of the metric matrix, preventing numerical instability during Cholesky decomposition.

### Training Enhancements

**Double Q-Learning:** Twin critic networks with minimum Q-value selection for conservative value estimates and reduced overestimation bias.

**Conservative Q-Learning Penalty:** Adds penalty term to Q-value updates to prevent overestimation on out-of-distribution actions.

**Gradient Clipping:** Per-component gradient norm clipping to prevent exploding gradients, especially important for the coupled multi-objective optimization.

**Learning Rate Scheduling:** Cosine annealing schedules for dynamics and policy optimizers.

**Adaptive Beta Scheduling:** Stability weight beta automatically adjusts based on performance improvement, decreasing when rewards improve and increasing when they degrade.

---

## Experimental Results

### Pendulum-v1 (200 episodes)

Results from a complete training run with all enhanced features enabled. Note that results vary across random seeds.

**Key Observations:**

| Metric | Behavior | Interpretation |
|--------|----------|----------------|
| Episode Reward | Generally improving | Task learning progresses |
| Evaluation Reward | More stable than training | Deterministic policy reduces variance |
| Contraction Energy | Decreasing trend | Energy regularization has measurable effect |
| Beta Parameter | Adaptive changes | System responds to performance feedback |
| Metric Condition Number | Bounded | Numerically stable metric learning |
| Dynamics Loss | Decreasing | Model accuracy improves |
| Critic Loss | May fluctuate | Coupled optimization creates tension |

**Generated Visualizations:**

The system automatically produces:
1. Training rewards and evaluation performance over episodes
2. Curriculum stage progression during training
3. Beta and alpha parameter adaptation over time
4. Safety margin estimates per episode
5. All loss components (dynamics, metric, critic, actor)
6. Energy levels and geodesic regularization losses
7. Exploration rate schedule
8. Reward distribution histogram
9. Moving average performance
10. Success rate over training
11. Stability-performance tradeoff scatter plot
12. Contraction-performance tradeoff scatter plot
13. Gradient norms for all components
14. Training summary statistics

---

## Quick Start

### Installation

```bash
git clone https://github.com/sirraya-labs/CDM.git
cd CDM

python -m venv venv
source venv/bin/activate

pip install torch numpy gymnasium matplotlib
```

### Basic Usage

```python
from main import Config, EnhancedContractionDynamicsAgent
import gymnasium as gym

config = Config()
config.ENV_NAME = "Pendulum-v1"
config.TOTAL_EPISODES = 200

agent = EnhancedContractionDynamicsAgent(config)

train_env = gym.make(config.ENV_NAME)
eval_env = gym.make(config.ENV_NAME)

results = agent.train(train_env, eval_env)
```

### Feature Configuration

```python
config = Config()

# Toggle enhanced features
config.USE_CURRICULUM = True
config.USE_META_LEARNING = True
config.USE_ATTENTION_METRIC = True
config.USE_SAFETY_CONSTRAINTS = True
config.USE_GEODESIC_REGULARIZATION = True

# Adjust contraction parameters
config.CONTRACTION_RATE_ALPHA = 0.85
config.INITIAL_BETA = 0.3
config.EPSILON_METRIC = 0.05
```

### Output Structure

After training completes, results are saved to `cdm_enhanced_results/`:

```
cdm_enhanced_results/
    config.json                 # Full configuration
    training_metrics.json       # All tracked metrics (JSON)
    training_metrics.pkl        # All tracked metrics (pickle)
    best_policy.pth             # Best evaluation performance
    best_critic.pth
    best_dynamics.pth
    best_metric.pth
    best_target_policy.pth
    best_target_critic.pth
    final_policy.pth            # Final trained models
    final_critic.pth
    final_dynamics.pth
    final_metric.pth
    final_target_policy.pth
    final_target_critic.pth
    training_results.png        # Comprehensive visualization
```

---

## Architecture Details

### Dynamics Model

Each ensemble member is a residual network with two residual blocks:

```
Input: [state, action]
  -> Linear + LayerNorm + ReLU + Dropout
  -> Residual Block 1 (Linear + LayerNorm + ReLU + Dropout + Linear)
  -> Residual Block 2 (Linear + LayerNorm + ReLU + Dropout + Linear)
  -> Linear output: next_state
```

The ensemble aggregates predictions using learned softmax weights and computes both mean prediction and epistemic uncertainty (variance across ensemble members).

### Metric Network

Two variants available:

**Standard:** Direct feedforward network that outputs lower-triangular Cholesky factors L(x). The metric is constructed as M = LL^T + epsilon*I to guarantee positive-definiteness.

**Attention-Based:** Applies multi-head self-attention over state dimensions before the feedforward network, allowing the metric to focus on the most relevant state components for stability.

Diagonal entries use softplus activation with a minimum offset (0.01). Off-diagonal entries use tanh activation scaled to [-0.1, 0.1] for numerical stability.

### Policy Network

Gaussian policy with learned mean and standard deviation:
- Mean network: Two hidden layers with LayerNorm, ReLU, and Dropout
- Standard deviation: Learned log_std parameter initialized to -1.0
- Action scaling: Output multiplied by action_scale (2.0) and passed through tanh

### Critic Network

Double Q-network with twin critics:
- Each critic: Two hidden layers with LayerNorm, ReLU, and Dropout
- Minimum of two Q-values used for conservative updates
- Separate target networks with soft updates (tau=0.005)

---

## Training Process

### Episode Execution

1. Reset environment and exploration noise processes
2. Get curriculum parameters for current training stage
3. For each step in episode:
   - Select action using policy with exploration noise (scaled by curriculum)
   - Execute action in environment
   - Optionally compute safety margin of current state
   - Store transition in replay buffer
4. Update state normalization statistics
5. Update exploration strategy based on episode reward
6. Update meta-learning controller with reward and energy data
7. Perform multiple training steps if buffer has sufficient data

### Training Step

Each training step on a sampled batch:

1. **Update Dynamics:** Minimize weighted MSE between predicted and actual next states, with uncertainty regularization
2. **Update Metric:** Minimize contraction-inspired energy loss with geodesic regularization and consistency across perturbed trajectories
3. **Update Critic:** Double Q-learning with conservative penalty and TD-error based prioritized replay updates
4. **Update Policy:** Maximize Q-values with contraction bonus, entropy bonus, safety penalty, and stability penalties

### Beta Adaptation

After each episode:
- If reward improved: beta *= 0.995 (reduce stability focus)
- If reward degraded: beta *= 1.02 (increase stability focus)
- Beta clipped to [BETA_MIN, BETA_MAX]

Curriculum overrides manual beta adaptation, providing stage-appropriate values with smooth interpolation between stages.

---

## Research Methodology

### Current Status

This project is in the **hypothesis testing phase**. The implementation demonstrates that contraction-inspired regularization can be integrated into MBRL, but has not yet established:

1. Whether the regularization provides measurable benefits over standard MBRL
2. Whether the learned metrics capture meaningful geometry
3. Whether the approach scales to more complex environments
4. How it compares to established methods

### Planned Validation

**Ablation Studies:** Systematically remove each component (contraction loss, metric learning, ensemble, curriculum) to isolate effects.

**Baseline Comparisons:** Implement and compare against SAC, TD3, MBPO, PETS, Dreamer on identical sample budgets.

**Robustness Testing:** Evaluate perturbation recovery time, performance under observation noise, sensitivity to parameter variation.

**Scale Testing:** Extend from Pendulum to MuJoCo environments (HalfCheetah, Hopper, Walker2d, Ant).

**Statistical Validation:** Run multiple seeds (minimum 5) with confidence intervals and significance tests.

### Suggested Experiments

**Contraction Effect Isolation:**
```python
configs = [
    Config(INITIAL_BETA=0.3),  # With contraction
    Config(INITIAL_BETA=0.0),  # Without contraction
]
```

**Feature Ablation:**
```python
configs = [
    Config(USE_CURRICULUM=True, USE_META_LEARNING=False),
    Config(USE_CURRICULUM=False, USE_META_LEARNING=True),
    Config(USE_CURRICULUM=False, USE_META_LEARNING=False),
]
```

**Architecture Comparison:**
```python
configs = [
    Config(USE_ATTENTION_METRIC=True),   # Attention-based metric
    Config(USE_ATTENTION_METRIC=False),  # Standard metric
]
```

**Robustness Evaluation:**
```python
def evaluate_robustness(agent, env, perturbation_scale=0.1, num_episodes=10):
    """Measure recovery after state perturbation."""
    results = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        state += np.random.randn(*state.shape) * perturbation_scale
        episode_reward = 0
        for step in range(env.spec.max_episode_steps):
            action = agent.select_action(state, deterministic=True, use_exploration=False)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        results.append(episode_reward)
    return np.mean(results), np.std(results)
```

---

## Configuration Reference

### Environment Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| ENV_NAME | "Pendulum-v1" | Gymnasium environment |
| STATE_DIM | 3 | State space dimension |
| ACTION_DIM | 1 | Action space dimension |
| MAX_EPISODE_LENGTH | 200 | Maximum steps per episode |

### Network Architecture
| Parameter | Default | Description |
|-----------|---------|-------------|
| DYNAMICS_HIDDEN_DIM | 128 | Dynamics model hidden size |
| POLICY_HIDDEN_DIM | 256 | Policy network hidden size |
| METRIC_HIDDEN_DIM | 128 | Metric network hidden size |
| CRITIC_HIDDEN_DIM | 256 | Critic network hidden size |
| ENSEMBLE_SIZE | 7 | Number of dynamics models |
| ATTENTION_HEADS | 3 | Attention heads for metric |

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| TOTAL_EPISODES | 200 | Training episodes |
| BATCH_SIZE | 256 | Samples per update |
| GAMMA | 0.99 | Discount factor |
| TAU | 0.005 | Target network update rate |
| LEARNING_START | 1000 | Steps before training begins |
| GRADIENT_STEPS | 40 | Updates per episode |

### Contraction Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| CONTRACTION_RATE_ALPHA | 0.85 | Target contraction rate |
| CONTRACTION_RATE_MIN | 0.7 | Minimum alpha |
| CONTRACTION_RATE_MAX | 0.95 | Maximum alpha |
| INITIAL_BETA | 0.3 | Initial stability weight |
| BETA_MIN | 0.05 | Minimum beta |
| BETA_MAX | 2.0 | Maximum beta |
| EPSILON_METRIC | 0.05 | Metric regularization |
| TARGET_CONDITION_NUMBER | 100.0 | Max condition number |

### Learning Rates
| Parameter | Default | Description |
|-----------|---------|-------------|
| ACTOR_LR | 3e-4 | Policy learning rate |
| CRITIC_LR | 3e-4 | Critic learning rate |
| DYNAMICS_LR | 1e-3 | Dynamics learning rate |
| METRIC_LR | 5e-5 | Metric learning rate |

### Feature Toggles
| Parameter | Default | Description |
|-----------|---------|-------------|
| USE_CURRICULUM | True | Progressive training stages |
| USE_META_LEARNING | True | Adaptive contraction rate |
| USE_ATTENTION_METRIC | True | Attention-based metric |
| USE_SAFETY_CONSTRAINTS | True | Safety margin monitoring |
| USE_GEODESIC_REGULARIZATION | True | Metric smoothness constraint |

---

## Troubleshooting

**Numerical instability in metric:**
Increase EPSILON_METRIC (try 0.1) or METRIC_REGULARIZATION (try 0.01). Check that condition numbers remain bounded.

**Policy converges to boundary actions:**
Reduce ACTOR_LR or increase entropy bonus weight. Check if contraction penalty is dominating the value loss.

**Dynamics model overfits:**
Increase dropout rate, reduce DYNAMICS_HIDDEN_DIM, or add weight decay to dynamics optimizer.

**Critic loss diverges:**
Reduce CRITIC_LR, increase batch size, or check for exploding Q-values. Conservative penalty may need adjustment.

**Memory usage too high:**
Reduce ENSEMBLE_SIZE, REPLAY_BUFFER_SIZE, or hidden dimensions. Consider using gradient checkpointing.

**Training too slow:**
Reduce GRADIENT_STEPS, increase UPDATE_EVERY, or use a smaller ensemble. Profile to identify bottleneck.

---

## Code Structure

```
main.py
    Config                              # All configuration parameters
    CurriculumStage                     # Stage definition for curriculum
    
    EnhancedDynamics                    # Single dynamics model
    DynamicsEnsemble                    # Ensemble with uncertainty
    
    AttentionBasedMetric                # Metric with self-attention
    RobustContractionMetric             # Standard metric network
    
    EnhancedPolicyNetwork               # Gaussian policy
    EnhancedValueNetwork                # Double Q-network
    
    PrioritizedReplayBuffer             # PER implementation
    AdaptiveExploration                 # Multi-strategy noise
    
    EnhancedRiemannianOperations        # Metric computations
        compute_energy                  # E = x^T M x
        condition_metric                # Bounded condition number
        compute_geodesic_regularization # Metric smoothness
        compute_contraction_loss        # Energy-based loss
        generate_displacements          # Virtual perturbations
        compute_safety_margin           # Distance to boundary
    
    CurriculumScheduler                 # Stage-based progression
    MetaLearningController              # Adaptive alpha tuning
    
    EnhancedContractionDynamicsAgent    # Main agent class
        _initialize_networks            # Build all models
        _initialize_optimizers          # Setup optimizers
        select_action                   # Action selection
        update_dynamics                 # Train dynamics model
        update_metric                   # Train metric network
        update_critic                   # Train Q-networks
        update_policy                   # Train policy
        train_step                      # Single update cycle
        train_episode                   # One episode execution
        train                           # Main training loop
        evaluate                        # Deterministic evaluation
        save_models / load_models       # Checkpointing
        plot_training_results           # Visualization
```

---

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Gymnasium
- Matplotlib
- Standard library: collections, random, pickle, json, os, pathlib, time, dataclasses, typing

---

## License

MIT License. See LICENSE file for details.

---



## Contact

**Author:** Amir Hameed  
**Organization:** Sirraya Labs  
**Email:** amir@sirraya.org

For questions or collaborations, open a GitHub issue or contact directly.

---

*This is a research prototype at the hypothesis-testing phase. Claims of effectiveness are pending rigorous empirical validation and baseline comparisons. Contributions, critiques, and collaborations are welcome.*
```
