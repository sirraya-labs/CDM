

# CR-MBRL: Contraction-Regularized Model-Based Reinforcement Learning

**Investigating whether contraction-inspired geometric regularization improves empirical robustness in model-based RL.**

[Python 3.8+](https://www.python.org/downloads/) | [PyTorch 2.0+](https://pytorch.org/) | [License: MIT](https://opensource.org/licenses/MIT)

---

## Problem

Model-based reinforcement learning can be sample-efficient but often produces policies that are brittle to perturbations. Standard MBRL optimizes for expected return without considering trajectory stability.

## Hypothesis

Adding contraction-inspired energy regularization to the MBRL objective may improve empirical robustness to state perturbations relative to an unregularized MBRL baseline.

**Secondary question:** Whether any observed robustness improvement comes at a performance cost.

## What This Is

A research prototype for testing one specific idea: can a learned Riemannian metric, used as a stability regularizer, measurably improve policy robustness? The emphasis is empirical evaluation rather than theoretical guarantees.

The metric \(M(x) = L(x)L(x)^T + \epsilon I\) is learned jointly with a dynamics model and policy. The regularization term is added to the policy objective, weighted by an adaptive coefficient.

## What This Is Not

- A provably stable controller. The contraction loss is a surrogate, not a formal certificate.
- A complete RL system ready for deployment.
- Benchmarked against existing methods.
- Validated across multiple environments or seeds.
- Evidence of causal contribution from individual components (ablation study pending).

## Evaluation Criteria

The hypothesis will be considered supported if:

1. Policies trained with contraction regularization show statistically significant improvement in perturbation recovery relative to the unregularized baseline across multiple random seeds using appropriate statistical testing.

2. Robustness gains do not come with substantial degradation in asymptotic return.

3. Observed effects persist across at least two additional continuous-control environments (HalfCheetah-v4, Hopper-v4).

The hypothesis will be considered not supported by the current experiments if robustness improvements disappear under controlled baselines, fail to generalize across environments, or come at prohibitive performance cost.

## Research Roadmap

| Phase | Goal | Success Condition |
|-------|------|-------------------|
| Phase 1 | Pendulum proof of concept | Robustness signal over baseline |
| Phase 2 | Multi-seed validation | Consistent effect across seeds |
| Phase 3 | MuJoCo generalization | Effect persists on harder tasks |
| Phase 4 | Ablation analysis | Isolate causal mechanism |

**Current status:** Phase 1. Single-seed results on Pendulum-v1. Baselines and multi-seed validation pending. The present objective is signal detection rather than performance optimization.

## Negative Results Policy

If controlled experiments fail to show robustness improvements over baseline methods, the repository will document those outcomes explicitly. The goal of this project is hypothesis evaluation rather than confirmation.

---

## Method

### Core Mechanism

Four learned components:

1. **Dynamics ensemble**: Multiple models with residual connections, providing mean prediction and epistemic uncertainty
2. **Policy network**: Gaussian policy with learned mean and variance
3. **Critic network**: Twin Q-networks with double Q-learning
4. **Metric network**: Maps states to positive-definite matrices via Cholesky decomposition

The policy is optimized using a reward objective augmented with contraction-inspired regularization. A simplified schematic objective is:

```
policy_loss = value_loss - beta * contraction_bonus + entropy_bonus
```

where `contraction_bonus` encourages policies that reduce energy along trajectories. The implementation includes additional regularization terms documented in the code.

### Metric Learning

The metric network outputs lower-triangular matrices \(L(x)\) with positive diagonal entries (enforced via softplus). The metric is constructed as:

\[
M(x) = L(x)L(x)^T + \epsilon I
\]

The regularizer penalizes positive energy growth between successive states using a softplus surrogate:

\[
\mathcal{L}_{\text{contraction}} = \text{softplus}\left(\beta \cdot \left(E(x_{t+1}) - \alpha^2 E(x_t)\right)\right)
\]

where \(E(x) = x^T M(x) x\) and \(\alpha \in (0,1)\) is the target contraction rate.

This objective encourages contraction-like behavior in learned trajectories but does not certify contraction or stability.

### Adaptive Regularization

The regularization weight `beta` is adjusted after each episode: decreased when reward improves, increased when it degrades. This allows the system to automatically trade off stability and performance based on training progress.

### Exploratory Mechanisms

The implementation includes several supporting components, included as exploratory mechanisms and evaluated through ablation to determine whether they contribute to observed effects:

- Curriculum schedule for staged training
- Adaptive contraction rate based on energy trends
- Geodesic regularization for metric smoothness
- Attention in metric network for state-dependent weighting
- Prioritized experience replay

---

## Experimental Design

### Robustness Definition

Robustness is operationalized as recovery performance following injected perturbations to the environment state during evaluation. Metrics include return degradation, recovery time to baseline performance, and variance across perturbation magnitudes.

### Primary Experiment

**Does contraction regularization improve robustness?**

Compare two conditions:
- `beta=0.3` (with regularization)
- `beta=0.0` (standard MBRL, no regularization)

Measure: reward recovery after random state perturbation at evaluation time, across multiple perturbation scales.

### Secondary Experiments

| Question | Method |
|----------|--------|
| Does the learned metric exhibit interpretable structure? | Visualize eigenvalue smoothness and anisotropy across state space |
| Does adaptive beta help? | Compare fixed vs. adaptive beta schedules |
| Does ensemble uncertainty matter? | Compare K=7 vs K=1 dynamics models |
| Do curriculum stages help? | Compare staged vs. fixed training |
| Does robustness trade off with reward? | Compare final evaluation return and perturbation recovery |

### Environment

Currently evaluated on Pendulum-v1 (3D state, 1D action). Extension to MuJoCo environments planned.

### Metrics Tracked

- Episode reward (training and evaluation)
- Contraction energy \(E(x) = x^T M(x) x\)
- Metric condition number
- Gradient norms per component

---

## Results

Results from a 200-episode run on Pendulum-v1 with all components enabled.

Note: single seed, no baselines. These are preliminary observations, not conclusions.

### Training Behavior

The system learns to solve the task. The adaptive beta parameter decreases over training as reward improves under the current scheduling rule.

### Metric Learning

The learned metric maintains positive-definiteness (enforced by construction) and bounded condition numbers. Energy levels along trajectories show a decreasing trend, consistent with the regularization objective.

### Loss Dynamics

The multi-component optimization shows coupled behavior: dynamics loss decreases steadily, metric loss fluctuates within bounds, critic loss may drift, and actor loss reflects the tension between value maximization and stability constraints.

### What These Results Do Not Show

- Statistical significance (single seed)
- Comparison to baselines (no SAC, TD3, or MBPO runs)
- Scaling behavior (single environment)
- Causality (no ablation results yet)

---

## Quick Start

### Installation

```bash
git clone https://github.com/sirraya-labs/CDM.git
cd CDM
pip install torch numpy gymnasium matplotlib
```

### Running

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

### Running Ablations

```python
# Without contraction regularization
config_no_reg = Config(INITIAL_BETA=0.0)

# Without metric learning
config_fixed = Config(USE_ATTENTION_METRIC=False)
# Then modify metric network to return identity

# Single dynamics model
config_single = Config(ENSEMBLE_SIZE=1)

# Without curriculum
config_no_curriculum = Config(USE_CURRICULUM=False)
```

### Output

Training produces:
```
cdm_enhanced_results/
    config.json
    training_metrics.json
    best_*.pth
    final_*.pth
    training_results.png
```

---

## Configuration

Key parameters for experiments:

| Parameter | Default | Role |
|-----------|---------|------|
| INITIAL_BETA | 0.3 | Contraction regularization weight |
| CONTRACTION_RATE_ALPHA | 0.85 | Target energy reduction rate |
| EPSILON_METRIC | 0.05 | Metric diagonal regularization |
| ENSEMBLE_SIZE | 7 | Number of dynamics models |
| TOTAL_EPISODES | 200 | Training duration |

Feature toggles for ablation:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| USE_CURRICULUM | True | Staged training |
| USE_META_LEARNING | True | Adaptive alpha |
| USE_ATTENTION_METRIC | True | Attention in metric |
| USE_GEODESIC_REGULARIZATION | True | Metric smoothness |
| USE_SAFETY_CONSTRAINTS | True | Safety margin tracking |

---

## Known Limitations

- Evaluated only on Pendulum-v1, a low-dimensional continuous control task
- No comparison against established baselines (SAC, TD3, MBPO, PETS, Dreamer)
- Single-seed results; statistical significance not established
- Contraction loss is a surrogate; does not guarantee mathematical contraction
- Sensitivity to hyperparameters (beta, alpha, learning rates) not characterized
- Computational overhead from ensemble and metric learning not benchmarked
- No ablation results isolating individual component contributions

---

## Next Steps

**Immediate:**
- Multi-seed evaluation (5+ seeds) for statistical significance
- Implement SAC and MBPO baselines on identical sample budgets
- Conduct perturbation recovery experiments at multiple scales

**Short-term:**
- Ablation study isolating each component
- Extension to HalfCheetah-v4, Hopper-v4
- Hyperparameter sensitivity analysis

**Medium-term:**
- Robustness benchmarks across environments
- Compute overhead analysis
- Baseline comparisons with statistical tests

**Long-term exploratory:**
- Investigate conditions under which learned metrics admit formal verification
- Comparison with Lyapunov-based stability methods

---

## Citation

```bibtex
@software{crmbrl2024,
  author = {Amir Hameed},
  title = {CR-MBRL: Contraction-Regularized Model-Based Reinforcement Learning},
  year = {2024},
  organization = {Sirraya Labs},
  url = {https://github.com/sirraya-labs/CDM},
  note = {Research prototype. Hypothesis testing phase.}
}
```

---

## Contact

Amir Hameed, Sirraya Labs  
amir@sirraya.org  
https://github.com/sirraya-labs/CDM

---

*This is a research prototype investigating a specific hypothesis about contraction-inspired regularization. Claims are pending rigorous empirical validation. Contributions, critiques, and collaborations are welcome.*
