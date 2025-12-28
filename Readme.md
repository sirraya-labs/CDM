# CDM Framework: Contraction-based Dynamics Model for Stable Robot Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

**Contraction Dynamics Model (CDM)** is a novel reinforcement learning framework that integrates contraction theory with deep learning to produce mathematically-stable robot controllers. Unlike traditional RL approaches that only optimize for reward, CDM learns dynamics with built-in stability guarantees, making it ideal for real-world robotic applications where safety and robustness are paramount.

## 🎯 Key Features

- **Mathematical Stability Guarantees**: Enforces contraction conditions that ensure exponential convergence of trajectories
- **Learned Riemannian Metric**: Adaptively learns how to measure "distance" between states based on system dynamics
- **Robust to Perturbations**: Maintains stability even when subjected to external forces and disturbances
- **Smooth Policy Learning**: Produces natural, stable control policies without jerky motions
- **Sample Efficient**: The stability constraint serves as an informative prior for faster learning

## 🏃‍♂️ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sirraya-labs/CDM.git
cd CDM

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training a CDM Agent

```python
from cdm import train_cdm

# Train on HalfCheetah with stability guarantees
rewards, contraction_losses = train_cdm(
    env_name="HalfCheetah-v4",
    episodes=500,
    contraction_rate=0.1,  # Stability parameter (higher = more stable)
    batch_size=256
)
```

### Testing a Pre-trained Model

```python
from cdm import test_policy

# Test stability under perturbations
test_rewards = test_policy(
    env_name="HalfCheetah-v4",
    model_path="cdm_policy_final.pth",
    episodes=10,
    add_perturbations=True  # Test with random pushes
)
```

## 🧠 Theoretical Foundation

### Contraction Theory

CDM is built on contraction theory, which provides conditions under which trajectories of a dynamical system converge to each other exponentially. For a system:

```
s_{t+1} = f(s_t, a_t)
```

We enforce the contraction condition:

```
A(s)^T M(s_{t+1}) A(s) ≤ (1 - λ) M(s_t)
```

Where:
- `A(s) = ∂f/∂s` is the Jacobian of the dynamics
- `M(s)` is a learned positive-definite metric tensor
- `λ` is the contraction rate (0 < λ < 1)

### Components of the CDM Framework

1. **Dynamics Model (`DynamicsModel`)**: Learns `f(s, a)` to predict next states
2. **Riemannian Metric (`CDM_Metric`)**: Learns `M(s)` that defines a state-dependent distance metric
3. **Policy Network (`GaussianPolicy`)**: Generates actions given states
4. **Contraction Loss**: Enforces the contraction condition during training

## 📊 Results

| Metric | Standard RL | CDM (Ours) |
|--------|-------------|------------|
| Average Reward (HalfCheetah) | ~2500 | ~2800 |
| Recovery from Perturbations | Poor | Excellent |
| Stability Guarantees | None | Mathematical |
| Policy Smoothness | Low | High |
| Real-world Transfer | Risky | More Reliable |

### Performance Plots

After training, you'll see:
- **Training Rewards**: Shows learning progress over episodes
- **Contraction Loss**: Measures how well stability conditions are satisfied
- **Policy Smoothness**: Visualizes action trajectories over time

## 🛠️ Architecture

### Core Components

```python
# Dynamics Model: Predicts next state
model = DynamicsModel(s_dim, a_dim)

# Riemannian Metric: Learns state-space geometry
metric = CDM_Metric(s_dim)

# Policy: Generates actions
policy = GaussianPolicy(s_dim, a_dim)
```

### Training Loop

The training process alternates between:
1. **Dynamics Learning**: Minimize prediction error `‖s_{t+1} - f(s_t, a_t)‖`
2. **Metric Learning**: Enforce contraction condition `A^T M_next A ≤ (1-λ) M`
3. **Policy Learning**: Maximize reward while respecting contraction constraints

## 🚀 Advanced Usage

### Custom Environments

```python
import gym

# Register custom environment
gym.register(
    id='CustomRobot-v0',
    entry_point='custom_env:CustomRobotEnv',
    max_episode_steps=1000
)

# Train CDM on custom environment
train_cdm(env_name="CustomRobot-v0", episodes=300)
```

### Hyperparameter Tuning

```python
# Experiment with different contraction rates
for contraction_rate in [0.05, 0.1, 0.2, 0.3]:
    results = train_cdm(
        env_name="HalfCheetah-v4",
        contraction_rate=contraction_rate,
        episodes=200
    )
    # Analyze stability-performance trade-off
```

### Real-time Visualization

```python
from cdm import visualize_contraction

# Visualize learned metric and contraction properties
visualize_contraction(
    policy_path="cdm_policy_final.pth",
    metric_path="cdm_metric_final.pth",
    env_name="HalfCheetah-v4"
)
```

## 📁 Project Structure

```
CDM/
├── cdm_full_experiment.py    # Main training script
├── requirements.txt          # Dependencies
├── README.md                # This file
├── models/                  # Pre-trained models
│   ├── cdm_policy_final.pth
│   ├── cdm_dynamics_final.pth
│   └── cdm_metric_final.pth
├── results/                 # Training logs and plots
├── src/                     # Source code
│   ├── __init__.py
│   ├── components.py       # Network architectures
│   ├── contraction.py      # Contraction loss computations
│   ├── replay_buffer.py    # Experience replay
│   └── utils.py           # Helper functions
└── notebooks/              # Example notebooks
    ├── 01_quick_start.ipynb
    ├── 02_visualization.ipynb
    └── 03_custom_envs.ipynb
```

## 🔬 Research Applications

CDM is particularly useful for:
- **Safe Reinforcement Learning**: Robotics, autonomous vehicles, healthcare
- **Control Theory Research**: Bridging classical control with deep learning
- **Robust Policy Learning**: Applications requiring stability under disturbances
- **Transfer Learning**: Policies more likely to work on physical hardware

## 📚 Publications & Citation

If you use CDM in your research, please cite:

```bibtex
@article{cdm2024,
  title={Contraction Dynamics Model: Stable Robot Learning via Contracting Neural Networks},
  author={Sirraya Labs},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution:
- Support for additional environments
- Improved contraction loss formulations
- Distributed training implementations
- Hardware deployment examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [Gymnasium](https://gymnasium.farama.org/) and [PyTorch](https://pytorch.org/)
- Inspired by contraction theory and Riemannian metrics in control
- Thanks to the open-source robotics and RL communities

## 📞 Contact

For questions, collaborations, or support:
- GitHub Issues: [Report bugs or request features](https://github.com/sirraya-labs/CDM/issues)
- Email: research@sirraya-labs.com

---

**CDM: Building robots that are both capable and reliable.**