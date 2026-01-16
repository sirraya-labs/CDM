# cdm_full_experiment_stable.py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import sys
import os

# ============================
# 1. CDM COMPONENTS - STABILIZED
# ============================

class CDM_Metric(nn.Module):
    """Learned Riemannian Metric via Cholesky Factor L - STABILIZED"""
    def __init__(self, dim, hidden_dim=64):  # Smaller network
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim * dim)
        )
        self.eye = torch.eye(dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        L = self.net(x).view(batch_size, self.dim, self.dim)
        # Add identity and small regularization
        M = torch.bmm(L, L.transpose(1,2)) + 1e-2 * self.eye.to(x.device)  # Increased regularization
        return M

class DynamicsModel(nn.Module):
    """Learn s_{t+1} = f(s, a) to provide Jacobian - STABILIZED"""
    def __init__(self, s_dim, a_dim, hidden_dim=64):  # Smaller network
        super().__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden_dim),
            nn.Tanh(),  # Changed from ReLU to Tanh for stability
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, s_dim)
        )
        
    def forward(self, s, a):
        # Normalize inputs for stability
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        sa = torch.cat([s, a], dim=-1)
        return self.net(sa)

class GaussianPolicy(nn.Module):
    """Gaussian policy for continuous control - STABILIZED"""
    def __init__(self, s_dim, a_dim, hidden_dim=64):  # Smaller network
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, a_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        # Initialize with smaller std
        self.log_std = nn.Parameter(torch.zeros(1, a_dim) * -1.0)  # Smaller initial std
        
    def forward(self, s):
        mu = self.mu_net(s)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std
    
    def sample(self, s):
        mu, std = self.forward(s)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, mu, std

# ============================
# 2. CONTRACTION LOSS - STABILIZED
# ============================

def compute_jacobian_stable(model, s, a):
    """Compute Jacobian df/ds with gradient clipping"""
    s.requires_grad_(True)
    s_next = model(s, a)
    
    s_dim = s_next.shape[1]
    jacobian = []
    
    for i in range(s_dim):
        grad_output = torch.zeros_like(s_next)
        grad_output[:, i] = 1.0
        
        if s.grad is not None:
            s.grad.zero_()
            
        s_next.backward(grad_output, retain_graph=True)
        # Clip gradients for stability
        jacobian.append(torch.clamp(s.grad.clone(), -10, 10))
    
    jacobian = torch.stack(jacobian, dim=1)
    return jacobian, s_next

def get_contraction_loss_stable(s, a, model, metric, contraction_rate=0.05, lambda_reg=0.01):  # Lower rate, higher reg
    """Compute contraction loss with better regularization"""
    A, s_next = compute_jacobian_stable(model, s, a)
    M_curr = metric(s)
    M_next = metric(s_next)
    
    # Use M_next with gradient (no detach) but clip values
    M_next_clipped = torch.clamp(M_next, 0.01, 100.0)  # Clip metric values
    
    stability_term = torch.bmm(torch.bmm(A.transpose(1, 2), M_next_clipped), A)
    contraction_cond = stability_term - (1 - contraction_rate) * M_curr
    
    # Use smoother activation (leaky relu instead of relu)
    pos_cond = torch.nn.functional.leaky_relu(contraction_cond, 0.01)
    
    # Use mean instead of norm for stability
    loss_frob = pos_cond.pow(2).mean()
    
    # Regularization: keep metric close to identity
    identity = torch.eye(M_curr.shape[1]).unsqueeze(0).to(M_curr.device)
    identity_reg = torch.norm(M_curr - identity, p='fro', dim=(1, 2)).mean()
    
    # Also regularize metric norm
    M_norm = torch.norm(M_curr, p='fro', dim=(1, 2)).mean()
    
    return loss_frob + lambda_reg * identity_reg + 0.001 * M_norm, loss_frob.item()

# ============================
# 3. REPLAY BUFFER (Optimized)
# ============================

class ReplayBuffer:
    def __init__(self, capacity, s_dim, a_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, s_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, a_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, s_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            torch.FloatTensor(self.states[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]).unsqueeze(1),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices]).unsqueeze(1)
        )
    
    def __len__(self):
        return self.size

# ============================
# 4. TRAINING LOOP - IMPROVED
# ============================

def train_cdm_stable(env_name="Pendulum-v1", episodes=100, max_steps=200, 
                     batch_size=64, buffer_size=50000, save_freq=20):
    
    print(f"\nCreating environment: {env_name}")
    
    env = gym.make(env_name)
    
    # Get environment properties
    s_dim = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Box):
        a_dim = env.action_space.shape[0]
        print(f"✓ Environment has CONTINUOUS action space (dim={a_dim})")
    else:
        print(f"✗ This version requires continuous action space")
        return [], []
    
    print(f"✓ State dimension: {s_dim}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    # Initialize models
    policy = GaussianPolicy(s_dim, a_dim).to(device)
    model = DynamicsModel(s_dim, a_dim).to(device)
    metric = CDM_Metric(s_dim).to(device)
    
    # Optimizers with weight decay for regularization
    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    model_optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    metric_optimizer = optim.Adam(metric.parameters(), lr=1e-4, weight_decay=1e-3)  # Lower lr for metric
    
    # Replay buffer
    replay_buffer = ReplayBuffer(buffer_size, s_dim, a_dim)
    
    # Training history
    rewards_history = []
    contraction_history = []
    avg_rewards = deque(maxlen=10)
    
    # Better exploration strategy
    exploration_noise = 0.5
    noise_decay = 0.998
    
    print(f"\nStarting STABLE training for {episodes} episodes...")
    print("-" * 60)
    
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        ep_contr_loss = 0
        step_count = 0
        
        for t in range(max_steps):
            # State to tensor
            s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get action with exploration
            with torch.no_grad():
                action_tensor, _, _, _ = policy.sample(s_tensor)
                action = action_tensor.cpu().numpy()[0]
            
            # Add exploration noise (decaying over time)
            noise = np.random.normal(0, exploration_noise, size=a_dim)
            action = action + noise
            
            # Clip action
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Training step (more frequent updates)
            if len(replay_buffer) >= batch_size:
                for _ in range(2):  # Multiple updates per step
                    # Sample batch
                    s_batch, a_batch, r_batch, s_next_batch, d_batch = replay_buffer.sample(batch_size)
                    s_batch = s_batch.to(device)
                    a_batch = a_batch.to(device)
                    s_next_batch = s_next_batch.to(device)
                    
                    # Normalize rewards for stability
                    r_batch_normalized = (r_batch - r_batch.mean()) / (r_batch.std() + 1e-8)
                    
                    # Update dynamics model
                    model_optimizer.zero_grad()
                    s_next_pred = model(s_batch, a_batch)
                    dyn_loss = nn.MSELoss()(s_next_pred, s_next_batch)
                    dyn_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                    model_optimizer.step()
                    
                    # Update metric with contraction loss
                    metric_optimizer.zero_grad()
                    contr_loss, contr_val = get_contraction_loss_stable(s_batch, a_batch, model, metric)
                    contr_loss.backward()
                    torch.nn.utils.clip_grad_norm_(metric.parameters(), 0.5)  # Tighter clipping for metric
                    metric_optimizer.step()
                    
                    # Update policy
                    policy_optimizer.zero_grad()
                    action_pred, log_prob, _, _ = policy.sample(s_batch)
                    with torch.no_grad():
                        advantage = r_batch_normalized.to(device)
                    
                    # Policy loss with entropy regularization
                    policy_loss = -(log_prob * advantage).mean() - 0.01 * log_prob.mean()  # Entropy bonus
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    policy_optimizer.step()
                    
                    ep_contr_loss += contr_val
            
            state = next_state
            ep_reward += reward
            step_count += 1
            
            if done:
                break
        
        # Decay exploration noise
        exploration_noise *= noise_decay
        
        # Record metrics
        rewards_history.append(ep_reward)
        avg_rewards.append(ep_reward)
        if step_count > 0:
            contraction_history.append(ep_contr_loss / step_count)
        else:
            contraction_history.append(0)
        
        # Print progress
        if ep % 10 == 0 or ep == episodes - 1:
            avg_reward = np.mean(avg_rewards) if avg_rewards else 0
            print(f"Episode {ep:4d} | Reward: {ep_reward:7.1f} | "
                  f"Avg Reward: {avg_reward:7.1f} | "
                  f"Contraction: {contraction_history[-1]:.4f} | "
                  f"Noise: {exploration_noise:.3f}")
        
        # Save models
        if ep % save_freq == 0 and ep > 0:
            torch.save(policy.state_dict(), f"cdm_stable_policy_ep{ep}.pth")
            torch.save(model.state_dict(), f"cdm_stable_dynamics_ep{ep}.pth")
            torch.save(metric.state_dict(), f"cdm_stable_metric_ep{ep}.pth")
    
    # Final save
    torch.save(policy.state_dict(), "cdm_stable_policy_final.pth")
    torch.save(model.state_dict(), "cdm_stable_dynamics_final.pth")
    torch.save(metric.state_dict(), "cdm_stable_metric_final.pth")
    
    env.close()
    
    # Plot results
    if rewards_history:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(rewards_history, alpha=0.6, label='Episode Reward')
        if len(rewards_history) >= 10:
            moving_avg = np.convolve(rewards_history, np.ones(10)/10, mode='valid')
            plt.plot(range(9, len(rewards_history)), moving_avg, 'r-', linewidth=2, label='10-episode Avg')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"CDM Stable Training on {env_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(contraction_history)
        plt.yscale('log')  # Log scale for contraction loss
        plt.xlabel("Episode")
        plt.ylabel("Contraction Loss (log scale)")
        plt.title("Stable Contraction Learning")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        # Show last 20 rewards
        last_rewards = rewards_history[-20:] if len(rewards_history) >= 20 else rewards_history
        plt.bar(range(len(last_rewards)), last_rewards, alpha=0.6)
        plt.axhline(y=np.mean(last_rewards), color='r', linestyle='--', label=f'Avg: {np.mean(last_rewards):.1f}')
        plt.xlabel("Recent Episodes")
        plt.ylabel("Reward")
        plt.title("Recent Performance")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"cdm_stable_training_{env_name}.png", dpi=100)
        plt.show()
    
    print("\n" + "=" * 60)
    print("STABLE TRAINING COMPLETE!")
    print(f"Final average reward: {np.mean(rewards_history[-10:]):.1f}")
    print(f"Best reward: {np.max(rewards_history):.1f}")
    print("=" * 60)
    
    return rewards_history, contraction_history

# ============================
# 5. TESTING FUNCTION
# ============================

def test_policy_stable(env_name="Pendulum-v1", model_path="cdm_stable_policy_final.pth", episodes=5):
    """Test the trained policy"""
    print(f"\nTesting stable policy on {env_name}...")
    
    env = gym.make(env_name, render_mode="human")
    
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    
    policy = GaussianPolicy(s_dim, a_dim)
    
    # Safe loading
    policy.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    print(f"✓ Policy loaded from {model_path}")
    
    policy.eval()
    
    test_rewards = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            s_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_tensor, _, _, _ = policy.sample(s_tensor)
                action = action_tensor.numpy()[0]
                action = np.clip(action, env.action_space.low, env.action_space.high)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            ep_reward += reward
        
        test_rewards.append(ep_reward)
        print(f"  Episode {ep}: Reward = {ep_reward:.1f}")
    
    print(f"\n✓ Average test reward: {np.mean(test_rewards):.1f} ± {np.std(test_rewards):.1f}")
    print(f"✓ Best test reward: {np.max(test_rewards):.1f}")
    
    env.close()
    return test_rewards

# ============================
# 6. MAIN ENTRY POINT
# ============================

if __name__ == "__main__":
    print("=" * 60)
    print("CDM STABLE FRAMEWORK - Contraction Dynamics Model")
    print("=" * 60)
    
    # Use Pendulum-v1 (always works and has continuous actions)
    env_name = "Pendulum-v1"
    
    # Note about Pendulum reward
    print("\nNote: Pendulum-v1 has negative rewards.")
    print("Goal: Get close to 0 (best possible is around -200 to -100)")
    print("Initial random policy gives around -1200 to -1000")
    print("-" * 60)
    
    # Training
    print("\n" + "=" * 60)
    print("Starting STABLE CDM Training")
    print("=" * 60)
    
    start_time = time.time()
    
    # Train with stable version
    rewards, contraction = train_cdm_stable(
        env_name=env_name,
        episodes=150,      # More episodes for learning
        max_steps=200,
        batch_size=64,
        buffer_size=50000,
        save_freq=25
    )
    
    training_time = time.time() - start_time
    print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
    
    # Test
    print("\n" + "=" * 60)
    print("Testing Stable Trained Policy")
    print("=" * 60)
    
    if os.path.exists("cdm_stable_policy_final.pth"):
        test_rewards = test_policy_stable(
            env_name=env_name,
            model_path="cdm_stable_policy_final.pth",
            episodes=3
        )
    
    print("\n" + "=" * 60)
    print("STABLE CDM DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("\nImprovements in this version:")
    print("1. Gradient clipping for stability")
    print("2. Better regularization for metric learning")
    print("3. Lower learning rate for metric network")
    print("4. Tanh activations instead of ReLU")
    print("5. Reward normalization")
    print("6. Entropy regularization for exploration")
    print("\nExpected results:")
    print("- Contraction loss should stabilize (not explode)")
    print("- Rewards should improve from -1200 to -500 or better")
    print("- Policy should show stable pendulum control")