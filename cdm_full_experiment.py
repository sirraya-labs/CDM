# cdm_full_experiment.py - FIXED FOR WINDOWS
import gymnasium as gym  # Changed from "import gym"
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import sys
import os

# Try to import pybullet for Windows compatibility
try:
    import pybullet_envs
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("Note: PyBullet not installed. Using standard gym environments if available.")

# ============================
# 1. CDM COMPONENTS (NO CHANGES NEEDED HERE)
# ============================

class CDM_Metric(nn.Module):
    """Learned Riemannian Metric via Cholesky Factor L"""
    def __init__(self, dim, hidden_dim=128):
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
        M = torch.bmm(L, L.transpose(1,2)) + 1e-4 * self.eye.to(x.device)
        return M

class DynamicsModel(nn.Module):
    """Learn s_{t+1} = f(s, a) to provide Jacobian"""
    def __init__(self, s_dim, a_dim, hidden_dim=128):
        super().__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, s_dim)
        )
        
    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.net(sa)

class GaussianPolicy(nn.Module):
    """Gaussian policy for continuous control"""
    def __init__(self, s_dim, a_dim, hidden_dim=128):
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, a_dim),
            nn.Tanh()  # Keep actions in [-1, 1] range
        )
        self.log_std = nn.Parameter(torch.zeros(1, a_dim) * 0.1)
        
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
# 2. CONTRACTION LOSS (NO CHANGES)
# ============================

def compute_jacobian(model, s, a):
    """Compute Jacobian df/ds efficiently"""
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
        jacobian.append(s.grad.clone())
    
    jacobian = torch.stack(jacobian, dim=1)
    return jacobian, s_next

def get_contraction_loss(s, a, model, metric, contraction_rate=0.1, lambda_reg=1e-4):
    """Compute contraction loss with regularization"""
    A, s_next = compute_jacobian(model, s, a)
    M_curr = metric(s)
    M_next = metric(s_next)
    
    M_next_detached = M_next.detach()
    stability_term = torch.bmm(torch.bmm(A.transpose(1, 2), M_next_detached), A)
    contraction_cond = stability_term - (1 - contraction_rate) * M_curr
    
    pos_cond = torch.relu(contraction_cond)
    loss_frob = torch.norm(pos_cond, p='fro', dim=(1, 2)).mean()
    
    M_norm = torch.norm(M_curr, p='fro', dim=(1, 2)).mean()
    
    return loss_frob + lambda_reg * M_norm, loss_frob.item()

# ============================
# 3. REPLAY BUFFER (NO CHANGES)
# ============================

class ReplayBuffer:
    def __init__(self, capacity, s_dim, a_dim):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.s_dim = s_dim
        self.a_dim = a_dim
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)

# ============================
# 4. TRAINING LOOP (NO CHANGES NEEDED)
# ============================

def train_cdm(env_name="HalfCheetah-v4", episodes=200, max_steps=1000, 
              batch_size=256, buffer_size=100000, save_freq=50):
    
    # Make environment
    env = gym.make(env_name, render_mode=None)  # Added render_mode for gymnasium
    
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    
    print(f"Environment: {env_name}")
    print(f"State dimension: {s_dim}, Action dimension: {a_dim}")
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    policy = GaussianPolicy(s_dim, a_dim).to(device)
    model = DynamicsModel(s_dim, a_dim).to(device)
    metric = CDM_Metric(s_dim).to(device)
    
    # Separate optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    model_optimizer = optim.Adam(model.parameters(), lr=3e-4)
    metric_optimizer = optim.Adam(metric.parameters(), lr=3e-4)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(buffer_size, s_dim, a_dim)
    
    # Training history
    rewards_history = []
    contraction_history = []
    avg_rewards = deque(maxlen=10)
    
    # Scale action perturbation
    action_noise_std = 0.5  # Reduced from 1.0 for stability
    noise_decay = 0.995
    
    for ep in range(episodes):
        state, _ = env.reset()  # gymnasium returns (state, info)
        ep_reward = 0
        ep_contr_loss = 0
        step_count = 0
        
        for t in range(max_steps):
            # Prepare state tensor
            s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Get action from policy
            with torch.no_grad():
                action_tensor, _, mu, std = policy.sample(s_tensor)
                action = action_tensor.cpu().numpy()[0]
            
            # Add exploration noise with decay
            if t % 50 == 0:
                noise = np.random.normal(0, action_noise_std, size=a_dim)
                action = action + noise
            
            # Clip action
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Train on batch from replay buffer
            if len(replay_buffer) >= batch_size:
                # Sample batch
                s_batch, a_batch, r_batch, s_next_batch, d_batch = replay_buffer.sample(batch_size)
                s_batch = s_batch.to(device)
                a_batch = a_batch.to(device)
                s_next_batch = s_next_batch.to(device)
                
                # Update dynamics model
                model_optimizer.zero_grad()
                s_next_pred = model(s_batch, a_batch)
                dyn_loss = nn.MSELoss()(s_next_pred, s_next_batch)
                dyn_loss.backward()
                model_optimizer.step()
                
                # Update metric with contraction loss
                metric_optimizer.zero_grad()
                contr_loss, contr_val = get_contraction_loss(s_batch, a_batch, model, metric)
                contr_loss.backward()
                metric_optimizer.step()
                
                # Update policy
                policy_optimizer.zero_grad()
                action_pred, log_prob, _, _ = policy.sample(s_batch)
                with torch.no_grad():
                    advantage = r_batch.to(device)
                policy_loss = -(log_prob * advantage).mean()
                policy_loss.backward()
                policy_optimizer.step()
                
                ep_contr_loss += contr_val
            
            state = next_state
            ep_reward += reward
            step_count += 1
            
            if done:
                break
        
        # Decay exploration noise
        action_noise_std *= noise_decay
        
        # Record metrics
        rewards_history.append(ep_reward)
        avg_rewards.append(ep_reward)
        if step_count > 0:
            contraction_history.append(ep_contr_loss / step_count)
        else:
            contraction_history.append(0)
        
        # Print progress
        if ep % 10 == 0:
            avg_reward = np.mean(avg_rewards) if avg_rewards else 0
            print(f"Episode {ep:4d} | Reward: {ep_reward:7.1f} | "
                  f"Avg Reward: {avg_reward:7.1f} | "
                  f"Contraction: {contraction_history[-1]:.4f} | "
                  f"Noise STD: {action_noise_std:.3f}")
        
        # Save models periodically
        if ep % save_freq == 0 and ep > 0:
            torch.save(policy.state_dict(), f"cdm_policy_ep{ep}.pth")
            torch.save(model.state_dict(), f"cdm_dynamics_ep{ep}.pth")
            torch.save(metric.state_dict(), f"cdm_metric_ep{ep}.pth")
    
    # Save final models
    torch.save(policy.state_dict(), "cdm_policy_final.pth")
    torch.save(model.state_dict(), "cdm_dynamics_final.pth")
    torch.save(metric.state_dict(), "cdm_metric_final.pth")
    
    env.close()
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history, alpha=0.6)
    plt.plot(np.convolve(rewards_history, np.ones(10)/10, mode='valid'), 'r-', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(contraction_history)
    plt.xlabel("Episode")
    plt.ylabel("Contraction Loss")
    plt.title("CDM Contraction Metric")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    hist_100 = np.array(rewards_history[-100:]) if len(rewards_history) >= 100 else np.array(rewards_history)
    plt.boxplot(hist_100)
    plt.title(f"Last {len(hist_100)} Episodes Reward Distribution")
    plt.ylabel("Reward")
    
    plt.tight_layout()
    plt.savefig("cdm_training_results.png", dpi=100)
    plt.show()
    
    print("\nTraining completed!")
    print(f"Final average reward: {np.mean(rewards_history[-10:]):.1f}")
    print("Models saved as: cdm_policy_final.pth, cdm_dynamics_final.pth, cdm_metric_final.pth")
    
    return rewards_history, contraction_history

# ============================
# 5. TESTING FUNCTION (UPDATED)
# ============================

def test_policy(env_name="HalfCheetah-v4", model_path="cdm_policy_final.pth", episodes=5):
    """Test the trained policy"""
    env = gym.make(env_name, render_mode="human")  # Show visualization
    
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    
    policy = GaussianPolicy(s_dim, a_dim)
    policy.load_state_dict(torch.load(model_path, map_location='cpu'))
    policy.eval()
    
    print(f"\nTesting policy: {model_path}")
    
    test_rewards = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            s_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = policy.sample(s_tensor)
            
            next_state, reward, terminated, truncated, _ = env.step(action.numpy()[0])
            done = terminated or truncated
            
            state = next_state
            ep_reward += reward
        
        test_rewards.append(ep_reward)
        print(f"Test Episode {ep}: Reward = {ep_reward:.1f}")
    
    print(f"\nAverage test reward: {np.mean(test_rewards):.1f} ± {np.std(test_rewards):.1f}")
    env.close()
    
    return test_rewards

# ============================
# 6. RUN - UPDATED FOR WINDOWS
# ============================

if __name__ == "__main__":
    print("=" * 60)
    print("CDM Framework - Contraction Dynamics Model")
    print("=" * 60)
    
    # Choose environment based on availability
    if PYBULLET_AVAILABLE:
        print("PyBullet detected. Using PyBullet HalfCheetah environment.")
        env_name = "HalfCheetahBulletEnv-v0"
    else:
        print("PyBullet not available. Trying standard gym environments.")
        try:
            # Test if standard HalfCheetah is available
            import gymnasium as gym
            env = gym.make("HalfCheetah-v4", render_mode=None)
            env.close()
            env_name = "HalfCheetah-v4"
            print("Standard HalfCheetah environment available.")
        except:
            print("No suitable environments found.")
            print("Please install PyBullet: pip install pybullet")
            sys.exit(1)
    
    # Training
    print("\n" + "=" * 60)
    print("Training CDM Agent")
    print("=" * 60)
    
    start_time = time.time()
    
    # Start with small training for testing
    rewards, contraction = train_cdm(
        env_name=env_name,
        episodes=50,  # Start small for testing
        max_steps=500,  # Shorter episodes for testing
        batch_size=64,  # Smaller batch size
        buffer_size=50000,
        save_freq=10
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    
    # Testing
    print("\n" + "=" * 60)
    print("Testing Trained Policy")
    print("=" * 60)
    
    test_rewards = test_policy(env_name=env_name, episodes=3)