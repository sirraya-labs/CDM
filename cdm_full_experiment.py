import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt

# Enable anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)

# ============================
# NETWORK ARCHITECTURES
# ============================

class FixedPolicy(nn.Module):
    """Actor network"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.action_scale = 2.0
    
    def forward(self, state):
        return self.net(state) * self.action_scale

class ValueNetwork(nn.Module):
    """Critic network"""
    def __init__(self, state_dim, hidden_dim=64):
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

class SimpleDynamics(nn.Module):
    """Dynamics model for contraction analysis"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class ContractionMetric(nn.Module):
    """Simplified Riemannian metric M(x)"""
    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.state_dim = state_dim
        
        # Output positive diagonal elements
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Get positive diagonal elements
        diag = self.net(x) + 0.1  # Ensure > 0
        
        # Create diagonal matrix
        M = torch.zeros(batch_size, self.state_dim, self.state_dim, device=x.device)
        for i in range(self.state_dim):
            M[:, i, i] = diag[:, i]
        
        return M

# ============================
# UTILITY FUNCTIONS
# ============================

def compute_energy(state, metric):
    """Compute energy E = x^T * M(x) * x"""
    M = metric(state)
    # x^T * M * x
    energy = torch.bmm(
        torch.bmm(state.unsqueeze(1), M),
        state.unsqueeze(-1)
    ).squeeze()
    return energy, M

def soft_update(target, source, tau):
    """Soft update target network"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# ============================
# REPLAY BUFFER
# ============================

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

# ============================
# ORNSTEIN-UHLENBECK NOISE
# ============================

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

# ============================
# FIXED TRAINING FUNCTION
# ============================

def train_cdm_with_contraction_fixed():
    """Fixed version without in-place operation errors"""
    
    env = gym.make("Pendulum-v1")
    state_dim = 3
    action_dim = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize networks
    actor = FixedPolicy(state_dim, action_dim).to(device)
    critic = ValueNetwork(state_dim).to(device)
    dynamics = SimpleDynamics(state_dim, action_dim).to(device)
    metric = ContractionMetric(state_dim).to(device)
    
    # Target networks
    target_actor = FixedPolicy(state_dim, action_dim).to(device)
    target_critic = ValueNetwork(state_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    
    # Optimizers
    actor_optim = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optim = optim.Adam(critic.parameters(), lr=1e-3)
    dynamics_optim = optim.Adam(dynamics.parameters(), lr=1e-3)
    metric_optim = optim.Adam(metric.parameters(), lr=1e-4)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=20000)
    
    # Training parameters
    gamma = 0.99
    tau = 0.005
    batch_size = 64  # Smaller batch for debugging
    lambda_contraction = 0.1
    
    # Noise for exploration
    ou_noise = OUNoise(action_dim)
    
    print("\nStarting CDM training...")
    print("-" * 70)
    print(f"{'Episode':>8} {'Reward':>10} {'Max|θ̇|':>8} {'Cosθ':>8}")
    print("-" * 70)
    
    rewards_history = []
    
    for episode in range(200):
        state, _ = env.reset()
        ou_noise.reset()
        
        # Decay exploration
        noise_scale = max(0.05, 0.3 * (1 - episode / 150))
        ou_noise.sigma = noise_scale * 0.2
        
        episode_reward = 0
        episode_max_velocity = 0
        steps = 0
        
        while True:
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_mean = actor(state_tensor).cpu().numpy()[0]
            
            # Add exploration noise
            noise = ou_noise.sample() * noise_scale
            action = action_mean + noise
            action = np.clip(action, -2.0, 2.0)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Update episode metrics
            episode_reward += reward
            episode_max_velocity = max(episode_max_velocity, abs(state[2]))
            steps += 1
            
            # Train if we have enough samples
            if len(replay_buffer) > batch_size:
                # Sample batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Convert to tensors (DETACHED for separate computation)
                states_t = torch.FloatTensor(states).to(device).detach().requires_grad_(False)
                actions_t = torch.FloatTensor(actions).to(device)
                rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states_t = torch.FloatTensor(next_states).to(device).detach().requires_grad_(False)
                dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                # ===== 1. TRAIN DYNAMICS MODEL (separate) =====
                dynamics_optim.zero_grad()
                # Create fresh tensor for dynamics training
                states_dyn = torch.FloatTensor(states).to(device).requires_grad_(True)
                next_states_pred = dynamics(states_dyn, actions_t)
                dynamics_loss = nn.MSELoss()(next_states_pred, next_states_t)
                dynamics_loss.backward()
                dynamics_optim.step()
                
                # ===== 2. TRAIN CRITIC (separate) =====
                critic_optim.zero_grad()
                
                with torch.no_grad():
                    next_actions = target_actor(next_states_t)
                    next_values = target_critic(next_states_t)
                    target_values = rewards_t + gamma * next_values * (1 - dones_t)
                
                # Fresh tensor for critic
                states_critic = torch.FloatTensor(states).to(device).requires_grad_(True)
                current_values = critic(states_critic)
                critic_loss = nn.MSELoss()(current_values, target_values)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optim.step()
                
                # ===== 3. TRAIN METRIC (separate) =====
                metric_optim.zero_grad()
                
                # Fresh tensor for metric training
                states_metric = torch.FloatTensor(states).to(device).requires_grad_(True)
                
                # Compute energy
                energy_curr, M = compute_energy(states_metric, metric)
                
                # Metric loss: encourage lower energy for upright states
                cos_theta = states_metric[:, 0]
                upright = cos_theta > 0.5
                
                metric_loss = torch.tensor(0.0, device=device)
                
                if upright.any():
                    # Upright states should have low energy (stable)
                    metric_loss += energy_curr[upright].mean() * 0.01
                
                # Regularization
                identity = torch.eye(state_dim, device=device).unsqueeze(0)
                reg_loss = torch.norm(M - identity, dim=(1, 2)).mean() * 0.001
                metric_loss += reg_loss
                
                metric_loss.backward()
                torch.nn.utils.clip_grad_norm_(metric.parameters(), 0.5)
                metric_optim.step()
                
                # ===== 4. TRAIN ACTOR WITH CONTRACTION (separate) =====
                actor_optim.zero_grad()
                
                # Fresh tensor for actor training
                states_actor = torch.FloatTensor(states).to(device).requires_grad_(True)
                
                # Get actor's actions
                actor_actions = actor(states_actor)
                
                # Get value estimates (using updated critic)
                actor_values = critic(states_actor)
                
                # Calculate contraction bonus using dynamics prediction
                with torch.no_grad():
                    # Predict next state
                    next_states_pred = dynamics(states_actor, actor_actions)
                    # Compute energies
                    energy_curr, _ = compute_energy(states_actor, metric)
                    energy_next, _ = compute_energy(next_states_pred, metric)
                    # Contraction: ΔE = E_current - E_next (positive means contracting)
                    delta_energy = energy_curr - energy_next
                    contraction_bonus = torch.tanh(delta_energy / 100.0).mean()
                
                # Velocity penalty
                velocity = states_actor[:, 2]
                velocity_penalty = torch.relu(torch.abs(velocity) - 3.0).mean() * 0.05
                
                # Action penalty for smoothness
                action_penalty = torch.abs(actor_actions).mean() * 0.001
                
                # Total actor loss
                actor_loss = -actor_values.mean() - lambda_contraction * contraction_bonus
                actor_loss += velocity_penalty + action_penalty
                
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optim.step()
                
                # ===== 5. UPDATE TARGET NETWORKS =====
                soft_update(target_actor, actor, tau)
                soft_update(target_critic, critic, tau)
            
            state = next_state
            
            if done or steps >= 200:
                break
        
        # Store episode metrics
        rewards_history.append(episode_reward)
        final_cos_theta = state[0]
        
        # Log progress
        if episode % 10 == 0 or episode < 20 or episode == 199:
            print(f"{episode:8d} {episode_reward:10.1f} {episode_max_velocity:8.2f} "
                  f"{final_cos_theta:8.2f}")
            
            # Success indicators
            if episode_reward > -800 and final_cos_theta > 0:
                print(f"  ↻ Learning: reward = {episode_reward:.1f}, cosθ = {final_cos_theta:.2f}")
        
        # Early stopping if learning well
        if episode >= 50 and episode_reward > -600:
            print(f"\n✓ Early stopping: Good performance achieved at episode {episode}")
            break
    
    env.close()
    
    # Save models
    torch.save(actor.state_dict(), "cdm_final_actor.pth")
    torch.save(critic.state_dict(), "cdm_final_critic.pth")
    torch.save(metric.state_dict(), "cdm_final_metric.pth")
    torch.save(dynamics.state_dict(), "cdm_final_dynamics.pth")
    
    # Plot results
    plot_simple_results(rewards_history)
    
    return actor, critic, metric, dynamics

def plot_simple_results(rewards):
    """Simple plot of training results"""
    plt.figure(figsize=(10, 5))
    
    episodes = range(len(rewards))
    
    # Raw rewards
    plt.plot(episodes, rewards, 'b-', alpha=0.3, linewidth=0.5)
    
    # Moving average
    if len(rewards) >= 10:
        window = 10
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2)
    
    plt.axhline(y=-800, color='g', linestyle='--', alpha=0.5, label='Good (-800)')
    plt.axhline(y=-500, color='orange', linestyle='--', alpha=0.5, label='Excellent (-500)')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('CDM Training Progress')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Summary text
    plt.figtext(0.02, 0.02, 
                f"Episodes: {len(rewards)}\nBest: {np.max(rewards):.1f}\nFinal: {rewards[-1]:.1f}\nAvg last 10: {np.mean(rewards[-10:]):.1f}",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('cdm_results.png', dpi=100)
    plt.show()

# ============================
# SIMPLE TEST FUNCTION
# ============================

def test_policy():
    """Test the trained policy"""
    env = gym.make("Pendulum-v1", render_mode="human")
    state_dim = 3
    action_dim = 1
    
    # Load actor
    actor = FixedPolicy(state_dim, action_dim)
    try:
        actor.load_state_dict(torch.load("cdm_final_actor.pth", map_location='cpu'))
        print("✓ Loaded trained policy")
    except:
        print("✗ Could not load policy")
        return
    
    actor.eval()
    
    print("\nTesting policy for 5 episodes...")
    
    for ep in range(5):
        state, _ = env.reset()
        total_reward = 0
        max_velocity = 0
        
        for step in range(200):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_tensor).numpy()[0]
            
            next_state, reward, done, _, _ = env.step(action)
            
            total_reward += reward
            max_velocity = max(max_velocity, abs(state[2]))
            state = next_state
            
            if done:
                break
        
        # Calculate angle
        cos_theta = state[0]
        sin_theta = state[1]
        angle = np.arctan2(sin_theta, cos_theta)
        
        print(f"Episode {ep}: Reward = {total_reward:7.1f}, "
              f"Max |θ̇| = {max_velocity:5.2f}, "
              f"Final angle = {np.degrees(angle):5.1f}°")
    
    env.close()

# ============================
# MAIN EXECUTION
# ============================

if __name__ == "__main__":
    print("="*70)
    print("CDM - SIMPLIFIED AND FIXED")
    print("="*70)
    
    try:
        print("\nTraining for 200 episodes...")
        actor, critic, metric, dynamics = train_cdm_with_contraction_fixed()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        
        # Test the trained policy
        test_policy()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try a very simple approach as last resort
        print("\nTrying ultra-simple training...")
        train_ultra_simple()

def train_ultra_simple():
    """Ultra-simple training for debugging"""
    env = gym.make("Pendulum-v1")
    state_dim = 3
    action_dim = 1
    
    # Simple policy
    policy = nn.Sequential(
        nn.Linear(state_dim, 32),
        nn.ReLU(),
        nn.Linear(32, action_dim),
        nn.Tanh()
    )
    
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    print("\nUltra-simple training...")
    print("-" * 50)
    
    for episode in range(100):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(200):
            # Get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = policy(state_tensor)
            action = action_tensor.detach().numpy()[0]
            
            # Add noise
            action = action + np.random.normal(0, 0.5)
            action = np.clip(action, -2.0, 2.0)
            
            # Take step
            next_state, reward, done, _, _ = env.step(action)
            
            # Simple learning: try to reduce angle
            cos_theta = state[0]
            sin_theta = state[1]
            angle = np.arctan2(sin_theta, cos_theta)
            
            # Target: proportional control
            target_action = -0.8 * angle
            
            # Update
            optimizer.zero_grad()
            loss = nn.MSELoss()(action_tensor, torch.FloatTensor([target_action]))
            loss.backward()
            optimizer.step()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward = {total_reward:7.1f}")
    
    env.close()
    torch.save(policy.state_dict(), "simple_policy.pth")
    print("\nSimple training complete.")