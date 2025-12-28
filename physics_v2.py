import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random

# ============================
# FIXED VALUE NETWORK (CRITIC)
# ============================

class ValueNetwork(nn.Module):
    """Critic that estimates state value V(s)"""
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

# ============================
# FIXED POLICY NETWORK (ACTOR)
# ============================

class FixedPolicy(nn.Module):
    """Actor that outputs mean action"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Scale to action range [-2, 2]
        self.action_scale = 2.0
    
    def forward(self, state):
        return torch.tanh(self.net(state)) * self.action_scale

# ============================
# SIMPLIFIED CONTRACTION METRIC
# ============================

class SimpleContractionMetric(nn.Module):
    """Simplified metric for stability"""
    def __init__(self, state_dim, hidden_dim=16):
        super().__init__()
        self.state_dim = state_dim
        
        # Learn position-dependent scaling
        self.scale_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus()
        )
    
    def forward(self, state):
        batch_size = state.shape[0]
        
        # Get scale for each dimension
        scales = self.scale_net(state) + 0.1  # Ensure > 0
        
        # Create diagonal metric M = diag(scale)
        M = torch.zeros(batch_size, self.state_dim, self.state_dim, device=state.device)
        
        # Fill diagonal with scales
        for i in range(self.state_dim):
            M[:, i, i] = scales[:, i]
        
        return M

# ============================
# REPLAY BUFFER
# ============================

class ReplayBuffer:
    def __init__(self, capacity=10000):
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
    """Ornstein-Uhlenbeck process for exploration"""
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
# SOFT UPDATE FUNCTION
# ============================

def soft_update(target, source, tau):
    """Soft update target network parameters"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# ============================
# FIXED TRAINING FUNCTION
# ============================

def train_cdm_proper():
    """Proper training with actor-critic and contraction"""
    
    env = gym.make("Pendulum-v1")
    state_dim = 3
    action_dim = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize networks
    actor = FixedPolicy(state_dim, action_dim).to(device)
    critic = ValueNetwork(state_dim).to(device)
    metric = SimpleContractionMetric(state_dim).to(device)
    
    # Target networks for stability
    target_actor = FixedPolicy(state_dim, action_dim).to(device)
    target_critic = ValueNetwork(state_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    
    # Optimizers
    actor_optim = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optim = optim.Adam(critic.parameters(), lr=1e-3)
    metric_optim = optim.Adam(metric.parameters(), lr=1e-4)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=10000)
    
    # Training parameters
    gamma = 0.99  # Discount factor
    tau = 0.005   # Target network update rate
    batch_size = 64
    
    # Exploration noise
    ou_noise = OUNoise(action_dim)
    
    print("Starting proper CDM training...")
    print("-" * 70)
    print(f"{'Episode':>8} {'Reward':>10} {'Max|θ̇|':>8} {'Cosθ':>8}")
    print("-" * 70)
    
    for episode in range(200):  # More episodes
        state, _ = env.reset()
        ou_noise.reset()
        
        episode_reward = 0
        episode_max_velocity = 0
        episode_states = []
        
        for step in range(200):
            # Convert to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Select action with exploration
            with torch.no_grad():
                action_mean = actor(state_tensor).cpu().numpy()[0]
            
            # Add exploration noise
            noise = ou_noise.sample()
            action = action_mean + noise
            action = np.clip(action, -2.0, 2.0)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Update metrics
            episode_reward += reward
            episode_max_velocity = max(episode_max_velocity, abs(state[2]))
            episode_states.append(state)
            
            # Train if enough samples
            if len(replay_buffer) > batch_size:
                # Sample batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Convert to tensors
                states_t = torch.FloatTensor(states).to(device)
                actions_t = torch.FloatTensor(actions).to(device)
                rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states_t = torch.FloatTensor(next_states).to(device)
                dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                # ===== TRAIN CRITIC =====
                critic_optim.zero_grad()
                
                with torch.no_grad():
                    next_actions = target_actor(next_states_t)
                    next_values = target_critic(next_states_t)
                    target_values = rewards_t + gamma * next_values * (1 - dones_t)
                
                current_values = critic(states_t)
                critic_loss = nn.MSELoss()(current_values, target_values)
                critic_loss.backward()
                critic_optim.step()
                
                # ===== TRAIN ACTOR =====
                actor_optim.zero_grad()
                
                # Get actions from current policy
                actor_actions = actor(states_t)
                
                # Get value estimates for these actions
                # IMPORTANT: Need to recompute values with updated critic
                actor_values = critic(states_t)
                
                # Actor loss: maximize Q-value (negative because we want to maximize)
                actor_loss = -actor_values.mean()
                
                # Add velocity penalty
                velocity_penalty = torch.abs(states_t[:, 2]).mean() * 0.01
                actor_loss += velocity_penalty
                
                # Add contraction-based regularization
                with torch.no_grad():
                    M = metric(states_t)
                    # Get metric eigenvalues
                    eigvals = torch.linalg.eigvalsh(M)
                    # Penalize large metrics in unstable states
                    cos_theta = states_t[:, 0]
                    unstable = cos_theta < 0
                    if unstable.any():
                        metric_penalty = eigvals[unstable].mean() * 0.001
                        actor_loss += metric_penalty
                
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optim.step()
                
                # ===== TRAIN METRIC =====
                metric_optim.zero_grad()
                
                # Recompute metric for this batch
                M = metric(states_t)
                
                # Metric should provide more "stiffness" for unstable states
                cos_theta = states_t[:, 0]
                
                # Separate stable and unstable states
                stable_mask = cos_theta > 0.3  # Somewhat upright
                unstable_mask = cos_theta < -0.3  # Somewhat inverted
                
                metric_loss = torch.tensor(0.0, device=device)
                
                if stable_mask.any():
                    M_stable = M[stable_mask]
                    # For stable states, want smaller metric (less correction needed)
                    metric_stable_loss = torch.diagonal(M_stable, dim1=1, dim2=2).mean()
                    metric_loss += metric_stable_loss * 0.001
                
                if unstable_mask.any():
                    M_unstable = M[unstable_mask]
                    # For unstable states, want larger metric (more aggressive correction)
                    metric_unstable_loss = -torch.diagonal(M_unstable, dim1=1, dim2=2).mean()
                    metric_loss += metric_unstable_loss * 0.01
                
                # Regularization: keep metric close to identity
                identity = torch.eye(state_dim, device=device).unsqueeze(0)
                reg_loss = torch.norm(M - identity, dim=(1, 2)).mean() * 0.001
                metric_loss += reg_loss
                
                if metric_loss != 0:
                    metric_loss.backward()
                    torch.nn.utils.clip_grad_norm_(metric.parameters(), 0.5)
                    metric_optim.step()
                
                # ===== UPDATE TARGET NETWORKS =====
                soft_update(target_actor, actor, tau)
                soft_update(target_critic, critic, tau)
            
            state = next_state
            
            if done:
                break
        
        # Log progress
        if len(episode_states) > 0:
            final_cos_theta = episode_states[-1][0]
        else:
            final_cos_theta = 0
        
        if episode % 10 == 0 or episode < 10:
            print(f"{episode:8d} {episode_reward:10.1f} {episode_max_velocity:8.2f} "
                  f"{final_cos_theta:8.2f}")
            
            # Early success detection
            if episode_reward > -500:  # Good performance
                print(f"  ✓ Learning successful! Reward: {episode_reward:.1f}")
    
    env.close()
    return actor, critic, metric

# ============================
# EVALUATION FUNCTION
# ============================

def evaluate_policy(actor, env_name="Pendulum-v1", episodes=5, render=False):
    """Evaluate trained policy"""
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)
    
    device = next(actor.parameters()).device
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    all_rewards = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        max_velocity = 0
        
        for step in range(200):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy()[0]
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            total_reward += reward
            max_velocity = max(max_velocity, abs(state[2]))
            state = next_state
            
            if terminated or truncated:
                break
        
        all_rewards.append(total_reward)
        
        # Analyze performance
        cos_theta = state[0]
        sin_theta = state[1]
        angle = np.arctan2(sin_theta, cos_theta)
        
        print(f"Episode {ep}:")
        print(f"  Reward: {total_reward:7.1f}")
        print(f"  Max |θ̇|: {max_velocity:6.2f}")
        print(f"  Final angle: {angle:6.2f} rad ({np.degrees(angle):5.0f}°)")
        print(f"  Final cosθ: {cos_theta:6.2f}")
        
        # Success criteria
        if abs(angle) < 0.35:  # Within ±20 degrees of upright
            print("  ✓ BALANCED: Within target range")
        else:
            print("  ✗ NOT BALANCED")
        print()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY:")
    print(f"Average reward: {np.mean(all_rewards):.1f}")
    print(f"Best reward: {np.max(all_rewards):.1f}")
    print(f"Worst reward: {np.min(all_rewards):.1f}")
    
    if np.mean(all_rewards) > -800:
        print("✓ POLICY IS LEARNING")
    else:
        print("✗ POLICY NEEDS MORE TRAINING")
    
    env.close()

# ============================
# MAIN EXECUTION
# ============================

if __name__ == "__main__":
    print("="*70)
    print("CDM PROPER IMPLEMENTATION")
    print("="*70)
    
    # Train properly
    try:
        actor, critic, metric = train_cdm_proper()
        
        # Save models
        torch.save(actor.state_dict(), "cdm_proper_actor.pth")
        torch.save(critic.state_dict(), "cdm_proper_critic.pth")
        torch.save(metric.state_dict(), "cdm_proper_metric.pth")
        print("\n✓ Models saved successfully!")
        
        # Evaluate with rendering
        evaluate_policy(actor, episodes=5, render=True)
        
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        print("\nTrying simplified approach...")
        
        # Fallback to simpler training
        train_simple_cdm()

def train_simple_cdm():
    """Simplified training for debugging"""
    env = gym.make("Pendulum-v1")
    state_dim = 3
    action_dim = 1
    
    device = torch.device("cpu")
    
    # Very simple policy
    policy = nn.Sequential(
        nn.Linear(state_dim, 32),
        nn.ReLU(),
        nn.Linear(32, action_dim),
        nn.Tanh()
    ).to(device)
    
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    print("\nStarting simple training...")
    print("-" * 70)
    
    for episode in range(50):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(200):
            # Get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_tensor = policy(state_tensor)
            action = action_tensor.detach().cpu().numpy()[0]
            
            # Add noise
            action = action + np.random.normal(0, 0.3)
            action = np.clip(action, -2.0, 2.0)
            
            # Take step
            next_state, reward, done, _, _ = env.step(action)
            
            # Very simple learning: try to reduce angle
            cos_theta = state[0]
            sin_theta = state[1]
            angle = np.arctan2(sin_theta, cos_theta)
            
            # Learn to produce torque opposite to angle
            target_action = -0.5 * angle  # Simple proportional control
            
            # Update policy
            optimizer.zero_grad()
            action_loss = nn.MSELoss()(action_tensor, 
                                      torch.FloatTensor([target_action]).to(device))
            action_loss.backward()
            optimizer.step()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        if episode % 10 == 0:
            print(f"Episode {episode:3d}: Reward = {total_reward:7.1f}")
    
    env.close()
    print("\nSimple training complete.")