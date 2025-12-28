import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt

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

class FixedValueNetwork(nn.Module):
    """Critic network with output clipping"""
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
        value = self.net(state)
        return torch.clamp(value, -100, 0)

class SimpleDynamics(nn.Module):
    """Dynamics model"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def forward(self, state, action):
        # Ensure proper dimensions
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        elif action.dim() == 3:
            action = action.squeeze(1)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class ContractionMetric(nn.Module):
    """Simplified Riemannian metric M(x)"""
    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus()
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        diag = self.net(x) + 0.1
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
    # Ensure proper dimensions for batch matrix multiplication
    if state.dim() == 2:
        state_unsqueezed = state.unsqueeze(1)  # [batch, 1, state_dim]
        energy = torch.bmm(torch.bmm(state_unsqueezed, M), state_unsqueezed.transpose(1, 2)).squeeze()
    else:
        state_unsqueezed = state.unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
        M_expanded = M.unsqueeze(0) if M.dim() == 2 else M
        energy = torch.bmm(torch.bmm(state_unsqueezed, M_expanded), state_unsqueezed.transpose(1, 2)).squeeze()
    return energy, M

def soft_update(target, source, tau):
    """Soft update target network"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def normalize_state(state):
    """Normalize pendulum state"""
    cos_theta, sin_theta, theta_dot = state
    normalized_theta_dot = theta_dot / 8.0  # Pendulum max velocity is ±8
    return np.array([cos_theta, sin_theta, normalized_theta_dot])

def scale_reward(reward):
    """Scale pendulum reward for better learning"""
    return reward / 10.0  # Reasonable scaling

# ============================
# IMPROVED ADAPTIVE NOISE
# ============================

class ImprovedAdaptiveNoise:
    def __init__(self, action_dim, base_sigma=0.3):
        self.action_dim = action_dim
        self.base_sigma = base_sigma
        self.sigma = base_sigma
        self.state = np.zeros(action_dim)
        self.recent_rewards = deque(maxlen=10)
        
    def reset(self):
        self.state = np.zeros(self.action_dim)
    
    def sample(self):
        # Ornstein-Uhlenbeck process
        self.state += -0.1 * self.state + self.sigma * np.random.randn(self.action_dim)
        return self.state
    
    def update(self, reward, episode):
        self.recent_rewards.append(reward)
        
        if len(self.recent_rewards) >= 5:
            avg_reward = np.mean(list(self.recent_rewards))
            
            # Adaptive sigma based on performance AND episode
            if avg_reward > -700:
                self.sigma = self.base_sigma * 0.1  # Very low noise when doing well
            elif avg_reward > -900:
                self.sigma = self.base_sigma * 0.3  # Low noise
            elif avg_reward > -1100:
                self.sigma = self.base_sigma * 0.6  # Medium noise
            else:
                self.sigma = self.base_sigma  # High noise when doing poorly
        
        # Also decay over time
        episode_decay = max(0.1, 1.0 - episode / 200)
        self.sigma *= episode_decay
        self.sigma = max(0.05, self.sigma)  # Minimum noise

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
# COMPLETE WORKING TRAINING FUNCTION
# ============================

def train_cdm_complete():
    """Complete working version with all fixes"""
    
    env = gym.make("Pendulum-v1")
    state_dim = 3
    action_dim = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize networks
    actor = FixedPolicy(state_dim, action_dim, hidden_dim=128).to(device)
    critic = FixedValueNetwork(state_dim, hidden_dim=128).to(device)
    dynamics = SimpleDynamics(state_dim, action_dim, hidden_dim=64).to(device)
    metric = ContractionMetric(state_dim, hidden_dim=32).to(device)
    
    # Target networks
    target_actor = FixedPolicy(state_dim, action_dim, hidden_dim=128).to(device)
    target_critic = FixedValueNetwork(state_dim, hidden_dim=128).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    
    # Optimizers with stable settings
    actor_optim = optim.Adam(actor.parameters(), lr=2e-4)
    critic_optim = optim.Adam(critic.parameters(), lr=5e-4)
    dynamics_optim = optim.Adam(dynamics.parameters(), lr=1e-3)
    metric_optim = optim.Adam(metric.parameters(), lr=1e-4)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=30000)
    
    # Training parameters
    gamma = 0.99
    tau = 0.005
    batch_size = 128
    lambda_contraction = 0.05  # Small contraction weight to start
    
    # Improved adaptive noise
    adaptive_noise = ImprovedAdaptiveNoise(action_dim, base_sigma=0.4)
    
    print("\nStarting COMPLETE CDM training...")
    print("-" * 70)
    print(f"{'Episode':>8} {'Reward':>10} {'Max|θ̇|':>8} {'Cosθ':>8} {'Noise':>8} {'Best':>8}")
    print("-" * 70)
    
    rewards_history = []
    best_reward = -float('inf')
    best_actor_state = None
    
    for episode in range(300):
        state, _ = env.reset()
        adaptive_noise.reset()
        
        episode_reward = 0
        episode_max_velocity = 0
        steps = 0
        
        # Episode buffer for batch addition
        episode_transitions = []
        
        while True:
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_mean = actor(state_tensor).cpu().numpy()[0]
            
            # Add adaptive noise with episode-based scaling
            noise_scale = 1.0
            if episode < 50:  # More exploration early
                noise_scale = 1.0
            elif episode < 150:
                noise_scale = 0.5
            else:
                noise_scale = 0.2
            
            noise = adaptive_noise.sample() * noise_scale
            action = action_mean + noise
            action = np.clip(action, -2.0, 2.0)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Scale reward
            scaled_reward = scale_reward(reward)
            
            # Store transition for this episode
            episode_transitions.append((state.copy(), action, scaled_reward, next_state.copy(), done))
            
            # Update metrics
            episode_reward += reward
            episode_max_velocity = max(episode_max_velocity, abs(state[2]))
            steps += 1
            
            state = next_state
            
            if done or steps >= 200:
                break
        
        # Add episode transitions to replay buffer
        for transition in episode_transitions:
            replay_buffer.push(*transition)
        
        # Update adaptive noise based on this episode's performance
        adaptive_noise.update(episode_reward, episode)
        
        # Store episode metrics
        rewards_history.append(episode_reward)
        final_cos_theta = state[0]
        
        # Update best reward and save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_actor_state = actor.state_dict().copy()
            torch.save(actor.state_dict(), "cdm_best_actor.pth")
            torch.save({
                'episode': episode,
                'reward': episode_reward,
                'actor': actor.state_dict(),
                'critic': critic.state_dict()
            }, "cdm_best_checkpoint.pth")
        
        # Train networks if we have enough samples
        if len(replay_buffer) > batch_size * 2:
            # Sample batch from replay buffer
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            # Convert to tensors - FIXED DIMENSIONS
            states_t = torch.FloatTensor(states).to(device)  # [batch, 3]
            actions_t = torch.FloatTensor(actions).to(device).unsqueeze(1)  # [batch, 1] - FIXED
            rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)  # [batch, 1]
            next_states_t = torch.FloatTensor(next_states).to(device)  # [batch, 3]
            dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)  # [batch, 1]
            
            # ===== 1. TRAIN CRITIC =====
            critic_optim.zero_grad()
            
            with torch.no_grad():
                next_actions = target_actor(next_states_t)
                next_values = target_critic(next_states_t)
                # Clip target values to prevent explosion
                target_values = rewards_t + gamma * torch.clamp(next_values, -20, 0) * (1 - dones_t)
                target_values = torch.clamp(target_values, -20, 0)
            
            current_values = critic(states_t)
            critic_loss = nn.MSELoss()(current_values, target_values)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_optim.step()
            
            # ===== 2. TRAIN DYNAMICS =====
            dynamics_optim.zero_grad()
            # Ensure actions_t has correct dimensions [batch, 1]
            next_states_pred = dynamics(states_t, actions_t)  # FIXED: removed .unsqueeze(1)
            dynamics_loss = nn.MSELoss()(next_states_pred, next_states_t)
            dynamics_loss.backward()
            torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
            dynamics_optim.step()
            
            # ===== 3. TRAIN METRIC =====
            metric_optim.zero_grad()
            energy_curr, M = compute_energy(states_t, metric)
            
            # Simple metric loss: encourage moderate energy
            cos_theta = states_t[:, 0]
            upright = cos_theta > 0.5
            
            metric_loss = torch.tensor(0.0, device=device)
            if upright.any():
                # For upright states, want moderate energy
                energy_upright = energy_curr[upright]
                target_energy = 0.5
                metric_loss += torch.abs(energy_upright.mean() - target_energy) * 0.01
            
            # Regularization
            identity = torch.eye(state_dim, device=device).unsqueeze(0)
            reg_loss = torch.norm(M - identity, dim=(1, 2)).mean() * 0.001
            metric_loss += reg_loss
            
            if metric_loss != 0:
                metric_loss.backward()
                torch.nn.utils.clip_grad_norm_(metric.parameters(), 0.5)
                metric_optim.step()
            
            # ===== 4. TRAIN ACTOR WITH CONTRACTION =====
            actor_optim.zero_grad()
            
            # Get actor's actions
            actor_actions = actor(states_t)  # [batch, 1]
            
            # Get value estimates
            actor_values = critic(states_t)
            
            # Calculate contraction bonus (optional, can be disabled)
            with torch.no_grad():
                # Predict next state using dynamics
                next_states_pred = dynamics(states_t, actor_actions)
                # Compute energies
                energy_curr, _ = compute_energy(states_t, metric)
                energy_next, _ = compute_energy(next_states_pred, metric)
                # Contraction measure
                delta_energy = energy_curr - energy_next
                contraction_bonus = torch.tanh(delta_energy / 10.0).mean()
            
            # Velocity penalty (encourage controlled movement)
            velocity = states_t[:, 2] * 8.0  # De-normalize
            velocity_penalty = torch.relu(torch.abs(velocity) - 2.0).mean() * 0.01
            
            # Action smoothness penalty
            action_penalty = torch.abs(actor_actions).mean() * 0.001
            
            # Angle penalty (encourage upright)
            angle = torch.atan2(states_t[:, 1], states_t[:, 0])
            angle_penalty = torch.abs(angle).mean() * 0.005
            
            # Total actor loss
            actor_loss = -actor_values.mean() * 0.1  # Conservative scaling
            actor_loss -= lambda_contraction * contraction_bonus * 0.1  # Small contraction influence
            actor_loss += velocity_penalty + action_penalty + angle_penalty
            
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_optim.step()
            
            # ===== 5. UPDATE TARGET NETWORKS =====
            soft_update(target_actor, actor, tau)
            soft_update(target_critic, critic, tau)
        
        # Log progress
        if episode % 10 == 0 or episode < 20 or episode == 299:
            print(f"{episode:8d} {episode_reward:10.1f} {episode_max_velocity:8.2f} "
                  f"{final_cos_theta:8.2f} {adaptive_noise.sigma:8.2f} {best_reward:8.1f}")
            
            # Performance feedback
            if episode_reward > -700:
                print(f"  ✓ EXCELLENT! Reward: {episode_reward:.1f}")
            elif episode_reward > -900:
                print(f"  ↻ Good progress")
            elif episode_reward == best_reward:
                print(f"  ★ NEW BEST!")
        
        # Early stopping if consistently good
        if episode >= 100 and episode_reward > -600:
            print(f"\n✓ Early stopping: Excellent performance at episode {episode}")
            break
    
    env.close()
    
    # Save final models
    torch.save(actor.state_dict(), "cdm_final_actor.pth")
    torch.save(critic.state_dict(), "cdm_final_critic.pth")
    torch.save(metric.state_dict(), "cdm_final_metric.pth")
    torch.save(dynamics.state_dict(), "cdm_final_dynamics.pth")
    
    # Plot comprehensive results
    plot_complete_results(rewards_history, best_reward)
    
    return actor, critic, metric, dynamics

def plot_complete_results(rewards, best_reward):
    """Plot complete training results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    episodes = range(len(rewards))
    
    # Plot 1: Raw rewards
    axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.3, linewidth=0.5)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Raw Training Rewards')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Moving averages
    axes[0, 1].plot(episodes, rewards, 'b-', alpha=0.2, linewidth=0.3)
    
    colors = ['r-', 'g-', 'purple']
    windows = [5, 10, 20]
    for i, window in enumerate(windows):
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(episodes[window-1:], moving_avg, colors[i], linewidth=2, 
                           label=f'{window}-ep MA')
    
    axes[0, 1].axhline(y=-800, color='green', linestyle='--', alpha=0.5, label='Good')
    axes[0, 1].axhline(y=-600, color='orange', linestyle='--', alpha=0.5, label='Excellent')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Moving Averages')
    axes[0, 1].legend(loc='upper left', fontsize='small')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Histogram
    axes[0, 2].hist(rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 2].axvline(x=-800, color='green', linestyle='--', alpha=0.5)
    axes[0, 2].axvline(x=-600, color='orange', linestyle='--', alpha=0.5)
    axes[0, 2].axvline(x=best_reward, color='red', linestyle='-', alpha=0.8, linewidth=2)
    axes[0, 2].set_xlabel('Reward')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Reward Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Success rate over time
    if len(rewards) >= 20:
        success_rates = []
        window = 20
        
        for i in range(len(rewards) - window + 1):
            window_rewards = rewards[i:i+window]
            successes = sum(1 for r in window_rewards if r > -800)
            success_rates.append(successes / window * 100)
        
        axes[1, 0].plot(range(window-1, len(rewards)), success_rates, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].set_title(f'Success Rate ({window}-ep window)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 100])
    
    # Plot 5: Cumulative best
    cumulative_best = []
    current_best = -float('inf')
    for r in rewards:
        current_best = max(current_best, r)
        cumulative_best.append(current_best)
    
    axes[1, 1].plot(episodes, cumulative_best, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Best Reward')
    axes[1, 1].set_title('Cumulative Best Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    axes[1, 2].axis('off')
    
    if len(rewards) >= 20:
        first_10 = rewards[:10]
        last_10 = rewards[-10:]
        
        summary_text = f"""
        CDM COMPLETE TRAINING SUMMARY
        
        Statistics:
        - Total Episodes: {len(rewards)}
        - Best Reward: {best_reward:.1f}
        - Final Reward: {rewards[-1]:.1f}
        - Average: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}
        
        Progress:
        - First 10: {np.mean(first_10):.1f}
        - Last 10: {np.mean(last_10):.1f}
        - Improvement: {np.mean(last_10) - np.mean(first_10):.1f}
        
        Performance:
        - > -1000: {sum(1 for r in rewards if r > -1000)} ({sum(1 for r in rewards if r > -1000)/len(rewards)*100:.1f}%)
        - > -800: {sum(1 for r in rewards if r > -800)} ({sum(1 for r in rewards if r > -800)/len(rewards)*100:.1f}%)
        - > -600: {sum(1 for r in rewards if r > -600)} ({sum(1 for r in rewards if r > -600)/len(rewards)*100:.1f}%)
        
        Assessment:
        """
        
        avg_last_10 = np.mean(last_10)
        if avg_last_10 > -500:
            assessment = "✅ OUTSTANDING: Perfect control!"
        elif avg_last_10 > -600:
            assessment = "✓ EXCELLENT: Very good balance"
        elif avg_last_10 > -800:
            assessment = "↻ GOOD: Learning successful"
        elif np.mean(last_10) > np.mean(first_10) + 200:
            assessment = "↻ PROGRESSING: Clear learning"
        else:
            assessment = "⚠ NEEDS TUNING"
        
        summary_text += assessment
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Contraction Dynamics Model - Complete Training Results', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('cdm_complete_results.png', dpi=100, bbox_inches='tight')
    plt.show()

# ============================
# DEMONSTRATION FUNCTION
# ============================

def demonstrate_cdm():
    """Demonstrate the trained CDM"""
    env = gym.make("Pendulum-v1", render_mode="human")
    state_dim = 3
    action_dim = 1
    
    # Load actor
    actor = FixedPolicy(state_dim, action_dim)
    
    try:
        # Try to load best model
        checkpoint = torch.load("cdm_best_checkpoint.pth", map_location='cpu', weights_only=False)
        actor.load_state_dict(checkpoint['actor'])
        episode_num = checkpoint['episode']
        reward = checkpoint['reward']
        print(f"✓ Loaded BEST model from episode {episode_num}")
        print(f"  Training reward: {reward:.1f}")
    except:
        try:
            actor.load_state_dict(torch.load("cdm_best_actor.pth", map_location='cpu', weights_only=False))
            print("✓ Loaded best actor")
        except:
            try:
                actor.load_state_dict(torch.load("cdm_final_actor.pth", map_location='cpu', weights_only=False))
                print("✓ Loaded final actor")
            except:
                print("✗ Could not load any model")
                return
    
    actor.eval()
    
    print("\n" + "="*70)
    print("CDM DEMONSTRATION")
    print("="*70)
    
    test_results = []
    
    for ep in range(5):
        state, _ = env.reset()
        total_reward = 0
        max_velocity = 0
        upright_time = 0
        steps = 0
        
        for step in range(200):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_tensor).numpy()[0]
            
            next_state, reward, done, _, _ = env.step(action)
            
            total_reward += reward
            max_velocity = max(max_velocity, abs(state[2]))
            
            # Track upright time
            if state[0] > 0.8:
                upright_time += 1
            
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Calculate metrics
        cos_theta = state[0]
        sin_theta = state[1]
        angle = np.arctan2(sin_theta, cos_theta)
        
        test_results.append({
            'reward': total_reward,
            'max_velocity': max_velocity,
            'upright_percent': (upright_time / steps) * 100,
            'final_angle': np.degrees(angle),
            'final_cos': cos_theta
        })
        
        print(f"\nEpisode {ep}:")
        print(f"  Reward: {total_reward:7.1f}")
        print(f"  Max |θ̇|: {max_velocity:6.2f} rad/s")
        print(f"  Upright: {upright_time/steps*100:5.1f}% of time")
        print(f"  Final angle: {np.degrees(angle):5.1f}°")
        print(f"  Final cosθ: {cos_theta:6.2f}")
        
        if abs(angle) < 0.35:  # Within 20 degrees
            print("  ✓ SUCCESS: Pendulum balanced!")
    
    env.close()
    
    # Summary
    print("\n" + "="*70)
    print("DEMONSTRATION SUMMARY")
    print("="*70)
    
    rewards = [r['reward'] for r in test_results]
    upright_percents = [r['upright_percent'] for r in test_results]
    
    print(f"Average Reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Average Upright Time: {np.mean(upright_percents):.1f}%")
    print(f"Success Rate (reward > -800): {sum(1 for r in rewards if r > -800)/len(rewards)*100:.1f}%")
    print(f"Excellent Rate (reward > -600): {sum(1 for r in rewards if r > -600)/len(rewards)*100:.1f}%")
    
    if np.mean(rewards) > -600:
        print("\n✅ CDM IS HIGHLY EFFECTIVE!")
    elif np.mean(rewards) > -800:
        print("\n✓ CDM IS EFFECTIVE")
    else:
        print("\n⚠ CDM NEEDS IMPROVEMENT")

# ============================
# MAIN EXECUTION
# ============================

if __name__ == "__main__":
    print("="*70)
    print("CONTRACTION DYNAMICS MODEL - COMPLETE WORKING VERSION")
    print("="*70)
    print("All fixes implemented:")
    print("1. Proper adaptive noise with performance-based adjustment")
    print("2. Reward scaling (÷10) for stable learning")
    print("3. Value clipping to prevent explosion")
    print("4. Conservative actor updates (scaled by 0.1)")
    print("5. Episode-based training for stability")
    print("6. Comprehensive logging and visualization")
    print("="*70)
    
    try:
        # Train the complete model
        print("\nStarting complete training...")
        actor, critic, metric, dynamics = train_cdm_complete()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        
        # Demonstrate the trained model
        print("\nStarting demonstration...")
        demonstrate_cdm()
        
        print("\n" + "="*70)
        print("ALL OPERATIONS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nOutput files created:")
        print("  - cdm_best_actor.pth (best policy)")
        print("  - cdm_best_checkpoint.pth (best checkpoint with metadata)")
        print("  - cdm_final_actor.pth (final policy)")
        print("  - cdm_final_critic.pth (final critic)")
        print("  - cdm_final_metric.pth (final metric)")
        print("  - cdm_final_dynamics.pth (final dynamics)")
        print("  - cdm_complete_results.png (comprehensive plots)")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Models have been saved at the last checkpoint")
    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()