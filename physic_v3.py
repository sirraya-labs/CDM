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
    
    def forward(self, state, action):
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
    energy = torch.bmm(
        torch.bmm(state.unsqueeze(1), M),
        state.unsqueeze(-1)
    ).squeeze()
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
# STABLE TRAINING FUNCTION
# ============================

def train_cdm_stable():
    """Stable training with proper gradient separation"""
    
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
    
    # Optimizers with better settings
    actor_optim = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optim = optim.Adam(critic.parameters(), lr=1e-3)
    dynamics_optim = optim.Adam(dynamics.parameters(), lr=1e-3)
    metric_optim = optim.Adam(metric.parameters(), lr=1e-4)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=20000)
    
    # Training parameters
    gamma = 0.99
    tau = 0.005
    batch_size = 128  # Increased for stability
    lambda_contraction = 0.1
    
    # Noise
    ou_noise = OUNoise(action_dim, sigma=0.1)  # Lower noise
    
    print("\nStarting STABLE CDM training...")
    print("-" * 70)
    print(f"{'Episode':>8} {'Reward':>10} {'Max|θ̇|':>8} {'Cosθ':>8} {'Q-Value':>8}")
    print("-" * 70)
    
    rewards_history = []
    best_reward = -float('inf')
    
    for episode in range(300):
        state, _ = env.reset()
        ou_noise.reset()
        
        # Normalize initial state
        state_norm = normalize_state(state)
        
        # Decay exploration slowly
        noise_scale = max(0.02, 0.2 * (1 - episode / 250))
        
        episode_reward = 0
        episode_max_velocity = 0
        episode_avg_q = 0
        steps = 0
        
        while True:
            # Get action from policy
            state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_mean = actor(state_tensor).cpu().numpy()[0]
            
            # Add exploration noise
            if episode < 100:  # More exploration early
                noise = ou_noise.sample() * noise_scale
            else:
                noise = ou_noise.sample() * (noise_scale * 0.3)  # Less noise later
            
            action = action_mean + noise
            action = np.clip(action, -2.0, 2.0)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Normalize next state
            next_state_norm = normalize_state(next_state)
            
            # Store experience
            replay_buffer.push(state_norm, action, reward, next_state_norm, done)
            
            # Update metrics
            episode_reward += reward
            episode_max_velocity = max(episode_max_velocity, abs(state[2]))
            steps += 1
            
            # Train if we have enough samples
            if len(replay_buffer) > batch_size * 2:  # Wait for more samples
                # ===== SEPARATE TRAINING STEPS =====
                
                # 1. Sample batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Convert to tensors
                states_t = torch.FloatTensor(states).to(device)
                actions_t = torch.FloatTensor(actions).to(device)
                rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states_t = torch.FloatTensor(next_states).to(device)
                dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                # 2. TRAIN DYNAMICS (separate, no shared tensors)
                dynamics_optim.zero_grad()
                states_dyn = states_t.detach().clone().requires_grad_(True)
                next_states_pred = dynamics(states_dyn, actions_t)
                dynamics_loss = nn.MSELoss()(next_states_pred, next_states_t.detach())
                dynamics_loss.backward()
                torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
                dynamics_optim.step()
                
                # 3. TRAIN METRIC (separate)
                metric_optim.zero_grad()
                states_metric = states_t.detach().clone().requires_grad_(True)
                energy_curr, M = compute_energy(states_metric, metric)
                
                # Metric loss: encourage appropriate energy levels
                cos_theta = states_metric[:, 0]
                velocity = states_metric[:, 2] * 8.0  # De-normalize
                
                # For upright states, want moderate energy
                upright = cos_theta > 0.3
                metric_loss = torch.tensor(0.0, device=device)
                
                if upright.any():
                    energy_upright = energy_curr[upright]
                    # Target energy for upright states
                    target_energy = 0.5
                    metric_loss += torch.abs(energy_upright.mean() - target_energy) * 0.01
                
                # For high velocity states, want higher energy (easier to contract)
                high_vel = torch.abs(velocity) > 3.0
                if high_vel.any():
                    energy_high_vel = energy_curr[high_vel]
                    metric_loss += -energy_high_vel.mean() * 0.01  # Negative to maximize
                
                # Regularization
                identity = torch.eye(state_dim, device=device).unsqueeze(0)
                reg_loss = torch.norm(M - identity, dim=(1, 2)).mean() * 0.001
                metric_loss += reg_loss
                
                if metric_loss != 0:
                    metric_loss.backward()
                    torch.nn.utils.clip_grad_norm_(metric.parameters(), 0.5)
                    metric_optim.step()
                
                # 4. TRAIN CRITIC (separate)
                critic_optim.zero_grad()
                
                with torch.no_grad():
                    next_actions = target_actor(next_states_t)
                    next_values = target_critic(next_states_t)
                    target_values = rewards_t + gamma * next_values * (1 - dones_t)
                
                states_critic = states_t.detach().clone().requires_grad_(True)
                current_values = critic(states_critic)
                critic_loss = nn.MSELoss()(current_values, target_values.detach())
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optim.step()
                
                episode_avg_q += current_values.mean().item()
                
                # 5. TRAIN ACTOR (separate, with contraction bonus)
                actor_optim.zero_grad()
                
                states_actor = states_t.detach().clone().requires_grad_(True)
                
                # Get actor's actions
                actor_actions = actor(states_actor)
                
                # Get value estimates
                actor_values = critic(states_actor.detach())  # Detach to prevent critic gradient flow
                
                # Calculate contraction bonus (using dynamics and metric)
                with torch.no_grad():
                    # Predict next state
                    next_states_pred = dynamics(states_actor.detach(), actor_actions.detach())
                    # Compute energies
                    energy_curr, _ = compute_energy(states_actor.detach(), metric)
                    energy_next, _ = compute_energy(next_states_pred.detach(), metric)
                    # Contraction measure
                    delta_energy = energy_curr - energy_next
                    contraction_bonus = torch.tanh(delta_energy / 50.0).mean()
                
                # Velocity penalty (softer)
                velocity = states_actor[:, 2] * 8.0  # De-normalize
                velocity_penalty = torch.relu(torch.abs(velocity) - 2.5).mean() * 0.02
                
                # Action penalty for smoothness
                action_penalty = torch.abs(actor_actions).mean() * 0.0005
                
                # Angle penalty (encourage upright)
                angle = torch.atan2(states_actor[:, 1], states_actor[:, 0])
                angle_penalty = torch.abs(angle).mean() * 0.005
                
                # Total actor loss
                actor_loss = -actor_values.mean() - lambda_contraction * contraction_bonus
                actor_loss += velocity_penalty + action_penalty + angle_penalty
                
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optim.step()
                
                # 6. UPDATE TARGET NETWORKS
                soft_update(target_actor, actor, tau)
                soft_update(target_critic, critic, tau)
            
            # Update state for next step
            state = next_state
            state_norm = next_state_norm
            
            if done or steps >= 200:
                break
        
        # Store episode metrics
        rewards_history.append(episode_reward)
        final_cos_theta = state[0]
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            # Save best model
            torch.save(actor.state_dict(), "cdm_best_actor.pth")
        
        # Calculate average Q-value
        avg_q = episode_avg_q / max(1, steps)
        
        # Log progress
        if episode % 10 == 0 or episode < 20 or episode == 299:
            print(f"{episode:8d} {episode_reward:10.1f} {episode_max_velocity:8.2f} "
                  f"{final_cos_theta:8.2f} {avg_q:8.2f}")
            
            # Success indicators
            if episode_reward > -1000 and episode_reward < best_reward:
                print(f"  ↻ Improving: {episode_reward:.1f} (Best: {best_reward:.1f})")
            if episode_reward > -800:
                print(f"  ✓ Good performance!")
            if episode_reward > -600 and final_cos_theta > 0.5:
                print(f"  ✓✓ Excellent! Upright and stable!")
        
        # Early stopping if excellent performance
        if episode >= 100 and episode_reward > -500:
            print(f"\n✓ Early stopping: Excellent performance at episode {episode}")
            break
    
    env.close()
    
    # Save final models
    torch.save(actor.state_dict(), "cdm_final_actor.pth")
    torch.save(critic.state_dict(), "cdm_final_critic.pth")
    torch.save(metric.state_dict(), "cdm_final_metric.pth")
    torch.save(dynamics.state_dict(), "cdm_final_dynamics.pth")
    
    # Plot results
    plot_comprehensive_results(rewards_history, best_reward)
    
    return actor, critic, metric, dynamics

def plot_comprehensive_results(rewards, best_reward):
    """Plot comprehensive training results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    episodes = range(len(rewards))
    
    # Plot 1: Raw rewards and moving average
    axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.3, linewidth=0.5, label='Raw')
    
    if len(rewards) >= 10:
        window = 10
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, 
                       label=f'{window}-ep MA')
    
    axes[0, 0].axhline(y=-1000, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    axes[0, 0].axhline(y=-800, color='green', linestyle='--', alpha=0.5, label='Good')
    axes[0, 0].axhline(y=-600, color='orange', linestyle='--', alpha=0.5, label='Excellent')
    axes[0, 0].axhline(y=-500, color='red', linestyle='--', alpha=0.5, label='Outstanding')
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend(loc='upper left', fontsize='small')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Success rate over time
    if len(rewards) >= 20:
        success_rates = []
        window = 20
        
        for i in range(len(rewards) - window + 1):
            window_rewards = rewards[i:i+window]
            successes = sum(1 for r in window_rewards if r > -800)
            success_rates.append(successes / window * 100)
        
        axes[0, 1].plot(range(window-1, len(rewards)), success_rates, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_title(f'Success Rate ({window}-ep window)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 100])
    
    # Plot 3: Histogram of rewards
    axes[1, 0].hist(rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axvline(x=-800, color='green', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=-600, color='orange', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=best_reward, color='red', linestyle='-', alpha=0.8, linewidth=2)
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    axes[1, 1].axis('off')
    
    if len(rewards) >= 20:
        first_10_avg = np.mean(rewards[:10])
        last_10_avg = np.mean(rewards[-10:])
        improvement = last_10_avg - first_10_avg
    else:
        first_10_avg = np.mean(rewards)
        last_10_avg = np.mean(rewards)
        improvement = 0
    
    summary_text = f"""
    CDM TRAINING SUMMARY
    
    Statistics:
    - Total Episodes: {len(rewards)}
    - Best Reward: {best_reward:.1f}
    - Final Reward: {rewards[-1]:.1f}
    - Average Reward: {np.mean(rewards):.1f}
    
    Progress:
    - First 10 avg: {first_10_avg:.1f}
    - Last 10 avg: {last_10_avg:.1f}
    - Improvement: {improvement:.1f}
    
    Performance Levels:
    - Rewards > -1000: {sum(1 for r in rewards if r > -1000)} episodes
    - Rewards > -800: {sum(1 for r in rewards if r > -800)} episodes
    - Rewards > -600: {sum(1 for r in rewards if r > -600)} episodes
    - Rewards > -500: {sum(1 for r in rewards if r > -500)} episodes
    
    Assessment:
    """
    
    if last_10_avg > -500:
        assessment = "✅ OUTSTANDING: Excellent balance control"
    elif last_10_avg > -600:
        assessment = "✓ EXCELLENT: Good balance with some oscillation"
    elif last_10_avg > -800:
        assessment = "↻ GOOD: Learning to balance but needs improvement"
    elif improvement > 200:
        assessment = "↻ PROGRESSING: Learning is occurring"
    else:
        assessment = "⚠ NEEDS IMPROVEMENT: Consider tuning parameters"
    
    summary_text += assessment
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Contraction Dynamics Model - Final Results', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('cdm_final_results.png', dpi=100, bbox_inches='tight')
    plt.show()

# ============================
# TEST FUNCTION
# ============================

def test_policy_comprehensive():
    """Comprehensive policy testing"""
    env = gym.make("Pendulum-v1", render_mode="human")
    state_dim = 3
    action_dim = 1
    
    # Load actor
    actor = FixedPolicy(state_dim, action_dim)
    try:
        actor.load_state_dict(torch.load("cdm_best_actor.pth", map_location='cpu'))
        print("✓ Loaded BEST trained policy")
    except:
        try:
            actor.load_state_dict(torch.load("cdm_final_actor.pth", map_location='cpu'))
            print("✓ Loaded final trained policy")
        except:
            print("✗ Could not load any policy")
            return
    
    actor.eval()
    
    print("\n" + "="*70)
    print("COMPREHENSIVE POLICY TESTING")
    print("="*70)
    
    test_results = []
    
    for ep in range(10):
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
            
            # Track upright time (cosθ > 0.8 means very upright)
            if state[0] > 0.8:
                upright_time += 1
            
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Calculate final metrics
        cos_theta = state[0]
        sin_theta = state[1]
        angle = np.arctan2(sin_theta, cos_theta)
        final_velocity = state[2]
        
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
            print("  ✓ BALANCED SUCCESSFULLY")
    
    env.close()
    
    # Summary statistics
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    rewards = [r['reward'] for r in test_results]
    upright_percents = [r['upright_percent'] for r in test_results]
    
    print(f"Average Reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Average Upright Time: {np.mean(upright_percents):.1f}%")
    print(f"Success Rate (reward > -800): {sum(1 for r in rewards if r > -800)/len(rewards)*100:.1f}%")
    print(f"Excellent Rate (reward > -600): {sum(1 for r in rewards if r > -600)/len(rewards)*100:.1f}%")
    
    if np.mean(rewards) > -600:
        print("\n✅ POLICY IS HIGHLY EFFECTIVE")
    elif np.mean(rewards) > -800:
        print("\n✓ POLICY IS EFFECTIVE")
    else:
        print("\n⚠ POLICY NEEDS IMPROVEMENT")

# ============================
# MAIN EXECUTION
# ============================

if __name__ == "__main__":
    print("="*70)
    print("CONTRACTION DYNAMICS MODEL - FINAL VERSION")
    print("="*70)
    print("Features:")
    print("1. Actor-Critic with Contraction Bonus")
    print("2. Stable Gradient Flow")
    print("3. Proper Network Separation")
    print("4. Comprehensive Evaluation")
    print("="*70)
    
    try:
        # Train the model
        print("\nStarting training phase...")
        actor, critic, metric, dynamics = train_cdm_stable()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        
        # Test the trained policy
        print("\nStarting testing phase...")
        test_policy_comprehensive()
        
        print("\n" + "="*70)
        print("ALL OPERATIONS COMPLETE")
        print("="*70)
        print("\nGenerated files:")
        print("  - cdm_best_actor.pth (best performing policy)")
        print("  - cdm_final_actor.pth (final policy)")
        print("  - cdm_final_critic.pth")
        print("  - cdm_final_metric.pth")
        print("  - cdm_final_dynamics.pth")
        print("  - cdm_final_results.png (training plots)")
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()