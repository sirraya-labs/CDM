import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        return torch.clamp(value, -50, 0)

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
    """Full Riemannian metric M(x) = L(x)L(x)^T + epsilon*I"""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim
        # We need (n*(n+1))/2 outputs to fill a lower triangular matrix
        self.output_dim = (state_dim * (state_dim + 1)) // 2
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )
        self.epsilon = 0.1
        self.softplus = nn.Softplus()

    def forward(self, x):
        batch_size = x.shape[0]
        l_params = self.net(x)
        
        # 1. Initialize L as a batch of zero matrices
        L = torch.zeros(batch_size, self.state_dim, self.state_dim, device=x.device)
        
        # 2. Fill L with lower triangular values
        # We use softplus on the diagonals to ensure they are strictly positive
        k = 0
        for i in range(self.state_dim):
            for j in range(i + 1):
                val = l_params[:, k]
                if i == j:
                    L[:, i, j] = self.softplus(val) + 0.1  # Strictly positive diagonal
                else:
                    L[:, i, j] = val
                k += 1
        
        # 3. Compute M = L*L^T + epsilon*I
        # This guarantees positive definiteness
        LT = L.transpose(1, 2)
        M = torch.bmm(L, LT) + self.epsilon * torch.eye(self.state_dim, device=x.device).unsqueeze(0)
        
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

def compute_contraction_rate(states, next_states, metric, lambda_min=0.01):
    """Compute contraction rate: x^T M(x) x - x'^T M(x') x'"""
    energy_curr, M_curr = compute_energy(states, metric)
    energy_next, M_next = compute_energy(next_states, metric)
    
    # Ensure contraction: energy_curr should decrease to energy_next
    delta_energy = energy_curr - energy_next
    
    # Add small regularization to ensure positive definiteness
    M_curr_det = torch.det(M_curr)
    M_next_det = torch.det(M_next)
    
    # Penalty for non-positive determinants (shouldn't happen with our construction)
    det_penalty = F.relu(-M_curr_det + lambda_min) + F.relu(-M_next_det + lambda_min)
    
    return delta_energy, energy_curr, energy_next, det_penalty.mean()

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
        # Ornstein-Uhlenbeck process for correlated noise
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
    """Complete working version with full Riemannian metric"""
    
    env = gym.make("Pendulum-v1")
    state_dim = 3
    action_dim = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize networks
    actor = FixedPolicy(state_dim, action_dim, hidden_dim=128).to(device)
    critic = FixedValueNetwork(state_dim, hidden_dim=128).to(device)
    dynamics = SimpleDynamics(state_dim, action_dim, hidden_dim=64).to(device)
    metric = ContractionMetric(state_dim, hidden_dim=64).to(device)
    
    # Target networks
    target_actor = FixedPolicy(state_dim, action_dim, hidden_dim=128).to(device)
    target_critic = FixedValueNetwork(state_dim, hidden_dim=128).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    
    # Optimizers with stable settings
    actor_optim = optim.Adam(actor.parameters(), lr=2e-4, weight_decay=1e-5)
    critic_optim = optim.Adam(critic.parameters(), lr=5e-4, weight_decay=1e-5)
    dynamics_optim = optim.Adam(dynamics.parameters(), lr=1e-3, weight_decay=1e-5)
    metric_optim = optim.Adam(metric.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=30000)
    
    # Training parameters
    gamma = 0.99
    tau = 0.005
    batch_size = 128
    lambda_contraction = 0.1  # Contraction weight
    
    # Improved adaptive noise
    adaptive_noise = ImprovedAdaptiveNoise(action_dim, base_sigma=0.4)
    
    print("\nStarting COMPLETE CDM training with Full Riemannian Metric...")
    print("="*70)
    print(f"{'Episode':>8} {'Reward':>10} {'Energy':>8} {'Contr':>6} {'Noise':>7} {'Best':>8}")
    print("="*70)
    
    rewards_history = []
    energy_history = []
    contraction_history = []
    best_reward = -float('inf')
    best_actor_state = None
    
    for episode in range(300):
        state, _ = env.reset()
        adaptive_noise.reset()
        
        episode_reward = 0
        episode_energy = 0
        episode_contraction = 0
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
        
        # Update best reward and save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_actor_state = actor.state_dict().copy()
            torch.save({
                'episode': episode,
                'reward': episode_reward,
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'metric': metric.state_dict(),
                'dynamics': dynamics.state_dict()
            }, "cdm_best_checkpoint.pth")
        
        # Train networks if we have enough samples
        if len(replay_buffer) > batch_size * 2:
            # Sample batch from replay buffer
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            # Convert to tensors
            states_t = torch.FloatTensor(states).to(device)  # [batch, 3]
            actions_t = torch.FloatTensor(actions).to(device).unsqueeze(1)  # [batch, 1]
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
            next_states_pred = dynamics(states_t, actions_t)
            dynamics_loss = nn.MSELoss()(next_states_pred, next_states_t)
            dynamics_loss.backward()
            torch.nn.utils.clip_grad_norm_(dynamics.parameters(), 1.0)
            dynamics_optim.step()
            
            # ===== 3. TRAIN METRIC =====
            metric_optim.zero_grad()
            
            # Compute energies for current states
            energy_curr, M_curr = compute_energy(states_t, metric)
            
            # Compute predicted next states (using dynamics model)
            with torch.no_grad():
                actor_actions = actor(states_t)
                next_states_pred = dynamics(states_t, actor_actions)
            
            # Compute energies for predicted next states
            energy_next, M_next = compute_energy(next_states_pred, metric)
            
            # Contraction loss: encourage energy decrease along trajectories
            contraction_loss = F.relu(energy_curr - energy_next + 0.1).mean()
            
            # Regularization: encourage well-conditioned metrics
            identity = torch.eye(state_dim, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            reg_loss = torch.norm(M_curr - identity, dim=(1, 2)).mean() * 0.01
            
            # Determinant penalty: ensure positive definiteness
            det_curr = torch.det(M_curr)
            det_penalty = F.relu(0.01 - det_curr).mean()
            
            metric_loss = contraction_loss + reg_loss + det_penalty * 0.1
            metric_loss.backward()
            torch.nn.utils.clip_grad_norm_(metric.parameters(), 0.5)
            metric_optim.step()
            
            # Store for logging
            episode_energy = energy_curr.mean().item()
            episode_contraction = (energy_curr - energy_next).mean().item()
            
            # ===== 4. TRAIN ACTOR WITH CONTRACTION =====
            actor_optim.zero_grad()
            
            # Get actor's actions
            actor_actions = actor(states_t)  # [batch, 1]
            
            # Get value estimates
            actor_values = critic(states_t)
            
            # Compute contraction bonus using the learned metric
            with torch.no_grad():
                # Predict next state using dynamics
                next_states_pred = dynamics(states_t, actor_actions)
                # Compute energies
                energy_curr, _ = compute_energy(states_t, metric)
                energy_next, _ = compute_energy(next_states_pred, metric)
                # Contraction measure
                delta_energy = energy_curr - energy_next
                contraction_bonus = torch.tanh(delta_energy / 5.0).mean()
            
            # Velocity penalty (encourage controlled movement)
            velocity = states_t[:, 2] * 8.0  # De-normalize
            velocity_penalty = F.relu(torch.abs(velocity) - 2.0).mean() * 0.01
            
            # Action smoothness penalty
            action_penalty = torch.abs(actor_actions).mean() * 0.001
            
            # Angle penalty (encourage upright)
            cos_theta = states_t[:, 0]
            sin_theta = states_t[:, 1]
            angle = torch.atan2(sin_theta, cos_theta)
            angle_penalty = torch.abs(angle).mean() * 0.005
            
            # Total actor loss
            actor_loss = -actor_values.mean() * 0.1  # Conservative scaling
            actor_loss -= lambda_contraction * contraction_bonus * 0.5  # Contraction influence
            actor_loss += velocity_penalty + action_penalty + angle_penalty
            
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_optim.step()
            
            # ===== 5. UPDATE TARGET NETWORKS =====
            soft_update(target_actor, actor, tau)
            soft_update(target_critic, critic, tau)
            
            # Store metrics for history
            energy_history.append(episode_energy)
            contraction_history.append(episode_contraction)
        
        # Log progress
        if episode % 10 == 0 or episode < 20 or episode == 299:
            print(f"{episode:8d} {episode_reward:10.1f} {episode_energy:8.2f} "
                  f"{episode_contraction:6.2f} {adaptive_noise.sigma:7.2f} {best_reward:8.1f}")
            
            # Performance feedback
            if episode_reward > -600:
                print(f"  ✅ OUTSTANDING! Reward: {episode_reward:.1f}")
            elif episode_reward > -800:
                print(f"  ↻ Good progress")
            elif episode_reward == best_reward:
                print(f"  ★ NEW BEST!")
        
        # Early stopping if consistently good
        if episode >= 100 and episode_reward > -600:
            print(f"\n✅ Early stopping: Excellent performance at episode {episode}")
            break
    
    env.close()
    
    # Save final models
    torch.save(actor.state_dict(), "cdm_final_actor.pth")
    torch.save(critic.state_dict(), "cdm_final_critic.pth")
    torch.save(metric.state_dict(), "cdm_final_metric.pth")
    torch.save(dynamics.state_dict(), "cdm_final_dynamics.pth")
    
    # Plot comprehensive results
    plot_complete_results(rewards_history, energy_history, contraction_history, best_reward)
    
    return actor, critic, metric, dynamics

def plot_complete_results(rewards, energies, contractions, best_reward):
    """Plot complete training results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    episodes = range(len(rewards))
    
    # Plot 1: Raw rewards
    axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.3, linewidth=0.5)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Raw Training Rewards')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Energy and Contraction
    if energies and contractions:
        energy_window = min(50, len(energies))
        if len(energies) >= energy_window:
            energy_smooth = np.convolve(energies, np.ones(energy_window)/energy_window, mode='valid')
            contraction_smooth = np.convolve(contractions, np.ones(energy_window)/energy_window, mode='valid')
            
            ax2 = axes[0, 1].twinx()
            line1, = axes[0, 1].plot(range(energy_window-1, len(energies)), energy_smooth, 'g-', linewidth=2, label='Energy')
            line2, = ax2.plot(range(energy_window-1, len(contractions)), contraction_smooth, 'r-', linewidth=2, label='Contraction')
            
            axes[0, 1].set_xlabel('Training Step')
            axes[0, 1].set_ylabel('Energy', color='g')
            ax2.set_ylabel('Contraction', color='r')
            axes[0, 1].set_title('Metric Learning Progress')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Combined legend
            lines = [line1, line2]
            labels = [l.get_label() for l in lines]
            axes[0, 1].legend(lines, labels, loc='upper left')
    
    # Plot 3: Histogram
    axes[0, 2].hist(rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 2].axvline(x=-1000, color='red', linestyle='--', alpha=0.5, label='Poor')
    axes[0, 2].axvline(x=-800, color='orange', linestyle='--', alpha=0.5, label='Good')
    axes[0, 2].axvline(x=-600, color='green', linestyle='--', alpha=0.5, label='Excellent')
    axes[0, 2].axvline(x=best_reward, color='red', linestyle='-', alpha=0.8, linewidth=2, label='Best')
    axes[0, 2].set_xlabel('Reward')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Reward Distribution')
    axes[0, 2].legend()
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
        last_10 = rewards[-10:] if len(rewards) >= 10 else rewards[-len(rewards):]
        
        summary_text = f"""
        RIEMANNIAN CDM TRAINING SUMMARY
        
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
        - > -800 (Good): {sum(1 for r in rewards if r > -800)} ({sum(1 for r in rewards if r > -800)/len(rewards)*100:.1f}%)
        - > -600 (Excellent): {sum(1 for r in rewards if r > -600)} ({sum(1 for r in rewards if r > -600)/len(rewards)*100:.1f}%)
        
        Metric Learning:
        - Avg Energy: {np.mean(energies) if energies else 0:.2f}
        - Avg Contraction: {np.mean(contractions) if contractions else 0:.2f}
        
        Assessment:
        """
        
        avg_last_10 = np.mean(last_10)
        if avg_last_10 > -500:
            assessment = "✅ OUTSTANDING: Perfect control!"
        elif avg_last_10 > -600:
            assessment = "✅ EXCELLENT: Very good balance"
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
    
    plt.suptitle('Contraction Dynamics Model - Riemannian Metric Training Results', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('cdm_riemannian_results.png', dpi=100, bbox_inches='tight')
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
        print(f"✅ Loaded BEST model from episode {episode_num}")
        print(f"  Training reward: {reward:.1f}")
    except:
        try:
            actor.load_state_dict(torch.load("cdm_best_actor.pth", map_location='cpu', weights_only=False))
            print("✅ Loaded best actor")
        except:
            try:
                actor.load_state_dict(torch.load("cdm_final_actor.pth", map_location='cpu', weights_only=False))
                print("✅ Loaded final actor")
            except:
                print("❌ Could not load any model")
                return
    
    actor.eval()
    
    print("\n" + "="*70)
    print("RIEMANNIAN CDM DEMONSTRATION")
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
            print("  ✅ SUCCESS: Pendulum balanced!")
    
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
        print("\n✅ RIEMANNIAN CDM IS HIGHLY EFFECTIVE!")
    elif np.mean(rewards) > -800:
        print("\n✅ CDM IS EFFECTIVE")
    else:
        print("\n⚠ CDM NEEDS IMPROVEMENT")

# ============================
# METRIC ANALYSIS FUNCTION
# ============================

def analyze_metric():
    """Analyze the learned Riemannian metric"""
    try:
        metric = ContractionMetric(3, hidden_dim=64)
        metric.load_state_dict(torch.load("cdm_final_metric.pth", map_location='cpu', weights_only=False))
        metric.eval()
        print("✅ Loaded learned metric")
        
        # Analyze metric at different states
        test_states = [
            [1.0, 0.0, 0.0],     # Upright, zero velocity
            [-1.0, 0.0, 0.0],    # Downward, zero velocity
            [0.0, 1.0, 0.0],     # Horizontal
            [0.0, -1.0, 0.0],    # Horizontal opposite
            [0.7, 0.7, 0.5],     # Diagonal
        ]
        
        print("\n" + "="*70)
        print("METRIC ANALYSIS")
        print("="*70)
        
        for i, state in enumerate(test_states):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                M = metric(state_tensor)[0]
                energy, _ = compute_energy(state_tensor, metric)
            
            print(f"\nState {i+1}: [cosθ={state[0]:.2f}, sinθ={state[1]:.2f}, θ̇={state[2]:.2f}]")
            print(f"Energy: {energy.item():.4f}")
            print("Metric M(x):")
            for row in M.numpy():
                print("  [" + " ".join([f"{val:8.4f}" for val in row]) + "]")
            
            # Compute eigenvalues
            eigenvalues = torch.linalg.eigvals(M).real
            print(f"Eigenvalues: {eigenvalues.numpy()}")
            print(f"Condition number: {eigenvalues.max()/eigenvalues.min():.2f}")
            print(f"Determinant: {torch.det(M.unsqueeze(0)).item():.6f}")
            
    except Exception as e:
        print(f"❌ Could not analyze metric: {e}")

# ============================
# MAIN EXECUTION
# ============================

if __name__ == "__main__":
    print("="*70)
    print("CONTRACTION DYNAMICS MODEL - RIEMANNIAN METRIC VERSION")
    print("="*70)
    print("Robust Features:")
    print("1. Full Riemannian metric M(x) = L(x)L(x)^T + εI")
    print("2. Guaranteed positive definiteness")
    print("3. Adaptive noise with performance-based adjustment")
    print("4. Conservative actor updates")
    print("5. Comprehensive metric learning")
    print("6. Detailed analysis tools")
    print("="*70)
    
    try:
        # Train the complete model
        print("\nStarting training with Riemannian metric...")
        actor, critic, metric, dynamics = train_cdm_complete()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        
        # Analyze the learned metric
        analyze_metric()
        
        # Demonstrate the trained model
        print("\nStarting demonstration...")
        demonstrate_cdm()
        
        print("\n" + "="*70)
        print("ALL OPERATIONS COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nOutput files created:")
        print("  - cdm_best_checkpoint.pth (best checkpoint)")
        print("  - cdm_final_actor.pth (final policy)")
        print("  - cdm_final_critic.pth (final critic)")
        print("  - cdm_final_metric.pth (Riemannian metric)")
        print("  - cdm_final_dynamics.pth (final dynamics)")
        print("  - cdm_riemannian_results.png (comprehensive plots)")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Models have been saved at the last checkpoint")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()