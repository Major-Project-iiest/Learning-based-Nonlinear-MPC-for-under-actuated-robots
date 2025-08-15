"""
Soft Actor-Critic (SAC) Implementation
Modern implementation with entropy regularization and target networks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple, Optional, List
import copy
from collections import deque

class ReplayBuffer:
    """Prioritized Experience Replay Buffer for SAC"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = 0.4     # Importance sampling exponent (will be annealed)
        self.beta_increment = 0.001
        
        # Pre-allocate memory
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        # Priority arrays
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        self.size = 0
        self.ptr = 0
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool, td_error: float = None):
        """Add experience to buffer"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        # Set priority
        priority = self.max_priority if td_error is None else abs(td_error) + 1e-6
        self.priorities[self.ptr] = priority ** self.alpha
        self.max_priority = max(self.max_priority, priority)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch with importance sampling weights"""
        if self.size < batch_size:
            indices = np.arange(self.size)
        else:
            # Sample according to priorities
            probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
            indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Convert to tensors
        states = torch.FloatTensor(self.states[indices])
        actions = torch.FloatTensor(self.actions[indices])
        rewards = torch.FloatTensor(self.rewards[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        dones = torch.FloatTensor(self.dones[indices])
        is_weights = torch.FloatTensor(weights)
        
        return states, actions, rewards, next_states, dones, is_weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self):
        return self.size


class Actor(nn.Module):
    """SAC Actor network with reparameterization trick"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256],
                 max_action: float = 1.0, log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared layers
        layers = []
        dims = [state_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU()
            ])
        
        self.shared_net = nn.Sequential(*layers)
        
        # Output layers
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        x = self.shared_net(state)
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action using reparameterization trick"""
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = torch.tanh(mean) * self.max_action
            log_prob = None
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            
            # Reparameterization trick
            z = normal.rsample()  # Sample with gradients
            action = torch.tanh(z) * self.max_action
            
            # Compute log probability with correction for tanh squashing
            log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) / (self.max_action**2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """SAC Critic network (Twin Q-networks)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        # Q1 network
        q1_layers = []
        dims = [state_dim + action_dim] + hidden_dims + [1]
        for i in range(len(dims) - 1):
            q1_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                q1_layers.append(nn.ReLU())
        
        self.Q1 = nn.Sequential(*q1_layers)
        
        # Q2 network
        q2_layers = []
        for i in range(len(dims) - 1):
            q2_layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                q2_layers.append(nn.ReLU())
        
        self.Q2 = nn.Sequential(*q2_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both Q networks"""
        sa = torch.cat([state, action], dim=-1)
        
        q1 = self.Q1(sa)
        q2 = self.Q2(sa)
        
        return q1, q2
    
    def Q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q1 only"""
        sa = torch.cat([state, action], dim=-1)
        return self.Q1(sa)


class SAC:
    """Soft Actor-Critic Algorithm"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 max_action: float = 1.0,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 auto_entropy_tuning: bool = True,
                 target_entropy: float = None,
                 device: str = 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action=max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            self.target_entropy = target_entropy if target_entropy else -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha, device=self.device)
        
        # Training stats
        self.training_stats = {
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'alpha_loss': deque(maxlen=100),
            'alpha': deque(maxlen=100),
            'q1_value': deque(maxlen=100),
            'q2_value': deque(maxlen=100),
        }
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _ = self.actor.sample(state_tensor, deterministic=deterministic)
        
        return action.cpu().numpy()[0]
    
    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> Dict[str, float]:
        """Update networks using batch from replay buffer"""
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones, is_weights, indices = replay_buffer.sample(batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        is_weights = is_weights.to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next_target, q2_next_target = self.critic_target(next_states, next_actions)
            q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_next_target
        
        current_q1, current_q2 = self.critic(states, actions)
        
        # Compute TD errors for priority update
        td_errors = (current_q1 - target_q).abs().detach().cpu().numpy().flatten()
        replay_buffer.update_priorities(indices, td_errors)
        
        # Weighted critic loss
        q1_loss = (is_weights * F.mse_loss(current_q1, target_q, reduction='none')).mean()
        q2_loss = (is_weights * F.mse_loss(current_q2, target_q, reduction='none')).mean()
        critic_loss = q1_loss + q2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Update Actor
        sampled_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, sampled_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Update Alpha (entropy temperature)
        alpha_loss = torch.tensor(0.0)
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Soft update target network
        self._soft_update(self.critic_target, self.critic, self.tau)
        
        # Store training stats
        self.training_stats['actor_loss'].append(actor_loss.item())
        self.training_stats['critic_loss'].append(critic_loss.item())
        self.training_stats['alpha_loss'].append(alpha_loss.item())
        self.training_stats['alpha'].append(self.alpha.item())
        self.training_stats['q1_value'].append(current_q1.mean().item())
        self.training_stats['q2_value'].append(current_q2.mean().item())
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'q1_value': current_q1.mean().item(),
            'q2_value': current_q2.mean().item(),
        }
    
    def _soft_update(self, target_net: nn.Module, source_net: nn.Module, tau: float):
        """Soft update target network"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None,
        }, filepath)
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha.data = checkpoint['log_alpha'].data
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp()
    
    def get_stats(self) -> Dict[str, float]:
        """Get training statistics"""
        stats = {}
        for key, values in self.training_stats.items():
            if values:
                stats[f'mean_{key}'] = np.mean(values)
                stats[f'std_{key}'] = np.std(values)
        return stats


# Test the SAC implementation
if __name__ == "__main__":
    print("Testing SAC implementation...")
    
    # Create SAC agent
    sac = SAC(state_dim=4, action_dim=1, device='cpu')
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=10000, state_dim=4, action_dim=1)
    
    # Add some dummy data
    for _ in range(1000):
        state = np.random.randn(4)
        action = np.random.randn(1)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.random() > 0.9
        
        replay_buffer.add(state, action, reward, next_state, done)
    
    # Test action selection
    test_state = np.random.randn(4)
    action = sac.select_action(test_state)
    print(f"Selected action: {action}")
    
    # Test update
    if len(replay_buffer) >= 256:
        losses = sac.update(replay_buffer, batch_size=256)
        print(f"Training losses: {losses}")
    
    print("✓ SAC test completed successfully!")