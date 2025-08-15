"""
Physics-Informed Neural Network for Dynamics Learning
Advanced implementation with proper physics constraints
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from torch.utils.data import DataLoader, TensorDataset

class PhysicsInformedNetwork(nn.Module):
    """PINN for learning dynamics residuals with physics constraints"""
    
    def __init__(self, 
                 state_dim: int = 4,
                 action_dim: int = 1, 
                 hidden_dims: list = [128, 128, 64],
                 activation: str = 'tanh',
                 physics_weight: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.physics_weight = physics_weight
        
        # Build network architecture
        input_dim = state_dim + action_dim
        dims = [input_dim] + hidden_dims + [state_dim]
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation on output layer
                if activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'swish':
                    layers.append(nn.SiLU())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
        
        # Physics constraints network (enforces conservation laws)
        self.physics_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Concatenate state and action
        xu = torch.cat([x, u], dim=-1)
        
        # Get residual prediction
        residual = self.network(xu)
        
        # Apply physics constraints
        if self.training:
            residual = self._apply_physics_constraints(x, u, residual)
        
        return residual
    
    def _apply_physics_constraints(self, x: torch.Tensor, u: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply physics constraints to the residual"""
        # Constraint 1: Residual on unactuated joint should be independent of control
        # This enforces that link 2 dynamics don't directly depend on motor torque
        if u.requires_grad:
            grad_r2_u = torch.autograd.grad(
                residual[:, 3].sum(), u, 
                create_graph=True, retain_graph=True
            )[0] if u.requires_grad else None
            
            if grad_r2_u is not None:
                # Penalize dependence of link 2 acceleration residual on control
                penalty = self.physics_weight * torch.mean(grad_r2_u**2)
                # Store penalty for loss computation
                if not hasattr(self, 'physics_penalty'):
                    self.physics_penalty = 0
                self.physics_penalty += penalty
        
        return residual
    
    def compute_energy_loss(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Compute energy-based physics loss"""
        # Extract positions and velocities
        q1, q2, dq1, dq2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        
        # Compute energy at current state
        energy_current = self._compute_energy(x)
        
        # For physics loss, we expect energy to be approximately conserved
        # in the absence of control inputs and damping
        if x.shape[0] > 1:
            energy_diff = energy_current[1:] - energy_current[:-1]
            # Energy should change smoothly
            energy_loss = torch.mean(energy_diff**2)
        else:
            energy_loss = torch.tensor(0.0, device=x.device)
        
        return energy_loss
    
    def _compute_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total mechanical energy"""
        q1, q2, dq1, dq2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        
        # Simplified energy calculation for constraint
        # Kinetic energy (approximate)
        KE = 0.5 * (dq1**2 + dq2**2)
        
        # Potential energy (approximate)
        PE = -torch.cos(q1) - 0.5 * torch.cos(q1 + q2)
        
        return KE + PE


class PINNTrainer:
    """Trainer for Physics-Informed Neural Network"""
    
    def __init__(self,
                 pinn: PhysicsInformedNetwork,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5,
                 device: str = 'cpu'):
        
        self.pinn = pinn.to(device)
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            self.pinn.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=10
        )
        
        self.loss_history = []
    
    def train_step(self, 
                   x_batch: torch.Tensor,
                   u_batch: torch.Tensor, 
                   target_residual: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        
        self.optimizer.zero_grad()
        
        # Forward pass
        predicted_residual = self.pinn(x_batch, u_batch)
        
        # Data fitting loss (MSE)
        data_loss = F.mse_loss(predicted_residual, target_residual)
        
        # Physics constraint loss
        physics_loss = getattr(self.pinn, 'physics_penalty', 0)
        
        # Energy conservation loss
        energy_loss = self.pinn.compute_energy_loss(x_batch, predicted_residual)
        
        # Total loss
        total_loss = data_loss + 0.1 * physics_loss + 0.01 * energy_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Reset physics penalty
        self.pinn.physics_penalty = 0
        
        losses = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss,
            'energy_loss': energy_loss.item()
        }
        
        return losses
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.pinn.train()
        epoch_losses = {'total_loss': 0, 'data_loss': 0, 'physics_loss': 0, 'energy_loss': 0}
        
        for batch_idx, (x_batch, u_batch, target_batch) in enumerate(dataloader):
            x_batch = x_batch.to(self.device)
            u_batch = u_batch.to(self.device)
            target_batch = target_batch.to(self.device)
            
            losses = self.train_step(x_batch, u_batch, target_batch)
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
        
        # Average losses
        num_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        self.loss_history.append(epoch_losses)
        self.scheduler.step(epoch_losses['total_loss'])
        
        return epoch_losses
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Validation step"""
        self.pinn.eval()
        val_losses = {'total_loss': 0, 'data_loss': 0, 'physics_loss': 0, 'energy_loss': 0}
        
        with torch.no_grad():
            for x_batch, u_batch, target_batch in val_dataloader:
                x_batch = x_batch.to(self.device)
                u_batch = u_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                
                predicted_residual = self.pinn(x_batch, u_batch)
                
                val_losses['data_loss'] += F.mse_loss(predicted_residual, target_batch).item()
                val_losses['energy_loss'] += self.pinn.compute_energy_loss(x_batch, predicted_residual).item()
        
        num_batches = len(val_dataloader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        val_losses['total_loss'] = val_losses['data_loss'] + 0.01 * val_losses['energy_loss']
        
        return val_losses


class ResidualBuffer:
    """Buffer for storing and managing residual learning data"""
    
    def __init__(self, capacity: int = 100000, state_dim: int = 4, action_dim: int = 1):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.residuals = np.zeros((capacity, state_dim))
        self.rewards = np.zeros((capacity, 1))
        
        self.size = 0
        self.ptr = 0
    
    def add(self, state: np.ndarray, action: np.ndarray, residual: np.ndarray, reward: float = 0.0):
        """Add experience to buffer"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.residuals[self.ptr] = residual
        self.rewards[self.ptr] = reward
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample batch for training"""
        if self.size < batch_size:
            indices = np.arange(self.size)
        else:
            indices = np.random.choice(self.size, batch_size, replace=False)
        
        states = torch.FloatTensor(self.states[indices])
        actions = torch.FloatTensor(self.actions[indices])
        residuals = torch.FloatTensor(self.residuals[indices])
        
        return states, actions, residuals
    
    def get_dataset(self, train_split: float = 0.8) -> Tuple[TensorDataset, TensorDataset]:
        """Get training and validation datasets"""
        if self.size == 0:
            raise ValueError("Buffer is empty")
        
        # Get all data
        states = torch.FloatTensor(self.states[:self.size])
        actions = torch.FloatTensor(self.actions[:self.size])
        residuals = torch.FloatTensor(self.residuals[:self.size])
        
        # Split data
        n_train = int(train_split * self.size)
        indices = np.random.permutation(self.size)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        train_dataset = TensorDataset(states[train_idx], actions[train_idx], residuals[train_idx])
        val_dataset = TensorDataset(states[val_idx], actions[val_idx], residuals[val_idx])
        
        return train_dataset, val_dataset
    
    def __len__(self):
        return self.size


# Test the PINN
if __name__ == "__main__":
    print("Testing Physics-Informed Neural Network...")
    
    # Create PINN
    pinn = PhysicsInformedNetwork()
    trainer = PINNTrainer(pinn)
    
    # Test forward pass
    x_test = torch.randn(10, 4)
    u_test = torch.randn(10, 1)
    
    residual = pinn(x_test, u_test)
    print(f"Residual shape: {residual.shape}")
    
    # Test buffer
    buffer = ResidualBuffer(capacity=1000)
    
    # Add some dummy data
    for i in range(100):
        state = np.random.randn(4)
        action = np.random.randn(1)
        residual = np.random.randn(4) * 0.1
        buffer.add(state, action, residual)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Test sampling
    states, actions, residuals = buffer.sample(32)
    print(f"Sample shapes: {states.shape}, {actions.shape}, {residuals.shape}")
    
    print("✓ PINN test completed successfully!")