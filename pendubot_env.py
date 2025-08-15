"""
Pendubot Gymnasium Environment
Clean implementation with proper physics and rendering
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from pendubot_model import PendubotModel


class PendubotEnv(gym.Env):
    """Pendubot environment for reinforcement learning"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, 
                 dt: float = 0.01,
                 max_episode_steps: int = 1000,
                 reward_type: str = 'swing_up',
                 noise_std: float = 0.0,
                 render_mode: Optional[str] = None):
        
        super().__init__()
        
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.noise_std = noise_std
        self.render_mode = render_mode
        
        # Create Pendubot model
        self.model = PendubotModel()
        
        # Action space: motor torque
        u_min, u_max = self.model.get_control_bounds()
        self.action_space = spaces.Box(low=u_min, high=u_max, dtype=np.float32)
        
        # Observation space: [q1, q2, dq1, dq2]
        x_min, x_max = self.model.get_state_bounds()
        self.observation_space = spaces.Box(low=x_min, high=x_max, dtype=np.float32)
        
        # Target states
        self.equilibria = self.model.get_equilibrium_points()
        self.target_state = self.equilibria['upright']
        
        # Environment state
        self.state = None
        self.steps = 0
        self.episode_return = 0.0
        
        # Rendering
        self.fig = None
        self.ax = None
        self.line1 = None
        self.line2 = None
        self.history = {'states': [], 'actions': [], 'rewards': []}
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        if options and 'initial_state' in options:
            self.state = np.array(options['initial_state'], dtype=np.float32)
        else:
            # Random initial state (downward with small perturbation)
            self.state = np.array([
                np.random.uniform(-0.2, 0.2),  # q1
                np.random.uniform(-0.2, 0.2),  # q2  
                np.random.uniform(-0.5, 0.5),  # dq1
                np.random.uniform(-0.5, 0.5),  # dq2
            ], dtype=np.float32)
        
        self.steps = 0
        self.episode_return = 0.0
        self.history = {'states': [self.state.copy()], 'actions': [], 'rewards': []}
        
        info = {
            'episode': {'l': 0, 'r': 0.0},
            'target_state': self.target_state,
            'current_energy': self.model.energy(self.state)
        }
        
        return self.state.copy(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Simulate dynamics
        self.state = self._simulate_step(self.state, action)
        
        # Add noise if specified
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=self.state.shape)
            self.state += noise
        
        # Compute reward
        reward = self._compute_reward(self.state, action)
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.steps >= self.max_episode_steps
        
        self.steps += 1
        self.episode_return += reward
        
        # Store history
        self.history['states'].append(self.state.copy())
        self.history['actions'].append(action.copy())
        self.history['rewards'].append(reward)
        
        # Info dictionary
        info = {
            'current_energy': self.model.energy(self.state),
            'target_energy': self.model.energy(self.target_state),
            'distance_to_target': np.linalg.norm(self.state - self.target_state),
            'episode': {'l': self.steps, 'r': self.episode_return} if terminated or truncated else {}
        }
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def _simulate_step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Simulate one step using the Pendubot model"""
        # Use model's simulation method
        next_state = self.model.simulate_step(state, action, self.dt)
        
        # Wrap angles to [-π, π]
        next_state[0] = self._wrap_angle(next_state[0])
        next_state[1] = self._wrap_angle(next_state[1])
        
        return next_state.astype(np.float32)
    
    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-π, π]"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def _compute_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Compute reward based on current state and action"""
        
        if self.reward_type == 'swing_up':
            return self._swing_up_reward(state, action)
        elif self.reward_type == 'balance':
            return self._balance_reward(state, action)
        elif self.reward_type == 'energy':
            return self._energy_reward(state, action)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def _swing_up_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Reward for swing-up task"""
        q1, q2, dq1, dq2 = state
        
        # Distance to upright position
        target_q1, target_q2 = np.pi, 0.0
        
        # Angular distance (considering wrapping)
        angle_dist_1 = abs(self._wrap_angle(q1 - target_q1))
        angle_dist_2 = abs(self._wrap_angle(q2 - target_q2))
        
        # Position reward (exponential decay with distance)
        pos_reward = np.exp(-2 * (angle_dist_1 + angle_dist_2))
        
        # Velocity penalty (want low velocities at target)
        vel_penalty = -0.1 * (dq1**2 + dq2**2)
        
        # Control effort penalty
        control_penalty = -0.001 * action[0]**2
        
        # Bonus for being close to upright
        if angle_dist_1 < 0.2 and angle_dist_2 < 0.2:
            upright_bonus = 10.0 * np.exp(-10 * (angle_dist_1 + angle_dist_2))
        else:
            upright_bonus = 0.0
        
        # Stability bonus (low velocities when upright)
        if angle_dist_1 < 0.1 and angle_dist_2 < 0.1:
            stability_bonus = 5.0 * np.exp(-5 * (dq1**2 + dq2**2))
        else:
            stability_bonus = 0.0
        
        total_reward = pos_reward + vel_penalty + control_penalty + upright_bonus + stability_bonus
        
        return float(total_reward)
    
    def _balance_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Reward for balancing task (assuming already near upright)"""
        q1, q2, dq1, dq2 = state
        
        # Distance from upright
        target_q1, target_q2 = np.pi, 0.0
        angle_dist_1 = abs(self._wrap_angle(q1 - target_q1))
        angle_dist_2 = abs(self._wrap_angle(q2 - target_q2))
        
        # High penalty for being far from upright
        if angle_dist_1 > 0.5 or angle_dist_2 > 0.5:
            return -100.0
        
        # Quadratic cost for deviations
        pos_cost = -(angle_dist_1**2 + angle_dist_2**2)
        vel_cost = -0.1 * (dq1**2 + dq2**2)
        control_cost = -0.001 * action[0]**2
        
        # Bonus for perfect balance
        if angle_dist_1 < 0.05 and angle_dist_2 < 0.05 and abs(dq1) < 0.1 and abs(dq2) < 0.1:
            balance_bonus = 1.0
        else:
            balance_bonus = 0.0
        
        return float(pos_cost + vel_cost + control_cost + balance_bonus)
    
    def _energy_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """Energy-based reward for swing-up"""
        current_energy = self.model.energy(state)
        target_energy = self.model.energy(self.target_state)
        
        # Energy error
        energy_error = abs(current_energy - target_energy)
        energy_reward = -energy_error
        
        # Control penalty
        control_penalty = -0.001 * action[0]**2
        
        # Bonus for correct energy and position
        q1, q2 = state[0], state[1]
        angle_dist_1 = abs(self._wrap_angle(q1 - np.pi))
        angle_dist_2 = abs(self._wrap_angle(q2 - 0.0))
        
        if energy_error < 0.5 and angle_dist_1 < 0.2 and angle_dist_2 < 0.2:
            success_bonus = 10.0
        else:
            success_bonus = 0.0
        
        return float(energy_reward + control_penalty + success_bonus)
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        # Terminate if successfully balanced for swing-up task
        if self.reward_type == 'swing_up':
            q1, q2, dq1, dq2 = self.state
            angle_dist_1 = abs(self._wrap_angle(q1 - np.pi))
            angle_dist_2 = abs(self._wrap_angle(q2 - 0.0))
            
            # Success condition: close to upright with low velocities
            if (angle_dist_1 < 0.1 and angle_dist_2 < 0.1 and 
                abs(dq1) < 0.2 and abs(dq2) < 0.2):
                return True
        
        # Check for unsafe states (very high velocities)
        if np.any(np.abs(self.state[2:]) > 15.0):
            return True
        
        return False
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
        
        if self.fig is None:
            self._init_render()
        
        self._update_render()
        
        if self.render_mode == "human":
            plt.pause(0.01)
        elif self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return buf
    
    def _init_render(self):
        """Initialize rendering"""
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Pendubot')
        
        # Create pendulum links
        self.line1, = self.ax.plot([], [], 'b-', linewidth=8, label='Link 1')
        self.line2, = self.ax.plot([], [], 'r-', linewidth=6, label='Link 2')
        
        # Joint markers
        self.joint1 = plt.Circle((0, 0), 0.05, color='black', zorder=10)
        self.joint2 = plt.Circle((0, 0), 0.03, color='gray', zorder=10)
        self.ax.add_patch(self.joint1)
        self.ax.add_patch(self.joint2)
        
        # Target position indicator
        target_x, target_y = self._get_end_effector_pos(self.target_state)
        self.target_marker = plt.Circle((target_x, target_y), 0.08, 
                                       color='green', alpha=0.5, label='Target')
        self.ax.add_patch(self.target_marker)
        
        self.ax.legend()
        plt.ion()
    
    def _update_render(self):
        """Update rendering with current state"""
        q1, q2 = self.state[0], self.state[1]
        l1, l2 = self.model.params['l1'], self.model.params['l2']
        
        # Link 1 coordinates
        x1 = l1 * np.sin(q1)
        y1 = -l1 * np.cos(q1)
        
        # Link 2 coordinates
        x2 = x1 + l2 * np.sin(q1 + q2)
        y2 = y1 - l2 * np.cos(q1 + q2)
        
        # Update lines
        self.line1.set_data([0, x1], [0, y1])
        self.line2.set_data([x1, x2], [y1, y2])
        
        # Update joint positions
        self.joint2.center = (x1, y1)
        
        # Update title with current info
        energy = self.model.energy(self.state)
        self.ax.set_title(f'Pendubot - Step: {self.steps}, Energy: {energy:.2f}')
    
    def _get_end_effector_pos(self, state: np.ndarray) -> Tuple[float, float]:
        """Get end effector position for given state"""
        q1, q2 = state[0], state[1]
        l1, l2 = self.model.params['l1'], self.model.params['l2']
        
        x1 = l1 * np.sin(q1)
        y1 = -l1 * np.cos(q1)
        x2 = x1 + l2 * np.sin(q1 + q2)
        y2 = y1 - l2 * np.cos(q1 + q2)
        
        return x2, y2
    
    def close(self):
        """Close rendering"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            plt.ioff()
    
    def get_episode_data(self) -> Dict:
        """Get episode history for analysis"""
        return {
            'states': np.array(self.history['states']),
            'actions': np.array(self.history['actions']),
            'rewards': np.array(self.history['rewards']),
            'episode_length': self.steps,
            'episode_return': self.episode_return
        }


# Test the environment
if __name__ == "__main__":
    print("Testing Pendubot Environment...")
    
    # Create environment
    env = PendubotEnv(render_mode="human", reward_type="swing_up")
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Test random actions
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        
        print(f"Step {i}: reward={reward:.3f}, terminated={terminated}")
        
        if terminated or truncated:
            print("Episode finished!")
            break
    
    # Get episode data
    episode_data = env.get_episode_data()
    print(f"Episode length: {episode_data['episode_length']}")
    print(f"Total return: {episode_data['episode_return']:.3f}")
    
    env.close()
    print("✓ Environment test completed successfully!")