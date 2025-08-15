"""
Utility functions for NMPC-SAC Pendubot project
"""
import os
import random
import logging
import numpy as np
import torch
import yaml
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_config(config: dict, save_path: str):
    """Save configuration to file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_config(config_path: str) -> dict:
    """Load configuration from file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_dir(directory: str):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)


class Logger:
    """Custom logger for training"""
    
    def __init__(self, log_dir: str, name: str = "training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(name)
        
    def info(self, message: str):
        self.logger.info(message)
        
    def warning(self, message: str):
        self.logger.warning(message)
        
    def error(self, message: str):
        self.logger.error(message)


class MetricsTracker:
    """Track and analyze training metrics"""
    
    def __init__(self):
        self.metrics = {}
        
    def add(self, metric_name: str, value: float, step: int):
        """Add metric value"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {'values': [], 'steps': []}
        
        self.metrics[metric_name]['values'].append(value)
        self.metrics[metric_name]['steps'].append(step)
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value of metric"""
        if metric_name in self.metrics and self.metrics[metric_name]['values']:
            return self.metrics[metric_name]['values'][-1]
        return None
    
    def get_mean(self, metric_name: str, last_n: int = 100) -> Optional[float]:
        """Get mean of last N values"""
        if metric_name in self.metrics and self.metrics[metric_name]['values']:
            values = self.metrics[metric_name]['values'][-last_n:]
            return np.mean(values)
        return None
    
    def plot_metrics(self, metric_names: list, save_path: str = None, show: bool = True):
        """Plot metrics"""
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4 * len(metric_names)))
        if len(metric_names) == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(metric_names):
            if metric_name in self.metrics:
                steps = self.metrics[metric_name]['steps']
                values = self.metrics[metric_name]['values']
                
                axes[i].plot(steps, values, alpha=0.7, label=metric_name)
                axes[i].set_xlabel('Step')
                axes[i].set_ylabel(metric_name)
                axes[i].set_title(f'{metric_name} over time')
                axes[i].grid(True, alpha=0.3)
                
                # Add moving average
                if len(values) > 50:
                    window = min(100, len(values) // 10)
                    moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                    moving_steps = steps[window-1:]
                    axes[i].plot(moving_steps, moving_avg, 'r-', alpha=0.8, 
                               label=f'Moving avg ({window})')
                
                axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save(self, file_path: str):
        """Save metrics to file"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.metrics, f)
    
    def load(self, file_path: str):
        """Load metrics from file"""
        with open(file_path, 'rb') as f:
            self.metrics = pickle.load(f)


def create_training_plots(results_dir: str, metrics_tracker: MetricsTracker):
    """Create comprehensive training plots"""
    plots_dir = Path(results_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Plot 1: Rewards and Success Rate
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Episode rewards
    if 'episode_reward' in metrics_tracker.metrics:
        steps = metrics_tracker.metrics['episode_reward']['steps']
        rewards = metrics_tracker.metrics['episode_reward']['values']
        
        ax1.plot(steps, rewards, alpha=0.6, color='blue', label='Episode Reward')
        
        # Moving average
        if len(rewards) > 50:
            window = min(100, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            moving_steps = steps[window-1:]
            ax1.plot(moving_steps, moving_avg, 'red', linewidth=2, 
                    label=f'Moving Average ({window})')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Success rate
    if 'success_rate' in metrics_tracker.metrics:
        steps = metrics_tracker.metrics['success_rate']['steps']
        success = metrics_tracker.metrics['success_rate']['values']
        
        ax2.plot(steps, success, alpha=0.7, color='green', label='Success Rate')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Success Rate Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(plots_dir / "training_progress.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: SAC Losses
    sac_metrics = ['sac_actor_loss', 'sac_critic_loss', 'sac_alpha']
    available_sac = [m for m in sac_metrics if m in metrics_tracker.metrics]
    
    if available_sac:
        fig, axes = plt.subplots(len(available_sac), 1, figsize=(12, 4 * len(available_sac)))
        if len(available_sac) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_sac):
            steps = metrics_tracker.metrics[metric]['steps']
            values = metrics_tracker.metrics[metric]['values']
            
            axes[i].plot(steps, values, alpha=0.7)
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel(metric.replace('sac_', '').replace('_', ' ').title())
            axes[i].set_title(f'SAC {metric.replace("sac_", "").replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "sac_losses.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: NMPC Statistics
    nmpc_metrics = ['nmpc_mean_solve_time', 'nmpc_success_rate']
    available_nmpc = [m for m in nmpc_metrics if m in metrics_tracker.metrics]
    
    if available_nmpc:
        fig, axes = plt.subplots(len(available_nmpc), 1, figsize=(12, 4 * len(available_nmpc)))
        if len(available_nmpc) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_nmpc):
            steps = metrics_tracker.metrics[metric]['steps']
            values = metrics_tracker.metrics[metric]['values']
            
            axes[i].plot(steps, values, alpha=0.7, color='orange')
            axes[i].set_xlabel('Episode')
            axes[i].set_ylabel(metric.replace('nmpc_', '').replace('_', ' ').title())
            axes[i].set_title(f'NMPC {metric.replace("nmpc_", "").replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "nmpc_stats.png", dpi=300, bbox_inches='tight')
        plt.close()


def analyze_episode_data(episode_data: dict, save_path: str = None):
    """Analyze and plot episode data"""
    states = episode_data['states']
    actions = episode_data['actions']
    rewards = episode_data['rewards']
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    time_steps = np.arange(len(states))
    
    # Plot states
    state_names = ['q1 (rad)', 'q2 (rad)', 'dq1 (rad/s)', 'dq2 (rad/s)']
    for i in range(4):
        row, col = i // 2, i % 2
        axes[row, col].plot(time_steps, states[:, i], label=state_names[i])
        axes[row, col].set_xlabel('Time Step')
        axes[row, col].set_ylabel(state_names[i])
        axes[row, col].set_title(f'State: {state_names[i]}')
        axes[row, col].grid(True, alpha=0.3)
        
        # Add target line for angles
        if i < 2:
            target = np.pi if i == 0 else 0.0
            axes[row, col].axhline(y=target, color='red', linestyle='--', 
                                 alpha=0.7, label='Target')
        axes[row, col].legend()
    
    # Plot actions
    if len(actions) > 0:
        axes[2, 0].plot(time_steps[:-1], actions[:, 0], 'g-', label='Control Torque')
        axes[2, 0].set_xlabel('Time Step')
        axes[2, 0].set_ylabel('Torque (Nm)')
        axes[2, 0].set_title('Control Actions')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].legend()
    
    # Plot rewards
    axes[2, 1].plot(time_steps[:-1], rewards, 'purple', label='Reward')
    axes[2, 1].set_xlabel('Time Step')
    axes[2, 1].set_ylabel('Reward')
    axes[2, 1].set_title('Rewards')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_algorithms(results_dict: Dict[str, dict], metric: str = 'episode_reward'):
    """Compare different algorithms"""
    plt.figure(figsize=(12, 8))
    
    for name, results in results_dict.items():
        if metric in results:
            steps = results[metric]['steps']
            values = results[metric]['values']
            
            # Plot raw data
            plt.plot(steps, values, alpha=0.3, label=f'{name} (raw)')
            
            # Plot moving average
            if len(values) > 50:
                window = min(100, len(values) // 10)
                moving_avg = np.convolve(values, np.ones(window)/window, mode='valid')
                moving_steps = steps[window-1:]
                plt.plot(moving_steps, moving_avg, linewidth=2, 
                        label=f'{name} (avg)')
    
    plt.xlabel('Episode' if 'episode' in metric else 'Step')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Algorithm Comparison: {metric.replace("_", " ").title()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def create_summary_report(results_dir: str, config: dict, final_results: dict):
    """Create summary report"""
    report_path = Path(results_dir) / "training_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# NMPC-SAC Pendubot Training Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Configuration\n\n")
        f.write("```yaml\n")
        yaml.dump(config, f, default_flow_style=False, indent=2)
        f.write("```\n\n")
        
        f.write("## Final Results\n\n")
        for key, value in final_results.items():
            f.write(f"- **{key.replace('_', ' ').title()}:** {value:.4f}\n")
        
        f.write("\n## Training Progress\n\n")
        f.write("![Training Progress](plots/training_progress.png)\n\n")
        
        f.write("## SAC Learning Curves\n\n")
        f.write("![SAC Losses](plots/sac_losses.png)\n\n")
        
        f.write("## NMPC Statistics\n\n")
        f.write("![NMPC Stats](plots/nmpc_stats.png)\n\n")


def evaluate_policy(env, agent, num_episodes: int = 10, render: bool = False):
    """Evaluate trained policy"""
    rewards = []
    successes = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
            
            if terminated or truncated:
                successes.append(terminated)  # Assuming terminated means success
                break
        
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Length = {episode_length}, Success = {terminated}")
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': np.mean(successes)
    }


def print_system_info():
    """Print system information"""
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    print(f"NumPy version: {np.__version__}")
    print("=" * 30)


# Test utilities
if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test metrics tracker
    tracker = MetricsTracker()
    
    # Add some dummy data
    for i in range(100):
        tracker.add('test_metric', np.sin(i * 0.1) + np.random.normal(0, 0.1), i)
    
    print(f"Latest value: {tracker.get_latest('test_metric')}")
    print(f"Mean of last 20: {tracker.get_mean('test_metric', 20)}")
    
    # Test plotting
    tracker.plot_metrics(['test_metric'], show=False)
    
    print("✓ Utilities test completed successfully!")