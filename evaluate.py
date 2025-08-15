"""
Complete Evaluation and Visualization Script
Evaluates trained models and creates comprehensive analysis
"""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from tqdm import tqdm

# Import our modules
from pendubot_model import PendubotModel
from pendubot_env import PendubotEnv
from sac_agent import SAC
from pinn_dynamics import PhysicsInformedNetwork
from nmpc_controller import NMPCController
from utils import load_config, evaluate_policy, analyze_episode_data, create_training_plots


class ModelEvaluator:
    """Comprehensive model evaluation and analysis"""
    
    def __init__(self, config_path: str, model_dir: str):
        self.config = load_config(config_path)
        self.model_dir = Path(model_dir)
        self.device = torch.device(self.config['device'])
        
        # Initialize components
        self._load_models()
        self._init_environment()
        
    def _load_models(self):
        """Load trained models"""
        print("Loading trained models...")
        
        # Load SAC agent
        self.sac_agent = SAC(
            state_dim=4,
            action_dim=1,
            max_action=5.0,
            device=self.device
        )
        
        sac_path = self.model_dir / 'sac_final.pt'
        if sac_path.exists():
            self.sac_agent.load(sac_path)
            print("✓ SAC model loaded")
        else:
            print("✗ SAC model not found")
            
        # Load PINN
        self.pinn = PhysicsInformedNetwork().to(self.device)
        pinn_path = self.model_dir / 'pinn_final.pt'
        if pinn_path.exists():
            self.pinn.load_state_dict(torch.load(pinn_path, map_location=self.device))
            self.pinn.eval()
            print("✓ PINN model loaded")
        else:
            print("✗ PINN model not found")
            self.pinn = None
        
        # Initialize NMPC (if available)
        try:
            self.model = PendubotModel()
            self.nmpc = NMPCController(
                model=self.model,
                prediction_horizon=self.config['nmpc']['prediction_horizon'],
                dt=self.config['nmpc']['dt']
            )
            print("✓ NMPC controller initialized")
        except Exception as e:
            print(f"✗ NMPC initialization failed: {e}")
            self.nmpc = None
    
    def _init_environment(self):
        """Initialize evaluation environment"""
        env_config = self.config['environment']
        self.env = PendubotEnv(
            dt=env_config['dt'],
            max_episode_steps=env_config['max_episode_steps'],
            reward_type=env_config['reward_type'],
            noise_std=0.0,  # No noise for evaluation
            render_mode="human"
        )
    
    def evaluate_controllers(self, num_episodes: int = 10) -> dict:
        """Evaluate different control strategies"""
        print(f"\\nEvaluating controllers over {num_episodes} episodes...")
        
        controllers = {
            'SAC': self._evaluate_sac,
            'NMPC': self._evaluate_nmpc,
            'NMPC+PINN': self._evaluate_nmpc_pinn,
            'Hybrid': self._evaluate_hybrid
        }
        
        results = {}
        
        for name, controller_fn in controllers.items():
            if name == 'NMPC' and self.nmpc is None:
                continue
            if name == 'NMPC+PINN' and (self.nmpc is None or self.pinn is None):
                continue
            if name == 'Hybrid' and (self.nmpc is None or self.pinn is None):
                continue
                
            print(f"\\nEvaluating {name}...")
            episode_data = []
            
            for episode in tqdm(range(num_episodes), desc=f"{name} evaluation"):
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                success = False
                
                states = [obs.copy()]
                actions = []
                rewards = []
                
                while True:
                    # Get action from controller
                    action = controller_fn(obs)
                    
                    # Execute action
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    # Store data
                    states.append(next_obs.copy())
                    actions.append(action.copy())
                    rewards.append(reward)
                    
                    episode_reward += reward
                    episode_length += 1
                    obs = next_obs
                    
                    if terminated or truncated:
                        success = terminated
                        break
                
                episode_data.append({
                    'reward': episode_reward,
                    'length': episode_length,
                    'success': success,
                    'states': np.array(states),
                    'actions': np.array(actions),
                    'rewards': np.array(rewards)
                })
            
            # Compute statistics
            rewards = [ep['reward'] for ep in episode_data]
            lengths = [ep['length'] for ep in episode_data]
            successes = [ep['success'] for ep in episode_data]
            
            results[name] = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'mean_length': np.mean(lengths),
                'std_length': np.std(lengths),
                'success_rate': np.mean(successes),
                'episode_data': episode_data
            }
            
            print(f"{name} Results:")
            print(f"  Mean Reward: {results[name]['mean_reward']:.2f} ± {results[name]['std_reward']:.2f}")
            print(f"  Success Rate: {results[name]['success_rate']:.2%}")
            print(f"  Mean Length: {results[name]['mean_length']:.1f} ± {results[name]['std_length']:.1f}")
        
        return results
    
    def _evaluate_sac(self, obs: np.ndarray) -> np.ndarray:
        """Pure SAC controller"""
        return self.sac_agent.select_action(obs, deterministic=True)
    
    def _evaluate_nmpc(self, obs: np.ndarray) -> np.ndarray:
        """Pure NMPC controller"""
        try:
            action, _ = self.nmpc.solve(obs)
            return action
        except:
            return np.array([0.0])  # Fallback
    
    def _evaluate_nmpc_pinn(self, obs: np.ndarray) -> np.ndarray:
        """NMPC with PINN residual"""
        try:
            action, _ = self.nmpc.solve(obs, residual_predictor=self.pinn)
            return action
        except:
            return np.array([0.0])  # Fallback
    
    def _evaluate_hybrid(self, obs: np.ndarray) -> np.ndarray:
        """Hybrid SAC+NMPC controller"""
        # Use SAC near the target, NMPC for swing-up
        q1, q2 = obs[0], obs[1]
        
        # Distance from upright
        target_dist = abs(q1 - np.pi) + abs(q2)
        
        if target_dist < 0.5:  # Close to target - use SAC
            return self.sac_agent.select_action(obs, deterministic=True)
        else:  # Far from target - use NMPC
            try:
                action, _ = self.nmpc.solve(obs, residual_predictor=self.pinn)
                return action
            except:
                return self.sac_agent.select_action(obs, deterministic=True)
    
    def create_comparison_plots(self, results: dict, save_dir: str):
        """Create comprehensive comparison plots"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Plot 1: Performance comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        controllers = list(results.keys())
        
        # Rewards
        rewards = [results[c]['mean_reward'] for c in controllers]
        reward_stds = [results[c]['std_reward'] for c in controllers]
        
        bars1 = ax1.bar(controllers, rewards, yerr=reward_stds, capsize=5)
        ax1.set_ylabel('Mean Episode Reward')
        ax1.set_title('Episode Rewards Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Color bars based on performance
        for i, bar in enumerate(bars1):
            if rewards[i] == max(rewards):
                bar.set_color('gold')
            else:
                bar.set_color('skyblue')
        
        # Success rates
        success_rates = [results[c]['success_rate'] for c in controllers]
        bars2 = ax2.bar(controllers, success_rates)
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Success Rate Comparison')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        
        # Color bars
        for i, bar in enumerate(bars2):
            if success_rates[i] == max(success_rates):
                bar.set_color('gold')
            else:
                bar.set_color('lightcoral')
        
        # Episode lengths
        lengths = [results[c]['mean_length'] for c in controllers]
        length_stds = [results[c]['std_length'] for c in controllers]
        
        bars3 = ax3.bar(controllers, lengths, yerr=length_stds, capsize=5)
        ax3.set_ylabel('Mean Episode Length')
        ax3.set_title('Episode Length Comparison')
        ax3.grid(True, alpha=0.3)
        
        # Reward distributions
        for i, controller in enumerate(controllers):
            episode_rewards = [ep['reward'] for ep in results[controller]['episode_data']]
            ax4.hist(episode_rewards, alpha=0.7, label=controller, bins=15)
        
        ax4.set_xlabel('Episode Reward')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Reward Distributions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'controller_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Learning curves for best episodes
        self._plot_best_episodes(results, save_dir)
        
        # Plot 3: Phase portraits
        self._plot_phase_portraits(results, save_dir)
    
    def _plot_best_episodes(self, results: dict, save_dir: Path):
        """Plot trajectories of best episodes for each controller"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, (controller, data) in enumerate(results.items()):
            # Find best episode
            episode_rewards = [ep['reward'] for ep in data['episode_data']]
            best_idx = np.argmax(episode_rewards)
            best_episode = data['episode_data'][best_idx]
            
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            states = best_episode['states']
            time_steps = np.arange(len(states)) * self.config['environment']['dt']
            
            # Plot angles
            ax.plot(time_steps, states[:, 0], 'b-', label='q1 (joint 1)', linewidth=2)
            ax.plot(time_steps, states[:, 1], 'r-', label='q2 (joint 2)', linewidth=2)
            
            # Target lines
            ax.axhline(y=np.pi, color='blue', linestyle='--', alpha=0.7, label='q1 target')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='q2 target')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Angle (rad)')
            ax.set_title(f'{controller} - Best Episode (Reward: {best_episode["reward"]:.1f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'best_episodes.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_phase_portraits(self, results: dict, save_dir: Path):
        """Plot phase portraits for different controllers"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, (controller, data) in enumerate(results.items()):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            # Combine states from all episodes
            all_states = []
            for episode in data['episode_data']:
                all_states.append(episode['states'])
            
            all_states = np.vstack(all_states)
            
            # Plot phase portrait (q1 vs dq1)
            scatter = ax.scatter(all_states[:, 0], all_states[:, 2], 
                               c=range(len(all_states)), cmap='viridis', 
                               alpha=0.6, s=1)
            
            # Mark target
            ax.plot(np.pi, 0, 'r*', markersize=15, label='Target')
            
            ax.set_xlabel('q1 (rad)')
            ax.set_ylabel('dq1 (rad/s)')
            ax.set_title(f'{controller} - Phase Portrait (Joint 1)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'phase_portraits.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_detailed_analysis(self, results: dict, save_dir: str):
        """Create detailed analysis report"""
        save_dir = Path(save_dir)
        
        # Create markdown report
        report_path = save_dir / 'evaluation_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# NMPC-SAC Pendubot Evaluation Report\\n\\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            f.write("## Controller Performance Summary\\n\\n")
            f.write("| Controller | Mean Reward | Success Rate | Mean Length |\\n")
            f.write("|------------|-------------|--------------|-------------|\\n")
            
            for controller, data in results.items():
                f.write(f"| {controller} | {data['mean_reward']:.2f} ± {data['std_reward']:.2f} | ")
                f.write(f"{data['success_rate']:.2%} | {data['mean_length']:.1f} ± {data['std_length']:.1f} |\\n")
            
            f.write("\\n## Key Findings\\n\\n")
            
            # Find best controller
            best_reward = max(results[c]['mean_reward'] for c in results)
            best_controller = [c for c in results if results[c]['mean_reward'] == best_reward][0]
            
            f.write(f"- **Best performing controller:** {best_controller} with mean reward {best_reward:.2f}\\n")
            
            best_success = max(results[c]['success_rate'] for c in results)
            best_success_controller = [c for c in results if results[c]['success_rate'] == best_success][0]
            
            f.write(f"- **Highest success rate:** {best_success_controller} with {best_success:.2%} success rate\\n")
            
            f.write("\\n## Visualizations\\n\\n")
            f.write("![Controller Comparison](controller_comparison.png)\\n\\n")
            f.write("![Best Episodes](best_episodes.png)\\n\\n")
            f.write("![Phase Portraits](phase_portraits.png)\\n\\n")
        
        print(f"Detailed analysis saved to {report_path}")
    
    def interactive_demo(self):
        """Run interactive demonstration"""
        print("\\nStarting interactive demonstration...")
        print("Controls:")
        print("  'q' - Quit")
        print("  '1' - SAC controller")
        print("  '2' - NMPC controller")
        print("  '3' - NMPC+PINN controller")
        print("  '4' - Hybrid controller")
        print("  'r' - Reset environment")
        
        current_controller = self._evaluate_sac
        controller_name = "SAC"
        
        obs, _ = self.env.reset()
        
        try:
            while True:
                # Get user input (non-blocking would be better)
                print(f"\\nCurrent controller: {controller_name}")
                print(f"Current state: q1={obs[0]:.3f}, q2={obs[1]:.3f}, dq1={obs[2]:.3f}, dq2={obs[3]:.3f}")
                
                action = current_controller(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                self.env.render()
                
                if terminated or truncated:
                    print(f"Episode finished! Reward: {reward:.2f}")
                    obs, _ = self.env.reset()
                
                # Simple command interface (in practice, use keyboard input)
                import time
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\\nDemo ended by user")
        finally:
            self.env.close()


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate NMPC-SAC Pendubot models')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--model-dir', type=str, default='results/models',
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--num-episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    parser.add_argument('--demo', action='store_true',
                       help='Run interactive demonstration')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.config):
        print(f"Config file {args.config} not found!")
        return
    
    if not os.path.exists(args.model_dir):
        print(f"Model directory {args.model_dir} not found!")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(args.config, args.model_dir)
    
    if args.demo:
        # Run interactive demo
        evaluator.interactive_demo()
    else:
        # Run comprehensive evaluation
        print("=== NMPC-SAC Pendubot Model Evaluation ===")
        
        # Evaluate controllers
        results = evaluator.evaluate_controllers(args.num_episodes)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create plots and analysis
        evaluator.create_comparison_plots(results, args.output_dir)
        evaluator.create_detailed_analysis(results, args.output_dir)
        
        print(f"\\nEvaluation completed! Results saved to {args.output_dir}")
        
        # Print summary
        print("\\n=== SUMMARY ===")
        for controller, data in results.items():
            print(f"{controller:12} | Reward: {data['mean_reward']:6.1f} | "
                  f"Success: {data['success_rate']:5.1%} | "
                  f"Length: {data['mean_length']:5.1f}")


if __name__ == "__main__":
    main()