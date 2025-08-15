"""
Complete Training Script for NMPC-SAC Pendubot Control
Integrates all components: PINN, SAC, NMPC, and Environment
"""
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
import wandb

# Import our modules
from pendubot_model import PendubotModel
from pendubot_env import PendubotEnv
from sac_agent import SAC, ReplayBuffer
from pinn_dynamics import PhysicsInformedNetwork, PINNTrainer, ResidualBuffer
from nmpc_controller import NMPCController
from utils import Logger, set_seed, save_config


class NMPCSACTrainer:
    """Main trainer for NMPC-SAC hybrid system"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Set random seeds
        set_seed(config['seed'])
        
        # Create directories
        self.log_dir = Path(config['log_dir'])
        self.model_dir = Path(config['model_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._init_model()
        self._init_environment()
        self._init_sac_agent()
        self._init_pinn()
        self._init_nmpc()
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_reward = -np.inf
        
        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
        
        # Wandb logging
        if config.get('use_wandb', False):
            wandb.init(
                project="nmpc-sac-pendubot",
                config=config,
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Training started")
        self.logger.info(f"Configuration: {self.config}")
    
    def _init_model(self):
        """Initialize Pendubot model"""
        self.model = PendubotModel()
        self.logger.info("Pendubot model initialized")
    
    def _init_environment(self):
        """Initialize training and evaluation environments"""
        env_config = self.config['environment']
        
        self.train_env = PendubotEnv(
            dt=env_config['dt'],
            max_episode_steps=env_config['max_episode_steps'],
            reward_type=env_config['reward_type'],
            noise_std=env_config.get('noise_std', 0.0)
        )
        
        self.eval_env = PendubotEnv(
            dt=env_config['dt'],
            max_episode_steps=env_config['max_episode_steps'],
            reward_type=env_config['reward_type'],
            noise_std=0.0,  # No noise during evaluation
            render_mode=env_config.get('render_mode', None)
        )
        
        self.logger.info("Environments initialized")
    
    def _init_sac_agent(self):
        """Initialize SAC agent and replay buffer"""
        sac_config = self.config['sac']
        
        self.sac_agent = SAC(
            state_dim=4,
            action_dim=1,
            max_action=5.0,
            lr=sac_config['learning_rate'],
            gamma=sac_config['gamma'],
            tau=sac_config['tau'],
            alpha=sac_config['alpha'],
            auto_entropy_tuning=sac_config['auto_entropy_tuning'],
            device=self.device
        )
        
        self.replay_buffer = ReplayBuffer(
            capacity=sac_config['buffer_size'],
            state_dim=4,
            action_dim=1,
            alpha=sac_config.get('priority_alpha', 0.6)
        )
        
        self.logger.info("SAC agent initialized")
    
    def _init_pinn(self):
        """Initialize Physics-Informed Neural Network"""
        pinn_config = self.config['pinn']
        
        self.pinn = PhysicsInformedNetwork(
            state_dim=4,
            action_dim=1,
            hidden_dims=pinn_config['hidden_dims'],
            activation=pinn_config['activation'],
            physics_weight=pinn_config['physics_weight']
        ).to(self.device)
        
        self.pinn_trainer = PINNTrainer(
            self.pinn,
            learning_rate=pinn_config['learning_rate'],
            weight_decay=pinn_config['weight_decay'],
            device=self.device
        )
        
        self.residual_buffer = ResidualBuffer(
            capacity=pinn_config['buffer_size'],
            state_dim=4,
            action_dim=1
        )
        
        self.logger.info("PINN initialized")
    
    def _init_nmpc(self):
        """Initialize NMPC controller"""
        nmpc_config = self.config['nmpc']
        
        try:
            self.nmpc = NMPCController(
                model=self.model,
                prediction_horizon=nmpc_config['prediction_horizon'],
                dt=nmpc_config['dt'],
                Q=np.diag(nmpc_config['Q_weights']),
                R=np.array([[nmpc_config['R_weight']]]),
                solver_type=nmpc_config['solver_type'],
                max_iter=nmpc_config['max_iter']
            )
            self.use_nmpc = True
            self.logger.info("NMPC controller initialized")
        except Exception as e:
            self.logger.warning(f"NMPC initialization failed: {e}. Using SAC only.")
            self.nmpc = None
            self.use_nmpc = False
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # Training parameters
        max_episodes = self.config['training']['max_episodes']
        eval_freq = self.config['training']['eval_freq']
        save_freq = self.config['training']['save_freq']
        pinn_update_freq = self.config['training']['pinn_update_freq']
        
        # Training loop
        pbar = tqdm(range(max_episodes), desc="Training")
        
        for episode in pbar:
            self.episode = episode
            
            # Training episode
            episode_stats = self._run_episode(mode='train')
            
            # Update statistics
            self.episode_rewards.append(episode_stats['return'])
            self.episode_lengths.append(episode_stats['length'])
            self.success_rate.append(episode_stats['success'])
            
            # Update progress bar
            pbar.set_postfix({
                'Reward': f"{episode_stats['return']:.2f}",
                'Success': f"{np.mean(self.success_rate):.2f}",
                'Steps': f"{self.total_steps}"
            })
            
            # Periodic evaluation
            if episode % eval_freq == 0:
                eval_stats = self._evaluate()
                self._log_statistics(episode_stats, eval_stats)
            
            # Update PINN periodically
            if episode % pinn_update_freq == 0 and len(self.residual_buffer) > 100:
                self._update_pinn()
            
            # Save models
            if episode % save_freq == 0:
                self._save_models(episode)
            
            # Early stopping check
            if np.mean(self.episode_rewards) > self.config['training'].get('target_reward', 800):
                self.logger.info(f"Target reward reached at episode {episode}")
                break
        
        # Final evaluation and save
        final_eval = self._evaluate(num_episodes=10)
        self._save_models(episode, is_final=True)
        
        self.logger.info("Training completed")
        return final_eval
    
    def _run_episode(self, mode='train') -> dict:
        """Run single episode"""
        env = self.train_env if mode == 'train' else self.eval_env
        
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        success = False
        
        # Episode data for residual learning
        episode_states = []
        episode_actions = []
        episode_next_states = []
        
        while True:
            # Select action
            if mode == 'train' and np.random.random() < self.config['training'].get('exploration_rate', 0.1):
                # Pure SAC exploration
                action = self.sac_agent.select_action(obs, deterministic=False)
            elif self.use_nmpc and np.random.random() < self.config['training'].get('nmpc_rate', 0.5):
                # Use NMPC with PINN residual
                try:
                    action, _ = self.nmpc.solve(obs, residual_predictor=self.pinn)
                except:
                    action = self.sac_agent.select_action(obs, deterministic=False)
            else:
                # Use SAC
                action = self.sac_agent.select_action(obs, deterministic=(mode != 'train'))
            
            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            if mode == 'train':
                # Store for SAC
                self.replay_buffer.add(obs, action, reward, next_obs, done)
                
                # Store for PINN (compute residual)
                if len(episode_states) > 0:
                    predicted_next = self.model.simulate_step(obs, action, env.dt)
                    actual_next = next_obs
                    residual = (actual_next - predicted_next) / env.dt
                    
                    self.residual_buffer.add(obs, action, residual)
                
                episode_states.append(obs)
                episode_actions.append(action)
                episode_next_states.append(next_obs)
            
            # Update SAC
            if mode == 'train' and len(self.replay_buffer) > self.config['sac']['batch_size']:
                if self.total_steps % self.config['training'].get('sac_update_freq', 1) == 0:
                    for _ in range(self.config['training'].get('sac_updates_per_step', 1)):
                        self.sac_agent.update(self.replay_buffer, self.config['sac']['batch_size'])
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            
            if mode == 'train':
                self.total_steps += 1
            
            if done:
                success = info.get('success', terminated)
                break
        
        return {
            'return': episode_reward,
            'length': episode_length,
            'success': success,
            'total_steps': self.total_steps
        }
    
    def _evaluate(self, num_episodes=5) -> dict:
        """Evaluate current policy"""
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        
        for _ in range(num_episodes):
            episode_stats = self._run_episode(mode='eval')
            eval_rewards.append(episode_stats['return'])
            eval_lengths.append(episode_stats['length'])
            eval_successes.append(episode_stats['success'])
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': np.mean(eval_successes)
        }
    
    def _update_pinn(self):
        """Update PINN using collected residual data"""
        if len(self.residual_buffer) < 100:
            return
        
        try:
            # Get datasets
            train_dataset, val_dataset = self.residual_buffer.get_dataset()
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=self.config['pinn']['batch_size'],
                shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config['pinn']['batch_size'],
                shuffle=False
            )
            
            # Train for a few epochs
            for epoch in range(self.config['pinn']['epochs_per_update']):
                train_losses = self.pinn_trainer.train_epoch(train_loader)
                
                if epoch % 5 == 0:
                    val_losses = self.pinn_trainer.validate(val_loader)
                    self.logger.info(f"PINN Epoch {epoch}: Train Loss {train_losses['total_loss']:.4f}, "
                                   f"Val Loss {val_losses['total_loss']:.4f}")
        
        except Exception as e:
            self.logger.warning(f"PINN update failed: {e}")
    
    def _log_statistics(self, episode_stats, eval_stats):
        """Log training statistics"""
        stats = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'episode_reward': episode_stats['return'],
            'episode_length': episode_stats['length'],
            'mean_reward_100': np.mean(self.episode_rewards),
            'success_rate_100': np.mean(self.success_rate),
            'eval_mean_reward': eval_stats['mean_reward'],
            'eval_success_rate': eval_stats['success_rate'],
        }
        
        # Add SAC statistics
        sac_stats = self.sac_agent.get_stats()
        stats.update({f'sac_{k}': v for k, v in sac_stats.items()})
        
        # Add NMPC statistics
        if self.use_nmpc:
            nmpc_stats = self.nmpc.get_stats()
            stats.update({f'nmpc_{k}': v for k, v in nmpc_stats.items()})
        
        # Log to wandb
        if self.config.get('use_wandb', False):
            wandb.log(stats)
        
        # Log to file
        self.logger.info(f"Episode {self.episode}: "
                        f"Reward {episode_stats['return']:.2f}, "
                        f"Eval Reward {eval_stats['mean_reward']:.2f}, "
                        f"Success Rate {eval_stats['success_rate']:.2f}")
    
    def _save_models(self, episode, is_final=False):
        """Save trained models"""
        suffix = 'final' if is_final else f'episode_{episode}'
        
        # Save SAC
        sac_path = self.model_dir / f'sac_{suffix}.pt'
        self.sac_agent.save(sac_path)
        
        # Save PINN
        pinn_path = self.model_dir / f'pinn_{suffix}.pt'
        torch.save(self.pinn.state_dict(), pinn_path)
        
        # Save training state
        state_path = self.model_dir / f'training_state_{suffix}.pt'
        torch.save({
            'episode': self.episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'episode_rewards': list(self.episode_rewards),
            'config': self.config
        }, state_path)
        
        self.logger.info(f"Models saved at episode {episode}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function"""
    # Load configuration
    config_path = "config/config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found. Creating default config.")
        create_default_config(config_path)
    
    config = load_config(config_path)
    
    # Create trainer and start training
    trainer = NMPCSACTrainer(config)
    
    try:
        final_results = trainer.train()
        print(f"Training completed successfully!")
        print(f"Final evaluation results: {final_results}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        trainer._save_models(trainer.episode, is_final=True)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if config.get('use_wandb', False):
            wandb.finish()


def create_default_config(config_path: str):
    """Create default configuration file"""
    default_config = {
        'seed': 42,
        'device': 'cpu',
        'log_dir': 'results/logs',
        'model_dir': 'results/models',
        'use_wandb': False,
        
        'environment': {
            'dt': 0.01,
            'max_episode_steps': 1000,
            'reward_type': 'swing_up',
            'noise_std': 0.0,
            'render_mode': None
        },
        
        'sac': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'alpha': 0.2,
            'auto_entropy_tuning': True,
            'buffer_size': 100000,
            'batch_size': 256,
            'priority_alpha': 0.6
        },
        
        'pinn': {
            'hidden_dims': [128, 128, 64],
            'activation': 'tanh',
            'physics_weight': 0.1,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'buffer_size': 50000,
            'batch_size': 64,
            'epochs_per_update': 10
        },
        
        'nmpc': {
            'prediction_horizon': 40,
            'dt': 0.01,
            'Q_weights': [30.0, 10.0, 2.0, 1.0],
            'R_weight': 0.1,
            'solver_type': 'SQP_RTI',
            'max_iter': 100
        },
        
        'training': {
            'max_episodes': 2000,
            'eval_freq': 50,
            'save_freq': 100,
            'pinn_update_freq': 25,
            'exploration_rate': 0.1,
            'nmpc_rate': 0.3,
            'sac_update_freq': 1,
            'sac_updates_per_step': 1,
            'target_reward': 800
        }
    }
    
    # Create config directory
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"Default configuration saved to {config_path}")


if __name__ == "__main__":
    main()