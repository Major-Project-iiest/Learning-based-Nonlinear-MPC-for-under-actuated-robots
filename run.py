#!/usr/bin/env python3
"""
Complete Run Script for NMPC-SAC Pendubot Project
This script provides a unified interface to run training, evaluation, and visualization
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path
import yaml
import shutil
from datetime import datetime

def create_project_structure():
    """Create the complete project directory structure"""
    dirs = [
        'config',
        'results/logs',
        'results/models', 
        'results/plots',
        'results/evaluation',
        'workspace'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Project directory structure created")

def create_default_configs():
    """Create default configuration files"""
    
    # Main configuration
    config = {
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
    
    # Save main config
    with open('config/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    # Create hyperparameter sweep config
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'eval_mean_reward', 'goal': 'maximize'},
        'parameters': {
            'sac.learning_rate': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 1e-2},
            'sac.alpha': {'distribution': 'uniform', 'min': 0.1, 'max': 0.5},
            'pinn.physics_weight': {'distribution': 'uniform', 'min': 0.01, 'max': 0.5},
            'training.nmpc_rate': {'distribution': 'uniform', 'min': 0.1, 'max': 0.7}
        }
    }
    
    with open('config/sweep_config.yaml', 'w') as f:
        yaml.dump(sweep_config, f, default_flow_style=False, indent=2)
    
    print("✓ Default configuration files created")

def create_requirements():
    """Create requirements.txt file"""
    requirements = [
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "gymnasium>=0.26.0",
        "casadi>=3.6.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "wandb>=0.15.0",
        "tensorboard>=2.12.0",
        "rich>=13.0.0",
        "control>=0.9.0",
        "slycot>=0.5.0"
    ]
    
    with open('requirements.txt', 'w') as f:
        for req in requirements:
            f.write(f"{req}\\n")
    
    print("✓ Requirements file created")

def setup_environment():
    """Setup the Python environment"""
    print("Setting up Python environment...")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        print("⚠️  Not in a virtual environment. Consider creating one:")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # Linux/Mac")
        print("   venv\\Scripts\\activate     # Windows")
        print()
    
    # Install basic requirements
    print("Installing Python requirements...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("✓ Python requirements installed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False
    
    return True

def install_acados():
    """Install acados (if not already installed)"""
    try:
        import acados_template
        print("✓ acados already installed")
        return True
    except ImportError:
        pass
    
    print("Installing acados...")
    print("⚠️  acados installation requires manual setup. Please run:")
    print("   bash setup/install_acados.sh")
    print("   This will build acados from source and set up the Python interface.")
    return False

def run_training(config_path: str = 'config/config.yaml', **kwargs):
    """Run the training script"""
    print(f"\\n{'='*50}")
    print("STARTING TRAINING")
    print(f"{'='*50}")
    
    cmd = [sys.executable, 'train.py', '--config', config_path]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\\n⚠️  Training interrupted by user")
        return False

def run_evaluation(config_path: str = 'config/config.yaml', 
                  model_dir: str = 'results/models',
                  output_dir: str = 'results/evaluation',
                  num_episodes: int = 10,
                  demo: bool = False):
    """Run the evaluation script"""
    print(f"\\n{'='*50}")
    print("STARTING EVALUATION")
    print(f"{'='*50}")
    
    cmd = [
        sys.executable, 'evaluate.py',
        '--config', config_path,
        '--model-dir', model_dir,
        '--output-dir', output_dir,
        '--num-episodes', str(num_episodes)
    ]
    
    if demo:
        cmd.append('--demo')
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Evaluation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Evaluation failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\\n⚠️  Evaluation interrupted by user")
        return False

def run_tests():
    """Run basic tests to verify installation"""
    print(f"\\n{'='*50}")
    print("RUNNING TESTS")
    print(f"{'='*50}")
    
    tests = [
        ("Testing Pendubot model", "from pendubot_model import PendubotModel; m=PendubotModel(); print('✓ Model OK')"),
        ("Testing SAC agent", "from sac_agent import SAC; a=SAC(4,1); print('✓ SAC OK')"),
        ("Testing PINN", "from pinn_dynamics import PhysicsInformedNetwork; p=PhysicsInformedNetwork(); print('✓ PINN OK')"),
        ("Testing Environment", "from pendubot_env import PendubotEnv; e=PendubotEnv(); print('✓ Env OK')"),
        ("Testing Utils", "from utils import set_seed; set_seed(42); print('✓ Utils OK')")
    ]
    
    for test_name, test_code in tests:
        print(f"\\n{test_name}...")
        try:
            subprocess.run([sys.executable, '-c', test_code], check=True, capture_output=True)
            print("  ✓ PASSED")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ FAILED: {e}")
            return False
    
    # Test acados (optional)
    print("\\nTesting acados integration...")
    try:
        test_code = """
try:
    from nmpc_controller import NMPCController
    from pendubot_model import PendubotModel
    m = PendubotModel()
    c = NMPCController(m)
    print('✓ NMPC OK')
except Exception as e:
    print(f'⚠️  NMPC not available: {e}')
"""
        subprocess.run([sys.executable, '-c', test_code], check=True)
    except:
        print("  ⚠️  acados tests failed (expected if not installed)")
    
    print("\\n✓ All basic tests passed!")
    return True

def create_demo_notebook():
    """Create a Jupyter notebook for demonstration"""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# NMPC-SAC Pendubot Control Demo\\n",
                    "\\n",
                    "This notebook demonstrates the NMPC-SAC hybrid control system for the Pendubot."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import numpy as np\\n",
                    "import matplotlib.pyplot as plt\\n",
                    "from pendubot_model import PendubotModel\\n",
                    "from pendubot_env import PendubotEnv\\n",
                    "from sac_agent import SAC\\n",
                    "\\n",
                    "# Create environment\\n",
                    "env = PendubotEnv(render_mode='human')\\n",
                    "print('Environment created successfully!')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Test random policy\\n",
                    "obs, info = env.reset()\\n",
                    "\\n",
                    "for i in range(100):\\n",
                    "    action = env.action_space.sample()\\n",
                    "    obs, reward, terminated, truncated, info = env.step(action)\\n",
                    "    \\n",
                    "    if terminated or truncated:\\n",
                    "        print(f'Episode finished at step {i}')\\n",
                    "        break\\n",
                    "\\n",
                    "env.close()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    import json
    with open('demo_notebook.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print("✓ Demo notebook created: demo_notebook.ipynb")

def print_usage():
    """Print usage information"""
    print("""
NMPC-SAC Pendubot Control System
===============================

Usage: python run.py [command] [options]

Commands:
  setup      - Setup the project (create dirs, configs, install deps)
  train      - Run training with default or custom config
  evaluate   - Evaluate trained models
  demo       - Run interactive demonstration
  test       - Run basic functionality tests
  clean      - Clean up generated files
  help       - Show this help message

Examples:
  python run.py setup                    # Initial setup
  python run.py train                    # Train with default config
  python run.py train --episodes 1000    # Train for 1000 episodes
  python run.py evaluate                 # Evaluate trained models
  python run.py demo                     # Run interactive demo
  python run.py test                     # Run tests

For more options, use: python run.py [command] --help
""")

def clean_project():
    """Clean up generated files"""
    print("Cleaning up project files...")
    
    dirs_to_clean = [
        'results/logs',
        'results/models',
        'results/plots',
        'results/evaluation',
        '__pycache__',
        '.pytest_cache'
    ]
    
    files_to_clean = [
        'acados_ocp.json',
        '*.pyc',
        '*.log'
    ]
    
    import glob
    
    for directory in dirs_to_clean:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"  Removed {directory}/")
    
    for pattern in files_to_clean:
        for file_path in glob.glob(pattern, recursive=True):
            os.remove(file_path)
            print(f"  Removed {file_path}")
    
    print("✓ Cleanup completed")

def main():
    parser = argparse.ArgumentParser(description='NMPC-SAC Pendubot Control System')
    parser.add_argument('command', choices=['setup', 'train', 'evaluate', 'demo', 'test', 'clean', 'help'],
                       help='Command to execute')
    
    # Training options
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--episodes', type=int, help='Number of training episodes')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device to use')
    
    # Evaluation options  
    parser.add_argument('--model-dir', type=str, default='results/models',
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--num-episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    
    args = parser.parse_args()
    
    if args.command == 'help':
        print_usage()
        return
    
    print(f"""
╔══════════════════════════════════════════════════════╗
║           NMPC-SAC Pendubot Control System           ║
║                                                      ║
║  Advanced Model Predictive Control with              ║
║  Soft Actor-Critic and Physics-Informed Learning    ║
╚══════════════════════════════════════════════════════╝

Command: {args.command}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
    
    if args.command == 'setup':
        print("Setting up NMPC-SAC Pendubot project...")
        create_project_structure()
        create_default_configs()
        create_requirements()
        create_demo_notebook()
        
        print("\\n" + "="*50)
        print("SETUP COMPLETE!")
        print("="*50)
        print("Next steps:")
        print("1. Install acados: bash setup/install_acados.sh")
        print("2. Install Python deps: pip install -r requirements.txt") 
        print("3. Run tests: python run.py test")
        print("4. Start training: python run.py train")
        
    elif args.command == 'train':
        # Update config with command line args
        kwargs = {}
        if args.episodes:
            kwargs['max_episodes'] = args.episodes
        if args.device:
            kwargs['device'] = args.device
            
        success = run_training(args.config, **kwargs)
        if success:
            print("\\n🎉 Training completed successfully!")
            print("Next steps:")
            print("- Evaluate results: python run.py evaluate")
            print("- Run demo: python run.py demo")
        
    elif args.command == 'evaluate':
        success = run_evaluation(
            args.config, 
            args.model_dir, 
            args.output_dir,
            args.num_episodes,
            demo=False
        )
        if success:
            print(f"\\n📊 Evaluation completed! Results in {args.output_dir}")
        
    elif args.command == 'demo':
        success = run_evaluation(
            args.config,
            args.model_dir,
            args.output_dir,
            args.num_episodes,
            demo=True
        )
        
    elif args.command == 'test':
        success = run_tests()
        if success:
            print("\\n✅ All tests passed! System ready for training.")
        else:
            print("\\n❌ Some tests failed. Check installation.")
        
    elif args.command == 'clean':
        clean_project()
        print("\\n🧹 Project cleaned!")

if __name__ == "__main__":
    main()