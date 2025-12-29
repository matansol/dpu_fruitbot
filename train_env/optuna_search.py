#!/usr/bin/env python
"""
Optuna-based hyperparameter optimization for PPO on FruitBot.

This script uses Bayesian optimization to efficiently search the hyperparameter
space and find the best configuration for your shaped reward function.

Key Features:
- Smart search (Bayesian optimization, not grid search)
- Automatic pruning of unpromising trials
- Integration with wandb for visualization
- Saves best model and hyperparameters
- Resume interrupted searches

Usage:
    # Quick search (20 trials, 500K steps each)
    python optuna_search.py --n-trials 20 --total-steps 500000 --study-name quick_search
    
    # Deep search (50 trials, 2M steps each) - Run after quick search
    python optuna_search.py --n-trials 50 --total-steps 2000000 --study-name deep_search
    
    # Resume previous search
    python optuna_search.py --n-trials 30 --study-name quick_search --resume
    
    # With wandb integration
    python optuna_search.py --n-trials 20 --wandb-project fruitbot-optuna

Requirements:
    pip install optuna optuna-dashboard wandb
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import json

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np

# Optional wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠ wandb not installed. Install with: pip install wandb")

# Import from train.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate import evaluate_agent_fast

from stable_baselines3 import PPO
import gym


def suggest_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define the hyperparameter search space for Optuna.
    
    Optuna will intelligently sample from these ranges using Bayesian optimization.
    """
    return {
        # Learning hyperparameters
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True),
        'gamma': trial.suggest_float('gamma', 0.95, 0.99),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
        
        # PPO-specific
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'ent_coef': trial.suggest_float('ent_coef', 0.001, 0.15, log=True),
        'vf_coef': trial.suggest_float('vf_coef', 0.3, 0.7),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0),
        
        # Training parameters
        'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
        'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024, 2048]),
        
        # # Reward shaping (FruitBot specific)
        # 'fruitbot_reward_completion': trial.suggest_float('fruitbot_reward_completion', 5.0, 20.0),
        # 'fruitbot_reward_positive': trial.suggest_float('fruitbot_reward_positive', 0.5, 5.0),
        # 'fruitbot_reward_negative': trial.suggest_float('fruitbot_reward_negative', -5.0, -0.5),
        # 'fruitbot_reward_wall_hit': trial.suggest_float('fruitbot_reward_wall_hit', -10.0, -0.5),
        # 'fruitbot_reward_step': trial.suggest_float('fruitbot_reward_step', 0.0, 0.1),
        'fruitbot_reward_completion': 10.0,
        'fruitbot_reward_positive': 1.0,
        'fruitbot_reward_negative': -1.0,
        'fruitbot_reward_wall_hit': -5.0,
        'fruitbot_reward_step': 0.1,
    }


def train_and_evaluate(
    hyperparams: Dict[str, Any],
    total_steps: int,
    n_envs: int,
    device: str,
    trial_number: int,
    save_dir: Path,
    eval_episodes: int = 50,
    num_levels: int = 100,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
) -> float:
    """
    Train a model with given hyperparameters and return evaluation score.
    
    Returns:
        Mean episode reward over evaluation episodes (higher is better)
    """
    import subprocess
    
    # Create unique directory for this trial
    trial_dir = save_dir / f"trial_{trial_number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Build training command
    train_script = Path(__file__).parent / "train.py"
    cmd = [
        sys.executable, str(train_script),
        "--env", "fruitbot",
        "--use-source",
        "--device", device,
        "--total-steps", str(total_steps),
        "--n-envs", str(n_envs),
        "--num-levels", str(num_levels),
        "--distribution-mode", "easy",
        "--save-freq", str(total_steps + 1),  # No intermediate saves
        "--seed", str(trial_number),  # Different seed per trial
    ]
    
    # Add hyperparameters to command
    for key, value in hyperparams.items():
        param_name = key.replace('_', '-')
        cmd.extend([f"--{param_name}", str(value)])
    
    # Add wandb integration if available
    if use_wandb and WANDB_AVAILABLE:
        cmd.extend([
            "--wandb-project", wandb_project or "fruitbot-optuna",
            "--wandb-run-name", f"trial_{trial_number:03d}",
        ])
    
    print(f"\n{'='*80}")
    print(f"🔬 Trial {trial_number}: Training with hyperparameters")
    print(f"{'='*80}")
    for key, value in hyperparams.items():
        print(f"  {key:30s}: {value}")
    print(f"{'='*80}\n")
    
    # Train the model
    try:
        # Change to script directory
        original_dir = os.getcwd()
        os.chdir(Path(__file__).parent)
        
        # Set UTF-8 encoding for subprocess on Windows
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace problematic characters
            env=env,
        )
        
        os.chdir(original_dir)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for trial {trial_number}")
        print(f"Error: {e.stderr}")
        return -float('inf')  # Return worst possible score
    
    # Find the trained model (most recent in models/fruitbot/)
    models_dir = Path("models") / "fruitbot"
    subdirs = sorted([d for d in models_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)
    
    if not subdirs:
        print(f"❌ No model found for trial {trial_number}")
        return -float('inf')
    
    latest_model_dir = subdirs[-1]
    model_path = latest_model_dir / "ppo_final.zip"
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return -float('inf')
    
    # Evaluate the trained model
    print(f"\n📊 Evaluating trial {trial_number}...")
    
    try:
        # Load model
        model = PPO.load(str(model_path), device=device)
        
        # Create evaluation environment
        import procgen
        env_kwargs = {
            'env_name': 'fruitbot',
            'distribution_mode': 'easy',
            'num_levels': 0,  # Test on unseen levels
            'start_level': 10000,  # Different seed range than training
        }
        env_id = "procgen-fruitbot-v0"
        env = gym.make(env_id, render_mode=None, **env_kwargs)
        
        # Run evaluation
        episode_rewards = []
        for ep in range(eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
        
        env.close()
        
        # Calculate metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        print(f"✅ Trial {trial_number} Results:")
        print(f"   Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   Min: {np.min(episode_rewards):.2f}, Max: {np.max(episode_rewards):.2f}")
        
        # Save trial info
        trial_info = {
            'trial_number': trial_number,
            'hyperparameters': hyperparams,
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'model_path': str(model_path),
        }
        
        with open(trial_dir / 'trial_info.json', 'w') as f:
            json.dump(trial_info, f, indent=2)
        
        return mean_reward
        
    except Exception as e:
        print(f"❌ Evaluation failed for trial {trial_number}: {e}")
        import traceback
        traceback.print_exc()
        return -float('inf')


def objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    study_dir: Path,
) -> float:
    """
    Optuna objective function: train and evaluate a model.
    
    Returns the metric to optimize (higher is better).
    """
    # Get hyperparameters for this trial
    hyperparams = suggest_hyperparameters(trial)
    
    # Ensure batch_size <= n_steps
    if hyperparams['batch_size'] > hyperparams['n_steps']:
        hyperparams['batch_size'] = hyperparams['n_steps']
    
    # Train and evaluate
    score = train_and_evaluate(
        hyperparams=hyperparams,
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        device=args.device,
        trial_number=trial.number,
        save_dir=study_dir,
        eval_episodes=args.eval_episodes,
        num_levels=args.num_levels,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )
    
    # Report intermediate value for pruning (optional)
    trial.report(score, step=args.total_steps)
    
    # Optuna can prune unpromising trials early
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return score


def parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter optimization with Optuna")
    
    # Optuna settings
    p.add_argument("--n-trials", type=int, default=20, help="Number of trials to run")
    p.add_argument("--study-name", type=str, default="fruitbot_optuna", help="Name of the Optuna study")
    p.add_argument("--resume", action="store_true", help="Resume an existing study")
    p.add_argument("--storage", type=str, default=None, help="Database URL for distributed optimization (e.g., sqlite:///optuna.db)")
    
    # Training settings
    p.add_argument("--total-steps", type=int, default=200_000, help="Training steps per trial")
    p.add_argument("--n-envs", type=int, default=96, help="Number of parallel environments")
    p.add_argument("--num-levels", type=int, default=100, help="Number of training levels")
    p.add_argument("--device", type=str, default="cuda", help="PyTorch device")
    
    # Evaluation settings
    p.add_argument("--eval-episodes", type=int, default=50, help="Number of episodes for evaluation")
    
    # Wandb integration
    p.add_argument("--use-wandb", action="store_true", help="Log trials to Weights & Biases")
    p.add_argument("--wandb-project", type=str, default="fruitbot-optuna", help="W&B project name")
    
    # Pruning
    p.add_argument("--no-pruning", action="store_true", help="Disable automatic pruning of bad trials")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Create study directory
    study_dir = Path("optuna_studies") / args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🔬 Optuna Hyperparameter Optimization")
    print(f"{'='*80}")
    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Steps per trial: {args.total_steps:,}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Device: {args.device}")
    print(f"Results directory: {study_dir}")
    if args.use_wandb:
        print(f"W&B project: {args.wandb_project}")
    print(f"{'='*80}\n")
    
    # Create Optuna study
    storage = args.storage or f"sqlite:///{study_dir / 'optuna.db'}"
    
    sampler = TPESampler(seed=42)  # Reproducible sampling
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3) if not args.no_pruning else None
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",  # We want to maximize reward
        sampler=sampler,
        pruner=pruner,
        load_if_exists=args.resume,
    )
    
    if args.resume and len(study.trials) > 0:
        print(f"📂 Resuming study with {len(study.trials)} existing trials")
        print(f"   Best value so far: {study.best_value:.2f}")
        print(f"   Best params: {study.best_params}\n")
    
    # Run optimization
    try:
        study.optimize(
            lambda trial: objective(trial, args, study_dir),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠ Optimization interrupted by user")
    
    # Print results
    print(f"\n{'='*80}")
    print(f"🎉 Optimization Complete!")
    print(f"{'='*80}")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"\n🏆 Best trial:")
    print(f"   Value (mean reward): {study.best_value:.2f}")
    print(f"\n   Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"      {key:30s}: {value}")
    print(f"{'='*80}\n")
    
    # Save best hyperparameters
    best_params_file = study_dir / "best_hyperparameters.json"
    with open(best_params_file, 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
        }, f, indent=2)
    
    print(f"💾 Best hyperparameters saved to: {best_params_file}")
    
    # Generate training command with best parameters
    print(f"\n📋 To train with best hyperparameters, run:")
    print(f"\npython train.py \\")
    print(f"  --env fruitbot \\")
    print(f"  --use-source \\")
    print(f"  --device {args.device} \\")
    print(f"  --n-envs {args.n_envs} \\")
    print(f"  --total-steps 5000000 \\")
    for key, value in study.best_params.items():
        param_name = key.replace('_', '-')
        print(f"  --{param_name} {value} \\")
    print()
    
    # Optuna dashboard tip
    print(f"💡 To visualize results, run:")
    print(f"   optuna-dashboard {storage}")
    print(f"   Then open: http://localhost:8080\n")


if __name__ == "__main__":
    main()
