#!/usr/bin/env python
"""
Hyperparameter grid search for PPO training on Procgen FruitBot.

This script trains multiple agents with different hyperparameter combinations,
evaluates them, and saves the results to help you find the best configuration.

Usage:
    python grid_search.py --total-steps 500000 --eval-episodes 50 --n-envs 8

The script will:
1. Train agents with different hyperparameter combinations
2. Evaluate each trained agent
3. Save results to a CSV file with performance metrics
4. Print a summary of the best configurations
"""

import argparse
import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from itertools import product

# Import evaluation function from evaluate.py
from evaluate import evaluate_agent_fast
import gym
from stable_baselines3 import PPO


def get_hyperparameter_grid() -> Dict[str, List[Any]]:
    """
    Define the hyperparameter grid to search over.
    
    Adjust these based on what you want to optimize.
    Start with a smaller grid for faster iteration.
    """
    return {
        # Learning hyperparameters
        'learning_rate': [5e-4, 1e-3, 2.5e-3],
        'gamma': [0.95, 0.99],
        'gae_lambda': [0.9, 0.95, 0.98],
        
        # PPO-specific
        'clip_range': [0.1, 0.2, 0.3],
        'ent_coef': [0.0, 0.01, 0.02],
        
        # Training parameters
        'n_steps': [128, 256, 512],
        'batch_size': [64, 128, 256],
        
        # Reward shaping (FruitBot specific)
        'fruitbot_reward_positive': [1.0, 2.0, 3.0],
        'fruitbot_reward_negative': [-0.5, -1.0, -2.0],
        'fruitbot_reward_wall_hit': [-1.0, -2.0, -5.0],
    }


def get_default_hyperparameters() -> Dict[str, Any]:
    """Default hyperparameters that remain constant."""
    return {
        'env': 'fruitbot',
        'distribution_mode': 'easy',
        'num_levels': 0,
        'start_level': 0,
        'n_envs': 8,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'device': 'auto',
        'save_freq': 1000000,  # Don't save intermediate checkpoints
        'fruitbot_reward_completion': 10.0,
        'fruitbot_reward_step': 0.0,
    }


def train_with_config(
    config: Dict[str, Any],
    total_steps: int,
    config_id: int,
    results_dir: Path,
    seed: int = None
) -> Path:
    """
    Train a model with the given hyperparameter configuration.
    
    Args:
        config: Dictionary of hyperparameters
        total_steps: Total training timesteps
        config_id: Unique ID for this configuration
        results_dir: Directory to save results
        seed: Random seed
    
    Returns:
        Path to the trained model
    """
    # Build command line arguments
    cmd = [
        'python', 'train.py',
        '--total-steps', str(total_steps),
    ]
    
    # Add all config parameters
    for key, value in config.items():
        param_name = key.replace('_', '-')
        cmd.extend([f'--{param_name}', str(value)])
    
    if seed is not None:
        cmd.extend(['--seed', str(seed)])
    
    print(f"\n{'='*80}")
    print(f"Training Configuration {config_id}")
    print(f"{'='*80}")
    print(json.dumps(config, indent=2))
    print(f"\nCommand: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Training failed for config {config_id}:")
        print(e.stdout)
        print(e.stderr)
        raise
    
    training_time = time.time() - start_time
    
    # Find the most recently created model directory
    models_dir = Path('models') / config['env']
    model_dirs = sorted(models_dir.glob(f"*_{config['distribution_mode']}"), key=lambda x: x.stat().st_mtime)
    
    if not model_dirs:
        raise RuntimeError(f"No model directory found after training config {config_id}")
    
    latest_model_dir = model_dirs[-1]
    model_path = latest_model_dir / "ppo_final.zip"
    
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")
    
    print(f"\nTraining completed in {training_time:.1f}s")
    print(f"Model saved to: {model_path}")
    
    return model_path, training_time


def evaluate_config(
    model_path: Path,
    config: Dict[str, Any],
    num_episodes: int = 100,
    num_parallel: int = 16
) -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the trained model
        config: Hyperparameter configuration used for training
        num_episodes: Number of episodes to evaluate
        num_parallel: Number of parallel environments
    
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\nEvaluating model: {model_path}")
    
    env_id = f"procgen-{config['env']}-v0"
    
    env_kwargs = {
        "distribution_mode": config['distribution_mode'],
        "num_levels": config.get('num_levels', 0),
        "start_level": config.get('start_level', 0),
        "fruitbot_reward_completion": config.get('fruitbot_reward_completion', 10.0),
        "fruitbot_reward_positive": config['fruitbot_reward_positive'],
        "fruitbot_reward_negative": config['fruitbot_reward_negative'],
        "fruitbot_reward_wall_hit": config['fruitbot_reward_wall_hit'],
        "fruitbot_reward_step": config.get('fruitbot_reward_step', 0.0),
    }
    
    stats = evaluate_agent_fast(
        model=PPO.load(str(model_path).replace('.zip', '')),
        env_id=env_id,
        env_kwargs=env_kwargs,
        num_episodes=num_episodes,
        num_parallel=num_parallel,
        fruitbot_reward_positive=config['fruitbot_reward_positive'],
        fruitbot_reward_negative=config['fruitbot_reward_negative'],
        fruitbot_reward_wall_hit=config['fruitbot_reward_wall_hit'],
    )
    
    print(f"Evaluation results:")
    print(f"  Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean steps: {stats['mean_length']:.1f}")
    print(f"  Avg good food: {stats['avg_good_food']:.2f}")
    print(f"  Avg bad food: {stats['avg_bad_food']:.2f}")
    print(f"  Avg wall hits: {stats['avg_wall_hits']:.2f}")
    
    return stats


def random_search(
    grid: Dict[str, List[Any]],
    defaults: Dict[str, Any],
    n_samples: int,
    total_steps: int,
    eval_episodes: int,
    results_dir: Path,
    seed: int = None
) -> pd.DataFrame:
    """
    Perform random search over hyperparameter space.
    More efficient than exhaustive grid search for large spaces.
    """
    results = []
    
    # Generate random configurations
    np.random.seed(seed)
    
    for i in range(n_samples):
        # Sample random hyperparameters
        config = defaults.copy()
        for param, values in grid.items():
            config[param] = np.random.choice(values)
        
        try:
            # Train
            model_path, training_time = train_with_config(
                config=config,
                total_steps=total_steps,
                config_id=i+1,
                results_dir=results_dir,
                seed=seed + i if seed is not None else None
            )
            
            # Evaluate
            eval_stats = evaluate_config(
                model_path=model_path,
                config=config,
                num_episodes=eval_episodes,
                num_parallel=16
            )
            
            # Store results
            result = {
                'config_id': i+1,
                'model_path': str(model_path),
                'training_time': training_time,
                **config,
                **eval_stats
            }
            results.append(result)
            
            # Save incremental results
            df = pd.DataFrame(results)
            df.to_csv(results_dir / 'grid_search_results.csv', index=False)
            print(f"\nProgress: {i+1}/{n_samples} configurations completed")
            
        except Exception as e:
            print(f"\nError with configuration {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return pd.DataFrame(results)


def exhaustive_grid_search(
    grid: Dict[str, List[Any]],
    defaults: Dict[str, Any],
    total_steps: int,
    eval_episodes: int,
    results_dir: Path,
    max_configs: int = None,
    seed: int = None
) -> pd.DataFrame:
    """
    Perform exhaustive grid search over all hyperparameter combinations.
    Warning: Can be very slow for large grids!
    """
    # Generate all combinations
    param_names = list(grid.keys())
    param_values = list(grid.values())
    all_combinations = list(product(*param_values))
    
    total_configs = len(all_combinations)
    print(f"\nTotal configurations to try: {total_configs}")
    
    if max_configs and total_configs > max_configs:
        print(f"Warning: Grid has {total_configs} configs, limiting to {max_configs}")
        np.random.seed(seed)
        selected_indices = np.random.choice(total_configs, max_configs, replace=False)
        all_combinations = [all_combinations[i] for i in selected_indices]
        total_configs = max_configs
    
    results = []
    
    for i, param_combo in enumerate(all_combinations):
        # Build config
        config = defaults.copy()
        for param_name, param_value in zip(param_names, param_combo):
            config[param_name] = param_value
        
        try:
            # Train
            model_path, training_time = train_with_config(
                config=config,
                total_steps=total_steps,
                config_id=i+1,
                results_dir=results_dir,
                seed=seed + i if seed is not None else None
            )
            
            # Evaluate
            eval_stats = evaluate_config(
                model_path=model_path,
                config=config,
                num_episodes=eval_episodes,
                num_parallel=16
            )
            
            # Store results
            result = {
                'config_id': i+1,
                'model_path': str(model_path),
                'training_time': training_time,
                **config,
                **eval_stats
            }
            results.append(result)
            
            # Save incremental results
            df = pd.DataFrame(results)
            df.to_csv(results_dir / 'grid_search_results.csv', index=False)
            print(f"\nProgress: {i+1}/{total_configs} configurations completed")
            
        except Exception as e:
            print(f"\nError with configuration {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return pd.DataFrame(results)


def print_summary(results_df: pd.DataFrame, top_n: int = 5):
    """Print summary of grid search results."""
    if len(results_df) == 0:
        print("No results to summarize!")
        return
    
    print("\n" + "="*80)
    print("GRID SEARCH SUMMARY")
    print("="*80)
    
    # Sort by mean reward
    results_df = results_df.sort_values('mean_reward', ascending=False)
    
    print(f"\nTop {top_n} Configurations by Mean Reward:")
    print("-" * 80)
    
    for i, row in results_df.head(top_n).iterrows():
        print(f"\nRank {row.name + 1}:")
        print(f"  Config ID: {row['config_id']}")
        print(f"  Mean Reward: {row['mean_reward']:.2f} ± {row['std_reward']:.2f}")
        print(f"  Mean Steps: {row['mean_length']:.1f}")
        print(f"  Good Food: {row['avg_good_food']:.2f}")
        print(f"  Bad Food: {row['avg_bad_food']:.2f}")
        print(f"  Wall Hits: {row['avg_wall_hits']:.2f}")
        print(f"  Training Time: {row['training_time']:.1f}s")
        print(f"  Model: {row['model_path']}")
        print(f"  Key Hyperparameters:")
        print(f"    - Learning Rate: {row['learning_rate']}")
        print(f"    - Gamma: {row['gamma']}")
        print(f"    - GAE Lambda: {row['gae_lambda']}")
        print(f"    - Clip Range: {row['clip_range']}")
        print(f"    - Entropy Coef: {row['ent_coef']}")
        print(f"    - N Steps: {row['n_steps']}")
        print(f"    - Batch Size: {row['batch_size']}")
        print(f"    - Good Food Reward: {row['fruitbot_reward_positive']}")
        print(f"    - Bad Food Penalty: {row['fruitbot_reward_negative']}")
        print(f"    - Wall Hit Penalty: {row['fruitbot_reward_wall_hit']}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter grid search for PPO")
    parser.add_argument("--total-steps", type=int, default=500000, 
                        help="Training steps per configuration")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Episodes to evaluate each model")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel training environments")
    parser.add_argument("--search-type", choices=['random', 'exhaustive'], default='random',
                        help="Search strategy: random sampling or exhaustive grid")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of random samples (for random search)")
    parser.add_argument("--max-configs", type=int, default=None,
                        help="Max configs to try (for exhaustive search)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_dir = Path("grid_search_results") / timestamp
    
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Get hyperparameter grid and defaults
    grid = get_hyperparameter_grid()
    defaults = get_default_hyperparameters()
    defaults['n_envs'] = args.n_envs
    
    # Save grid configuration
    with open(results_dir / 'grid_config.json', 'w') as f:
        json.dump({
            'grid': {k: [str(v) for v in vals] for k, vals in grid.items()},
            'defaults': {k: str(v) for k, v in defaults.items()},
            'args': vars(args)
        }, f, indent=2)
    
    # Run search
    if args.search_type == 'random':
        print(f"\nRunning random search with {args.n_samples} samples...")
        results_df = random_search(
            grid=grid,
            defaults=defaults,
            n_samples=args.n_samples,
            total_steps=args.total_steps,
            eval_episodes=args.eval_episodes,
            results_dir=results_dir,
            seed=args.seed
        )
    else:
        print(f"\nRunning exhaustive grid search...")
        results_df = exhaustive_grid_search(
            grid=grid,
            defaults=defaults,
            total_steps=args.total_steps,
            eval_episodes=args.eval_episodes,
            results_dir=results_dir,
            max_configs=args.max_configs,
            seed=args.seed
        )
    
    # Save final results
    results_df.to_csv(results_dir / 'grid_search_results_final.csv', index=False)
    
    # Print summary
    print_summary(results_df)
    
    print(f"\n✓ Grid search complete! Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
