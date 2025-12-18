#!/usr/bin/env python
"""
Quick hyperparameter search - smaller grid for faster iteration.

This is a simplified version that focuses on the most impactful hyperparameters
for FruitBot learning.

Usage:
    python quick_grid_search.py

This will train ~10-20 models with different configurations and show you
which combinations work best.
"""

import subprocess
import time
import json
from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO
from evaluate import evaluate_agent_fast


# Small, focused grid for quick iteration
CONFIGS = [
    # Baseline
    {
        'name': 'baseline',
        'learning_rate': 5e-4,
        'gamma': 0.99,
        'ent_coef': 0.01,
        'clip_range': 0.2,
        'n_steps': 256,
        'batch_size': 256,
        'reward_positive': 2.0,
        'reward_negative': -1.0,
        'reward_wall': -2.0,
    },
    # Higher learning rate
    {
        'name': 'high_lr',
        'learning_rate': 2e-3,
        'gamma': 0.99,
        'ent_coef': 0.01,
        'clip_range': 0.2,
        'n_steps': 256,
        'batch_size': 256,
        'reward_positive': 2.0,
        'reward_negative': -1.0,
        'reward_wall': -2.0,
    },
    # More exploration (higher entropy)
    {
        'name': 'high_entropy',
        'learning_rate': 5e-4,
        'gamma': 0.99,
        'ent_coef': 0.05,
        'clip_range': 0.2,
        'n_steps': 256,
        'batch_size': 256,
        'reward_positive': 2.0,
        'reward_negative': -1.0,
        'reward_wall': -2.0,
    },
    # Stronger penalties for bad behavior
    {
        'name': 'strong_penalties',
        'learning_rate': 5e-4,
        'gamma': 0.99,
        'ent_coef': 0.01,
        'clip_range': 0.2,
        'n_steps': 256,
        'batch_size': 256,
        'reward_positive': 3.0,
        'reward_negative': -2.0,
        'reward_wall': -5.0,
    },
    # Higher gamma (more long-term thinking)
    {
        'name': 'high_gamma',
        'learning_rate': 5e-4,
        'gamma': 0.995,
        'ent_coef': 0.01,
        'clip_range': 0.2,
        'n_steps': 256,
        'batch_size': 256,
        'reward_positive': 2.0,
        'reward_negative': -1.0,
        'reward_wall': -2.0,
    },
    # More steps per update
    {
        'name': 'more_steps',
        'learning_rate': 5e-4,
        'gamma': 0.99,
        'ent_coef': 0.01,
        'clip_range': 0.2,
        'n_steps': 512,
        'batch_size': 512,
        'reward_positive': 2.0,
        'reward_negative': -1.0,
        'reward_wall': -2.0,
    },
    # Conservative clipping
    {
        'name': 'conservative_clip',
        'learning_rate': 5e-4,
        'gamma': 0.99,
        'ent_coef': 0.01,
        'clip_range': 0.1,
        'n_steps': 256,
        'batch_size': 256,
        'reward_positive': 2.0,
        'reward_negative': -1.0,
        'reward_wall': -2.0,
    },
    # Aggressive clipping
    {
        'name': 'aggressive_clip',
        'learning_rate': 5e-4,
        'gamma': 0.99,
        'ent_coef': 0.01,
        'clip_range': 0.3,
        'n_steps': 256,
        'batch_size': 256,
        'reward_positive': 2.0,
        'reward_negative': -1.0,
        'reward_wall': -2.0,
    },
    # Combined: Strong penalties + high entropy
    {
        'name': 'penalties_exploration',
        'learning_rate': 5e-4,
        'gamma': 0.99,
        'ent_coef': 0.03,
        'clip_range': 0.2,
        'n_steps': 256,
        'batch_size': 256,
        'reward_positive': 3.0,
        'reward_negative': -2.0,
        'reward_wall': -5.0,
    },
    # Combined: High LR + more steps
    {
        'name': 'high_lr_more_steps',
        'learning_rate': 2e-3,
        'gamma': 0.99,
        'ent_coef': 0.01,
        'clip_range': 0.2,
        'n_steps': 512,
        'batch_size': 512,
        'reward_positive': 2.0,
        'reward_negative': -1.0,
        'reward_wall': -2.0,
    },
]


def train_config(config, total_steps=500000, n_envs=8, seed=42):
    """Train a model with the given config."""
    cmd = [
        'python', 'train.py',
        '--env', 'fruitbot',
        '--distribution-mode', 'easy',
        '--total-steps', str(total_steps),
        '--n-envs', str(n_envs),
        '--learning-rate', str(config['learning_rate']),
        '--gamma', str(config['gamma']),
        '--ent-coef', str(config['ent_coef']),
        '--clip-range', str(config['clip_range']),
        '--n-steps', str(config['n_steps']),
        '--batch-size', str(config['batch_size']),
        '--fruitbot-reward-positive', str(config['reward_positive']),
        '--fruitbot-reward-negative', str(config['reward_negative']),
        '--fruitbot-reward-wall-hit', str(config['reward_wall']),
        '--seed', str(seed),
    ]
    
    print(f"\n{'='*80}")
    print(f"Training: {config['name']}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start
    
    if result.returncode != 0:
        print(f"ERROR: Training failed for {config['name']}")
        print(result.stderr)
        return None, duration
    
    # Find model path
    models_dir = Path('models/fruitbot')
    model_dirs = sorted(models_dir.glob("*_easy"), key=lambda x: x.stat().st_mtime)
    if model_dirs:
        model_path = model_dirs[-1] / "ppo_final.zip"
        return model_path, duration
    
    return None, duration


def evaluate_model(model_path, config, num_episodes=50):
    """Evaluate a trained model."""
    if not model_path or not model_path.exists():
        return None
    
    print(f"\nEvaluating {config['name']}...")
    
    import gym
    env_kwargs = {
        "distribution_mode": "easy",
        "num_levels": 0,
        "start_level": 0,
        "fruitbot_reward_positive": config['reward_positive'],
        "fruitbot_reward_negative": config['reward_negative'],
        "fruitbot_reward_wall_hit": config['reward_wall'],
    }
    
    stats = evaluate_agent_fast(
        model=PPO.load(str(model_path).replace('.zip', '')),
        env_id="procgen-fruitbot-v0",
        env_kwargs=env_kwargs,
        num_episodes=num_episodes,
        num_parallel=8,
        fruitbot_reward_positive=config['reward_positive'],
        fruitbot_reward_negative=config['reward_negative'],
        fruitbot_reward_wall_hit=config['reward_wall'],
    )
    
    return stats


def main():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = Path("grid_search_results") / f"quick_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Quick Grid Search - Testing {len(CONFIGS)} configurations")
    print(f"Results will be saved to: {results_dir}\n")
    
    results = []
    
    for i, config in enumerate(CONFIGS, 1):
        print(f"\n{'#'*80}")
        print(f"Configuration {i}/{len(CONFIGS)}: {config['name']}")
        print(f"{'#'*80}")
        
        # Train
        model_path, train_time = train_config(config, total_steps=500000)
        
        if model_path is None:
            print(f"Skipping {config['name']} - training failed")
            continue
        
        # Evaluate
        eval_stats = evaluate_model(model_path, config, num_episodes=50)
        
        if eval_stats is None:
            print(f"Skipping {config['name']} - evaluation failed")
            continue
        
        # Store results
        result = {
            'config_name': config['name'],
            'model_path': str(model_path),
            'train_time': train_time,
            'mean_reward': eval_stats['mean_reward'],
            'std_reward': eval_stats['std_reward'],
            'mean_steps': eval_stats['mean_length'],
            'good_food': eval_stats['avg_good_food'],
            'bad_food': eval_stats['avg_bad_food'],
            'wall_hits': eval_stats['avg_wall_hits'],
            **config
        }
        results.append(result)
        
        # Save incremental
        df = pd.DataFrame(results)
        df.to_csv(results_dir / 'results.csv', index=False)
        
        print(f"\n✓ {config['name']} complete:")
        print(f"  Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
        print(f"  Good food: {eval_stats['avg_good_food']:.2f}")
        print(f"  Bad food: {eval_stats['avg_bad_food']:.2f}")
        print(f"  Wall hits: {eval_stats['avg_wall_hits']:.2f}")
    
    # Final summary
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('mean_reward', ascending=False)
        
        print("\n" + "="*80)
        print("FINAL RESULTS (sorted by mean reward)")
        print("="*80 + "\n")
        
        print(df[['config_name', 'mean_reward', 'good_food', 'bad_food', 'wall_hits', 'mean_steps']].to_string(index=False))
        
        print("\n" + "="*80)
        print("TOP 3 CONFIGURATIONS:")
        print("="*80)
        
        for i, row in df.head(3).iterrows():
            print(f"\n{i+1}. {row['config_name']}")
            print(f"   Reward: {row['mean_reward']:.2f} ± {row['std_reward']:.2f}")
            print(f"   Good food: {row['good_food']:.2f}, Bad food: {row['bad_food']:.2f}, Wall hits: {row['wall_hits']:.2f}")
            print(f"   LR: {row['learning_rate']}, Gamma: {row['gamma']}, Ent: {row['ent_coef']}")
            print(f"   Model: {row['model_path']}")
        
        df.to_csv(results_dir / 'final_results.csv', index=False)
        print(f"\n✓ Results saved to: {results_dir}")
    else:
        print("\n❌ No successful runs!")


if __name__ == "__main__":
    main()
