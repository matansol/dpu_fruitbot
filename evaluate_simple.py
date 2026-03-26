#!/usr/bin/env python
"""
Simple Evaluation Script for FruitBot Agents

A simplified version of evaluate_comprehensive.py that's easy to understand and modify.
Evaluates models on different levels using dpu_clf.get_config_by_index().

Usage:
    python evaluate_simple.py --num-levels 10 --config-index 0
    python evaluate_simple.py --num-levels 100 --config-index 1 --output results/my_eval.csv
"""

import os
os.environ['PROCGEN_NO_BUILD'] = '1'

import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import gym
import torch
from stable_baselines3 import PPO

import dpu_clf


# Model definitions - same as evaluate_comprehensive.py
MODELS = {
    1: {'path': 'models/fruitbot/20260116-074523_easy/ppo_final.zip', 'name': 'no_doors_collect_all'},
    2: {'path': 'models/fruitbot/20260116-210051_easy/ppo_final.zip', 'name': 'no_doors_junk_only'},
    3: {'path': 'models/fruitbot/20260117-134142_easy/ppo_final.zip', 'name': 'no_doors_fruits_only'},
    4: {'path': 'models/fruitbot/20260103-073446_easy/ppo_final.zip', 'name': 'open_doors_avoid_food'},
    5: {'path': 'models/fruitbot/20260105-075949_easy/ppo_final.zip', 'name': 'mostly_fruits_open_doors'},
    6: {'path': 'models/fruitbot/20251231-174002_easy/ppo_final.zip', 'name': 'open_doors_fruits_only'},
}


def evaluate_model_on_level(model_path, config_index, start_level, rand_seed=0):
    """
    Evaluate a single model on a single level.
    
    Args:
        model_path: Path to the model .zip file
        config_index: Configuration index (0-6)
        start_level: Level to evaluate on
        rand_seed: Random seed
    
    Returns:
        Dictionary with episode statistics
    """
    # Get environment configuration using dpu_clf
    env_config = dpu_clf.get_config_by_index(config_index, rand_seed)
    
    # Set level parameters - THIS IS KEY!
    env_config['num_levels'] = 1  # Always 1 so start_level determines the environment
    env_config['start_level'] = start_level  # This is what creates different environments
    env_config['rand_seed'] = rand_seed
    
    # Load model
    model = PPO.load(model_path, device='cpu')
    model.policy.eval()
    
    # Create environment (note: procgen: prefix is required)
    env = gym.make("procgen:procgen-fruitbot-v0", render_mode=None, **env_config)
    
    # Run episode
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs
    
    done = False
    truncated = False
    total_reward = 0.0
    steps = 0
    good_food = 0
    bad_food = 0
    wall_hits = 0
    completed = False
    
    TOL = 1e-3
    reward_positive = env_config.get('fruitbot_reward_positive', 2.0)
    reward_negative = env_config.get('fruitbot_reward_negative', -1.0)
    reward_wall_hit = env_config.get('fruitbot_reward_wall_hit', -3.0)
    reward_completion = env_config.get('fruitbot_reward_completion', 10.0)
    
    with torch.no_grad():
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            result = env.step(action)
            
            if len(result) == 4:
                obs, reward, done, info = result
                truncated = False
            else:
                obs, reward, done, truncated, info = result
            
            r = float(np.asarray(reward).item()) if hasattr(reward, 'item') else float(reward)
            total_reward += r
            steps += 1
            
            # Track events
            if np.isclose(r, reward_positive, atol=TOL, rtol=0.0):
                good_food += 1
            elif np.isclose(r, reward_negative, atol=TOL, rtol=0.0):
                bad_food += 1
            elif np.isclose(r, reward_wall_hit, atol=TOL, rtol=0.0):
                wall_hits += 1
            elif np.isclose(r, reward_completion, atol=TOL, rtol=0.0):
                completed = True
    
    env.close()
    
    return {
        'total_reward': total_reward,
        'episode_length': steps,
        'good_food_collected': good_food,
        'bad_food_touched': bad_food,
        'wall_hits': wall_hits,
        'completed': completed,
    }


def main():
    parser = argparse.ArgumentParser(description="Simple FruitBot agent evaluation")
    
    parser.add_argument("--num-levels", type=int, default=3,
                        help="Number of different levels to evaluate")
    parser.add_argument("--level-offset", type=int, default=0,
                        help="Starting level number")
    parser.add_argument("--config-index", type=int, default=0,
                        help="Configuration index (0=no_walls, 1=walls_food, 2=walls_fruits, 3=walls_doors)")
    parser.add_argument("--rand-seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--output", type=str, default="results/simple_eval.csv",
                        help="Output CSV file path")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SIMPLE FRUITBOT EVALUATION")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    all_results = []
    for config_index in range(4):
        print(f"\nConfiguration:")
        print(f"  Config index: {config_index}")
        print(f"  Levels: {args.level_offset} to {args.level_offset + args.num_levels - 1}")
        print(f"  Random seed: {args.rand_seed}")
        print(f"  Models: {len(MODELS)}")
        print()
        
        # Get config info from dpu_clf
        config_info = dpu_clf.ENV_CONFIG_DEFINITIONS.get(config_index, {})
        config_name = config_info.get('option_name', f'config_{config_index}')
        print(f"Config name: {config_name}")
        print(f"  Walls: {config_info.get('fruitbot_num_walls', '?')}")
        print(f"  Good food: {config_info.get('fruitbot_num_good_min', '?')}")
        print(f"  Bad food: {config_info.get('fruitbot_num_bad_min', '?')}")
        print(f"  Door prob: {config_info.get('fruitbot_door_prob_pct', 0)}%")
        print()
        
        # Collect results
        
        total_episodes = len(MODELS) * args.num_levels
        completed = 0
        
        start_time = time.time()
        
        # Loop through all models and levels
        for model_idx, model_info in MODELS.items():
            model_path = model_info['path']
            model_name = model_info['name']
            
            # Check if model exists
            if not Path(model_path).exists():
                print(f"Warning: Model not found: {model_path}")
                continue
            
            print(f"\nEvaluating model {model_idx}: {model_name}")
            
            for level_num in range(args.level_offset, args.level_offset + args.num_levels):
                start_level = level_num
                
                # Evaluate this model on this level
                try:
                    result = evaluate_model_on_level(
                        model_path=model_path,
                        config_index=config_index,
                        start_level=start_level,
                        rand_seed=args.rand_seed,
                    )
                    
                    # Add metadata
                    result['model_index'] = model_idx
                    result['model_name'] = model_name
                    result['config_index'] = config_index
                    result['config_name'] = config_name
                    result['start_level'] = start_level
                    
                    all_results.append(result)
                    completed += 1
                    
                    # Progress update
                    if completed % 10 == 0 or completed == total_episodes:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total_episodes - completed) / rate if rate > 0 else 0
                        print(f"  Progress: {completed}/{total_episodes} ({100*completed/total_episodes:.1f}%) | "
                            f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
                
                except Exception as e:
                    print(f"  Error on level {start_level}: {e}")
                    continue
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"Evaluation complete!")
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"Results saved to: {output_path}")
        print(f"Total episodes: {len(df)}")
        print(f"\nColumns: {list(df.columns)}")
        
        # Quick summary
        print(f"\nQuick summary:")
        print(f"  Mean reward: {df['total_reward'].mean():.2f}")
        print(f"  Mean good food: {df['good_food_collected'].mean():.2f}")
        print(f"  Mean bad food: {df['bad_food_touched'].mean():.2f}")
        print(f"  Mean wall hits: {df['wall_hits'].mean():.2f}")
        print(f"  Completion rate: {df['completed'].mean()*100:.1f}%")
        print(f"{'='*60}")
    else:
        print("\nNo results collected!")


if __name__ == "__main__":
    main()
