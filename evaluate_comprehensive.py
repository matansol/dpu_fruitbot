#!/usr/bin/env python
"""
Comprehensive Parallel Evaluation Script for FruitBot Agents

This script evaluates multiple trained models across specific environment options
to find interesting setups that highlight behavioral differences between agents.

Environment Configurations (from dpu_clf.ENV_CONFIG_DEFINITIONS):
  Config 0: basic (no walls) - No walls, 4 good food, 4 bad food
  Config 1: basic (with walls) - 3 walls, 3 good food, 3 bad food  
  Config 2: walls_fruits - 4 walls, 6 good food, 0 bad food
  Config 3: walls_doors - 3 walls with 80% door prob, 4 good/bad food

Usage:
    # Quick test with 10 levels
    python evaluate_comprehensive.py --num-levels 10
    
    # Full evaluation with 100 levels
    python evaluate_comprehensive.py --num-levels 100
    
    # Run only specific option
    python evaluate_comprehensive.py --num-levels 10 --options basic walls_fruits
"""

import os
os.environ['PROCGEN_NO_BUILD'] = '1'

import argparse
import time
import json
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd
import gym
import torch

from stable_baselines3 import PPO

# Import centralized environment configurations from dpu_clf
from dpu_clf import (
    ENV_CONFIG_DEFINITIONS,
    BASE_ENV_CONFIG,
    get_config_by_index,
)

# Suppress some warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class EnvConfig:
    """Environment configuration dataclass"""
    start_level: int
    config_name: str = ""  # Name of the configuration (e.g., "no_walls")
    config_index: int = 0  # Index of the configuration (0-3)
    distribution_mode: str = "easy"
    num_levels: int = 1  # Always 1 so start_level determines the environment
    rand_seed: int = 0  # Random seed for environment generation
    
    # FruitBot layout params
    fruitbot_num_walls: int = 3
    fruitbot_num_good_min: int = 5
    fruitbot_num_good_range: int = 2
    fruitbot_num_bad_min: int = 5
    fruitbot_num_bad_range: int = 2
    fruitbot_wall_gap_pct: int = 40
    fruitbot_door_prob_pct: int = 0
    food_diversity: int = 6
    fruitbot_force_no_walls: bool = False

    
    # Reward shaping
    fruitbot_reward_positive: float = 2.0
    fruitbot_reward_negative: float = -1.0
    fruitbot_reward_wall_hit: float = -3.0
    fruitbot_reward_completion: float = 5.0
    fruitbot_reward_step: float = 0.0
    
    def to_env_kwargs(self) -> Dict:
        """Convert to gym.make kwargs"""
        kwargs = {
            "distribution_mode": self.distribution_mode,
            "num_levels": self.num_levels,
            "start_level": self.start_level,
            "fruitbot_num_walls": self.fruitbot_num_walls,
            "fruitbot_num_good_min": self.fruitbot_num_good_min,
            "fruitbot_num_good_range": self.fruitbot_num_good_range,
            "fruitbot_num_bad_min": self.fruitbot_num_bad_min,
            "fruitbot_num_bad_range": self.fruitbot_num_bad_range,
            "fruitbot_wall_gap_pct": self.fruitbot_wall_gap_pct,
            "fruitbot_door_prob_pct": self.fruitbot_door_prob_pct,
            "fruitbot_force_no_walls": self.fruitbot_force_no_walls,
            "food_diversity": self.food_diversity,
            "fruitbot_reward_positive": self.fruitbot_reward_positive,
            "fruitbot_reward_negative": self.fruitbot_reward_negative,
            "fruitbot_reward_wall_hit": self.fruitbot_reward_wall_hit,
            "fruitbot_reward_completion": self.fruitbot_reward_completion,
            "fruitbot_reward_step": self.fruitbot_reward_step,
            "use_discrete_action_wrapper": True,
            "use_stay_bonus_wrapper": False,
            "stay_bonus": 0,
            "rand_seed": self.rand_seed,
        }
        return kwargs
    


@dataclass
class EpisodeResult:
    """Single episode evaluation result"""
    model_index: int
    model_name: str
    model_path: str
    start_level: int
    episode: int
    
    # Environment config info
    config_name: str
    config_index: int
    
    # Performance metrics
    total_reward: float
    episode_length: int
    good_food_collected: int
    bad_food_touched: int
    wall_hits: int
    completed: bool  # Did the agent get the completion reward?
    
    # Environment config (stored for analysis)
    distribution_mode: str
    num_walls: int
    num_good_min: int
    num_good_range: int
    num_bad_min: int
    num_bad_range: int
    door_prob_pct: int
    food_diversity: int
    force_no_walls: bool
    
    # Timing
    eval_time_seconds: float = 0.0


def discover_models(models_dir: Path) -> List[Tuple[int, str, Path]]:
    """
    Discover all trained models from the predefined easy_models_dict.
    
    Returns:
        List of (model_index, model_name, model_path) tuples
    """
    
    easy_models_dict = {
    # Behavior 1: don't open doors and collect all food
    1: {'path': 'models/fruitbot/20260116-074523_easy/ppo_final.zip', 'index': 1, 'name': 'no_doors_collect_all'},

    # Behavior 2: do not open doors and collect only junk
    2: {"path": "models/fruitbot/20260116-210051_easy/ppo_final.zip", 'index': 2, 'name': 'no_doors_junk_only'},
    
    # Behavior 3: don't open doors and collect only fruits
    3: {'path': "models/fruitbot/20260117-134142_easy/ppo_final.zip", 'index': 3, 'name': 'no_doors_fruits_only'},
    
    # Behavior 4: open doors and avoid all foods  
    4: {'path': "models/fruitbot/20260103-073446_easy/ppo_final.zip", 'index': 4, 'name': 'open_doors_avoid_food'},
    
    # Behavior 5: open doors and collect all food - mostly fruits
    5: {'path': "models/fruitbot/20260105-075949_easy/ppo_final.zip", 'index': 5, 'name': 'mostly_fruits_open_doors'}, 

     # Behavior 6: collect only fruits and open doors
    6: {'path': "models/fruitbot/20251231-174002_easy/ppo_final.zip", 'index': 6, 'name': 'open_doors_fruits_only'},
}

    models = []
    for model_info in easy_models_dict.values():
        if model_info is not None:
            model_path = Path(model_info['path'])
            model_name = model_info['name']
            model_index = model_info['index']
            if model_path.exists():
                models.append((model_index, model_name, model_path))
            else:
                print(f"Warning: Model file not found: {model_path}")
    
    return sorted(models, key=lambda x: x[0])


def generate_env_configs(
    num_levels: int = 100,
    level_offset: int = 0,
    include_variations: bool = True,
    quick_mode: bool = False,
    options: Optional[List[str]] = None,
    rand_seed: int = 0,
) -> List[EnvConfig]:
    """
    Generate environment configurations based on specific option types.
    
    Uses centralized configuration definitions from dpu_clf.py.
    
    Args:
        num_levels: Number of different levels to use per variant
        level_offset: Starting level offset
        include_variations: Ignored (kept for API compatibility)
        quick_mode: Ignored (kept for API compatibility)
        options: List of option names to include (default: all)
        rand_seed: Random seed for environment generation (default: 0)
    
    Returns:
        List of EnvConfig objects
    """
    if options is None:
        options = []
    
    configs = []
    levels = list(range(level_offset, level_offset + num_levels))
    
    # Iterate through flat indexed ENV_CONFIG_DEFINITIONS
    for config_idx, config_def in ENV_CONFIG_DEFINITIONS.items():
        config_name = config_def['option_name']
        
        # Generate configs for all levels
        for level in levels:
            # Get the full config from dpu_clf (includes BASE_ENV_CONFIG merged)
            full_config = get_config_by_index(config_idx, rand_seed)
            
            # Create EnvConfig from the full config
            # Extract the fruitbot params from full_config
            config = EnvConfig(
                start_level=level,
                config_name=config_name,
                config_index=config_idx,
                distribution_mode=full_config.get('distribution_mode', 'easy'),
                num_levels=1,  # Always 1 so start_level determines the environment
                rand_seed=rand_seed,
                fruitbot_num_walls=config_def.get('fruitbot_num_walls', 0),
                fruitbot_num_good_min=config_def.get('fruitbot_num_good_min', 4),
                fruitbot_num_good_range=config_def.get('fruitbot_num_good_range', 0),
                fruitbot_num_bad_min=config_def.get('fruitbot_num_bad_min', 4),
                fruitbot_num_bad_range=config_def.get('fruitbot_num_bad_range', 0),
                fruitbot_wall_gap_pct=config_def.get('fruitbot_wall_gap_pct', 0),
                fruitbot_door_prob_pct=config_def.get('fruitbot_door_prob_pct', 0),
                food_diversity=full_config.get('food_diversity', 6),
                fruitbot_force_no_walls=config_def.get('fruitbot_force_no_walls', False),
                fruitbot_reward_positive=full_config.get('fruitbot_reward_positive', 2.0),
                fruitbot_reward_negative=full_config.get('fruitbot_reward_negative', -1.0),
                fruitbot_reward_wall_hit=full_config.get('fruitbot_reward_wall_hit', -3.0),
                fruitbot_reward_completion=full_config.get('fruitbot_reward_completion', 5.0),
                fruitbot_reward_step=full_config.get('fruitbot_reward_step', 0.0),
            )
            configs.append(config)
    
    return configs


def evaluate_single_episode(
    model_path: str,
    env_kwargs: dict,
    reward_positive: float = 2.0,
    reward_negative: float = -1.0,
    reward_wall_hit: float = -3.0,
) -> Dict[str, Any]:
    """
    Evaluate a single episode of the agent.
    
    This function is designed to be called in a separate process.
    
    Returns:
        Dict with episode statistics
    """
    import procgen  # noqa: F401
    
    # Load model
    model_path_str = str(model_path)
    if model_path_str.endswith('.zip'):
        model_path_str = model_path_str[:-4]
    
    model = PPO.load(model_path_str, device='cpu')
    model.policy.eval()
    
    # Create environment
    env = gym.make("procgen-fruitbot-v0", render_mode=None, **env_kwargs)
    
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
    
    TOL = 1e-3
    
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
            
            # Track food/wall events
            if np.isclose(r, reward_positive, atol=TOL, rtol=0.0):
                good_food += 1
            elif np.isclose(r, reward_negative, atol=TOL, rtol=0.0):
                bad_food += 1
            elif np.isclose(r, reward_wall_hit, atol=TOL, rtol=0.0):
                wall_hits += 1
    
    env.close()
    
    return {
        'total_reward': total_reward,
        'episode_length': steps,
        'good_food_collected': good_food,
        'bad_food_touched': bad_food,
        'wall_hits': wall_hits,
    }


def evaluate_model_on_config(
    model_index: int,
    model_name: str,
    model_path: Path,
    config: EnvConfig,
    num_episodes: int = 1,
) -> List[EpisodeResult]:
    """
    Evaluate a model on a specific environment configuration.
    
    Args:
        model_index: Index of the model
        model_name: Name of the model
        model_path: Path to the model file
        config: Environment configuration
        num_episodes: Number of episodes to run
    
    Returns:
        List of EpisodeResult objects
    """
    import procgen  # noqa: F401
    
    results = []
    env_kwargs = config.to_env_kwargs()
    
    start_time = time.time()
    
    try:
        # Load model once for all episodes
        model_path_str = str(model_path)
        if model_path_str.endswith('.zip'):
            model_path_str = model_path_str[:-4]
        
        model = PPO.load(model_path_str, device='cpu')
        model.policy.eval()
        
        for ep in range(num_episodes):
            ep_start = time.time()
            
            # Create fresh environment for each episode
            env = gym.make("procgen-fruitbot-v0", render_mode=None, **env_kwargs)
            
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
            
            with torch.no_grad():
                while not (done or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    result = env.step(action)
                    
                    if len(result) == 4:
                        obs, reward, done, info = result
                        truncated = False
                    else:
                        obs, reward, done, truncated, info = result
                    
                    try:
                        r = float(np.asarray(reward).item())
                    except:
                        r = float(reward)
                    
                    total_reward += r
                    steps += 1
                    
                    # Track food/wall events and completion
                    if np.isclose(r, config.fruitbot_reward_positive, atol=TOL, rtol=0.0):
                        good_food += 1
                    elif np.isclose(r, config.fruitbot_reward_negative, atol=TOL, rtol=0.0):
                        bad_food += 1
                    elif np.isclose(r, config.fruitbot_reward_wall_hit, atol=TOL, rtol=0.0):
                        wall_hits += 1
                    elif np.isclose(r, config.fruitbot_reward_completion, atol=TOL, rtol=0.0):
                        completed = True
            
            env.close()
            
            ep_time = time.time() - ep_start
            
            result = EpisodeResult(
                model_index=model_index,
                model_name=model_name,
                model_path=str(model_path),
                start_level=config.start_level,
                episode=ep,
                config_name=config.config_name,
                config_index=config.config_index,
                total_reward=total_reward,
                episode_length=steps,
                good_food_collected=good_food,
                bad_food_touched=bad_food,
                wall_hits=wall_hits,
                completed=completed,
                distribution_mode=config.distribution_mode,
                num_walls=config.fruitbot_num_walls,
                num_good_min=config.fruitbot_num_good_min,
                num_good_range=config.fruitbot_num_good_range,
                num_bad_min=config.fruitbot_num_bad_min,
                num_bad_range=config.fruitbot_num_bad_range,
                door_prob_pct=config.fruitbot_door_prob_pct,
                food_diversity=config.food_diversity,
                force_no_walls=config.fruitbot_force_no_walls,
                eval_time_seconds=ep_time,
            )
            results.append(result)
            
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        # Return empty results on error
    
    return results


def worker_evaluate_batch(args: Tuple) -> List[EpisodeResult]:
    """
    Worker function for parallel evaluation.
    
    Evaluates multiple (model, config) pairs.
    """
    tasks, num_episodes = args
    all_results = []
    
    for model_index, model_name, model_path, config in tasks:
        results = evaluate_model_on_config(
            model_index=model_index,
            model_name=model_name,
            model_path=model_path,
            config=config,
            num_episodes=num_episodes,
        )
        all_results.extend(results)
    
    return all_results


def run_parallel_evaluation(
    models: List[Tuple[int, str, Path]],
    configs: List[EnvConfig],
    num_episodes_per_config: int = 1,
    num_workers: Optional[int] = None,
    batch_size: int = 10,
) -> List[EpisodeResult]:
    """
    Run parallel evaluation across all model-config combinations.
    
    Args:
        models: List of (model_index, model_name, model_path) tuples
        configs: List of EnvConfig objects
        num_episodes_per_config: Episodes to run per model-config pair
        num_workers: Number of parallel workers (default: CPU count - 1)
        batch_size: Number of tasks per worker batch
    
    Returns:
        List of all EpisodeResult objects
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    # Generate all task combinations
    all_tasks = []
    for model_index, model_name, model_path in models:
        for config in configs:
            all_tasks.append((model_index, model_name, model_path, config))
    
    total_tasks = len(all_tasks)
    print(f"\nTotal evaluation tasks: {total_tasks}")
    print(f"  Models: {len(models)}")
    print(f"  Configs: {len(configs)}")
    print(f"  Episodes per config: {num_episodes_per_config}")
    print(f"  Workers: {num_workers}")
    print(f"  Batch size: {batch_size}")
    
    # Split tasks into batches for workers
    batches = []
    for i in range(0, len(all_tasks), batch_size):
        batch = all_tasks[i:i + batch_size]
        batches.append((batch, num_episodes_per_config))
    
    print(f"  Total batches: {len(batches)}")
    print()
    
    all_results = []
    completed_batches = 0
    start_time = time.time()
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches
        future_to_batch = {
            executor.submit(worker_evaluate_batch, batch): i 
            for i, batch in enumerate(batches)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                results = future.result()
                all_results.extend(results)
                completed_batches += 1
                
                # Progress update
                if completed_batches % 10 == 0 or completed_batches == len(batches):
                    elapsed = time.time() - start_time
                    rate = completed_batches / elapsed if elapsed > 0 else 0
                    eta = (len(batches) - completed_batches) / rate if rate > 0 else 0
                    print(f"Progress: {completed_batches}/{len(batches)} batches "
                          f"({100*completed_batches/len(batches):.1f}%) | "
                          f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
                    
            except Exception as e:
                print(f"Batch {batch_idx} failed: {e}")
    
    total_time = time.time() - start_time
    print(f"\nEvaluation complete!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Total episodes: {len(all_results)}")
    if total_time > 0:
        print(f"Rate: {len(all_results)/total_time:.2f} episodes/second")
    
    return all_results


def save_results(
    results: List[EpisodeResult],
    output_path: Path,
    format: str = "csv",
) -> None:
    """
    Save evaluation results to file.
    
    Args:
        results: List of EpisodeResult objects
        output_path: Path to output file
        format: Output format ('csv', 'parquet', 'json')
    """
    # Convert to DataFrame
    df = pd.DataFrame([asdict(r) for r in results])
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "parquet":
        df.to_parquet(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient="records", indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    print(f"\nResults saved to: {output_path}")
    print(f"Total rows: {len(df)}")


def generate_summary_report(results: List[EpisodeResult], output_dir: Path) -> None:
    """
    Generate summary statistics and analysis from results.
    """
    df = pd.DataFrame([asdict(r) for r in results])
    
    if len(df) == 0:
        print("No results to summarize")
        return
    
    # Model-level summary
    model_summary = df.groupby('model_name').agg({
        'total_reward': ['mean', 'std', 'min', 'max'],
        'episode_length': ['mean', 'std'],
        'good_food_collected': ['mean', 'std'],
        'bad_food_touched': ['mean', 'std'],
        'wall_hits': ['mean', 'std'],
        'start_level': 'count',
    }).round(3)
    model_summary.columns = ['_'.join(col).strip() for col in model_summary.columns.values]
    model_summary = model_summary.rename(columns={'start_level_count': 'num_episodes'})
    model_summary = model_summary.sort_values('total_reward_mean', ascending=False)
    
    # Save model summary
    model_summary_path = output_dir / "model_summary.csv"
    model_summary.to_csv(model_summary_path)
    print(f"\nModel summary saved to: {model_summary_path}")
    
    # Config-level summary by option/variant
    config_summary = df.groupby(['config_name', 'start_level']).agg({
        'total_reward': ['mean', 'std'],
        'good_food_collected': 'mean',
        'bad_food_touched': 'mean',
        'model_name': 'count',
    }).round(3)
    config_summary.columns = ['_'.join(col).strip() for col in config_summary.columns.values]
    config_summary = config_summary.rename(columns={'model_name_count': 'num_models_tested'})
    config_summary = config_summary.sort_values('total_reward_std', ascending=False)
    
    # Save config summary
    config_summary_path = output_dir / "config_summary.csv"
    config_summary.to_csv(config_summary_path)
    print(f"Config summary saved to: {config_summary_path}")
    
    # Interesting environments analysis
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print("\n--- Top 5 Models (by mean reward) ---")
    for idx, (model_name, row) in enumerate(model_summary.head().iterrows()):
        print(f"{idx+1}. {model_name}: {row['total_reward_mean']:.2f} ± {row['total_reward_std']:.2f}")
    
    print("\n--- Model Performance Variance ---")
    print("(High variance = model performs inconsistently across environments)")
    variance_ranking = df.groupby('model_name')['total_reward'].std().sort_values(ascending=False)
    for model_name, std in variance_ranking.head().items():
        print(f"  {model_name}: std={std:.2f}")
    
    # Find environments where models disagree the most (by config/level)
    print("\n--- Most Discriminative Environments (by config) ---")
    print("(High std across models = environments that differentiate agents)")
    env_discrimination = df.groupby(['config_name', 'start_level'])['total_reward'].std().sort_values(ascending=False)
    for idx, std in env_discrimination.head(10).items():
        config_name, level = idx
        print(f"  {config_name}/level{level}: cross-model std={std:.2f}")


def analyze_level_completion(
    results: List[EpisodeResult],
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Analyze levels by counting how many models completed them without hitting walls.
    
    Args:
        results: List of EpisodeResult objects
        output_dir: Directory to save the results
    
    Returns:
        DataFrame with level completion statistics
    """
    df = pd.DataFrame([asdict(r) for r in results])
    
    if len(df) == 0:
        print("No results to analyze")
        return pd.DataFrame()
    
    print("\n" + "="*60)
    print("LEVEL COMPLETION ANALYSIS")
    print("="*60)
    
    # For each level, count how many models completed it without hitting walls
    level_stats = []
    
    for (config_name, level), level_df in df.groupby(['config_name', 'start_level']):
        # Count models that completed and didn't hit walls
        completed_no_walls = level_df[(level_df['completed'] == True) & (level_df['wall_hits'] == 0)]
        num_completed_no_walls = completed_no_walls['model_index'].nunique()
        
        # Count models that completed (regardless of wall hits)
        completed = level_df[level_df['completed'] == True]
        num_completed = completed['model_index'].nunique()
        
        # Count models that hit walls
        hit_walls = level_df[level_df['wall_hits'] > 0]
        num_hit_walls = hit_walls['model_index'].nunique()
        
        # Total models tested
        total_models = level_df['model_index'].nunique()
        
        level_stats.append({
            'config_name': config_name,
            'start_level': level,
            'total_models': total_models,
            'models_completed': num_completed,
            'models_completed_no_walls': num_completed_no_walls,
            'models_hit_walls': num_hit_walls,
            'completion_rate': num_completed / total_models if total_models > 0 else 0,
            'no_wall_completion_rate': num_completed_no_walls / total_models if total_models > 0 else 0,
        })
    
    level_completion_df = pd.DataFrame(level_stats)
    level_completion_df = level_completion_df.sort_values('models_completed_no_walls', ascending=False)
    
    # Save to file
    if output_dir:
        levels_csv_path = output_dir / "level_completion_analysis.csv"
        level_completion_df.to_csv(levels_csv_path, index=False)
        print(f"\nLevel completion analysis saved to: {levels_csv_path}")
    
    # Print summary
    print("\n--- Top 10 Levels by Models Completing Without Walls ---")
    for idx, row in level_completion_df.head(10).iterrows():
        print(f"  {row['config_name']}/level{row['start_level']}: "
              f"{row['models_completed_no_walls']}/{row['total_models']} models "
              f"({row['no_wall_completion_rate']*100:.0f}%)")
    
    return level_completion_df


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive parallel evaluation of FruitBot agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Options:
  OPTION: BASIC
  ────────────────────────────────────────────────────────────────────────────────
    [0] g4_b4 - Walls: 0 | Force no walls: True
    [1] g4_b4 - Walls: 3 | Force no walls: False

  OPTION: WALLS_FRUITS
  ────────────────────────────────────────────────────────────────────────────────
    [2] g6_b0 - Walls: 4 | Force no walls: False

  OPTION: WALLS_DOORS
  ────────────────────────────────────────────────────────────────────────────────
    [3] d80_g4_b4 - Walls: 3 (with 80% door prob) | Force no walls: False

Examples:
    # Quick test with 10 levels, find top 2 interesting per option
    python evaluate_comprehensive.py --num-levels 10 --top-k 2
    
    # Full evaluation with 100 levels, find top 10 interesting per option
    python evaluate_comprehensive.py --num-levels 100 --top-k 10
    
    # Run only specific options
    python evaluate_comprehensive.py --num-levels 10 --options basic walls_fruits
        """
    )
    
    parser.add_argument("--models-dir", type=str, default="models/fruitbot",
                        help="Directory containing trained models")
    parser.add_argument("--output", type=str, default="results/comprehensive_eval.csv",
                        help="Output file path for results")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "parquet", "json"],
                        help="Output format")
    
    # Evaluation parameters
    parser.add_argument("--num-levels", type=int, default=100,
                        help="Number of different levels to evaluate per variant")
    parser.add_argument("--level-offset", type=int, default=0,
                        help="Starting level offset")
    parser.add_argument("--rand-seed", type=int, default=0,
                        help="Random seed for environment generation (default: 0)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="(Deprecated) Kept for compatibility")
    parser.add_argument("--episodes-per-config", type=int, default=1,
                        help="Number of episodes per model-config pair")
    parser.add_argument("--options", type=str, nargs="+", default=None,
                        choices=[],
                        help="Which environment options to evaluate (default: all)")
    
    # Parallelization
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count - 1)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Tasks per worker batch")
    
    # Convenience flags
    parser.add_argument("--no-summary", action="store_true",
                        help="Skip summary report generation")
    
    args = parser.parse_args()
    
    print("="*60)
    print("COMPREHENSIVE FRUITBOT AGENT EVALUATION")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Discover models
    models_dir = Path(args.models_dir)
    models = discover_models(models_dir)
    
    if not models:
        print(f"\nError: No models found in {models_dir}")
        print("Please check the models directory path.")
        return
    
    print(f"\nDiscovered {len(models)} models:")
    for idx, name, path in models:
        print(f"  - [{idx}] {name}")
    
    # Generate environment configs
    configs = generate_env_configs(
        num_levels=args.num_levels,
        level_offset=args.level_offset,
        rand_seed=args.rand_seed,
    )
    
    print(f"Levels per variant: {args.num_levels}")
    print(f"Total configurations: {len(configs)}")
    
    # Count configs per option
    from collections import Counter
    config_counts = Counter(c.config_name for c in configs)
    for config_name, count in sorted(config_counts.items()):
        print(f"  - {config_name}: {count} configs")
    
    # Estimate total work
    total_episodes = len(models) * len(configs) * args.episodes_per_config
    print(f"\nTotal episodes to evaluate: {total_episodes}")
    
    # Run evaluation
    results = run_parallel_evaluation(
        models=models,
        configs=configs,
        num_episodes_per_config=args.episodes_per_config,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    
    # Save results
    output_path = Path(args.output)
    save_results(results, output_path, format=args.format)
    
    # Save detailed models_result.csv only if we have results
    if len(results) > 0:
        models_result_path = output_path.parent / "models_result.csv"
        df_detailed = pd.DataFrame([asdict(r) for r in results])
        # Select key columns for models_result
        cols_to_save = ['model_index', 'model_name', 'config_name', 'config_index', 'start_level', 
                        'total_reward', 'good_food_collected', 'bad_food_touched', 
                        'wall_hits', 'completed', 'episode_length']
        df_detailed[cols_to_save].to_csv(models_result_path, index=False)
        print(f"\nDetailed models results saved to: {models_result_path}")
    else:
        print("\nNo results to save. Check configuration and model paths.")
    
    # Generate summary report
    if not args.no_summary and len(results) > 0:
        generate_summary_report(results, output_path.parent)
    
    # Analyze level completion
    if len(results) > 0:
        level_completion_df = analyze_level_completion(
            results,
            output_dir=output_path.parent,
        )
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()
