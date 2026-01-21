#!/usr/bin/env python
"""
Comprehensive Parallel Evaluation Script for FruitBot Agents

This script evaluates multiple trained models across specific environment options
to find interesting setups that highlight behavioral differences between agents.

Environment Options:
  1. BASIC (no walls, all food):
     - 0 walls with (good, bad) foods: (3,6), (6,2), (6,6) with range 1
     
  2. WALLS_FRUITS (walls, only good fruits):
     - 4 walls with (good, bad) foods: (4,0), (8,0) with range 0
     
  3. WALLS_DOORS (walls with doors, all food):
     - 3 walls with 20/60% door prob, (good, bad): (6,2), (6,6) with range 1

Usage:
    # Quick test with 10 seeds, find top 2 interesting per option
    python evaluate_comprehensive.py --num-seeds 10 --top-k 2
    
    # Full evaluation with 100 seeds, find top 10 interesting per option
    python evaluate_comprehensive.py --num-seeds 100 --top-k 10
    
    # Run only specific option
    python evaluate_comprehensive.py --num-seeds 10 --options basic walls_fruits
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
    ENV_OPTION_BASIC,
    ENV_OPTION_WALLS_FRUITS,
    ENV_OPTION_WALLS_DOORS,
    ALL_ENV_OPTIONS,
    ENV_CONFIG_DEFINITIONS,
    BASE_ENV_CONFIG,
    get_env_config,
    get_all_variants,
)

# Suppress some warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class EnvConfig:
    """Environment configuration dataclass"""
    seed: int
    option_name: str = ""  # Name of the environment option (basic, walls_fruits, walls_doors)
    option_variant: str = ""  # Specific variant within the option (e.g., "g3_b6", "d20_g6_b2")
    distribution_mode: str = "easy"
    num_levels: int = 1
    start_level: int = 0
    
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
            "rand_seed": self.seed,
        }
        return kwargs
    
    def config_id(self) -> str:
        """Generate a unique ID for this configuration"""
        if self.option_name and self.option_variant:
            return f"{self.option_name}_{self.option_variant}_s{self.seed}"
        parts = [
            f"s{self.seed}",
            f"w{self.fruitbot_num_walls}",
            f"g{self.fruitbot_num_good_min}-{self.fruitbot_num_good_range}",
            f"b{self.fruitbot_num_bad_min}-{self.fruitbot_num_bad_range}",
            f"d{self.fruitbot_door_prob_pct}",
        ]

        return "_".join(parts)


@dataclass
class EpisodeResult:
    """Single episode evaluation result"""
    model_name: str
    model_path: str
    config_id: str
    seed: int
    episode: int
    
    # Environment option info
    option_name: str
    option_variant: str
    
    # Performance metrics
    total_reward: float
    episode_length: int
    good_food_collected: int
    bad_food_touched: int
    wall_hits: int
    
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


def discover_models(models_dir: Path) -> List[Tuple[str, Path]]:
    """
    Discover all trained models from the predefined easy_models_dict.
    
    Returns:
        List of (model_name, model_path) tuples
    """
    
    easy_models_dict = {
    # Behavior 1: avoid walls and randomly collect food
    0: {'path': "models/fruitbot/20251223-133810_easy/ppo_final.zip", 'index': 0, 'name': 'random_food_avoid_walls'},
    
    # Behavior 2: 
    1: {'path': "models/fruitbot/20251223-133810_easy/ppo_final.zip", 'index': 1, 'name': 'avoid_walls_random_food'},
    
    # Behavior 3: don't open doors and collect all food
    2: {'path': 'models/fruitbot/20260116-074523_easy/ppo_final.zip', 'index': 2, 'name': 'no_doors_collect_all'},
    
    # Behavior 4: don't open doors and collect only fruits
    3: {'path': "models/fruitbot/20260117-134142_easy/ppo_final.zip", 'index': 3, 'name': 'no_doors_fruits_only'},
    
    # Behavior 5: collect only fruits and open doors
    4: {'path': "models/fruitbot/20251231-174002_easy/ppo_final.zip", 'index': 4, 'name': 'open_doors_fruits_only'},
    
    # Behavior 6: open doors and collect all foods
    5: {'path': "models/fruitbot/20260121-152950_easy/ppo_final.zip", 'index': 5, 'name': 'open_doors_collect_all'},
    
    # Behavior 7: open doors and avoid all foods  
    6: {'path': "models/fruitbot/20260103-073446_easy/ppo_final.zip", 'index': 6, 'name': 'open_doors_avoid_food'},
    
    # Behavior 8: try to open doors and collect only fruits
    7: {'path': "models/fruitbot/20260105-075949_easy/ppo_final.zip", 'index': 7, 'name': 'only_fruits_tries_open_doors'},

    # Behavior 9: do not open doors and collect only junk
    8: {"path": "models/fruitbot/20260116-210051_easy/ppo_final.zip", 'index': 8, 'name': 'no_doors_junk_only'},
}
    models = []
    for model_info in easy_models_dict.values():
        if model_info is not None:
            model_path = Path(model_info['path'])
            model_name = model_info['name']
            if model_path.exists():
                models.append((model_name, model_path))
            else:
                print(f"Warning: Model file not found: {model_path}")
    
    return sorted(models, key=lambda x: x[0])


def generate_env_configs(
    num_seeds: int = 100,
    seed_offset: int = 1000,
    include_variations: bool = True,
    quick_mode: bool = False,
    options: Optional[List[str]] = None,
) -> List[EnvConfig]:
    """
    Generate environment configurations based on specific option types.
    
    Uses centralized configuration definitions from dpu_clf.py.
    
    Args:
        num_seeds: Number of random seeds to use per variant
        seed_offset: Starting seed offset
        include_variations: Ignored (kept for API compatibility)
        quick_mode: Ignored (kept for API compatibility)
        options: List of option names to include (default: all)
    
    Returns:
        List of EnvConfig objects
    """
    if options is None:
        options = ALL_ENV_OPTIONS
    
    configs = []
    seeds = list(range(seed_offset, seed_offset + num_seeds))
    
    for option_name in options:
        if option_name not in ENV_CONFIG_DEFINITIONS:
            print(f"Warning: Unknown option {option_name}, skipping")
            continue
            
        option_def = ENV_CONFIG_DEFINITIONS[option_name]
        base_params = option_def['base_params']
        
        for seed in seeds:
            for variant in option_def['variants']:
                # Build config from centralized definitions
                config = EnvConfig(
                    seed=seed,
                    start_level=seed,
                    option_name=option_name,
                    option_variant=variant['name'],
                    fruitbot_num_walls=base_params.get('fruitbot_num_walls', 3),
                    fruitbot_force_no_walls=base_params.get('fruitbot_force_no_walls', False),
                    fruitbot_num_good_min=variant.get('fruitbot_num_good_min', 5),
                    fruitbot_num_good_range=base_params.get('fruitbot_num_good_range', 1),
                    fruitbot_num_bad_min=variant.get('fruitbot_num_bad_min', 5),
                    fruitbot_num_bad_range=base_params.get('fruitbot_num_bad_range', 1),
                    fruitbot_wall_gap_pct=base_params.get('fruitbot_wall_gap_pct', 40),
                    fruitbot_door_prob_pct=variant.get('fruitbot_door_prob_pct', base_params.get('fruitbot_door_prob_pct', 0)),
                    food_diversity=BASE_ENV_CONFIG.get('food_diversity', 4),
                )
                configs.append(config)
    
    return configs


def get_option_variants(option_name: str) -> List[str]:
    """Get all variant names for a given option using centralized definitions."""
    if option_name in ENV_CONFIG_DEFINITIONS:
        return [v['name'] for v in ENV_CONFIG_DEFINITIONS[option_name]['variants']]
    return []


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
    model_name: str,
    model_path: Path,
    config: EnvConfig,
    num_episodes: int = 1,
) -> List[EpisodeResult]:
    """
    Evaluate a model on a specific environment configuration.
    
    Args:
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
                    
                    # Track food/wall events
                    if np.isclose(r, config.fruitbot_reward_positive, atol=TOL, rtol=0.0):
                        good_food += 1
                    elif np.isclose(r, config.fruitbot_reward_negative, atol=TOL, rtol=0.0):
                        bad_food += 1
                    elif np.isclose(r, config.fruitbot_reward_wall_hit, atol=TOL, rtol=0.0):
                        wall_hits += 1
            
            env.close()
            
            ep_time = time.time() - ep_start
            
            result = EpisodeResult(
                model_name=model_name,
                model_path=str(model_path),
                config_id=config.config_id(),
                seed=config.seed,
                episode=ep,
                option_name=config.option_name,
                option_variant=config.option_variant,
                total_reward=total_reward,
                episode_length=steps,
                good_food_collected=good_food,
                bad_food_touched=bad_food,
                wall_hits=wall_hits,
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
        print(f"Error evaluating {model_name} on config {config.config_id()}: {e}")
        # Return empty results on error
    
    return results


def worker_evaluate_batch(args: Tuple) -> List[EpisodeResult]:
    """
    Worker function for parallel evaluation.
    
    Evaluates multiple (model, config) pairs.
    """
    tasks, num_episodes = args
    all_results = []
    
    for model_name, model_path, config in tasks:
        results = evaluate_model_on_config(
            model_name=model_name,
            model_path=model_path,
            config=config,
            num_episodes=num_episodes,
        )
        all_results.extend(results)
    
    return all_results


def run_parallel_evaluation(
    models: List[Tuple[str, Path]],
    configs: List[EnvConfig],
    num_episodes_per_config: int = 1,
    num_workers: Optional[int] = None,
    batch_size: int = 10,
) -> List[EpisodeResult]:
    """
    Run parallel evaluation across all model-config combinations.
    
    Args:
        models: List of (model_name, model_path) tuples
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
    for model_name, model_path in models:
        for config in configs:
            all_tasks.append((model_name, model_path, config))
    
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
        'seed': 'count',
    }).round(3)
    model_summary.columns = ['_'.join(col).strip() for col in model_summary.columns.values]
    model_summary = model_summary.rename(columns={'seed_count': 'num_episodes'})
    model_summary = model_summary.sort_values('total_reward_mean', ascending=False)
    
    # Save model summary
    model_summary_path = output_dir / "model_summary.csv"
    model_summary.to_csv(model_summary_path)
    print(f"\nModel summary saved to: {model_summary_path}")
    
    # Config-level summary by option/variant
    config_summary = df.groupby(['option_name', 'option_variant', 'seed']).agg({
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
    
    # Find environments where models disagree the most (by option/variant/seed)
    print("\n--- Most Discriminative Environments (by option/variant) ---")
    print("(High std across models = environments that differentiate agents)")
    env_discrimination = df.groupby(['option_name', 'option_variant', 'seed'])['total_reward'].std().sort_values(ascending=False)
    for idx, std in env_discrimination.head(10).items():
        opt, var, seed = idx
        print(f"  {opt}/{var}/seed{seed}: cross-model std={std:.2f}")


def find_interesting_seeds(
    results: List[EpisodeResult],
    top_k: int = 10,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, List[int]]]:
    """
    Find the most interesting seeds for each option/variant combination.
    
    "Interesting" is defined as seeds where there is high variance in agent
    performance (i.e., environments that differentiate between agents).
    
    Args:
        results: List of EpisodeResult objects
        top_k: Number of top seeds to select per variant
        output_dir: Directory to save the results
    
    Returns:
        Dict mapping option_name -> variant -> list of top_k seed numbers
    """
    df = pd.DataFrame([asdict(r) for r in results])
    
    if len(df) == 0:
        print("No results to analyze")
        return {}
    
    interesting_seeds = {}
    
    print("\n" + "="*60)
    print(f"FINDING TOP {top_k} INTERESTING SEEDS PER VARIANT")
    print("="*60)
    
    # Group by option_name and option_variant
    for option_name in df['option_name'].unique():
        option_df = df[df['option_name'] == option_name]
        interesting_seeds[option_name] = {}
        
        print(f"\n--- Option: {option_name} ---")
        
        for variant in option_df['option_variant'].unique():
            variant_df = option_df[option_df['option_variant'] == variant]
            
            # Calculate variance across models for each seed
            seed_variance = variant_df.groupby('seed')['total_reward'].std().sort_values(ascending=False)
            
            # Get top-k seeds with highest variance
            top_seeds = seed_variance.head(top_k).index.tolist()
            interesting_seeds[option_name][variant] = top_seeds
            
            print(f"\n  Variant: {variant}")
            print(f"  Top {top_k} seeds (by cross-model std):")
            for i, seed in enumerate(top_seeds):
                std_val = seed_variance[seed]
                # Get model scores for this seed
                seed_df = variant_df[variant_df['seed'] == seed]
                scores = seed_df.groupby('model_name')['total_reward'].mean()
                min_score = scores.min()
                max_score = scores.max()
                print(f"    {i+1}. Seed {seed}: std={std_val:.2f}, range=[{min_score:.1f}, {max_score:.1f}]")
    
    # Save interesting seeds to file
    if output_dir:
        seeds_path = output_dir / "interesting_seeds.json"
        with open(seeds_path, 'w') as f:
            json.dump(interesting_seeds, f, indent=2)
        print(f"\nInteresting seeds saved to: {seeds_path}")
        
        # Also save as a flat CSV for easy viewing
        rows = []
        for option, variants in interesting_seeds.items():
            for variant, seeds in variants.items():
                for rank, seed in enumerate(seeds, 1):
                    rows.append({
                        'option_name': option,
                        'option_variant': variant,
                        'rank': rank,
                        'seed': seed,
                    })
        seeds_df = pd.DataFrame(rows)
        seeds_csv_path = output_dir / "interesting_seeds.csv"
        seeds_df.to_csv(seeds_csv_path, index=False)
        print(f"Interesting seeds CSV saved to: {seeds_csv_path}")
    
    return interesting_seeds


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive parallel evaluation of FruitBot agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Options:
  basic         - 0 walls, foods: (3,6), (6,2), (6,6) with range 1
  walls_fruits  - 4 walls, only good foods: (4,0), (8,0) with range 0
  walls_doors   - 3 walls, 30/60%% doors, foods: (6,2), (6,6) with range 1

Examples:
    # Quick test with 10 seeds, find top 2 interesting per option
    python evaluate_comprehensive.py --num-seeds 10 --top-k 2
    
    # Full evaluation with 100 seeds, find top 10 interesting per option
    python evaluate_comprehensive.py --num-seeds 100 --top-k 10
    
    # Run only specific options
    python evaluate_comprehensive.py --num-seeds 10 --options basic walls_fruits
        """
    )
    
    parser.add_argument("--models-dir", type=str, default="models/fruitbot",
                        help="Directory containing trained models")
    parser.add_argument("--output", type=str, default="results/comprehensive_eval.csv",
                        help="Output file path for results")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "parquet", "json"],
                        help="Output format")
    
    # Evaluation parameters
    parser.add_argument("--num-seeds", type=int, default=50,
                        help="Number of different seeds to evaluate per variant")
    parser.add_argument("--seed-offset", type=int, default=1000,
                        help="Starting seed offset")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top interesting seeds to find per variant")
    parser.add_argument("--episodes-per-config", type=int, default=1,
                        help="Number of episodes per model-config pair")
    parser.add_argument("--options", type=str, nargs="+", default=None,
                        choices=ALL_ENV_OPTIONS,
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
    for name, path in models:
        print(f"  - {name}")
    
    # Generate environment configs
    options_to_use = args.options if args.options else ALL_ENV_OPTIONS
    configs = generate_env_configs(
        num_seeds=args.num_seeds,
        seed_offset=args.seed_offset,
        options=options_to_use,
    )
    
    print(f"\nEnvironment options to evaluate: {options_to_use}")
    print(f"Seeds per variant: {args.num_seeds}")
    print(f"Total configurations: {len(configs)}")
    
    # Count configs per option
    from collections import Counter
    option_counts = Counter(c.option_name for c in configs)
    for opt, count in sorted(option_counts.items()):
        print(f"  - {opt}: {count} configs")
    
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
    
    # Generate summary report
    if not args.no_summary:
        generate_summary_report(results, output_path.parent)
    
    # Find interesting seeds
    interesting_seeds = find_interesting_seeds(
        results,
        top_k=args.top_k,
        output_dir=output_path.parent,
    )
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"interesting_seeds = {interesting_seeds}")


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()
