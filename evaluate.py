#!/usr/bin/env python
"""
Evaluate a trained PPO agent on Procgen and optionally render gameplay.

Examples:
    # Evaluate and watch the agent play
    python evaluate.py --model models/fruitbot/20251116-195636/ppo_final.zip --env fruitbot --episodes 10 --render

    # Evaluate silently and get statistics
    python evaluate.py --model models/fruitbot/20251116-195636/ppo_final.zip --env fruitbot --episodes 100

    # Record video
    python evaluate.py --model models/fruitbot/20251116-195636/ppo_final.zip --env fruitbot --episodes 5 --render --record
"""

import argparse
import time
from pathlib import Path
import numpy as np

try:
    import gym
except Exception:
    gym = None

try:
    import gymnasium as gymn
except Exception:
    gymn = None

try:
    from stable_baselines3 import PPO
except Exception as e:
    raise SystemExit(f"stable-baselines3 not found: {e}")

import procgen  # noqa: F401

try:
    from procgen.gym_registration import make_env as pg_make_env
except Exception:
    pg_make_env = None

try:
    from procgen.wrappers import make_fruitbot_basic
except Exception:
    make_fruitbot_basic = None


def make_eval_env(env_name, render_mode=None, **kwargs):
    """Create a single environment for evaluation."""
    env_id = f"procgen:procgen-{env_name}-v0"
    
    # Try to build from source first
    env = None
    if pg_make_env is not None:
        try:
            env = pg_make_env(render_mode=render_mode, env_name=env_name, **kwargs)
        except Exception:
            pass
    
    # Fallback to gym.make
    if env is None:
        if gym is not None:
            env = gym.make(env_id, render_mode=render_mode, **kwargs)
        else:
            raise RuntimeError("Cannot create environment - gym not available")
    
    # Apply FruitBot wrapper if needed
    if env_name == "fruitbot" and make_fruitbot_basic is not None:
        env = make_fruitbot_basic(env)
        print("Applied FruitBot basic action wrapper")
    
    return env


def evaluate_agent(model, env, num_episodes=10, render=False, delay=0.2):
    """
    Evaluate agent and return statistics.
    
    Args:
        model: Trained PPO model
        env: Environment to evaluate on
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        delay: Delay between frames when rendering (seconds)
    
    Returns:
        dict with evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    completion_count = 0
    
    for ep in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs
        
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (done or truncated):
            if render:
                # Render using native window (no opencv needed)
                env.render()
                time.sleep(delay)
            
            action, _states = model.predict(obs, deterministic=True)
            result = env.step(action)
            
            # Handle both gym and gymnasium API
            if len(result) == 4:
                obs, reward, done, info = result
                truncated = False
            else:
                obs, reward, done, truncated, info = result
            
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Check if level was completed
        if isinstance(info, dict) and info.get('level_complete', False):
            completion_count += 1
        
        print(f"Episode {ep+1}/{num_episodes}: Reward={total_reward:.2f}, Steps={steps}, Completed={info.get('level_complete', False) if isinstance(info, dict) else 'N/A'}")
    
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'completion_rate': completion_count / num_episodes,
        'total_episodes': num_episodes
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent on Procgen")
    parser.add_argument("--model", required=True, help="Path to saved model (.zip file)")
    parser.add_argument("--env", required=True, help="Environment name (e.g., fruitbot)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true", help="Render the gameplay in a window")
    parser.add_argument("--delay", type=float, default=0.03, help="Delay between frames when rendering (seconds)")
    parser.add_argument("--distribution-mode", type=str, default="hard", choices=["easy", "hard", "extreme", "memory", "exploration"])
    parser.add_argument("--num-levels", type=int, default=0, help="Number of levels (0 = unlimited)")
    parser.add_argument("--start-level", type=int, default=0, help="Start level seed")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = PPO.load(args.model)
    print("Model loaded successfully!")
    
    # Create environment
    env_kwargs = {
        "distribution_mode": args.distribution_mode,
        "num_levels": args.num_levels,
        "start_level": args.start_level,
    }
    
    render_mode = "human" if args.render else None
    env = make_eval_env(args.env, render_mode=render_mode, **env_kwargs)
    
    if args.seed is not None:
        try:
            env.reset(seed=args.seed)
        except TypeError:
            # Old gym API doesn't accept seed in reset
            if hasattr(env, 'seed'):
                env.seed(args.seed)
    
    print(f"\nEvaluating on {args.env} ({args.distribution_mode} mode)")
    print(f"Episodes: {args.episodes}")
    print("-" * 60)
    
    # Run evaluation
    stats = evaluate_agent(model, env, num_episodes=args.episodes, render=args.render, delay=args.delay)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean Reward:       {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Reward Range:      [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
    print(f"Mean Episode Len:  {stats['mean_length']:.1f} steps")
    print(f"Completion Rate:   {stats['completion_rate']*100:.1f}% ({int(stats['completion_rate']*stats['total_episodes'])}/{stats['total_episodes']})")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
