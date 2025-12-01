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

import os
os.environ['PROCGEN_NO_BUILD'] = '1'

import argparse
import time
from pathlib import Path
import numpy as np
from typing import Dict, Any, Tuple, Optional
from procgen.wrappers import make_fruitbot_basic

import gym

from stable_baselines3 import PPO
import procgen  # noqa: F401 - Required for procgen environment registration (uses pre-built binaries)


def make_eval_env(env_name: str, render_mode: Optional[str] = None, **kwargs) -> gym.Env:
    """
    Create a single Procgen environment for evaluation using pre-built binaries.
    
    Note: This function uses the already-compiled procgen package. No rebuilding occurs.
    The procgen environments are registered when you import procgen above.
    
    Args:
        env_name: Name of the Procgen game (e.g., 'fruitbot')
        render_mode: 'human' for window rendering, 'rgb_array' for video, None for no rendering
        **kwargs: Additional environment arguments (distribution_mode, num_levels, etc.)
    
    Returns:
        Configured Procgen environment
    """
    env_id = f"procgen-{env_name}-v0"
    
    # Create the environment using gym.make
    env = gym.make(
        env_id,
        render_mode=render_mode,
        **kwargs
    )
    
    # Apply the FruitBot-specific wrappers if applicable
    if env_name == "fruitbot":
        env = make_fruitbot_basic(env)
    
    return env


def evaluate_agent(
    model: PPO,
    env: gym.Env,
    num_episodes: int = 6,
    render: bool = False,
    delay: float = 0.1,
    fruitbot_reward_positive: float = 2.0,
    fruitbot_reward_negative: float = -1.0,
    fruitbot_reward_wall_hit: float = -2.0,
) -> Dict[str, Any]:
    """
    Evaluate agent and return statistics.

    Added:
      - Counts of good food, bad food, and wall hits by comparing per-step rewards
        to the configured event reward values (with a small tolerance).

    Args:
        model: Trained PPO model
        env: Environment to evaluate on
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        delay: Delay between frames when rendering (seconds)
        fruitbot_reward_positive: reward value for collecting good food (optional)
        fruitbot_reward_negative: reward value for touching bad food (optional)
        fruitbot_reward_wall_hit: reward value for hitting walls/doors (optional)

    Returns:
        dict with evaluation statistics (includes new event counts)
    """
    
    episode_rewards = []
    episode_lengths = []
    completion_count = 0

    total_good_food = 0
    total_bad_food = 0
    total_wall_hits = 0

    for ep in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs

        done = False
        truncated = False
        total_reward = 0.0
        steps = 0

        ep_good = 0
        ep_bad = 0
        ep_wall = 0

        while not (done or truncated):
            if render:
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

            # Ensure reward is a scalar float (if it's array-like)
            try:
                r = float(np.asarray(reward).item())
            except Exception:
                # fallback
                r = float(reward)

            total_reward += r
            steps += 1
            TOL = 1e-3

            # print(r)

            # Compare reward to configured event rewards (with tolerance)
            if np.isclose(r, fruitbot_reward_positive, atol=TOL, rtol=0.0):
                ep_good += 1
            elif np.isclose(r, fruitbot_reward_negative, atol=TOL, rtol=0.0):
                ep_bad += 1
            elif np.isclose(r, fruitbot_reward_wall_hit, atol=TOL, rtol=0.0):
                ep_wall += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        total_good_food += ep_good
        total_bad_food += ep_bad
        total_wall_hits += ep_wall

        print(f"Episode {ep+1}/{num_episodes}: Reward={total_reward:.2f}, Steps={steps}, Good={ep_good}, Bad={ep_bad}, WallHits={ep_wall}")

    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'completion_rate': completion_count / num_episodes,
        'total_episodes': num_episodes,
        # New aggregated event counts
        'avg_good_food': total_good_food / num_episodes,
        'avg_bad_food': total_bad_food / num_episodes,
        'avg_wall_hits': total_wall_hits / num_episodes,
    }

    return stats

# best model so far - models\fruitbot\20251124-155020_easy\ppo_final.zip
def main() -> None:
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
    # FruitBot custom reward shaping
    parser.add_argument("--fruitbot-reward-completion", type=float, default=10.0, help="FruitBot: reward for reaching the goal")
    parser.add_argument("--fruitbot-reward-positive", type=float, default=2.0, help="FruitBot: reward for collecting good fruit")
    parser.add_argument("--fruitbot-reward-negative", type=float, default=-1.0, help="FruitBot: penalty for touching bad food")
    parser.add_argument("--fruitbot-reward-wall-hit", type=float, default=-2.0, help="FruitBot: penalty for hitting walls/doors")
    parser.add_argument("--fruitbot-reward-step", type=float, default=0.0, help="FruitBot: small reward for each step (encourages survival)")

    
    args = parser.parse_args()
    
    # Verify model file exists
    model_path = Path(args.model)
    
    # Check if path exists as-is
    if not model_path.exists():
        # Try adding .zip if not present
        if not str(model_path).endswith('.zip'):
            model_path = Path(str(model_path) + '.zip')
        
        if not model_path.exists():
            print(f"Error: Model file not found: {args.model}")
            print(f"Also tried: {model_path}")
            print("\nPlease check:")
            print("1. The file path is correct")
            print("2. The file exists")
            print(f"3. Current directory: {Path.cwd()}")
            return
    
    # Load model
    print(f"Loading model from {model_path}...")
    
    # Remove .zip extension if present since PPO.load() adds it automatically
    model_path_str = str(model_path)
    if model_path_str.endswith('.zip'):
        model_path_str = model_path_str[:-4]
    
    try:
        model = PPO.load(model_path_str)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        if "unsupported pickle protocol: 5" in str(e):
            print("Hint: The model uses pickle protocol 5 (Python 3.8+). Please update your environment to Python 3.8.")
        return
    
    # Create environment - just use gym.make, no need to build from source
    env_kwargs = {
        "distribution_mode": args.distribution_mode,
        "num_levels": args.num_levels,
        "start_level": args.start_level,
    }
    env_kwargs["fruitbot_reward_completion"] = args.fruitbot_reward_completion
    env_kwargs["fruitbot_reward_positive"] = args.fruitbot_reward_positive
    env_kwargs["fruitbot_reward_negative"] = args.fruitbot_reward_negative
    env_kwargs["fruitbot_reward_wall_hit"] = args.fruitbot_reward_wall_hit
    env_kwargs["fruitbot_reward_step"] = args.fruitbot_reward_step
    env_kwargs["use_discrete_action_wrapper"] = True
    env_kwargs["use_stay_bonus_wrapper"] = True
    env_kwargs["stay_bonus"] = 0
    
    render_mode = "human" if args.render else None
    env_id = f"procgen-{args.env}-v0"
    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
    # env = make_eval_env(args.env, render_mode=render_mode, **env_kwargs)
    
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
    stats = evaluate_agent(model, env, num_episodes=args.episodes, render=args.render, delay=args.delay, 
                           fruitbot_reward_negative=args.fruitbot_reward_negative,
                           fruitbot_reward_positive=args.fruitbot_reward_positive,
                           fruitbot_reward_wall_hit=args.fruitbot_reward_wall_hit)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean Reward:       {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Reward Range:      [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
    print(f"Mean Episode Len:  {stats['mean_length']:.1f} steps")
    print(f"Completion Rate:   {stats['completion_rate']*100:.1f}% ({int(stats['completion_rate']*stats['total_episodes'])}/{stats['total_episodes']})")

    print(f"Avg Good Food / episode: {stats['avg_good_food']:.2f}")
    print(f"Avg Bad  Food / episode: {stats['avg_bad_food']:.2f}")
    print(f"Avg Wall Hits / episode: {stats['avg_wall_hits']:.2f}")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
