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
    
    # Import rendering utilities only if needed
    if render:
        try:
            import cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False
            print("Warning: cv2 not available, rendering will be disabled")
            render = False
    
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
            action, _states = model.predict(obs, deterministic=True)
            result = env.step(action)

            # Handle both gym and gymnasium API
            if len(result) == 4:
                obs, reward, done, info = result
                truncated = False
            else:
                obs, reward, done, truncated, info = result

            # Render if requested - extract RGB from info
            if render and has_cv2 and 'rgb' in info:
                rgb = info['rgb']
                # Convert RGB to BGR for cv2 display
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow('Procgen Evaluation', bgr)
                cv2.waitKey(1)
                if delay > 0 and steps % 2 == 0:
                    time.sleep(delay)

            # Ensure reward is a scalar float (if it's array-like)
            try:
                r = float(np.asarray(reward).item())
            except Exception:
                # fallback
                r = float(reward)

            total_reward += r
            steps += 1
            TOL = 1e-3

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

    # Clean up rendering window
    if render and has_cv2:
        cv2.destroyAllWindows()

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

import torch

def evaluate_agent_fast(
    model: PPO,
    env_id: str,
    env_kwargs: dict,
    num_episodes: int = 100,
    num_parallel: int = 16,  # Evaluate 16 episodes in parallel
    fruitbot_reward_positive: float = 2.0,
    fruitbot_reward_negative: float = -1.0,
    fruitbot_reward_wall_hit: float = -2.0,
) -> Dict[str, Any]:
    """
    Fast parallel evaluation using vectorized environments.
    
    This runs multiple episodes simultaneously for much faster evaluation.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    import procgen
    
    # Create vectorized environment
    def make_env():
        return gym.make(env_id, **env_kwargs)
    
    # Use SubprocVecEnv for true parallelism (each env in separate process)
    # Or DummyVecEnv for single-process (faster for simple envs)
    vec_env = DummyVecEnv([make_env for _ in range(num_parallel)])
    
    episode_rewards = []
    episode_lengths = []
    episode_good_food = []
    episode_bad_food = []
    episode_wall_hits = []
    
    # Track per-env stats
    current_rewards = [0.0] * num_parallel
    current_lengths = [0] * num_parallel
    current_good = [0] * num_parallel
    current_bad = [0] * num_parallel
    current_wall = [0] * num_parallel
    
    obs = vec_env.reset()
    episodes_done = 0
    
    # Set model to eval mode
    if hasattr(model.policy, 'eval'):
        model.policy.eval()
    
    # Disable gradient computation for inference
    with torch.no_grad():
        while episodes_done < num_episodes:
            # Predict actions for all environments at once
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(actions)
            
            # Update stats for each parallel environment
            for i in range(num_parallel):
                r = float(rewards[i])
                current_rewards[i] += r
                current_lengths[i] += 1
                
                TOL = 1e-3
                if np.isclose(r, fruitbot_reward_positive, atol=TOL, rtol=0.0):
                    current_good[i] += 1
                elif np.isclose(r, fruitbot_reward_negative, atol=TOL, rtol=0.0):
                    current_bad[i] += 1
                elif np.isclose(r, fruitbot_reward_wall_hit, atol=TOL, rtol=0.0):
                    current_wall[i] += 1
                
                if dones[i] and episodes_done < num_episodes:
                    # Episode finished
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_good_food.append(current_good[i])
                    episode_bad_food.append(current_bad[i])
                    episode_wall_hits.append(current_wall[i])
                    
                    # Reset counters
                    current_rewards[i] = 0.0
                    current_lengths[i] = 0
                    current_good[i] = 0
                    current_bad[i] = 0
                    current_wall[i] = 0
                    
                    episodes_done += 1
                    if episodes_done % 10 == 0:
                        print(f"Progress: {episodes_done}/{num_episodes} episodes completed")
    
    vec_env.close()
    
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'completion_rate': 0.0,  # Not tracked in fast mode
        'total_episodes': num_episodes,
        'avg_good_food': np.mean(episode_good_food),
        'avg_bad_food': np.mean(episode_bad_food),
        'avg_wall_hits': np.mean(episode_wall_hits),
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
    parser.add_argument("--distribution-mode", type=str, default="easy", choices=["easy", "hard", "extreme", "memory", "exploration"])
    parser.add_argument("--num-levels", type=int, default=0, help="Number of levels (0 = unlimited)")
    parser.add_argument("--start-level", type=int, default=0, help="Start level seed")
    parser.add_argument("--compile", action="store_true", help="Compile the Procgen environment from source")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    # FruitBot custom reward shaping
    parser.add_argument("--fruitbot-reward-completion", type=float, default=10.0, help="FruitBot: reward for reaching the goal")
    parser.add_argument("--fruitbot-reward-positive", type=float, default=2.0, help="FruitBot: reward for collecting good fruit")
    parser.add_argument("--fruitbot-reward-negative", type=float, default=-1.0, help="FruitBot: penalty for touching bad food")
    parser.add_argument("--fruitbot-reward-wall-hit", type=float, default=-2.0, help="FruitBot: penalty for hitting walls/doors")
    parser.add_argument("--fruitbot-reward-step", type=float, default=0.0, help="FruitBot: small reward for each step (encourages survival)")
    parser.add_argument("--fast", action="store_true", help="Use fast parallel evaluation (no rendering)")
    parser.add_argument("--num-parallel", type=int, default=8, help="Number of parallel environments for fast evaluation")

    
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
    
    model = PPO.load(model_path_str)
    print("Model loaded successfully!")
    print(f"Model policy: {type(model.policy)}")
    print(f"Model action space: {model.action_space}")
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
    
    # REMOVED: These are not valid env kwargs - they should be wrappers applied after creation
    env_kwargs["use_discrete_action_wrapper"] = True
    env_kwargs["use_stay_bonus_wrapper"] = False
    env_kwargs["stay_bonus"] = 0
    
    # environment structuring
    env_kwargs['fruitbot_num_walls'] = 3
    env_kwargs['fruitbot_num_good_min'] = 5
    env_kwargs['fruitbot_num_good_range'] = 1
    env_kwargs['fruitbot_num_bad_min'] = 5
    env_kwargs['fruitbot_num_bad_range'] = 1
    env_kwargs['fruitbot_wall_gap_pct'] = 30
    env_kwargs['fruitbot_door_prob_pct'] = 20 #if args.distribution_mode in ['hard', 'extreme'] else 0
    env_kwargs['food_diversity'] = 2

    if args.compile:
        print("Compiling Procgen environment from source...")
        try:
            from procgen.builder import build
            build()
            print("Compilation complete!")
        except Exception as e:
            print(f"Compilation failed: {e}")
            import traceback
            traceback.print_exc()
            return

    # Create environment with proper error handling
    print("\nCreating environment...")
    print(f"Environment ID: procgen-{args.env}-v0")
    print(f"Render mode: {'human' if args.render else 'None'}")

    
    try:
        render_mode = None  # Don't use render_mode, we'll handle rendering manually
        env_id = f"procgen-{args.env}-v0"
        
        # Create base environment
        env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
        
        print("✓ Base environment created successfully!")
        
        # Apply custom wrappers if needed (these are not standard Procgen kwargs)
        # If you have custom wrappers, apply them here:
        # from procgen.wrappers import DiscreteActionWrapper  # example
        # env = DiscreteActionWrapper(env)
        
    except Exception as e:
        print(f"\n❌ Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test environment reset
    print("\nTesting environment reset...")
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, info = obs
            print(f"Reset successful! Obs shape: {obs.shape}, Info keys: {list(info.keys())}")
        else:
            print(f"Reset successful! Obs shape: {obs.shape}")
    except Exception as e:
        print(f"\n❌ Error resetting environment: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return
    
    if args.seed is not None:
        print(f"\nSetting seed: {args.seed}")
        try:
            env.reset(seed=args.seed)
        except TypeError:
            # Old gym API doesn't accept seed in reset
            if hasattr(env, 'seed'):
                env.seed(args.seed)
                env.reset()
    
    print(f"\n{'='*60}")
    print(f"Starting evaluation on {args.env} ({args.distribution_mode} mode)")
    print(f"Episodes: {args.episodes}")
    print(f"Render: {args.render}")
    print("=" * 60)
    
    # Run evaluation with error handling
    try:
        stats = evaluate_agent(
            model, 
            env, 
            num_episodes=args.episodes, 
            render=args.render, 
            delay=args.delay, 
            fruitbot_reward_negative=args.fruitbot_reward_negative,
            fruitbot_reward_positive=args.fruitbot_reward_positive,
            fruitbot_reward_wall_hit=args.fruitbot_reward_wall_hit
        )
        
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
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nClosing environment...")
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
