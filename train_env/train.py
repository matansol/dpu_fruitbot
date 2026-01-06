#!/usr/bin/env python
"""
Train a PPO agent on a selected Procgen environment and save checkpoints.

- Uses stable-baselines3.PPO with IMPALA-CNN (default) or CnnPolicy
- Accepts common training hyperparameters
- Saves under models/<env_name>/<timestamp>/

Examples:
    python train.py --env fruitbot --total-steps 2000000 --n-envs 8 --device auto
    python train.py --env coinrun  --total-steps 1000000 --learning-rate 2.5e-4 --seed 0
    python train.py --env fruitbot --use-impala --impala-depths 16 32 32 --impala-embedding 256

Notes:
- This repo registers Gym environments on import (procgen:procgen-<env>-v0).
- We depend on stable-baselines3 (SB3) and torch. Install if missing.
- Python 3.10 is supported by SB3 1.8.x (Gym API) and SB3 2.x (Gymnasium API).
  This script works with either, but defaults to using Gym.

IMPORTANT - Manual Rebuild Required:
- If you modify C++ files in procgen/src/, you MUST rebuild before training:
  python -c "from procgen.builder import build; build()"
- Python-only changes (wrappers, this script) do NOT require rebuild.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

# Prefer Gym for compatibility with procgen's registration.
# Gymnasium is optional; if you prefer it, pass --use-gymnasium.
try:
    import gym  # type: ignore
except Exception as e:
    gym = None  # may use gymnasium path below

# Optional Gymnasium support
try:
    import gymnasium as gymn  # type: ignore
    try:
        from gymnasium.wrappers import EnvCompatibility as GymnEnvCompatibility  # type: ignore
    except Exception:
        GymnEnvCompatibility = None
except Exception:
    gymn = None
    GymnEnvCompatibility = None

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback
except Exception as e:
    raise SystemExit(
        "stable-baselines3 not found. Please install it in your environment.\n"
        "Recommended for your setup (Python 3.10, Gymnasium present):\n"
        "  pip install stable-baselines3==2.3.2\n"
        "(or a compatible 2.x release).\n"
        f"Original import error: {e}"
    )

# Optional wandb integration
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Ensure procgen envs are registered with gym on import
import procgen  # noqa: F401
try:
    # direct constructor that bypasses gym.make registry
    from procgen.gym_registration import make_env as pg_make_env  # type: ignore
except Exception:
    pg_make_env = None


def make_env_fn(
    env_id: str,
    render_mode: Optional[str] = None,
    use_gymnasium: bool = False,
    use_source: bool = False,
    **kwargs,
):
    """Return a thunk that creates a single env instance when called.

    kwargs are passed through to gym.make (e.g., distribution_mode, num_levels, start_level).
    
    Note: use_source=True is only needed if you're actively developing Procgen's C++ code.
    For normal training, use_source=False (default) is recommended.
    """
    def _thunk():
        env = None
        
        if use_source and pg_make_env is not None:
            try:
                env = pg_make_env(render_mode=render_mode, **kwargs)
            except Exception as e:
                print(f"Warning: pg_make_env failed: {e}")
                env = None

        if env is None:
            if env_id.startswith("procgen:") and gym is not None:
                env = gym.make(env_id, render_mode=render_mode, **kwargs)
                if 'render_mode' not in getattr(env, 'metadata', {}):
                    try:
                        env.render_mode = render_mode
                    except Exception:
                        pass
            elif gym is not None:
                # Try without procgen: prefix
                env = gym.make(env_id, render_mode=render_mode, **kwargs)
        
        if env is None:
            raise RuntimeError(f"Failed to create environment: {env_id}")
        
        env = Monitor(env)
        
        # Add seed compatibility wrapper
        original_seed_method = getattr(env, 'seed', None)
        
        def seed_wrapper(seed=None):
            """Wrapper that handles seed() calls with or without arguments"""
            if seed is None:
                # No seed provided - call original if it exists
                if original_seed_method is not None:
                    try:
                        return original_seed_method()
                    except TypeError:
                        # Original seed() doesn't accept arguments, just call it
                        return original_seed_method()
                return [0]
            
            # Seed provided - try multiple approaches
            # 1. Try new gym API: reset(seed=seed)
            try:
                env.reset(seed=seed)
                return [seed]
            except TypeError:
                # 2. Try old gym API: seed(seed)
                if original_seed_method is not None:
                    try:
                        result = original_seed_method(seed)
                        return result if result is not None else [seed]
                    except TypeError:
                        # seed() method exists but doesn't accept arguments
                        try:
                            original_seed_method()
                            return [seed]
                        except:
                            return [seed]
                return [seed]
            except Exception:
                # Fallback - just return seed
                return [seed]
        
        # Replace seed method
        env.seed = seed_wrapper
        
        return env

    return _thunk


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on Procgen")
    p.add_argument("--env", default="fruitbot", help="Procgen env name, e.g., fruitbot, coinrun, jumper")
    p.add_argument("--total-steps", type=int, default=1_000_000, help="Total training timesteps")
    p.add_argument("--n-envs", type=int, default=24, help="Number of parallel envs (use 1 on low-end/Windows if needed)")
    p.add_argument("--n-steps", type=int, default=512, help="PPO n_steps")
    p.add_argument("--batch-size", type=int, default=256, help="PPO batch_size")
    p.add_argument("--learning-rate", type=float, default=5e-4, help="PPO learning rate")
    p.add_argument("--gamma", type=float, default=0.97, help="Discount factor")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip-range", type=float, default=0.2, help="Policy clip range")
    p.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    p.add_argument("--vf-coef", type=float, default=0.37, help="Value function coefficient")
    p.add_argument("--max-grad-norm", type=float, default=0.6, help="Gradient clipping")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--device", type=str, default="auto", help="PyTorch device: auto/cpu/cuda")
    p.add_argument("--save-freq", type=int, default=200_000, help="Checkpoint save frequency (timesteps)")
    p.add_argument("--from-model", type=str, default=None, help="Path to checkpoint to resume training from (e.g., models/fruitbot/20240115-120000/ppo_1000000_steps.zip)")
    p.add_argument("--use-gymnasium", action="store_true", default=False, help="Use gymnasium instead of gym if available")
    p.add_argument("--use-source", action="store_true", default=False, help="Build envs from procgen source (only needed for C++ development)")
    # Procgen-specific knobs
    p.add_argument("--distribution-mode", type=str, default="easy", choices=["easy","hard","extreme","memory","exploration"], help="Procgen difficulty mode")
    p.add_argument("--num-levels", type=int, default=0, help="Number of levels (0 = unlimited)")
    p.add_argument("--start-level", type=int, default=0, help="Start level (seed offset)")
    p.add_argument("--use-sequential-levels", action="store_true", help="Use sequential instead of random levels")
    p.add_argument("--frame-stack", type=int, default=1, help="Stack N frames (VecFrameStack)")
    # FruitBot custom reward shaping
    p.add_argument("--fruitbot-reward-completion", type=float, default=10.0, help="FruitBot: reward for reaching the goal")
    p.add_argument("--fruitbot-reward-positive", type=float, default=2.0, help="FruitBot: reward for collecting good fruit")
    p.add_argument("--fruitbot-reward-negative", type=float, default=-1.0, help="FruitBot: penalty for touching bad food")
    p.add_argument("--fruitbot-reward-wall-hit", type=float, default=-3.0, help="FruitBot: penalty for hitting walls/doors")
    p.add_argument("--fruitbot-reward-step", type=float, default=0.0, help="FruitBot: small reward for each step (encourages survival)")
    p.add_argument("--fruitbot-door-prob-pct", type=int, default=20, help="FruitBot: probability of door spawning (0-100)")
    # Wandb integration
    p.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name")
    p.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")
    p.add_argument("--wandb-tags", type=str, nargs="*", default=None, help="Weights & Biases tags")
    # GPU optimization
    p.add_argument("--n-epochs", type=int, default=15, help="Number of PPO epochs per update")
    return p.parse_args()


def main():
    args = parse_args()

    if args.use_source:
        print("Building procgen environment from source...")
        from procgen.builder import build
        build(debug=False)
        print("Build complete.")

    env_id = f"procgen-{args.env}-v0"

    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        
        # Display GPU/CPU device information
        print("\n" + "="*60)
        print("TRAINING DEVICE CONFIGURATION")
        print("="*60)
        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Device mode: auto -> {device}")
        else:
            device = args.device
            print(f"Device mode: {device} (manually specified)")
        
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: YES")
            print(f"  - GPU device count: {torch.cuda.device_count()}")
            print(f"  - GPU device name: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA version: {torch.version.cuda}")
            if device == "cpu":
                print("  [WARNING] CUDA is available but training on CPU")
        else:
            print(f"[X] CUDA available: NO")
            print(f"  - Training will use CPU (slower)")
            if device == "cuda":
                print("  [ERROR] CUDA requested but not available!")
        print("="*60 + "\n")
    except Exception:
        pass

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_root = Path("models") / args.env
    save_dir = save_root / f"{timestamp}_{args.distribution_mode}"
    save_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        env_name=args.env,
        distribution_mode=args.distribution_mode,
        num_levels=args.num_levels,
        start_level=args.start_level,
        use_sequential_levels=args.use_sequential_levels,
    )
    
    if args.env == "fruitbot":
        env_kwargs["fruitbot_reward_completion"] = args.fruitbot_reward_completion
        env_kwargs["fruitbot_reward_positive"] = args.fruitbot_reward_positive
        env_kwargs["fruitbot_reward_negative"] = args.fruitbot_reward_negative
        env_kwargs["fruitbot_reward_wall_hit"] = args.fruitbot_reward_wall_hit
        env_kwargs["fruitbot_reward_step"] = args.fruitbot_reward_step
        env_kwargs["use_discrete_action_wrapper"] = True
        env_kwargs["use_stay_bonus_wrapper"] = True
        env_kwargs["stay_bonus"] = 0.1
        env_kwargs['fruitbot_num_walls'] = 3
        env_kwargs['fruitbot_num_good_min'] = 5
        env_kwargs['fruitbot_num_good_range'] = 1
        env_kwargs['fruitbot_num_bad_min'] = 5
        env_kwargs['fruitbot_num_bad_range'] = 1
        env_kwargs['fruitbot_wall_gap_pct'] = 50
        env_kwargs['fruitbot_door_prob_pct'] = args.fruitbot_door_prob_pct
        env_kwargs['food_diversity'] = 6

        print(f"\nFruitBot Reward Configuration:")
        print(f"  Completion bonus: {args.fruitbot_reward_completion}")
        print(f"  Good fruit reward: {args.fruitbot_reward_positive}")
        print(f"  Bad food penalty: {args.fruitbot_reward_negative}")
        print(f"  Wall hit penalty: {args.fruitbot_reward_wall_hit}")
        print(f"  Step reward: {args.fruitbot_reward_step}")
        print()

    # Use SubprocVecEnv for parallel environments (much faster on Windows!)
    # Changed: Force SubprocVecEnv on Windows for better parallelization
    use_subproc = args.n_envs > 1  # Removed Windows check - SubprocVecEnv works fine now
    make_thunk = make_env_fn(
        env_id,
        render_mode=None,
        use_gymnasium=args.use_gymnasium,
        use_source=args.use_source,
        **env_kwargs,
    )

    if use_subproc:
        venv = SubprocVecEnv([
            make_env_fn(env_id, render_mode=None, use_gymnasium=args.use_gymnasium, use_source=args.use_source, **env_kwargs)
            for _ in range(args.n_envs)
        ])
    else:
        venv = DummyVecEnv([make_thunk for _ in range(args.n_envs)])

    if args.frame_stack and args.frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=args.frame_stack, channels_order="last")

    # Import IMPALA model
    from common.model import ImpalaModel
    
    print(f"\n{'='*60}")
    print(f"Using IMPALA-CNN Architecture (depths=[16,32,32], embedding=256)")
    print(f"{'='*60}\n")
    
    # Load model - now the seed wrapper is already in place
    if args.from_model:
        checkpoint_path = Path(args.from_model)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = PPO.load(
            str(checkpoint_path),
            env=venv,
            device=args.device,
        )
        print(f"Successfully loaded model. Continuing training...")
    else:
        model = PPO(
            policy=ImpalaModel,
            env=venv,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            device=args.device,
            verbose=1,
            tensorboard_log=str(save_dir / "tensorboard"),
        )

    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback - disabled to only save final model
    # checkpoint_cb = CheckpointCallback(
    #     save_freq=args.save_freq // max(1, args.n_envs),
    #     save_path=str(save_dir),
    #     save_vecnormalize=False,
    #     save_replay_buffer=False,
    #     name_prefix="ppo",
    # )
    # callbacks.append(checkpoint_cb)
    
    # Wandb callback
    if args.wandb_project and WANDB_AVAILABLE:
        wandb_run_name = args.wandb_run_name or f"{args.env}_{timestamp}"
        wandb_config = {
            'env': args.env,
            'total_steps': args.total_steps,
            'n_envs': args.n_envs,
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'gamma': args.gamma,
            'gae_lambda': args.gae_lambda,
            'clip_range': args.clip_range,
            'ent_coef': args.ent_coef,
            'vf_coef': args.vf_coef,
            'max_grad_norm': args.max_grad_norm,
            'n_epochs': args.n_epochs,
            'distribution_mode': args.distribution_mode,
            'num_levels': args.num_levels,
            'device': args.device,
        }
        
        if args.env == 'fruitbot':
            wandb_config.update({
                'fruitbot_reward_completion': args.fruitbot_reward_completion,
                'fruitbot_reward_positive': args.fruitbot_reward_positive,
                'fruitbot_reward_negative': args.fruitbot_reward_negative,
                'fruitbot_reward_wall_hit': args.fruitbot_reward_wall_hit,
                'fruitbot_reward_step': args.fruitbot_reward_step,
            })
        
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            tags=args.wandb_tags,
            config=wandb_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        
        wandb_cb = WandbCallback(
            model_save_path=str(save_dir / "wandb"),
            verbose=2,
        )
        callbacks.append(wandb_cb)
        
        print(f"\n[OK] Weights & Biases logging enabled")
        print(f"   Project: {args.wandb_project}")
        print(f"   Run: {wandb_run_name}")
        print(f"   URL: {wandb.run.url}\n")
    elif args.wandb_project and not WANDB_AVAILABLE:
        print(f"\n[WARNING] --wandb-project specified but wandb not installed")
        print(f"   Install with: pip install wandb\n")

    model.learn(total_timesteps=args.total_steps, progress_bar=True, callback=callbacks)

    final_path = save_dir / "ppo_final"
    model.save(str(final_path))
    print(f"Training complete. Saved checkpoints in: {save_dir}")
    
    # Close wandb run
    if args.wandb_project and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
