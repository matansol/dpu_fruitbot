#!/usr/bin/env python
"""
Train a PPO agent on a selected Procgen environment and save checkpoints.

- Uses stable-baselines3.PPO with CnnPolicy (image observations)
- Accepts common training hyperparameters
- Saves under models/<env_name>/<timestamp>/

Examples:
    python train.py --env fruitbot --total-steps 2000000 --n-envs 8 --device auto
    python train.py --env coinrun  --total-steps 1000000 --learning-rate 2.5e-4 --seed 0

Notes:
- This repo registers Gym environments on import (procgen:procgen-<env>-v0).
- We depend on stable-baselines3 (SB3) and torch. Install if missing.
- Python 3.10 is supported by SB3 1.8.x (Gym API) and SB3 2.x (Gymnasium API).
  This script works with either, but defaults to using Gym.
"""

import argparse
import os
import time
from pathlib import Path

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

# Ensure procgen envs are registered with gym on import
import procgen  # noqa: F401
try:
    # direct constructor that bypasses gym.make registry
    from procgen.gym_registration import make_env as pg_make_env  # type: ignore
except Exception:
    pg_make_env = None

# Import wrappers for action space reduction
try:
    from procgen.wrappers import make_fruitbot_basic  # type: ignore
except Exception:
    make_fruitbot_basic = None


def make_env_fn(
    env_id: str,
    render_mode: str | None = None,
    use_gymnasium: bool = False,
    use_source: bool = True,
    **kwargs,
):
    """Return a thunk that creates a single env instance when called.

    kwargs are passed through to gym.make (e.g., distribution_mode, num_levels, start_level).
    """
    def _thunk():
        # Preferred: build directly from source without going through gym.make registry
        # This uses procgen.gym_registration.make_env which returns a Gym-compatible env
        env = None
        if use_source and pg_make_env is not None:
            try:
                # pg_make_env expects env_name in kwargs and handles render_mode internally
                env = pg_make_env(render_mode=render_mode, **kwargs)
            except Exception:
                env = None

        if env is None:
            # Fallback paths: create via gymnasium or gym
            if env_id.startswith("procgen:") and gym is not None:
                env = gym.make(env_id, render_mode=render_mode, **kwargs)
                if 'render_mode' not in getattr(env, 'metadata', {}):
                    try:
                        env.render_mode = render_mode
                    except Exception:
                        pass
            else:
                lib = gymn if use_gymnasium and gymn is not None else gym
                if lib is None:
                    raise RuntimeError("No Gym/Gymnasium available to create the environment.")
                env = lib.make(env_id, render_mode=render_mode, **kwargs)

        # Skip EnvCompatibility wrapper - it causes seed() issues with Procgen
        # Our custom wrapper will handle space conversions instead
        
        # Apply FruitBot action space wrapper (4 basic actions: left, right, stay, throw)
        # This wrapper also converts spaces to gymnasium for SB3 2.x compatibility
        if kwargs.get("env_name") == "fruitbot" and make_fruitbot_basic is not None:
            env = make_fruitbot_basic(env)
            print(f"Applied FruitBot basic action wrapper - action space reduced to 4 actions")
        
        # Wrap with Monitor to record episode stats compatible with SB3
        if hasattr(env, "spec") and env.spec is not None and getattr(env.spec, "max_episode_steps", None):
            # Gym TimeLimit usually applied via spec
            pass
        env = Monitor(env)
        return env

    return _thunk


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on Procgen")
    p.add_argument("--env", required=True, help="Procgen env name, e.g., fruitbot, coinrun, jumper")
    p.add_argument("--total-steps", type=int, default=1_000_000, help="Total training timesteps")
    p.add_argument("--n-envs", type=int, default=8, help="Number of parallel envs (use 1 on low-end/Windows if needed)")
    p.add_argument("--n-steps", type=int, default=256, help="PPO n_steps")
    p.add_argument("--batch-size", type=int, default=512, help="PPO batch_size")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="PPO learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip-range", type=float, default=0.2, help="Policy clip range")
    p.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    p.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    p.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--device", type=str, default="auto", help="PyTorch device: auto/cpu/cuda")
    p.add_argument("--save-freq", type=int, default=100_000, help="Checkpoint save frequency (timesteps)")
    p.add_argument("--save-model", type=str, default=None, help="Path to checkpoint to resume training from (e.g., models/fruitbot/20240115-120000/ppo_1000000_steps.zip)")
    p.add_argument("--use-gymnasium", action="store_true", default=False, help="Use gymnasium instead of gym if available")
    p.add_argument("--no-source", dest="use_source", action="store_false", help="Do not build envs from procgen source; use gym/gymnasium make")
    # Procgen-specific knobs
    p.add_argument("--distribution-mode", type=str, default="hard", choices=["easy","hard","extreme","memory","exploration"], help="Procgen difficulty mode")
    p.add_argument("--num-levels", type=int, default=0, help="Number of levels (0 = unlimited)")
    p.add_argument("--start-level", type=int, default=0, help="Start level (seed offset)")
    p.add_argument("--use-sequential-levels", action="store_true", help="Use sequential instead of random levels")
    p.add_argument("--frame-stack", type=int, default=1, help="Stack N frames (VecFrameStack)")
    # FruitBot custom reward shaping
    p.add_argument("--fruitbot-reward-completion", type=float, default=10.0, help="FruitBot: reward for reaching the goal")
    p.add_argument("--fruitbot-reward-positive", type=float, default=2.0, help="FruitBot: reward for collecting good fruit")
    p.add_argument("--fruitbot-reward-negative", type=float, default=-1.0, help="FruitBot: penalty for touching bad food")
    p.add_argument("--fruitbot-reward-wall-hit", type=float, default=-2.0, help="FruitBot: penalty for hitting walls/doors")
    p.add_argument("--fruitbot-reward-step", type=float, default=0.0, help="FruitBot: small reward for each step (encourages survival)")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve env id registered by procgen on gym import
    env_id = f"procgen:procgen-{args.env}-v0"

    # Reproducibility
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except Exception:
        pass

    # Directory for saving models
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_root = Path("models") / args.env
    save_dir = save_root / f"{timestamp}_{args.distribution_mode}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build vectorized envs
    env_kwargs = dict(
        env_name=args.env,
        distribution_mode=args.distribution_mode,
        num_levels=args.num_levels,
        start_level=args.start_level,
        use_sequential_levels=args.use_sequential_levels,
    )
    
    # Add FruitBot custom rewards if training FruitBot
    if args.env == "fruitbot":
        env_kwargs["fruitbot_reward_completion"] = args.fruitbot_reward_completion
        env_kwargs["fruitbot_reward_positive"] = args.fruitbot_reward_positive
        env_kwargs["fruitbot_reward_negative"] = args.fruitbot_reward_negative
        env_kwargs["fruitbot_reward_wall_hit"] = args.fruitbot_reward_wall_hit
        env_kwargs["fruitbot_reward_step"] = args.fruitbot_reward_step
        print(f"\nFruitBot Reward Configuration:")
        print(f"  Completion bonus: {args.fruitbot_reward_completion}")
        print(f"  Good fruit reward: {args.fruitbot_reward_positive}")
        print(f"  Bad food penalty: {args.fruitbot_reward_negative}")
        print(f"  Wall hit penalty: {args.fruitbot_reward_wall_hit}")
        print(f"  Step reward: {args.fruitbot_reward_step}")
        print()

    # On Windows, SubprocVecEnv can be heavier; default to DummyVecEnv unless n_envs is large
    use_subproc = args.n_envs > 1 and os.name != "nt"
    make_thunk = make_env_fn(
        env_id,
        render_mode=None,
        use_gymnasium=args.use_gymnasium,
        use_source=args.use_source,
        **env_kwargs,
    )

    if use_subproc:
        venv = SubprocVecEnv([
            make_env_fn(env_id, use_gymnasium=args.use_gymnasium, use_source=args.use_source, **env_kwargs)
            for _ in range(args.n_envs)
        ])
    else:
        venv = DummyVecEnv([make_thunk for _ in range(args.n_envs)])

    if args.frame_stack and args.frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=args.frame_stack, channels_order="last")

    # PPO with CNN policy for image observations
    if args.save_model:
        # Load existing model and continue training
        checkpoint_path = Path(args.save_model)
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
        # Create new model from scratch
        model = PPO(
            policy="CnnPolicy",
            env=venv,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )

    # Checkpointing callback
    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq // max(1, args.n_envs),  # adjust for vec env steps
        save_path=str(save_dir),
        save_vecnormalize=False,
        save_replay_buffer=False,
        name_prefix="ppo",
    )

    model.learn(total_timesteps=args.total_steps, progress_bar=True, callback=checkpoint_cb)

    # Save final model
    final_path = save_dir / "ppo_final"
    model.save(str(final_path))
    print(f"Training complete. Saved checkpoints in: {save_dir}")


if __name__ == "__main__":
    main()
