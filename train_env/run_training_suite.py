#!/usr/bin/env python
"""
Template script to run multiple training configurations for FruitBot.

Easy to edit:
1. Modify TRAINING_CONFIGS list to change parameter sets
2. Set BASE_MODEL to a checkpoint path or None to train from scratch
3. Run: python run_training_suite.py
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# ============================================================================
# CONFIGURATION: Edit these to run different training combinations
# ============================================================================

# Base model to fine-tune from, or None to train from scratch
BASE_MODEL: Optional[str] = r"models\fruitbot\20251230-134845_easy\ppo_final.zip"
BASE_MODEL: Optional[str] = None  # Uncomment to train from scratch

# Training configurations - easily add/remove parameter sets here
TRAINING_CONFIGS = [
    {
        "name": "doors_fruits",
        "description": "collect fruits, open doors",
        "fruitbot_reward_negative": -2,
        "fruitbot_reward_positive": 2,
        "fruitbot_reward_wall_hit": -5,
        "fruitbot_reward_step": 0.1,
        "fruitbot_door_prob_pct": 30,
    },
    {
        "name": "only fruits, open doors",
        "description": "only fruits, open doors, want to stay",
        "fruitbot_reward_negative": -2,
        "fruitbot_reward_positive": 2,
        "fruitbot_reward_wall_hit": -3,
        "fruitbot_reward_step": 0.2,
        "fruitbot_door_prob_pct": 20,
    },
    # Add more configurations below:
    {
        "name": "only fruits, no doors",
        "description": "Balanced positive and negative rewards",
        "fruitbot_reward_negative": -1,
        "fruitbot_reward_positive": 1,
        "fruitbot_reward_wall_hit": -2,
        "fruitbot_reward_step": 0.05,
        "fruitbot_door_prob_pct": 0,
    },
]

# Common training hyperparameters (same for all configs)
COMMON_PARAMS = {
    "env": "fruitbot",
    "use_source": False,
    "n_envs": 32,
    "device": "cuda",
    "distribution_mode": "easy",
    "total_steps": 4000000,
}

# ============================================================================
# DO NOT EDIT BELOW (unless you know what you're doing)
# ============================================================================


def build_command(config: Dict[str, Any], base_model: Optional[str]) -> list:
    """Build a training command from config."""
    cmd = ["python", "train_env/train.py"]

    # Add common parameters
    for key, value in COMMON_PARAMS.items():
        if key == "use_source":
            # use_source is a boolean flag - only add if True
            if value:
                cmd.append("--use-source")
        else:
            cmd.append(f"--{key.replace('_', '-')}")
            cmd.append(str(value))

    # Add reward-specific parameters
    cmd.append(f"--fruitbot-reward-negative")
    cmd.append(str(config["fruitbot_reward_negative"]))

    cmd.append(f"--fruitbot-reward-positive")
    cmd.append(str(config["fruitbot_reward_positive"]))

    cmd.append(f"--fruitbot-reward-wall-hit")
    cmd.append(str(config["fruitbot_reward_wall_hit"]))

    cmd.append(f"--fruitbot-reward-step")
    cmd.append(str(config["fruitbot_reward_step"]))

    cmd.append(f"--fruitbot-door-prob-pct")
    cmd.append(str(config["fruitbot_door_prob_pct"]))

    # Add base model if specified
    if base_model:
        cmd.append("--from-model")
        cmd.append(base_model)

    return cmd


def print_config_summary(config: Dict[str, Any], base_model: Optional[str]):
    """Print a summary of the training configuration."""
    print("\n" + "=" * 80)
    print(f"Configuration: {config['name']}")
    print(f"Description: {config['description']}")
    print("=" * 80)
    print("\nReward Parameters:")
    print(f"  --fruitbot-reward-negative:   {config['fruitbot_reward_negative']}")
    print(f"  --fruitbot-reward-positive:   {config['fruitbot_reward_positive']}")
    print(f"  --fruitbot-reward-wall-hit:   {config['fruitbot_reward_wall_hit']}")
    print(f"  --fruitbot-reward-step:       {config['fruitbot_reward_step']}")
    print(f"  --fruitbot-door-prob-pct:     {config['fruitbot_door_prob_pct']}")
    print("\nCommon Parameters:")
    for key, value in COMMON_PARAMS.items():
        print(f"  --{key.replace('_', '-'):.<30} {value}")
    print("\nBase Model:")
    if base_model:
        print(f"  {base_model}")
    else:
        print("  None (training from scratch)")
    print("=" * 80 + "\n")


def main():
    """Run all training configurations."""
    workspace_root = Path(__file__).parent.parent
    os.chdir(workspace_root)

    if not TRAINING_CONFIGS:
        print("[ERROR] No training configurations defined in TRAINING_CONFIGS")
        sys.exit(1)

    print(f"\n{'#' * 80}")
    print(f"Training Suite: {len(TRAINING_CONFIGS)} configuration(s)")
    print(f"{'#' * 80}")

    successful = []
    failed = []

    for i, config in enumerate(TRAINING_CONFIGS, 1):
        print(f"\n[{i}/{len(TRAINING_CONFIGS)}] Starting training...")
        print_config_summary(config, BASE_MODEL)

        cmd = build_command(config, BASE_MODEL)
        cmd_str = " ".join(cmd)
        print(f"Command:\n  {cmd_str}\n")

        try:
            result = subprocess.run(cmd, cwd=workspace_root)
            if result.returncode == 0:
                successful.append(config["name"])
                print(f"\n✓ {config['name']} completed successfully")
            else:
                failed.append((config["name"], result.returncode))
                print(f"\n✗ {config['name']} failed with exit code {result.returncode}")
        except KeyboardInterrupt:
            print(f"\n⚠ Training interrupted by user")
            sys.exit(0)
        except Exception as e:
            failed.append((config["name"], str(e)))
            print(f"\n✗ {config['name']} failed with error: {e}")

    # Summary
    print(f"\n\n{'#' * 80}")
    print("TRAINING SUITE SUMMARY")
    print(f"{'#' * 80}")
    print(f"Total configurations: {len(TRAINING_CONFIGS)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\n✓ Successful:")
        for name in successful:
            print(f"  - {name}")

    if failed:
        print(f"\n✗ Failed:")
        for name, error in failed:
            print(f"  - {name}: {error}")

    print(f"{'#' * 80}\n")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    import os
    main()
