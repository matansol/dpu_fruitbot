#!/usr/bin/env python
"""
Generate correction triplets (obs, base_action, user_action) for fine-tuning.

Instead of a real user, we use a "smart agent" as a proxy for user corrections.
The base agent plays episodes on specified environment configs, and for each
observation we also record what the smart (user-proxy) agent would have done.

Usage:
    python train_env/generate_corrections.py --num-episodes 50 --config-index 1
    python train_env/generate_corrections.py --num-episodes 100 --seeds 0 1 2 3 4

Output:
    Saves a .npz file in train_env/correction_data/ with:
        - observations:  (N, 64, 64, 3) uint8
        - base_actions:  (N,) int64
        - user_actions:  (N,) int64
        - rewards:       (N,) float32
        - dones:         (N,) bool
        - episode_ids:   (N,) int64
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gym
import procgen  # noqa: F401  – registers procgen envs
from stable_baselines3 import PPO
from dpu_clf import get_config_by_index, BASE_ENV_CONFIG


# ── Agent paths ──────────────────────────────────────────────────────────────
BASE_AGENT_PATH = "models/fruitbot/20260116-074523_easy/ppo_final"       # no_doors_collect_all (index 1)
SMART_AGENT_PATH = "models/fruitbot/20251231-174002_easy/ppo_final"      # open_doors_fruits_only (index 6)

SAVE_DIR = Path("train_env/correction_data")


def load_agent(path: str) -> PPO:
    """Load a PPO agent, stripping .zip if present."""
    p = str(PROJECT_ROOT / path)
    if p.endswith(".zip"):
        p = p[:-4]
    if not os.path.exists(p + ".zip"):
        raise FileNotFoundError(f"Model not found: {p}.zip")
    model = PPO.load(p)
    print(f"  Loaded agent from {p}.zip")
    return model


def make_env(config_index: int, seed: int, start_level: int = 0) -> gym.Env:
    """Create a procgen fruitbot environment with a specific config."""
    config = get_config_by_index(config_index)
    config["rand_seed"] = seed
    config["num_levels"] = 1
    config["start_level"] = start_level
    env = gym.make("procgen-fruitbot-v0", **config)
    return env


def collect_episode(env: gym.Env, base_agent: PPO, smart_agent: PPO):
    """Run one episode with the base agent and record smart-agent corrections.

    Returns:
        dict with arrays: observations, base_actions, user_actions, rewards, dones
    """
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    observations = []
    base_actions = []
    user_actions = []
    rewards = []
    dones = []

    done = False
    while not done:
        # Base agent action
        b_action, _ = base_agent.predict(obs, deterministic=True)
        b_action = int(b_action.item()) if hasattr(b_action, "item") else int(b_action)

        # Smart (user-proxy) agent action on the SAME observation
        u_action, _ = smart_agent.predict(obs, deterministic=True)
        u_action = int(u_action.item()) if hasattr(u_action, "item") else int(u_action)

        observations.append(obs.copy())
        base_actions.append(b_action)
        user_actions.append(u_action)

        # Step the env using the BASE agent's action (the base agent drives the episode)
        result = env.step(b_action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        rewards.append(float(reward))
        dones.append(bool(done))

    return {
        "observations": np.array(observations, dtype=np.uint8),
        "base_actions": np.array(base_actions, dtype=np.int64),
        "user_actions": np.array(user_actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=bool),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Generate correction triplets for fine-tuning")
    p.add_argument("--config-index", type=int, default=1,
                   help="Environment config index (0-3). Default 1 = walls_food")
    p.add_argument("--num-episodes", type=int, default=50,
                   help="Number of episodes to collect")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Specific env seeds to use. If None, uses range(num_episodes)")
    p.add_argument("--start-levels", type=int, nargs="*", default=None,
                   help="Specific start levels. If None, uses range(num_episodes)")
    p.add_argument("--base-agent", type=str, default=BASE_AGENT_PATH,
                   help="Path to base agent model")
    p.add_argument("--smart-agent", type=str, default=SMART_AGENT_PATH,
                   help="Path to smart (user-proxy) agent model")
    p.add_argument("--output-name", type=str, default=None,
                   help="Custom output filename (without extension)")
    p.add_argument("--only-disagreements", action="store_true", default=False,
                   help="Only save timesteps where base and smart agents disagree")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("CORRECTION DATA GENERATOR")
    print("=" * 70)
    print(f"  Config index:    {args.config_index}")
    print(f"  Num episodes:    {args.num_episodes}")
    print(f"  Base agent:      {args.base_agent}")
    print(f"  Smart agent:     {args.smart_agent}")
    print(f"  Only disagree:   {args.only_disagreements}")
    print()

    # Load agents
    print("Loading agents...")
    base_agent = load_agent(args.base_agent)
    smart_agent = load_agent(args.smart_agent)
    print()

    # Determine seeds and levels
    seeds = args.seeds if args.seeds else list(range(args.num_episodes))
    start_levels = args.start_levels if args.start_levels else list(range(args.num_episodes))

    # Pad to match num_episodes
    while len(seeds) < args.num_episodes:
        seeds.append(seeds[-1] + 1 if seeds else 0)
    while len(start_levels) < args.num_episodes:
        start_levels.append(start_levels[-1] + 1 if start_levels else 0)

    all_obs = []
    all_base = []
    all_user = []
    all_rewards = []
    all_dones = []
    all_episode_ids = []

    total_steps = 0
    total_disagreements = 0

    for ep_idx in range(args.num_episodes):
        seed = seeds[ep_idx]
        level = start_levels[ep_idx]

        env = make_env(args.config_index, seed=seed, start_level=level)
        data = collect_episode(env, base_agent, smart_agent)
        env.close()

        n_steps = len(data["base_actions"])
        disagree_mask = data["base_actions"] != data["user_actions"]
        n_disagree = disagree_mask.sum()

        total_steps += n_steps
        total_disagreements += n_disagree

        if args.only_disagreements:
            # Filter to only keep disagreement timesteps
            if n_disagree > 0:
                all_obs.append(data["observations"][disagree_mask])
                all_base.append(data["base_actions"][disagree_mask])
                all_user.append(data["user_actions"][disagree_mask])
                all_rewards.append(data["rewards"][disagree_mask])
                all_dones.append(data["dones"][disagree_mask])
                all_episode_ids.append(np.full(n_disagree, ep_idx, dtype=np.int64))
        else:
            all_obs.append(data["observations"])
            all_base.append(data["base_actions"])
            all_user.append(data["user_actions"])
            all_rewards.append(data["rewards"])
            all_dones.append(data["dones"])
            all_episode_ids.append(np.full(n_steps, ep_idx, dtype=np.int64))

        if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
            print(f"  Episode {ep_idx + 1}/{args.num_episodes} | "
                  f"steps={n_steps} | disagree={n_disagree} ({100*n_disagree/max(n_steps,1):.1f}%)")

    # Concatenate all episodes
    observations = np.concatenate(all_obs, axis=0)
    base_actions = np.concatenate(all_base, axis=0)
    user_actions = np.concatenate(all_user, axis=0)
    rewards_arr = np.concatenate(all_rewards, axis=0)
    dones_arr = np.concatenate(all_dones, axis=0)
    episode_ids = np.concatenate(all_episode_ids, axis=0)

    print()
    print("=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    print(f"  Total episodes:        {args.num_episodes}")
    print(f"  Total timesteps:       {total_steps}")
    print(f"  Total disagreements:   {total_disagreements} ({100*total_disagreements/max(total_steps,1):.1f}%)")
    print(f"  Saved timesteps:       {len(base_actions)}")
    print(f"  Observation shape:     {observations.shape}")
    print()

    # Action distribution
    from collections import Counter
    action_names = {0: "left", 1: "stay", 2: "right", 3: "throw"}
    base_dist = Counter(base_actions.tolist())
    user_dist = Counter(user_actions.tolist())
    print("  Base agent action distribution:")
    for a in sorted(base_dist):
        print(f"    {action_names.get(a, a):>6s}: {base_dist[a]:>6d} ({100*base_dist[a]/len(base_actions):.1f}%)")
    print("  Smart agent action distribution:")
    for a in sorted(user_dist):
        print(f"    {action_names.get(a, a):>6s}: {user_dist[a]:>6d} ({100*user_dist[a]/len(user_actions):.1f}%)")

    # Save
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        fname = args.output_name
    else:
        suffix = "_disagree" if args.only_disagreements else ""
        fname = f"corrections_cfg{args.config_index}_ep{args.num_episodes}{suffix}"

    save_path = SAVE_DIR / f"{fname}.npz"
    np.savez_compressed(
        save_path,
        observations=observations,
        base_actions=base_actions,
        user_actions=user_actions,
        rewards=rewards_arr,
        dones=dones_arr,
        episode_ids=episode_ids,
        # Metadata
        config_index=np.array(args.config_index),
        base_agent_path=np.array(args.base_agent),
        smart_agent_path=np.array(args.smart_agent),
    )
    print(f"\n  Saved to: {save_path}")
    print(f"  File size: {save_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
