#!/usr/bin/env python
"""
Measure how many training levels (episodes) are needed to make a weak/base agent
behave more like a good/smart agent using correction-based fine-tuning.

Workflow:
1) Collect corrections from `num_train_episodes` episodes (base vs smart).
2) Create 10 progressive checkpoints across those episodes.
3) For each checkpoint, train a fresh fine-tuned model from the base agent using
   corrections from episodes [0..checkpoint-1].
4) Evaluate each fine-tuned model on a fixed set of 10 held-out levels
   (disjoint from training levels), and measure:
   - mean score
   - action agreement with smart agent (%)

This script keeps the same training method as `finetune_from_corrections.py`:
behavioral cloning (cross-entropy) + optional KL regularization.
"""

import argparse
import copy
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gym
import procgen  # noqa: F401
from stable_baselines3 import PPO

from dpu_clf import get_config_by_index
from generate_corrections import make_env, collect_episode


BASE_AGENT_PATH = "models/fruitbot/20260116-074523_easy/ppo_final"  # weak/base
SMART_AGENT_PATH = "models/fruitbot/20251231-174002_easy/ppo_final"  # good/smart


class CorrectionDatasetFromArrays(Dataset):
    """Dataset of correction triplets from in-memory arrays."""

    def __init__(self, observations: np.ndarray, user_actions: np.ndarray, base_actions: np.ndarray):
        self.observations = torch.from_numpy(observations)   # uint8 NHWC
        self.user_actions = torch.from_numpy(user_actions).long()
        self.base_actions = torch.from_numpy(base_actions).long()

    def __len__(self):
        return len(self.user_actions)

    def __getitem__(self, idx):
        return self.observations[idx], self.user_actions[idx], self.base_actions[idx]


def preprocess_obs(obs_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = obs_batch.float().to(device) / 255.0
    return x.permute(0, 3, 1, 2)


def finetune(
    model: PPO,
    dataset: Dataset,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    bc_weight: float,
    kl_weight: float,
    device: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Same method as finetune_from_corrections.py (BC + optional KL)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model.policy.to(dev)

    frozen_policy = copy.deepcopy(model.policy)
    frozen_policy.eval()
    for p in frozen_policy.parameters():
        p.requires_grad = False

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_bc_loss": [], "train_kl_loss": [], "train_acc": []}

    for _epoch in range(1, epochs + 1):
        model.policy.train()
        ep_loss = ep_bc = ep_kl = 0.0
        ep_correct = ep_total = 0

        for obs_batch, user_act_batch, _base_act_batch in loader:
            obs_pp = preprocess_obs(obs_batch, dev)
            targets = user_act_batch.to(dev)

            features = model.policy.extract_features(obs_pp)
            if hasattr(model.policy, "mlp_extractor"):
                latent_pi, _ = model.policy.mlp_extractor(features)
            else:
                latent_pi = features
            logits = model.policy.action_net(latent_pi)

            bc_loss = ce_loss_fn(logits, targets)
            kl_loss = torch.tensor(0.0, device=dev)

            if kl_weight > 0:
                with torch.no_grad():
                    frozen_features = frozen_policy.extract_features(obs_pp)
                    if hasattr(frozen_policy, "mlp_extractor"):
                        frozen_latent, _ = frozen_policy.mlp_extractor(frozen_features)
                    else:
                        frozen_latent = frozen_features
                    frozen_logits = frozen_policy.action_net(frozen_latent)

                log_probs = F.log_softmax(logits, dim=-1)
                frozen_probs = F.softmax(frozen_logits, dim=-1)
                kl_loss = F.kl_div(log_probs, frozen_probs, reduction="batchmean")

            loss = bc_weight * bc_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), max_norm=0.5)
            optimizer.step()

            n = len(targets)
            ep_loss += loss.item() * n
            ep_bc += bc_loss.item() * n
            ep_kl += kl_loss.item() * n
            ep_correct += (logits.argmax(dim=-1) == targets).sum().item()
            ep_total += n

        scheduler.step()

        history["train_loss"].append(ep_loss / max(ep_total, 1))
        history["train_bc_loss"].append(ep_bc / max(ep_total, 1))
        history["train_kl_loss"].append(ep_kl / max(ep_total, 1))
        history["train_acc"].append(ep_correct / max(ep_total, 1))

    return history


def evaluate_agent_like_smart(
    agent: PPO,
    smart_agent: PPO,
    *,
    config_index: int,
    eval_seeds: List[int],
    eval_levels: List[int],
) -> Tuple[float, float]:
    """Return (mean_score, action_agreement_with_smart)."""
    scores = []
    agree = 0
    total = 0

    for seed, level in zip(eval_seeds, eval_levels):
        env = make_env(config_index, seed=seed, start_level=level)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        done = False
        ep_score = 0.0
        while not done:
            a_agent, _ = agent.predict(obs, deterministic=True)
            a_smart, _ = smart_agent.predict(obs, deterministic=True)
            a_agent = int(a_agent.item()) if hasattr(a_agent, "item") else int(a_agent)
            a_smart = int(a_smart.item()) if hasattr(a_smart, "item") else int(a_smart)

            agree += int(a_agent == a_smart)
            total += 1

            step_out = env.step(a_agent)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, _info = step_out
                done = terminated or truncated
            else:
                obs, reward, done, _info = step_out

            ep_score += float(reward)

        env.close()
        scores.append(ep_score)

    return float(np.mean(scores)), float(agree / max(total, 1))


def load_agent(path: str) -> PPO:
    full = str(PROJECT_ROOT / path)
    if full.endswith(".zip"):
        full = full[:-4]
    return PPO.load(full)


def parse_args():
    p = argparse.ArgumentParser(description="Progressive fine-tuning from smart corrections")

    # Agents and env
    p.add_argument("--base-agent", type=str, default=BASE_AGENT_PATH)
    p.add_argument("--smart-agent", type=str, default=SMART_AGENT_PATH)
    p.add_argument("--config-index", type=int, default=1, help="Env config index")

    # Training collection
    p.add_argument("--num-train-episodes", type=int, default=1000)
    p.add_argument("--train-seed-start", type=int, default=0)
    p.add_argument("--train-level-start", type=int, default=0)
    p.add_argument("--only-disagreements", action="store_true", default=True)

    # Progressive eval setup (10 checkpoints, 10 held-out levels)
    p.add_argument("--num-checkpoints", type=int, default=10)
    p.add_argument("--num-eval-levels", type=int, default=10)
    p.add_argument("--eval-seed-start", type=int, default=10000)
    p.add_argument("--eval-level-start", type=int, default=2000,
                   help="Must be outside training levels")

    # Fine-tune hyperparameters
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bc-weight", type=float, default=1.0)
    p.add_argument("--kl-weight", type=float, default=0.2)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # Output
    p.add_argument("--output-dir", type=str, default="train_env/finetuned_models")
    p.add_argument("--run-name", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Safety: held-out eval levels must not overlap training levels
    train_level_end = args.train_level_start + args.num_train_episodes - 1
    eval_level_end = args.eval_level_start + args.num_eval_levels - 1
    overlap = not (eval_level_end < args.train_level_start or args.eval_level_start > train_level_end)
    if overlap:
        raise ValueError(
            f"Eval levels [{args.eval_level_start}, {eval_level_end}] overlap training levels "
            f"[{args.train_level_start}, {train_level_end}]. Choose non-overlapping ranges."
        )

    print("=" * 80)
    print("PROGRESSIVE CORRECTION FINE-TUNING (BASE -> SMART)")
    print("=" * 80)
    print(f"Base agent:         {args.base_agent}")
    print(f"Smart agent:        {args.smart_agent}")
    print(f"Config index:       {args.config_index}")
    print(f"Train episodes:     {args.num_train_episodes}")
    print(f"Checkpoints:        {args.num_checkpoints}")
    print(f"Held-out eval lvls: {args.num_eval_levels} (from level {args.eval_level_start})")
    print(f"Device:             {device}")

    print("\nLoading base and smart agents...")
    base_agent = load_agent(args.base_agent)
    smart_agent = load_agent(args.smart_agent)

    # Fixed held-out evaluation set
    eval_seeds = [args.eval_seed_start + i for i in range(args.num_eval_levels)]
    eval_levels = [args.eval_level_start + i for i in range(args.num_eval_levels)]

    print("\nEvaluating base and smart references on held-out levels...")
    base_score, base_agree = evaluate_agent_like_smart(
        base_agent, smart_agent,
        config_index=args.config_index,
        eval_seeds=eval_seeds,
        eval_levels=eval_levels,
    )
    smart_score, smart_agree = evaluate_agent_like_smart(
        smart_agent, smart_agent,
        config_index=args.config_index,
        eval_seeds=eval_seeds,
        eval_levels=eval_levels,
    )
    print(f"  Base  -> score={base_score:.3f}, agreement={base_agree:.2%}")
    print(f"  Smart -> score={smart_score:.3f}, agreement={smart_agree:.2%}")

    print("\nCollecting correction data (1000 episodes by default)...")
    all_obs, all_base, all_user, all_ep_ids = [], [], [], []
    total_steps = 0
    total_disagree = 0

    for ep in range(args.num_train_episodes):
        seed = args.train_seed_start + ep
        level = args.train_level_start + ep

        env = make_env(args.config_index, seed=seed, start_level=level)
        data = collect_episode(env, base_agent, smart_agent)
        env.close()

        base_actions = data["base_actions"]
        user_actions = data["user_actions"]
        disagree_mask = base_actions != user_actions

        n_steps = len(base_actions)
        n_dis = int(disagree_mask.sum())
        total_steps += n_steps
        total_disagree += n_dis

        if args.only_disagreements:
            if n_dis > 0:
                all_obs.append(data["observations"][disagree_mask])
                all_base.append(base_actions[disagree_mask])
                all_user.append(user_actions[disagree_mask])
                all_ep_ids.append(np.full(n_dis, ep, dtype=np.int64))
        else:
            all_obs.append(data["observations"])
            all_base.append(base_actions)
            all_user.append(user_actions)
            all_ep_ids.append(np.full(n_steps, ep, dtype=np.int64))

        if (ep + 1) % 100 == 0 or ep == 0:
            print(
                f"  Episode {ep + 1:4d}/{args.num_train_episodes} | "
                f"steps={n_steps:3d} | disagree={n_dis:3d} ({100*n_dis/max(n_steps,1):.1f}%)"
            )

    if len(all_obs) == 0:
        raise RuntimeError("No correction samples collected (agents fully agree).")

    observations = np.concatenate(all_obs, axis=0)
    base_actions = np.concatenate(all_base, axis=0)
    user_actions = np.concatenate(all_user, axis=0)
    episode_ids = np.concatenate(all_ep_ids, axis=0)

    print("\nCollection summary:")
    print(f"  Total steps:         {total_steps}")
    print(f"  Total disagreements: {total_disagree} ({100*total_disagree/max(total_steps,1):.2f}%)")
    print(f"  Saved samples:       {len(user_actions)}")

    # 10 checkpoints with logarithmic growth from 1 to num_train_episodes
    checkpoints = np.logspace(
        0,
        np.log10(args.num_train_episodes),
        args.num_checkpoints,
        dtype=int,
    ).tolist()
    # Ensure unique values and remove duplicates from rounding
    checkpoints = sorted(list(set(checkpoints)))

    run_ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"progress_levels_{run_ts}"
    out_dir = PROJECT_ROOT / args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "progress_eval.csv"
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print("\nRunning progressive training/evaluation...")
    print("-" * 80)
    print("ckpt | episodes | samples | train_acc | heldout_score | smart_agreement")
    print("-" * 80)

    for i, ckpt in enumerate(checkpoints, start=1):
        mask = episode_ids < ckpt
        n_samples = int(mask.sum())

        if n_samples == 0:
            print(f"{i:>4d} | {ckpt:>8d} | {n_samples:>7d} |   n/a    |      n/a     |      n/a")
            continue

        dataset = CorrectionDatasetFromArrays(
            observations[mask],
            user_actions[mask],
            base_actions[mask],
        )

        ft_model = load_agent(args.base_agent)

        hist = finetune(
            ft_model,
            dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            bc_weight=args.bc_weight,
            kl_weight=args.kl_weight,
            device=device,
        )

        heldout_score, heldout_agree = evaluate_agent_like_smart(
            ft_model,
            smart_agent,
            config_index=args.config_index,
            eval_seeds=eval_seeds,
            eval_levels=eval_levels,
        )

        train_acc = float(hist["train_acc"][-1]) if len(hist["train_acc"]) > 0 else float("nan")

        ckpt_model_path = model_dir / f"finetuned_ep{ckpt}"
        ft_model.save(str(ckpt_model_path))

        row = {
            "checkpoint_index": i,
            "episodes_used": ckpt,
            "samples_used": n_samples,
            "train_acc": train_acc,
            "heldout_score": heldout_score,
            "heldout_smart_agreement": heldout_agree,
            "base_score": base_score,
            "base_smart_agreement": base_agree,
            "smart_score": smart_score,
            "smart_smart_agreement": smart_agree,
            "model_path": str(ckpt_model_path) + ".zip",
        }
        rows.append(row)

        print(
            f"{i:>4d} | {ckpt:>8d} | {n_samples:>7d} | "
            f"{train_acc:>8.3f} | {heldout_score:>11.3f} | {heldout_agree:>14.2%}"
        )

    if len(rows) == 0:
        raise RuntimeError("No checkpoints were evaluated.")

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda r: r["heldout_smart_agreement"])

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Saved CSV: {csv_path}")
    print(f"Best checkpoint by agreement:")
    print(
        f"  episodes={best['episodes_used']}, samples={best['samples_used']}, "
        f"agreement={best['heldout_smart_agreement']:.2%}, score={best['heldout_score']:.3f}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
