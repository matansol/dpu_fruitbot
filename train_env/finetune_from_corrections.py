#!/usr/bin/env python
"""
Fine-tune a PPO agent using correction data from user feedback (or smart-agent proxy).

Reads (obs, base_action, user_action) triplets saved by generate_corrections.py
and updates the base agent's policy network to move closer to the user's actions
using a behavioral cloning (cross-entropy) loss, optionally mixed with the
original PPO policy-distillation loss to prevent catastrophic forgetting.

Usage:
    # Basic fine-tuning with defaults
    python train_env/finetune_from_corrections.py

    # Custom settings
    python train_env/finetune_from_corrections.py \
        --data train_env/correction_data/corrections_cfg1_ep50.npz \
        --lr 1e-4 --epochs 20 --batch-size 64 --bc-weight 1.0 --kl-weight 0.5

    # Only fine-tune on disagreement steps
    python train_env/finetune_from_corrections.py \
        --data train_env/correction_data/corrections_cfg1_ep50_disagree.npz

Output:
    Saves fine-tuned model to train_env/finetuned_models/<timestamp>/
"""

import argparse
import os
import sys
import time
import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

# ── Defaults ─────────────────────────────────────────────────────────────────
BASE_AGENT_PATH = "models/fruitbot/20260116-074523_easy/ppo_final"  # no_doors_collect_all
DEFAULT_DATA = "train_env/correction_data/corrections_cfg1_ep50.npz"
SAVE_ROOT = Path("train_env/finetuned_models")


# ── Dataset ──────────────────────────────────────────────────────────────────

class CorrectionDataset(Dataset):
    """PyTorch dataset wrapping correction triplets.
    
    Each item is (observation, user_action) where observation is a float32
    tensor in CHW format and user_action is the target label.
    """

    def __init__(
        self,
        npz_path: str,
        only_disagreements: bool = True,
    ):
        data = np.load(npz_path, allow_pickle=True)
        
        observations = data["observations"]   # (N, H, W, C) uint8
        base_actions = data["base_actions"]    # (N,) int64
        user_actions = data["user_actions"]    # (N,) int64

        if only_disagreements:
            mask = base_actions != user_actions
            observations = observations[mask]
            base_actions = base_actions[mask]
            user_actions = user_actions[mask]
            print(f"  Filtered to disagreements: {mask.sum()} / {len(mask)} steps "
                  f"({100 * mask.sum() / max(len(mask), 1):.1f}%)")

        # Store as tensors
        # SB3 expects observations in NHWC uint8, but the policy internally
        # converts to float32 and transposes to NCHW.  We store NHWC uint8 here
        # and let the policy handle preprocessing.
        self.observations = torch.from_numpy(observations)  # uint8 NHWC
        self.base_actions = torch.from_numpy(base_actions).long()
        self.user_actions = torch.from_numpy(user_actions).long()

        print(f"  Dataset size: {len(self)} steps")
        print(f"  Observation shape: {self.observations.shape}")

    def __len__(self):
        return len(self.user_actions)

    def __getitem__(self, idx):
        return self.observations[idx], self.user_actions[idx], self.base_actions[idx]


# ── Fine-tuning logic ───────────────────────────────────────────────────────

def preprocess_obs(obs_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert NHWC uint8 observations to NCHW float32 [0,1] on device.
    
    This mirrors what SB3's CnnPolicy does internally.
    """
    x = obs_batch.float().to(device) / 255.0
    # NHWC -> NCHW
    x = x.permute(0, 3, 1, 2)
    return x


def finetune(
    model: PPO,
    dataset: CorrectionDataset,
    *,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    bc_weight: float = 1.0,
    kl_weight: float = 0.5,
    device: Optional[str] = None,
    save_dir: Optional[Path] = None,
):
    """Fine-tune a PPO model's policy using behavioral cloning on correction data.
    
    The loss is:
        L = bc_weight * CE(policy(obs), user_action)
          + kl_weight * KL(frozen_policy(obs) || policy(obs))
    
    The KL term prevents catastrophic forgetting of the original policy.
    
    Args:
        model: SB3 PPO model to fine-tune
        dataset: CorrectionDataset of (obs, user_action, base_action) triplets
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Learning rate for the fine-tuning optimizer
        bc_weight: Weight for the behavioral cloning (cross-entropy) loss
        kl_weight: Weight for the KL-divergence regularization loss
        device: torch device string
        save_dir: Directory to save checkpoints and final model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    
    # Move model to device
    model.policy.to(dev)
    
    # Freeze a copy of the original policy for KL regularization
    frozen_policy = copy.deepcopy(model.policy)
    frozen_policy.eval()
    for p in frozen_policy.parameters():
        p.requires_grad = False

    # Use entire dataset for training (no validation split)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Optimizer - only update the policy network (action_net + features_extractor)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ce_loss_fn = nn.CrossEntropyLoss()

    print(f"\n{'='*70}")
    print(f"FINE-TUNING")
    print(f"{'='*70}")
    print(f"  Device:       {dev}")
    print(f"  Train steps:  {len(dataset)}")
    print(f"  Epochs:       {epochs}")
    print(f"  Batch size:   {batch_size}")
    print(f"  LR:           {lr}")
    print(f"  BC weight:    {bc_weight}")
    print(f"  KL weight:    {kl_weight}")
    print(f"{'='*70}\n")

    history = {
        "train_loss": [], "train_bc_loss": [], "train_kl_loss": [],
        "train_acc": [],
    }

    for epoch in range(1, epochs + 1):
        # ── Training ─────────────────────────────────────────────────────
        model.policy.train()
        epoch_loss = 0.0
        epoch_bc_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for obs_batch, user_act_batch, base_act_batch in train_loader:
            obs_pp = preprocess_obs(obs_batch, dev)
            targets = user_act_batch.to(dev)

            # Forward through SB3 policy to get action logits
            # We access the features extractor + action net directly
            features = model.policy.extract_features(obs_pp)
            if hasattr(model.policy, 'mlp_extractor'):
                latent_pi, _ = model.policy.mlp_extractor(features)
            else:
                latent_pi = features
            logits = model.policy.action_net(latent_pi)

            # Behavioral cloning loss
            bc_loss = ce_loss_fn(logits, targets)

            # KL regularization against frozen policy
            kl_loss = torch.tensor(0.0, device=dev)
            if kl_weight > 0:
                with torch.no_grad():
                    frozen_features = frozen_policy.extract_features(obs_pp)
                    if hasattr(frozen_policy, 'mlp_extractor'):
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

            epoch_loss += loss.item() * len(targets)
            epoch_bc_loss += bc_loss.item() * len(targets)
            epoch_kl_loss += kl_loss.item() * len(targets)
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == targets).sum().item()
            epoch_total += len(targets)

        scheduler.step()

        avg_train_loss = epoch_loss / max(epoch_total, 1)
        avg_bc_loss = epoch_bc_loss / max(epoch_total, 1)
        avg_kl_loss = epoch_kl_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)

        history["train_loss"].append(avg_train_loss)
        history["train_bc_loss"].append(avg_bc_loss)
        history["train_kl_loss"].append(avg_kl_loss)
        history["train_acc"].append(train_acc)

        # Print progress
        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"loss={avg_train_loss:.4f} (bc={avg_bc_loss:.4f} kl={avg_kl_loss:.4f}) | "
              f"train_acc={train_acc:.3f}")

    return history


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune PPO agent from correction data")
    p.add_argument("--data", type=str, default=DEFAULT_DATA,
                   help="Path to .npz correction data file")
    p.add_argument("--base-agent", type=str, default=BASE_AGENT_PATH,
                   help="Path to base agent to fine-tune")
    p.add_argument("--epochs", type=int, default=15,
                   help="Number of fine-tuning epochs")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Mini-batch size")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate")
    p.add_argument("--bc-weight", type=float, default=1.0,
                   help="Weight for behavioral cloning loss")
    p.add_argument("--kl-weight", type=float, default=0.5,
                   help="Weight for KL divergence regularization (prevents forgetting)")
    p.add_argument("--only-disagreements", action="store_true", default=True,
                   help="Only train on steps where base and smart agents disagree")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto / cpu / cuda")
    p.add_argument("--output-name", type=str, default=None,
                   help="Custom output directory name")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("PPO FINE-TUNING FROM CORRECTION DATA")
    print("=" * 70)

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"  Device: {device}")

    # Check data file exists
    data_path = PROJECT_ROOT / args.data
    if not data_path.exists():
        print(f"\n  ERROR: Correction data not found at {data_path}")
        print(f"  Run generate_corrections.py first!")
        sys.exit(1)

    # Load dataset
    print(f"\nLoading correction data from {data_path}...")
    dataset = CorrectionDataset(str(data_path), only_disagreements=args.only_disagreements)

    if len(dataset) == 0:
        print("\n  ERROR: No correction data available (agents fully agree).")
        sys.exit(1)

    # Load base agent
    print(f"\nLoading base agent from {args.base_agent}...")
    agent_path = str(PROJECT_ROOT / args.base_agent)
    if agent_path.endswith(".zip"):
        agent_path = agent_path[:-4]
    model = PPO.load(agent_path, device=device)
    print(f"  Model loaded successfully.")

    # Create save directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dir_name = args.output_name or f"finetuned_{timestamp}"
    save_dir = SAVE_ROOT / dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save training config
    config = {
        "data_path": str(args.data),
        "base_agent": args.base_agent,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "bc_weight": args.bc_weight,
        "kl_weight": args.kl_weight,
        "only_disagreements": args.only_disagreements,
        "device": device,
        "dataset_size": len(dataset),
        "timestamp": timestamp,
    }
    import json
    with open(save_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Fine-tune
    history = finetune(
        model, dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        bc_weight=args.bc_weight,
        kl_weight=args.kl_weight,
        device=device,
        save_dir=save_dir,
    )

    # Save final model
    final_path = save_dir / "ppo_final"
    model.save(str(final_path))
    print(f"\n  Final model saved to: {final_path}")

    # Save training history
    np.savez(save_dir / "training_history.npz", **{k: np.array(v) for k, v in history.items()})
    print(f"  Training history saved to: {save_dir / 'training_history.npz'}")

    print(f"\n{'='*70}")
    print(f"FINE-TUNING COMPLETE")
    print(f"{'='*70}")
    print(f"  Output directory: {save_dir}")
    print(f"  Final train accuracy: {history['train_acc'][-1]:.3f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
