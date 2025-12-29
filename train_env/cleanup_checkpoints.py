#!/usr/bin/env python
"""
Clean up intermediate checkpoint files from model directories.
Keeps only ppo_final.zip and removes all ppo_XXXXXX_steps.zip files.

Usage:
    python cleanup_checkpoints.py                    # Dry run (shows what would be deleted)
    python cleanup_checkpoints.py --confirm          # Actually delete the files
    python cleanup_checkpoints.py --models-dir ../models  # Use custom models directory
"""

import argparse
import os
from pathlib import Path


def find_checkpoint_files(models_dir: Path):
    """Find all intermediate checkpoint files in model directories.
    
    Returns:
        List of tuples: (file_path, file_size_mb)
    """
    checkpoint_files = []
    
    # Walk through models directory
    for env_dir in models_dir.iterdir():
        if not env_dir.is_dir():
            continue
            
        # Each environment has timestamped training run directories
        for run_dir in env_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            # Find checkpoint files (ppo_XXXXXX_steps.zip)
            for file_path in run_dir.glob("ppo_*_steps.zip"):
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                checkpoint_files.append((file_path, file_size_mb))
    
    return checkpoint_files


def main():
    parser = argparse.ArgumentParser(description="Clean up intermediate training checkpoints")
    parser.add_argument("--models-dir", type=str, default="../models", 
                        help="Path to models directory (default: ../models)")
    parser.add_argument("--confirm", action="store_true", 
                        help="Actually delete files (default is dry run)")
    args = parser.parse_args()
    
    # Resolve models directory
    script_dir = Path(__file__).parent
    models_dir = (script_dir / args.models_dir).resolve()
    
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        return
    
    print(f"Scanning models directory: {models_dir}\n")
    
    # Find all checkpoint files
    checkpoint_files = find_checkpoint_files(models_dir)
    
    if not checkpoint_files:
        print("No intermediate checkpoint files found.")
        return
    
    # Calculate total size
    total_size_mb = sum(size for _, size in checkpoint_files)
    
    # Display what will be deleted
    print(f"Found {len(checkpoint_files)} checkpoint files (Total: {total_size_mb:.2f} MB):\n")
    
    for file_path, size_mb in checkpoint_files:
        relative_path = file_path.relative_to(models_dir)
        print(f"  {relative_path} ({size_mb:.2f} MB)")
    
    print(f"\nTotal space to reclaim: {total_size_mb:.2f} MB")
    
    # Delete or dry run
    if args.confirm:
        print("\n[DELETING FILES...]")
        deleted_count = 0
        for file_path, _ in checkpoint_files:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"  Error deleting {file_path.name}: {e}")
        
        print(f"\n✓ Deleted {deleted_count} checkpoint files")
        print(f"✓ Reclaimed {total_size_mb:.2f} MB of disk space")
    else:
        print("\n[DRY RUN] No files deleted.")
        print("Run with --confirm to actually delete these files:")
        print(f"  python {Path(__file__).name} --confirm")


if __name__ == "__main__":
    main()
