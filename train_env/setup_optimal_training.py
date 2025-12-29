#!/usr/bin/env python
"""
One-command setup for optimal FruitBot training.

This script:
1. Checks your environment
2. Installs missing packages
3. Runs GPU benchmark
4. Generates optimized training command
5. Optionally starts training

Usage:
    python setup_optimal_training.py
    python setup_optimal_training.py --auto-train  # Start training immediately
"""

import subprocess
import sys
from pathlib import Path
import json


def check_package(package_name):
    """Check if a package is installed."""
    try:
        __import__(package_name.replace('-', '_'))
        return True
    except ImportError:
        return False


def install_packages(packages):
    """Install missing packages."""
    if not packages:
        return True
    
    print(f"\n📦 Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", *packages
        ])
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install packages")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'available': True,
                'device_name': torch.cuda.get_device_name(0),
                'vram_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'cuda_version': torch.version.cuda,
            }
        else:
            return {'available': False}
    except ImportError:
        return {'available': False, 'error': 'torch not installed'}


def run_benchmark():
    """Run GPU benchmark to find optimal settings."""
    print(f"\n🔬 Running GPU benchmark...")
    print(f"This will take ~5 minutes...\n")
    
    try:
        benchmark_script = Path(__file__).parent / "gpu_benchmark.py"
        result = subprocess.run(
            [sys.executable, str(benchmark_script), "--device", "cuda", "--quick"],
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Try to parse benchmark results
        results_file = Path("gpu_benchmark_results.json")
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            
            successful = [r for r in results if r.get('success', False)]
            if successful:
                best = max(successful, key=lambda x: x['samples_per_sec'])
                return best
        
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Benchmark failed: {e}")
        return None


def generate_training_command(benchmark_result=None, use_wandb=True):
    """Generate optimal training command."""
    
    # Default values (conservative)
    n_envs = 32
    n_steps = 1024
    batch_size = 1024
    
    # Use benchmark results if available
    if benchmark_result:
        n_envs = benchmark_result.get('n_envs', 32)
        n_steps = benchmark_result.get('n_steps', 1024)
        batch_size = benchmark_result.get('batch_size', 1024)
    
    cmd_parts = [
        "python train_env/train.py",
        "--env fruitbot",
        "--use-source",
        "--device cuda",
        "--total-steps 5000000",
        f"--n-envs {n_envs}",
        f"--n-steps {n_steps}",
        f"--batch-size {batch_size}",
        "--n-epochs 10",
        "--learning-rate 5e-4",
        "--gamma 0.99",
        "--gae-lambda 0.95",
        "--clip-range 0.2",
        "--ent-coef 0.05",
        "--vf-coef 0.5",
        "--max-grad-norm 0.5",
        "--num-levels 200",
        "--distribution-mode easy",
        "--fruitbot-reward-completion 10.0",
        "--fruitbot-reward-positive 2.0",
        "--fruitbot-reward-negative -1.0",
        "--fruitbot-reward-wall-hit -2.0",
        "--fruitbot-reward-step 0.02",
        "--save-freq 200000",
    ]
    
    if use_wandb:
        cmd_parts.extend([
            "--wandb-project fruitbot-optimization",
            "--wandb-run-name optimal_training",
        ])
    
    # Windows-style multi-line
    return " `\n  ".join(cmd_parts)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Setup optimal FruitBot training")
    parser.add_argument("--auto-train", action="store_true", help="Start training immediately")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip GPU benchmark")
    parser.add_argument("--no-wandb", action="store_true", help="Don't use wandb")
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🚀 FruitBot Training Setup")
    print(f"{'='*80}\n")
    
    # Step 1: Check environment
    print(f"Step 1: Checking environment...")
    
    required_packages = {
        'torch': 'torch',
        'stable_baselines3': 'stable-baselines3',
        'gym': 'gym',
        'procgen': 'procgen',
    }
    
    optional_packages = {
        'wandb': 'wandb',
        'optuna': 'optuna',
    }
    
    missing_required = []
    for pkg_import, pkg_name in required_packages.items():
        if not check_package(pkg_import):
            missing_required.append(pkg_name)
            print(f"  ❌ {pkg_name} not installed")
        else:
            print(f"  ✅ {pkg_name} installed")
    
    if missing_required:
        print(f"\n❌ Missing required packages. Please install:")
        print(f"   pip install {' '.join(missing_required)}")
        return
    
    missing_optional = []
    for pkg_import, pkg_name in optional_packages.items():
        if not check_package(pkg_import):
            missing_optional.append(pkg_name)
            print(f"  ⚠️  {pkg_name} not installed (optional)")
        else:
            print(f"  ✅ {pkg_name} installed")
    
    # Step 2: Install optional packages
    if missing_optional:
        print(f"\n📦 Optional packages for optimization:")
        for pkg in missing_optional:
            print(f"   - {pkg}")
        
        response = input(f"\nInstall these packages? (y/n): ").lower()
        if response == 'y':
            if install_packages(missing_optional):
                print(f"✅ Packages installed successfully")
            else:
                print(f"⚠️  Some packages failed to install, continuing anyway...")
    
    # Step 3: Check CUDA
    print(f"\nStep 2: Checking CUDA...")
    cuda_info = check_cuda()
    
    if cuda_info['available']:
        print(f"  ✅ CUDA available")
        print(f"     Device: {cuda_info['device_name']}")
        print(f"     VRAM: {cuda_info['vram_gb']:.1f} GB")
        print(f"     CUDA version: {cuda_info['cuda_version']}")
    else:
        print(f"  ❌ CUDA not available")
        print(f"     Training will use CPU (much slower)")
        response = input(f"\nContinue with CPU training? (y/n): ").lower()
        if response != 'y':
            return
    
    # Step 4: Run benchmark
    benchmark_result = None
    if not args.skip_benchmark and cuda_info['available']:
        print(f"\nStep 3: Running GPU benchmark...")
        response = input(f"Run quick benchmark to find optimal settings? (y/n): ").lower()
        
        if response == 'y':
            benchmark_result = run_benchmark()
            
            if benchmark_result:
                print(f"\n✅ Benchmark complete!")
                print(f"   Optimal settings:")
                print(f"   - n_envs: {benchmark_result['n_envs']}")
                print(f"   - n_steps: {benchmark_result['n_steps']}")
                print(f"   - batch_size: {benchmark_result['batch_size']}")
                print(f"   - Throughput: {benchmark_result['samples_per_sec']:.0f} samples/sec")
                print(f"   - VRAM usage: {benchmark_result['peak_vram_mb']:.0f} MB")
            else:
                print(f"\n⚠️  Benchmark failed, using conservative defaults")
        else:
            print(f"   Skipping benchmark, using conservative defaults")
    
    # Step 5: Generate training command
    print(f"\nStep 4: Generating optimal training command...")
    
    use_wandb = not args.no_wandb and check_package('wandb')
    training_cmd = generate_training_command(benchmark_result, use_wandb)
    
    print(f"\n{'='*80}")
    print(f"🎯 OPTIMAL TRAINING COMMAND")
    print(f"{'='*80}\n")
    print(training_cmd)
    print(f"\n{'='*80}\n")
    
    if benchmark_result:
        print(f"Expected performance:")
        print(f"  - GPU utilization: 80-90%")
        print(f"  - Training speed: {benchmark_result['samples_per_sec']:.0f} samples/sec")
        print(f"  - Time for 5M steps: ~{5000000 / benchmark_result['samples_per_sec'] / 3600:.1f} hours\n")
    
    # Save command to file
    cmd_file = Path("optimal_training_command.txt")
    with open(cmd_file, 'w') as f:
        f.write(training_cmd)
    print(f"💾 Command saved to: {cmd_file}\n")
    
    # Step 6: Optionally start training
    if args.auto_train:
        print(f"🚀 Starting training...")
        cmd_list = training_cmd.replace(" `\n  ", " ").split()
        subprocess.run(cmd_list)
    else:
        print(f"To start training, run:")
        print(f"  {cmd_file.name}\n")
        print(f"Or run this script with --auto-train:\n")
        print(f"  python setup_optimal_training.py --auto-train\n")
    
    # Final tips
    print(f"{'='*80}")
    print(f"💡 NEXT STEPS")
    print(f"{'='*80}\n")
    print(f"1. Start training (copy command above)")
    print(f"2. Monitor GPU usage: nvidia-smi -l 1")
    
    if use_wandb:
        print(f"3. Watch training: https://wandb.ai/<your-username>/fruitbot-optimization")
    else:
        print(f"3. Watch training: tensorboard --logdir models/fruitbot")
    
    print(f"\nFor hyperparameter optimization:")
    print(f"  python optuna_search.py --n-trials 20 --use-wandb\n")
    
    print(f"📚 See OPTIMIZATION_GUIDE.md for full details\n")


if __name__ == "__main__":
    main()
