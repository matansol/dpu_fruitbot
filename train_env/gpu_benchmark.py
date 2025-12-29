#!/usr/bin/env python
"""
Quick GPU benchmark and optimization recommendations for your RTX 3060.

This script:
1. Tests different batch sizes to find optimal GPU utilization
2. Measures training throughput (samples/sec)
3. Monitors VRAM usage
4. Provides specific recommendations for your hardware

Usage:
    python gpu_benchmark.py --device cuda
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path
import sys
import psutil
import threading

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
import procgen


def benchmark_configuration(n_envs, n_steps, batch_size, n_epochs, device, duration_sec=30):
    """
    Benchmark a specific configuration.
    
    Returns:
        dict with metrics: samples_per_sec, gpu_utilization, vram_mb, cpu_percent
    """
    print(f"\n{'='*60}")
    print(f"Testing: n_envs={n_envs}, n_steps={n_steps}, batch={batch_size}, epochs={n_epochs}")
    print(f"Duration: {duration_sec}s")
    print(f"{'='*60}")
    print("Creating environments and model...")
    
    # Create environment
    def make_env():
        env = gym.make(
            "procgen-fruitbot-v0",
            render_mode=None,
            env_name='fruitbot',
            distribution_mode='easy',
            num_levels=10,
        )
        return env
    
    venv = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # Create model
    from common.model import ImpalaModel
    
    try:
        model = PPO(
            policy=ImpalaModel,
            env=venv,
            learning_rate=5e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            device=device,
            verbose=0,
        )
        
        # Measure initial VRAM
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_vram = torch.cuda.memory_allocated() / 1024**2
        
        # CPU monitoring in background
        cpu_samples = []
        stop_monitoring = threading.Event()
        
        def monitor_cpu():
            while not stop_monitoring.is_set():
                cpu_samples.append(psutil.cpu_percent(interval=0.5))
        
        cpu_thread = threading.Thread(target=monitor_cpu, daemon=True)
        cpu_thread.start()
        
        # Run training for specified duration
        print("Training started...")
        start_time = time.time()
        steps_to_run = 0
        iterations = 0
        
        while time.time() - start_time < duration_sec:
            model.learn(total_timesteps=n_envs * n_steps, reset_num_timesteps=False)
            steps_to_run += n_envs * n_steps
            iterations += 1
            if iterations % 3 == 0:  # Print every 3 iterations
                elapsed = time.time() - start_time
                print(f"  Progress: {elapsed:.1f}s / {duration_sec}s ({steps_to_run} steps)")
        
        # Stop CPU monitoring
        stop_monitoring.set()
        cpu_thread.join(timeout=1)
        
        elapsed = time.time() - start_time
        total_steps = steps_to_run
        
        # Calculate metrics
        samples_per_sec = total_steps / elapsed
        avg_cpu = np.mean(cpu_samples) if cpu_samples else 0
        
        if device == 'cuda' and torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / 1024**2
            current_vram = torch.cuda.memory_allocated() / 1024**2
        else:
            peak_vram = 0
            current_vram = 0
        
        # Clean up
        venv.close()
        del model
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        results = {
            'n_envs': n_envs,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'samples_per_sec': samples_per_sec,
            'vram_mb': current_vram,
            'peak_vram_mb': peak_vram,
            'cpu_percent': avg_cpu,
            'elapsed_sec': elapsed,
            'success': True,
        }
        
        print(f"✅ Success!")
        print(f"   Throughput: {samples_per_sec:.0f} samples/sec")
        print(f"   CPU Usage: {avg_cpu:.1f}%")
        print(f"   VRAM: {current_vram:.0f} MB (peak: {peak_vram:.0f} MB)")
        
        return results
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        
        # Clean up on error
        try:
            venv.close()
        except:
            pass
        
        return {
            'n_envs': n_envs,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'error': str(e),
            'success': False,
        }


def main():
    parser = argparse.ArgumentParser(description="GPU & CPU Benchmark for PPO Training")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--duration", type=int, default=10, help="Benchmark duration per config (seconds)")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer configurations")
    parser.add_argument("--sweep-envs", action="store_true", help="Sweep through different parallel env counts to find optimal")
    args = parser.parse_args()
    
    # Override duration for quick mode if not explicitly set
    if args.quick and args.duration == 10:
        args.duration = 5  # Quick mode uses 5 seconds per config
    
    print(f"\n{'='*80}")
    print(f"🚀 PPO Training Benchmark for Windows")
    print(f"{'='*80}\n")
    
    # System info
    print(f"CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count(logical=True)} threads)")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    # Check CUDA availability
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("❌ CUDA not available! Switching to CPU mode.")
            args.device = "cpu"
        else:
            print(f"✅ CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
            print(f"   CUDA Version: {torch.version.cuda}\n")
    
    # Define configurations to test
    # Focus on environment parallelism and epochs (Windows bottleneck)
    if args.sweep_envs:
        # Sweep through many environment counts to find optimal throughput
        print("\n🔍 PARALLEL ENVIRONMENT SWEEP MODE")
        print("Testing different parallel environment counts to maximize throughput...\n")
        configs = [
            # (n_envs, n_steps, batch_size, n_epochs)
            # Keep n_steps, batch_size, n_epochs constant, only vary n_envs
            (8, 256, 256, 10),
            (16, 256, 256, 10),
            (24, 256, 256, 10),
            (32, 256, 256, 10),
            (48, 256, 256, 10),
            (64, 256, 256, 10),
            # (96, 256, 256, 10),
            # (128, 256, 256, 10),
        ]
    elif args.quick:
        configs = [
            # (n_envs, n_steps, batch_size, n_epochs)
            (32, 256, 256, 10),    # Small baseline
            (64, 256, 256, 10),    # More parallel envs
            (64, 512, 512, 10),    # Larger buffers
            (96, 256, 256, 10),    # Max parallel
        ]
    else:
        configs = [
            # Test different n_envs (most important for Windows)
            (32, 512, 512, 10),
            (48, 512, 512, 10),
            (64, 512, 512, 10),
            (96, 512, 512, 10),
            (128, 512, 512, 10),
            
            # Test different n_steps
            (64, 256, 256, 10),
            (64, 1024, 1024, 10),
            (64, 2048, 2048, 10),
            
            # Test different n_epochs (sample efficiency)
            (64, 512, 512, 5),
            (64, 512, 512, 15),
            (64, 512, 512, 20),
        ]
    
    results = []
    total_configs = len(configs)
    
    for idx, (n_envs, n_steps, batch_size, n_epochs) in enumerate(configs, 1):
        print(f"\n[{idx}/{total_configs}] Starting benchmark...")
        
        result = benchmark_configuration(
            n_envs=n_envs,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            device=args.device,
            duration_sec=args.duration,
        )
        results.append(result)
        
        # Estimate remaining time
        if result.get('success'):
            remaining = total_configs - idx
            est_time_remaining = remaining * (args.duration + 3)  # +3 for overhead
            if remaining > 0:
                print(f"⏱️  Estimated time remaining: ~{est_time_remaining} seconds")
        
        # Wait between tests
        time.sleep(1)
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        print("❌ No successful configurations. Check your setup.")
        return
    
    # Sort by throughput
    successful_results.sort(key=lambda x: x['samples_per_sec'], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"📊 BENCHMARK RESULTS")
    print(f"{'='*80}\n")
    
    if args.sweep_envs:
        # Special display for environment sweep showing clear correlation
        print("PARALLEL ENVIRONMENTS vs THROUGHPUT:")
        print(f"\n{'n_envs':<10} {'Samples/sec':<15} {'CPU%':<10} {'VRAM':<15} {'Status':<10}")
        print(f"{'-'*10} {'-'*15} {'-'*10} {'-'*15} {'-'*10}")
        
        # Sort by n_envs for sweep mode
        sweep_results = sorted(successful_results, key=lambda x: x['n_envs'])
        
        for r in sweep_results:
            envs_str = str(r['n_envs'])
            throughput_str = f"{r['samples_per_sec']:.0f} samp/s"
            cpu_str = f"{r['cpu_percent']:.0f}%"
            vram_str = f"{r['peak_vram_mb']:.0f} MB"
            
            # Mark the best
            if r == successful_results[0]:
                status = "⭐ BEST"
            else:
                status = ""
            
            print(f"{envs_str:<10} {throughput_str:<15} {cpu_str:<10} {vram_str:<15} {status:<10}")
        
        print(f"\n{'='*80}")

    
    print(f"{'Config':<30} {'Throughput':<15} {'CPU%':<10} {'VRAM':<15}")
    print(f"{'-'*30} {'-'*15} {'-'*10} {'-'*15}")
    
    for r in successful_results:
        config_str = f"{r['n_envs']}e/{r['n_steps']}s/{r['batch_size']}b/{r['n_epochs']}ep"
        throughput_str = f"{r['samples_per_sec']:.0f} samp/s"
        cpu_str = f"{r['cpu_percent']:.0f}%"
        vram_str = f"{r['peak_vram_mb']:.0f} MB"
        print(f"{config_str:<30} {throughput_str:<15} {cpu_str:<10} {vram_str:<15}")
    
    # Best configuration
    best = successful_results[0]
    
    print(f"\n{'='*80}")
    print(f"🏆 RECOMMENDED CONFIGURATION")
    print(f"{'='*80}")
    print(f"\nBest throughput: {best['samples_per_sec']:.0f} samples/sec")
    print(f"Optimal parallel envs: {best['n_envs']}")
    print(f"CPU usage: {best['cpu_percent']:.0f}%")
    print(f"VRAM usage: {best['peak_vram_mb']:.0f} MB")
    
    # Analysis
    print(f"\n📊 Bottleneck Analysis:")
    if best['cpu_percent'] > 80:
        print(f"   ⚠️  CPU-bound (environment simulation is the bottleneck)")
        print(f"   → More parallel envs and higher n_epochs recommended")
    elif best['peak_vram_mb'] > 5000:
        print(f"   ⚠️  GPU memory-bound (close to VRAM limit)")
        print(f"   → Reduce batch_size or n_envs if OOM errors occur")
    elif best['cpu_percent'] < 30:
        print(f"   ⚠️  Very low CPU usage - potential I/O or memory bottleneck")
        print(f"   → Windows environment creation overhead may be the issue")
        print(f"   → Consider using SubprocVecEnv for better parallelization")
    else:
        print(f"   ✅ Balanced configuration")
    
    # Throughput assessment
    if best['samples_per_sec'] < 1000:
        print(f"\n⚠️  WARNING: Low throughput detected ({best['samples_per_sec']:.0f} samples/sec)")
        print(f"   This is significantly slower than expected.")
        print(f"   Possible causes:")
        print(f"   - Windows environment overhead (use DummyVecEnv vs SubprocVecEnv)")
        print(f"   - Disk I/O bottleneck")
        print(f"   - Background processes consuming resources")
        print(f"   - First-time JIT compilation overhead")
    
    print(f"\n💡 Optimal training command:")
    print(f"\npython train_env/train.py `")
    print(f"  --env fruitbot `")
    print(f"  --use-source `")
    print(f"  --device {args.device} `")
    print(f"  --n-envs {best['n_envs']} `")
    print(f"  --n-steps {best['n_steps']} `")
    print(f"  --batch-size {best['batch_size']} `")
    print(f"  --n-epochs {best['n_epochs']} `")
    print(f"  --total-steps 5000000 `")
    print(f"  --learning-rate 5e-4 `")
    print(f"  --ent-coef 0.05 `")
    print(f"  --num-levels 200")
    
    # Time estimate
    est_time_hours = 5_000_000 / best['samples_per_sec'] / 3600
    print(f"\n⏱️  Estimated time for 5M steps: {est_time_hours:.1f} hours")
    print(f"\n{'='*80}\n")
    
    # Save results
    import json
    results_file = Path("gpu_benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}\n")


if __name__ == "__main__":
    main()
