# 🚀 Quick Start: Optimize Your FruitBot Training

This guide gets you from 7% GPU utilization to 80%+ and helps you find the best hyperparameters fast.

## Step 1: Install Required Packages (5 minutes)

```powershell
# Install optimization tools
pip install wandb optuna optuna-dashboard

# Login to wandb (free account)
wandb login
```

## Step 2: Find Your Optimal GPU Configuration (10 minutes)

Run the GPU benchmark to find the best settings for your RTX 3060:

```powershell
python train_env/gpu_benchmark.py --device cuda --quick
```

This will test different configurations and tell you exactly what parameters to use.

**Expected Output:**
```
🏆 RECOMMENDED CONFIGURATION
Best throughput: 8500 samples/sec
VRAM usage: 4800 MB / 6144 MB (78.1%)

Optimal training command:
python train.py --n-envs 64 --n-steps 2048 --batch-size 2048 ...
```

## Step 3a: Quick Hyperparameter Search (6-8 hours)

Use Optuna to find promising hyperparameter combinations:

```powershell
python train_env/optuna_search.py `
  --n-trials 20 `
  --total-steps 500000 `
  --n-envs 64 `
  --device cuda `
  --study-name fruitbot_quick `
  --use-wandb `
  --wandb-project fruitbot-optimization
```

**What this does:**
- Tests 20 different hyperparameter combinations
- Each trial trains for 500K steps (~15-20 minutes)
- Uses smart Bayesian optimization (not grid search)
- Automatically prunes bad configurations
- Logs everything to wandb for visualization

**Monitor progress:**
```powershell
# In another terminal - visualize in real-time
optuna-dashboard sqlite:///optuna_studies/fruitbot_quick/optuna.db
# Open: http://localhost:8080
```

## Step 3b: OR Train with Recommended Settings (12-16 hours)

If you want to skip hyperparameter search and use optimized defaults:

```powershell
python train_env/train.py `
  --env fruitbot `
  --use-source `
  --device cuda `
  --total-steps 5000000 `
  --n-envs 64 `
  --n-steps 2048 `
  --batch-size 2048 `
  --learning-rate 5e-4 `
  --gamma 0.99 `
  --gae-lambda 0.95 `
  --clip-range 0.2 `
  --ent-coef 0.05 `
  --n-epochs 10 `
  --num-levels 200 `
  --distribution-mode easy `
  --fruitbot-reward-completion 10.0 `
  --fruitbot-reward-positive 2.0 `
  --fruitbot-reward-negative -1.0 `
  --fruitbot-reward-wall-hit -2.0 `
  --fruitbot-reward-step 0.02 `
  --save-freq 200000 `
  --wandb-project fruitbot-optimization `
  --wandb-run-name optimal_v1
```

## Step 4: Monitor Training

### Option A: Weights & Biases (Recommended)
```powershell
# Automatically tracked! Just open:
# https://wandb.ai/<your-username>/fruitbot-optimization
```

### Option B: TensorBoard
```powershell
tensorboard --logdir models/fruitbot/
# Open: http://localhost:6006
```

## Step 5: Evaluate Your Agent

After training completes:

```powershell
python train_env/evaluate.py `
  --model models/fruitbot/<timestamp>_easy/ppo_final.zip `
  --n-episodes 100 `
  --render
```

---

## 🎯 Key Improvements Over Your Current Setup

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| **GPU Utilization** | 7% | 80-95% | **13x** |
| **Batch Size** | 256 | 2048 | **8x** |
| **Parallel Envs** | 32 | 64 | **2x** |
| **Training Speed** | ~2000 samp/s | ~8000 samp/s | **4x** |
| **Hyperparameter Tuning** | Manual | Automated | ∞ |
| **Experiment Tracking** | None | wandb | ✅ |

---

## 🔍 What to Look For

### Good Signs:
- ✅ GPU utilization > 80%
- ✅ Episode reward increasing steadily
- ✅ Entropy > 0.5 (exploration happening)
- ✅ Policy loss decreasing
- ✅ Agent collects more good fruit over time

### Warning Signs:
- ⚠️ Reward plateaus early (< 1M steps)
- ⚠️ Entropy drops to ~0 (no exploration)
- ⚠️ High KL divergence (> 0.1) - policy updating too fast
- ⚠️ Agent avoids all walls/risks (wall penalty too high)

---

## 💡 Pro Tips

1. **Run overnight**: 5M steps = ~12-16 hours on RTX 3060
2. **Multiple seeds**: Best configs should be tested with 3 different `--seed` values
3. **Curriculum learning**: Start with `--num-levels 50`, increase to 200 after 1M steps
4. **Reward debugging**: Log individual reward components to see what agent optimizes
5. **Compare baselines**: Keep one run with original rewards to measure improvement

---

## 🆘 Troubleshooting

### GPU Shows Low Utilization (<30%)
```powershell
# Try increasing batch size more aggressively
--batch-size 4096 --n-steps 4096
```

### Out of Memory Error
```powershell
# Reduce batch size and envs
--batch-size 1024 --n-envs 48
```

### Training Too Slow
```powershell
# Use fewer levels during early training
--num-levels 50
```

### Agent Not Learning
```powershell
# Increase exploration
--ent-coef 0.1

# Or reduce learning rate
--learning-rate 1e-4

# Or adjust reward shaping
--fruitbot-reward-step 0.05  # More step rewards
```

---

## 📊 Expected Results

With optimal hyperparameters and shaped rewards, you should see:

- **After 500K steps**: Agent consistently collects 3-5 fruits
- **After 1M steps**: Agent reaches goal in 30-50% of episodes
- **After 3M steps**: Agent reaches goal in 70-80% of episodes
- **After 5M steps**: Near-optimal policy with 85-95% success rate

---

## 🎓 Next Steps

1. ✅ Run GPU benchmark
2. ✅ Start Optuna hyperparameter search
3. ✅ Monitor on wandb
4. ✅ Train best configuration for 5M steps
5. ✅ Evaluate on 100 test episodes
6. ✅ Visualize policy with `--render`
7. ✅ Share results! (wandb has great sharing features)

---

**Time Investment:**
- Setup: 15 minutes
- Quick HP search: 6-8 hours
- Final training: 12-16 hours
- **Total: ~1 day** to optimal policy

**vs. Manual tuning: weeks** ⏰

Good luck! 🚀
