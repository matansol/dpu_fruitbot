# 🎯 Immediate Actions: Maximize RTX 3060 for FruitBot Training

## TL;DR - Run This Now

```powershell
# 1. Install optimization tools (30 seconds)
pip install wandb optuna optuna-dashboard

# 2. Run GPU benchmark to find optimal settings (5 minutes)
python gpu_benchmark.py --device cuda --quick

# 3. Start hyperparameter search with optimal GPU settings (6-8 hours)
# Use the settings from step 2 benchmark results
python train_env/optuna_search.py `
  --n-trials 20 `
  --total-steps 500000 `
  --n-envs 64 `
  --device cuda `
  --study-name fruitbot_quick `
  --use-wandb `
  --wandb-project fruitbot-optimization
```

---

## 📈 Why Your GPU is Only at 7%

Your current training command uses:
- `--batch-size 256` → Too small! GPU barely working
- `--n-envs 32` → Could be higher
- `--n-steps 256` → Limits batch processing

**Your RTX 3060 has 6GB VRAM** but you're only using ~1.6GB for training.

---

## 🚀 Quick Wins (Immediate 10x+ Speedup)

### Change 1: Increase Batch Size (Most Important!)
```powershell
--batch-size 2048   # Up from 256 (8x increase)
--n-steps 2048      # Match batch size
```
**Expected GPU usage: 60-70%**

### Change 2: More Parallel Environments
```powershell
--n-envs 64         # Up from 32 (2x increase)
```
**Expected GPU usage: 75-85%**

### Change 3: Add PPO Epochs
```powershell
--n-epochs 10       # More training per batch (new parameter)
```
**Better sample efficiency**

### Combined Command:
```powershell
python train_env/train.py `
  --env fruitbot `
  --use-source `
  --device cuda `
  --total-steps 3000000 `
  --n-envs 64 `
  --n-steps 2048 `
  --batch-size 2048 `
  --n-epochs 10 `
  --learning-rate 5e-4 `
  --ent-coef 0.05 `
  --num-levels 100 `
  --fruitbot-reward-positive 2 `
  --fruitbot-reward-negative -1 `
  --fruitbot-reward-wall-hit -2 `
  --fruitbot-reward-step 0.02 `
  --save-freq 200000 `
  --wandb-project fruitbot-gpu-test `
  --wandb-run-name test_high_batch
```

**Expected Results:**
- ✅ GPU utilization: 80-90% (vs. 7% before)
- ✅ Training speed: 7000-9000 samples/sec (vs. ~2000 before)
- ✅ Training time: ~10-12 hours for 3M steps (vs. 30+ hours before)

---

## 🧪 Finding Best Hyperparameters

Your shaped reward function changes the optimization landscape. **Don't guess** - use automated search!

### Option 1: Optuna (Recommended - Smart Search)

**Why Optuna:**
- ✅ 10-100x faster than grid search
- ✅ Uses Bayesian optimization (learns from previous trials)
- ✅ Automatically prunes bad runs
- ✅ Beautiful dashboard

**Quick Search (Recommended First):**
```powershell
python train_env/optuna_search.py `
  --n-trials 20 `
  --total-steps 500000 `
  --n-envs 64 `
  --device cuda `
  --study-name fruitbot_quick `
  --use-wandb
```

**Time:** 6-8 hours (20 trials × 15-20 min each)  
**Result:** Top 3-5 hyperparameter combinations

**Deep Search (After Quick):**
```powershell
python train_env/optuna_search.py `
  --n-trials 50 `
  --total-steps 2000000 `
  --n-envs 64 `
  --device cuda `
  --study-name fruitbot_deep `
  --use-wandb
```

**Time:** ~24-30 hours  
**Result:** Near-optimal hyperparameters

### Option 2: Manual Grid Search (Slower)

```powershell
python train_env/grid_search.py `
  --total-steps 500000 `
  --n-envs 64 `
  --eval-episodes 50
```

⚠️ **Not recommended** - much slower than Optuna

---

## 📊 Weights & Biases Setup (5 minutes)

**Why wandb:**
- Track all experiments in one place
- Compare runs side-by-side
- Beautiful visualization
- Share results easily
- Free for academics/personal use

**Setup:**
```powershell
# Install
pip install wandb

# Login (creates free account)
wandb login

# Done! Your training commands will automatically log
```

**View results:**
```
https://wandb.ai/<your-username>/fruitbot-optimization
```

---

## 🎛️ Critical Hyperparameters for Your Reward Function

Your current reward shaping:
```
Completion:  +10.0
Good fruit:  +1.0
Bad fruit:   -1.0
Wall hit:    -3.0
Step:        +0.01
```

### Issues to Test:

1. **Learning Rate**: Your shaped rewards might need **lower LR**
   - Current: `1e-3`
   - Test: `[1e-4, 5e-4, 1e-3]`
   - Why: Shaped rewards create steeper gradients

2. **Entropy Coefficient**: Need **more exploration**
   - Current: `0.05`
   - Test: `[0.03, 0.05, 0.1]`
   - Why: Agent might converge to local optimum too fast

3. **Step Reward**: Might be **too small**
   - Current: `0.01` → only 5.0 reward over 500 steps
   - Test: `[0.01, 0.02, 0.05]`
   - Why: Should be meaningful fraction of completion bonus

4. **Wall Penalty**: Might **discourage exploration**
   - Current: `-3.0` (3x fruit reward!)
   - Test: `[-1.0, -2.0, -3.0]`
   - Why: Agent might avoid risky but optimal paths

### Optuna will test these automatically!

---

## 🔍 Monitoring Your Training

### Real-time GPU Check:
```powershell
# In another terminal, watch GPU usage
watch -n 1 nvidia-smi
# Or on Windows:
while ($true) { cls; nvidia-smi; Start-Sleep -Seconds 1 }
```

**Target:** 80-95% GPU utilization, 4-5GB VRAM usage

### Wandb Dashboard:
- Episode reward (should increase steadily)
- Entropy (should stay > 0.5)
- Policy loss (should decrease)
- Value loss (should decrease)
- GPU utilization (aim for 80%+)

---

## ⚡ Performance Comparison

| Configuration | GPU Util | Samples/sec | Time for 3M steps | GPU Usage |
|---------------|----------|-------------|-------------------|-----------|
| **Your current** | 7% | ~2000 | ~30 hours | ❌ Underutilized |
| **Recommended** | 85% | ~8000 | ~8 hours | ✅ Optimal |
| **Aggressive** | 95% | ~10000 | ~6 hours | ⚠️ Risk OOM |

---

## 🎓 Recommended Order

```
Day 1:
├─ 09:00 - Install tools (wandb, optuna)
├─ 09:15 - Run GPU benchmark
├─ 09:30 - Start Optuna quick search (20 trials, 500K steps)
└─ 17:00 - Review results, pick best 3 configs

Day 2:
├─ 09:00 - Train best config for 5M steps
└─ 21:00 - Training complete!

Day 3:
├─ 09:00 - Evaluate agent (100 episodes)
├─ 10:00 - Visualize policy
└─ Done! 🎉
```

---

## 🆘 Quick Troubleshooting

### "Out of Memory" error
```powershell
# Reduce batch size
--batch-size 1024 --n-envs 48
```

### GPU still low (<30%)
```powershell
# Increase batch more
--batch-size 4096 --n-steps 4096
```

### Training unstable (reward jumps around)
```powershell
# Lower learning rate
--learning-rate 1e-4
# And clip range
--clip-range 0.1
```

### Agent not improving
```powershell
# More exploration
--ent-coef 0.15
# Or check if wall penalty too harsh
--fruitbot-reward-wall-hit -1.0
```

---

## 📁 Files I Created for You

1. **`OPTIMIZATION_GUIDE.md`** - Comprehensive guide with all details
2. **`QUICKSTART.md`** - Step-by-step quick start
3. **`optuna_search.py`** - Automated hyperparameter optimization
4. **`gpu_benchmark.py`** - Find optimal GPU settings for your hardware
5. **`train.py`** - Updated with wandb integration and GPU optimizations
6. **`requirements.txt`** - Updated with wandb, optuna

---

## ✅ Action Checklist

- [ ] Install wandb and optuna: `pip install wandb optuna optuna-dashboard`
- [ ] Run GPU benchmark: `python train_env/gpu_benchmark.py --device cuda --quick`
- [ ] Start Optuna search: `python train_env/optuna_search.py --n-trials 20 ...`
- [ ] Monitor on wandb: Check https://wandb.ai/
- [ ] Review best hyperparameters after search completes
- [ ] Train final model with best settings for 5M steps
- [ ] Evaluate and celebrate! 🎉

---

**Questions?** Check `OPTIMIZATION_GUIDE.md` for detailed explanations!

**Ready to start?** Run the GPU benchmark first! ⚡
