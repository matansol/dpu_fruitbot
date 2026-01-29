# Memory Optimization Guide

## Problem
Server memory fills up quickly with multiple concurrent users, causing OOM crashes.

## Root Causes
1. **Large PPO models**: Each agent ~150-200 MB
2. **Episode data accumulation**: Images + frames + observations stored throughout session
3. **Multiple agents in memory**: Both current and previous agents kept loaded
4. **No automatic cleanup**: GameControl instances persist even after users finish

## Implemented Solutions

### 1. Episode Data Cleanup (`clear_episode_data()`)
**What it clears:**
- `episode_images` - Base64 encoded frame images
- `episode_frames` - Raw numpy arrays (64×64×3)
- `episode_obs` - Observation arrays
- `episode_actions` - Action history
- `episode_agent_locations` - Position tracking

**When it runs:**
- After sending `episode_data` to client in `play_entire_episode`
- On demand via `cleanup_all()`

**Memory saved:** ~50-100 MB per episode (depending on episode length)

### 2. Agent Cleanup (`cleanup_agents()`)
**What it clears:**
- Previous agent model (`prev_agent`)

**When it runs:**
- After each `next_episode` call
- On demand via `cleanup_all()`

**Memory saved:** ~150-200 MB per cleanup

### 3. Full Cleanup (`cleanup_all()`)
**What it clears:**
- Environment instance
- Current PPO agent
- Previous PPO agent
- All episode data

**When it runs:**
- When user completes 5 episodes
- When client disconnects
- On game finish

**Memory saved:** ~400-600 MB per user

### 4. GameControl Removal
**What happens:**
- After 5 episodes, entire GameControl instance is deleted
- User removed from `game_controls` dictionary
- All references cleared

**Memory saved:** Complete user session cleanup

### 5. Comparison Cleanup
**What it clears:**
- Temporary environments (`env1`, `env2`)
- Frame lists from `record_frames()`

**When it runs:**
- Immediately after generating comparison images in `agents_different_routs`

**Memory saved:** ~100-200 MB per comparison

## Memory Usage Estimates

### Per User (Worst Case)
- PPO agent: 200 MB
- Previous agent: 200 MB
- Episode data: 100 MB
- Environment: 50 MB
- **Total: ~550 MB per active user**

### With Optimizations
- PPO agent: 200 MB
- Episode data: 0 MB (cleared after sending)
- Previous agent: 0 MB (cleared after episode)
- Environment: 50 MB
- **Total: ~250 MB per active user**

### Server Capacity
- **Current (1 GB)**: 2 concurrent users maximum
- **Recommended (4 GB)**: 10-15 concurrent users with optimizations
- **Optimal (8 GB)**: 25-30 concurrent users comfortably

## Additional Optimization Opportunities

### Not Yet Implemented
1. **Lazy model loading**: Load agents only when needed, unload after use
2. **Frame resolution reduction**: 64×64 → 32×32 for internal storage
3. **Streaming instead of buffering**: Send frames as generated, don't store
4. **Model caching**: Share loaded models between users with same agent
5. **Limit concurrent compare_agents**: Queue requests to avoid memory spikes

### Configuration Options
Set these limits in `app.py` to restrict resource usage:

```python
MAX_CONCURRENT_USERS = 10  # Limit active sessions
MAX_EPISODES_PER_USER = 5  # Auto-cleanup after N episodes
ENABLE_EPISODE_DATA_CLEAR = True  # Clear data after sending
```

## Monitoring

### Check Memory Usage
Add logging to track memory consumption:

```python
import psutil
import os

process = psutil.Process(os.getpid())
mem_mb = process.memory_info().rss / (1024 * 1024)
print(f"Current memory usage: {mem_mb:.2f} MB")
```

### Key Metrics to Watch
- Memory per user: Should be ~250 MB with optimizations
- Total server memory: Should stay below 3.5 GB for 4 GB container
- Cleanup effectiveness: Memory should drop after each cleanup

## Testing

### Stress Test with Memory Monitoring
```powershell
# Test with 10 users
set N_CLIENTS=10
python tests\server_stress_test.py
```

### Expected Results
- Before: OOM crash at 3-4 users
- After: Stable with 10 users on 4GB container

## Deployment Checklist

- [x] Implement cleanup methods in GameControl
- [x] Add automatic cleanup after episodes
- [x] Add disconnect cleanup
- [x] Add cleanup in compare_agents
- [ ] Increase Azure Container to 4 GB RAM
- [ ] Test with 10 concurrent users
- [ ] Monitor memory usage in production
- [ ] Add memory usage logging/metrics
