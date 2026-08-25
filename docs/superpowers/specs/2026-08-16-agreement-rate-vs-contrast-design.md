# Agreement-rate board metric vs. Contrast method — Design

Date: 2026-08-16

## Motivation

The IUI paper's **Contrast** method selects the demonstration board (env config) that
maximizes the reward difference between two agents (the "before" and "after" agent shown
to the user). This is implemented via `models_neighbors[a][b]['contrast_config']` in
`app.py` and computed from per-`(model, config, level)` reward rollouts in
`evaluate_agents.ipynb` (reward-diff method).

We want to validate this choice against a second, independent notion of "which board best
distinguishes two agents": **action-agreement rate** (the Yotam-style measure of how often
two agents take the same action from the same state). If the two methods agree — i.e. the
Contrast-selected config is also the lowest-agreement config — that strengthens the paper's
board-selection story.

## Definitions

- **Board**: `(config_index ∈ {0,1,2,3}, start_level ∈ {0..99})`, built exactly like the
  app demonstration environment: `dpu_clf.get_config_by_index(config_index)`, with
  `num_levels=1`, `rand_seed=0`, `start_level=level`. Agents run with
  `deterministic=True`, so each agent has one fixed trajectory per board.
- **Agent pairs**: the 14 unique *undirected* pairs appearing in `models_neighbors`
  (agreement is symmetric, so directed duplicates are collapsed).
- **Agreement rate (symmetric two-pass)** for pair (A, B) on a board:
  - Roll out A → visited states `S_A`, own actions `act_A`.
  - Roll out B → visited states `S_B`, own actions `act_B`.
  - Query the other agent on each visited state: `B(S_A)`, `A(S_B)`.
  - `agreement = ( |{s∈S_A : A(s)=B(s)}| + |{s∈S_B : B(s)=A(s)}| ) / (|S_A| + |S_B|)`
  - `disagreement = 1 − agreement`.

## Compute plan

Outer loop over boards, inner over pairs, with a per-board trajectory cache so each agent
is rolled out once per board and reused across every pair it participates in:

```
for config in 0..3:
  for level in 0..99:
    cache = {}
    for agent_idx in agents_used:
        cache[agent_idx] = rollout_trajectory(agent[agent_idx], config, level)  # (obs_seq, act_seq)
    for (A, B) in unique_pairs:
        bA = agent[B].predict(stack(obs_A))   # batched
        aB = agent[A].predict(stack(obs_B))
        agree = (sum(act_A == bA) + sum(act_B == aB)) / (len_A + len_B)
        record row (pair, config, level, agree, disagree, len_A, len_B)
    del cache
```

Bounds memory to 6 trajectories at a time (~4 GB of stored observations avoided). ~317k env
steps for rollouts; cross-agent predictions batched for speed. All 6 agents loaded once up
front.

`rollout_trajectory(agent, config, level)`: create env, reset, step with deterministic
actions until `done` or a 1000-step safety cap; return the observation sequence and the
agent's own action sequence.

## Outputs

1. `results/agreement_by_board.csv` — one row per `(pair, config, level)`:
   `pair_a, pair_b, config_index, start_level, agreement, disagreement, len_a, len_b`,
   left-joined with `|reward_diff|` recomputed from `results/simple_eval100.csv`.
2. **Per-config aggregation**: mean agreement per `(pair, config)`; for each pair, the
   argmin-agreement config and the rank (1 = lowest agreement) of its `contrast_config`.
3. **Validation analysis**:
   - Exact-match rate: fraction of pairs where `contrast_config` == lowest-agreement config.
   - "Close" rate: `contrast_config` within the bottom-2 configs by agreement.
   - Spearman correlation between per-config mean `|reward_diff|` and mean disagreement
     (per pair and pooled).
4. **Per-level detail**: agreement of the selected `contrast_config` levels
   (`models_neighbors[a][b]['configs'][contrast_config]`) vs. that config's level distribution.
5. **Plots**: pair×config agreement heatmap; reward-diff vs. disagreement scatter;
   distribution of the `contrast_config` agreement rank across pairs.

### Level coverage

Primary analysis uses all 100 levels/config. A secondary variant restricts the per-config
aggregation to the finishing-level filter used by the reward-diff analysis (levels where ≥2
models complete, from `simple_eval100.csv`), reported alongside for robustness.

## Reuse vs. build

- **Reuse**: `dpu_clf.load_agent`, `dpu_clf.get_config_by_index`, `models_dict` and
  `models_neighbors` (copied from `app.py`), `results/simple_eval100.csv`.
- **Build**: `agreement_analysis.ipynb` containing `rollout_trajectory`, the cached board
  loop, aggregation/comparison, and plots.

## Non-goals

- No changes to `app.py` or the live board-selection logic.
- No retraining or re-evaluation of reward rollouts (reuse existing `simple_eval100.csv`).
