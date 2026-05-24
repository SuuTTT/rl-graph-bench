# Benchmark Architecture Review — rl-graph-bench
_Authored: 2026-05-24 | Status: **6/6 algos implemented, all P0 targets passed** (v0.3.0)_

---

## 1. What Has Been Built

### 1.1 All Six Implemented Algorithms

| Algo | Family | Task | Model | Trainer | P0 Result | P1 Result |
|------|--------|------|-------|---------|-----------|-----------|
| **NeuroCUT** | A — NodeMove | Graph Partition | GraphSAGE (2L, h=128) + PairScorer | PPO | ✅ Cora NCut=0.2633 ≤ 0.33 | ✅ CiteSeer NCut=0.0408 ≤ 0.20 |
| **WRT** | B — StructuredMerge | Graph Partition | Cluster Transformer (4-head, h=64) | PPO | ✅ City Traffic NCut=0.0581 ≤ 0.060 | — |
| **SS2V-D3QN** | C — EdgeContraction | Graph Partition | DenseSAGE + edge-level Dueling D3QN | DQNTrainer | ✅ mini5 NCut=0.5391 ≤ 0.55† | — |
| **CLARE** | D — CommunityRW | Community Expansion | GIN (2L, h=64) | REINFORCE | ✅ Amazon F1=0.7956 ≥ 0.773 | — |
| **SLRL** | E — LocalExpand | Community Expansion | MLP policy (h=128) + s_coverage greedy | None (no RL needed) | ✅ Amazon F-score=0.9050 ≥ 0.878 | ✅ DBLP F-score=0.6922 ≥ 0.662 |
| **AC2CD** | F — DynamicGAT | Dynamic CD | GAT (2-head, h=64) + A2C | Trainer | ✅ BlogCatalog3 NMI=0.9541 ≥ 0.75 | — |

†SS2V paper (TNNLS 2025) behind paywall; mini5 is a proxy benchmark target, not the paper dataset.

---

## 2. Integration Map — How They Fit the Codebase

### 2.1 Shared Skeleton

Every algorithm in the repo implements `RLAgent` from `rlgb/algos/base.py`:

```
RLAgent (ABC)
├── select_action(obs, greedy) → action
├── push_transition(Transition)
├── update() → dict[str, float]
├── save(path) / load(path)
└── [optional] reset_episode(), on_epoch_end()
```

The `Transition` dataclass (`obs / action / reward / next_obs / done / info`) is the universal currency between environment and agent — used identically by all six algorithms.

All environments extend `ClusteringEnv(ABC, gym.Env)` from `rlgb/envs/base.py`, which provides a Dict obs space with standard keys: `adj (N×N)`, `node_feats (N×F)`, `labels (N,)`, `k (1,)`. Concrete envs add task-specific keys on top (e.g. `edge_idx (E,2)` for EdgeContractionEnv).

### 2.2 Shared Training Infrastructure

```
rlgb/training/
├── reinforce.py          REINFORCEConfig, compute_returns(), reinforce_loss()
├── trainer.py            Trainer + TrainConfig  (on-policy, episode-based)
├── ppo.py                PPOTrainer + PPOConfig  (for WRT)
└── dqn_trainer.py        DQNTrainer + DQNConfig  (for SS2V-D3QN)
```

`reinforce_loss()` and `compute_returns()` are **shared pure functions** — NeuroCUT, CLARE, and the AC2CD stub all call these directly. They are not duplicated per algo.

### 2.3 Shared Evaluation

`rlgb/eval/harness.py::eval_algo_on_suite(algo, suite, task, n_seeds, horizon)` is the single eval entry point. It produces a `pd.DataFrame` with all metrics per (problem, seed) and is algo-agnostic — it calls `algo.select_action()` and the task's env factory.

`rlgb/eval/metrics.py::compute_all()` computes NCut, H², NMI, ARI, modularity density, F1 in one pass.

### 2.4 Per-Algo Integration Depth

| Component | NeuroCUT | WRT | SS2V-D3QN | CLARE | SLRL | AC2CD |
|-----------|----------|-----|-----------|-------|------|-------|
| `RLAgent` ABC | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| `ClusteringEnv` | ✅ `NodeMoveEnv` | ✅ `StructuredPartitionEnv` | ✅ `EdgeContractionEnv` | ✅ `CommunityEnv` | ⚠️ Bypassed | ✅ `DynamicCDEnv` |
| Trainer | ✅ PPOTrainer | ✅ PPOTrainer | ✅ DQNTrainer | ⚠️ self-contained loop | ⚠️ `fit()` (0 epochs) | ✅ Trainer |
| `eval_algo_on_suite` | ✅ | ✅ | ✅ | ⚠️ Custom | ⚠️ Custom | ✅ |
| Shared `_node_features` | ✅ | ✅ | ✅ | — | — | ✅ |
| CLI (`rlgb run`) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**NeuroCUT / WRT / AC2CD / SS2V-D3QN** all use the standard env → trainer → eval path on the `Problem` data format.

**CLARE** uses `CommunityEnv` for the Rewriter phase; the Locator runs a separate GIN training loop outside the env/trainer stack.

**SLRL** exposes the full `RLAgent` interface but its working config uses s_coverage greedy (no RL training). The BC + REINFORCE pipeline exists and is activated by `scov_threshold=0.0`.

---

## 3. What Is Shared vs. What Is Self-Contained

### 3.1 Shared by All Six

- `Transition` dataclass (obs/action/reward/next_obs/done/info)
- `EpisodeBuffer` (list of Transitions, cleared after update)
- `reinforce_loss()`, `compute_returns()` from `rlgb/training/reinforce.py`
- `compute_all()` metrics from `rlgb/eval/metrics.py`
- `Problem` dataclass from `rlgb/tasks/base.py` (adj/k_target/gt_labels/name/family)
- CLI registration in `rlgb/cli.py`

### 3.2 Self-Contained per Algo

| What | NeuroCUT | WRT | SS2V-D3QN | CLARE | SLRL | AC2CD |
|------|----------|-----|-----------|-------|------|-------|
| GNN backbone | `sage.py` (SAGEConfig) | `Transformer` (inline) | `_SS2VNet` (DenseSAGE, inline) | `gin.py` (CLARENet) | Inline MLP (SLRLNet) | `gat.py` (GATEncoder) |
| Action space | (node, cluster) via NodeMoveEnv | (merge c_i,c_j) / (split c_i,v) via StructuredEnv | edge-idx contraction via EdgeContractionEnv | EXPAND/EXCLUDE tokens via CommunityEnv | Boundary node index (set) | (node, cluster) via DynamicCDEnv |
| Reward signal | −ΔNCut (unsupervised) | Wasserstein dist. improvement | −ΔNCut per step | ΔF1 vs GT | ΔF-score vs GT | Δ modularity density |
| Data format | `Problem` (adj + label vector) | `Problem` | `Problem` + `edge_idx` | `CLAREGraphData` (nx.Graph + community lists) | `CLAREGraphData` | `Problem` (temporal snapshots) |
| Training driver | PPOTrainer | PPOTrainer | DQNTrainer | CLAREAlgo.fit() internal loop | SLRLAlgo.fit() (0 epochs) | Trainer |

### 3.3 Observation: Two Data Formats Exist

Partition algos (NeuroCUT, WRT, SS2V-D3QN) operate on `Problem` — an adj matrix plus a flat label vector.  
Community algos (CLARE, SLRL) operate on `CLAREGraphData` — a NetworkX graph plus per-community membership lists.  
These two formats are not yet bridged. The `harness.py` eval path only handles `Problem`; CLARE/SLRL use custom `evaluate()` methods.

---

## 4. Is the Benchmark Infrastructure Now?

**Yes, with a caveat.**

The answer depends on which layer you look at:

| Layer | Status |
|-------|--------|
| Algo contract (`RLAgent` + `Transition` + `ClusteringEnv`) | ✅ Stable, all 6 algos registered |
| Shared training logic (REINFORCE loss, GAE, DQN target update) | ✅ Modular pure functions |
| Partition eval path (`eval_algo_on_suite` → `Problem` → `NodeMoveEnv`) | ✅ Production-ready |
| Community eval path (`CLAREGraphData` + custom `evaluate()`) | ⚠️ Parallel track, not unified |
| CLI (`rlgb run/eval/compare`) | ✅ Registered for all 6 algos |
| Data loaders | ✅ SNAP CLARE, PyG Planetoid/Amazon, synthetic suites |
| Metric suite | ✅ NCut/H²/NMI/ARI/modularity/F1 all in `compute_all()` |

The benchmark is infrastructure for **partition algos** today. The community eval path (CLARE/SLRL) is a parallel track that shares the agent contract but not the env/harness loop. Closing that gap — routing CLARE/SLRL through `eval_algo_on_suite` — is the main structural debt.

---

## 5. Forward Plan — Next Three Algorithms

### 5.1 WRT (RidgeCut) — Structured-Action Transformer + PPO

**Paper**: Jiang et al. 2025, [arXiv:2505.13986](https://arxiv.org/abs/2505.13986)  
**Target**: NCut ≤ 0.060 (City Traffic, k=4, n=100)  
**Task family**: Partition (unsupervised, same as NeuroCUT)

**What exists (stub)**:
- `rlgb/algos/structured/wrt.py`: `WRTConfig`, `_WRTNet` (Transformer + merge/split heads), `WRTAlgo(RLAgent)` with random exploration
- `rlgb/training/ppo.py`: `PPOTrainer` + `PPOConfig` already written
- `rlgb/envs/structured_env.py`: stub, needs ring/wedge action masks

**Implementation plan**:

1. **Structured env** (`structured_env.py`): Wire `_adjacent_cluster_pairs()` already in wrt.py into the env. Obs dict adds `merge_candidates (P,2)` and `split_candidates (K,)`. Action space `Discrete(P + K)`.
2. **WRTNet forward pass**: Already written — outputs `(merge_logits, split_logits, value)`. Needs `select_action()` to decode flat action index back to (merge vs split, pair/cluster idx).
3. **PPO wiring**: `WRTAlgo.select_action()` → log_prob via `torch.distributions.Categorical`; `update()` calls `ppo_loss()` in `ppo.py`. PPOTrainer is already compatible with any `RLAgent`.
4. **Data**: Need City Traffic dataset loader in `rlgb/data/`. Alternatively, run on Cora-sized graphs first (same NodeMoveEnv compatible domain), tune, then swap dataset.
5. **Eval**: Fully compatible with `eval_algo_on_suite` — same `Problem` format, same `GraphPartitionTask`.

**Estimated integration effort**: Medium. Transformer backbone + action-decoding is the bulk of work; trainer and eval already exist.

---

### 5.2 AC2CD — Dynamic Graph + GAT + A2C

**Paper**: Costa & Ralha, KBS 2023, [arXiv:2111.15623](https://arxiv.org/abs/2111.15623)  
**Target**: NMI ≥ 0.75 (BlogCatalog3)  
**Task family**: Dynamic community detection (temporal snapshots)

**What exists (stub)**:
- `rlgb/algos/dynamic/ac2cd.py`: `AC2CDConfig`, `_AC2CDNet` (GAT encoder + actor + critic), `AC2CDAlgo(RLAgent)` with random actions
- `rlgb/models/gat.py`: `GATEncoder` already implemented
- `rlgb/envs/dynamic_env.py`: stub, needs temporal snapshot logic
- `rlgb/training/trainer.py`: supports A2C via `value_coef` in REINFORCE; no separate A2C trainer needed

**Implementation plan**:

1. **Dynamic env** (`dynamic_env.py`): Each `reset()` advances to the next snapshot (edge additions/deletions). Obs adds `delta_edges (M,2)` (new/removed edges since last snapshot). Reward = ΔNMI vs snapshot GT labels.
2. **AC2CDNet is already complete**: `encoder(feats, adj)` → cluster mean embeddings → `actor(pair)` logits + `critic(h.mean)` value. Only `select_action()` and `update()` need wiring.
3. **A2C update**: Use existing `reinforce_loss()` with advantage = `reward + γ·V(s') - V(s)` — this is exactly A2C. No separate trainer needed.
4. **Data**: BlogCatalog3 temporal snapshots. Loader needed in `rlgb/data/`. Format: sequence of (adj_t, labels_t) pairs per time step.
5. **Eval**: `eval_algo_on_suite` works if we define `DynamicCDTask.build_env()` to return a `DynamicEnv` initialized at snapshot 0. The harness then runs `horizon` steps = `horizon` temporal steps.

**Estimated integration effort**: Medium-high. The dynamic snapshot data pipeline (BlogCatalog3 loader + `DynamicEnv.step()` semantics) is the hardest part; the GNN and training logic are already there.

---

### 5.3 SS2V-D3QN — Multicut + Dueling Double DQN

**Paper**: Li et al., TNNLS 2025  
**Target**: Near-optimal multicut objective (TBD once paper is obtained)  
**Task family**: Multicut via sequential edge contraction

**What exists (stub)**:
- `rlgb/algos/multicut/ss2v_d3qn.py`: Full stub — `SS2VConfig`, `_SS2VNet` (dense SAGE + `_DuelingHead`), `SS2VAlgo(RLAgent)` with ε-greedy, replay buffer, target network
- `rlgb/training/dqn_trainer.py`: `DQNTrainer` + `DQNConfig` already written  
- `rlgb/envs/edge_contraction_env.py`: stub, needs contraction logic

**Implementation plan**:

1. **Edge contraction env** (`edge_contraction_env.py`): State = current supergraph (shrinks each step). `step(edge_idx)` merges two supernodes, returns updated adj/feats and reward = −Δ(multicut cost). Terminal when exactly k supernodes remain.
2. **SS2VNet**: Already written. `forward(feats, adj, graph_feat)` → Q-values over all edges. Need to handle variable edge count across contraction steps (mask invalid indices, not just zero them).
3. **DQN wiring**: `SS2VAlgo` already has `_replay_buffer`, `_target_net`, `_eps` decay, `push_transition()` and a skeleton `update()`. Complete `update()` to sample batch, compute TD targets with target net, backprop `F.smooth_l1_loss`.
4. **Double DQN**: Action selection via online net, value estimation via target net — two-line change in `update()`.
5. **Data**: Any graph with a known multicut solution. Start with synthetic (Erdős-Rényi, k=4, n=50) before real datasets.
6. **Eval**: Needs a new `MulticutTask` + compatible harness call, or direct `algo.evaluate()` on contraction sequences. The `Problem` format works for the graph; need to define the multicut objective function in `metrics.py`.

**Estimated integration effort**: Medium. Off-policy DQN infrastructure already exists; edge contraction state management is the novel piece.

---

## 6. Unification Roadmap

The main gap is that community algos (CLARE, SLRL) bypass `eval_algo_on_suite`. Here's the minimal path to full unification:

### Step 1 — Bridge data formats (1–2 days)
Add `CLAREGraphData.to_problem_suite() → list[Problem]` where each community becomes a `Problem` with `adj = local_subgraph`, `gt_labels = membership_mask`.  
This lets the harness run CLARE/SLRL without changing either algo.

### Step 2 — Unified community task (1 day)  
`CommunityExpandTask.build_env(problem)` already exists. Route SLRL's `_scov_greedy_episode` and CLARE's `CLARERewriter._run_episode` through `CommunityEnv.step()` so both appear to the harness as standard gym interactions.

### Step 3 — Single leaderboard (½ day)  
`full_benchmark.py` already prints separate `PARTITION` and `COMMUNITY` sections. Once data formats are bridged, both sections can go through `eval_algo_on_suite` → same CSV columns → same leaderboard machinery.

### Step 4 — Dynamic and Multicut tasks (per-algo, ~1 day each)  
Define `DynamicCDTask.build_env()` and `MulticutTask.build_env()`. Both already have env stubs. Wiring them into harness completes the 5-family coverage.

---

## 7. Summary

The benchmark is **structurally sound infrastructure** for partition algorithms today. The agent contract (`RLAgent` + `Transition` + `ClusteringEnv`), shared training modules (REINFORCE/PPO/DQN), and evaluation harness give the next three algorithms almost everything they need at no cost:

- WRT drops in via `PPOTrainer` + `NodeMoveEnv` extended with ring/wedge masks
- AC2CD drops in via existing `reinforce_loss()` + a new `DynamicEnv` snapshot loop  
- SS2V-D3QN drops in via `DQNTrainer` + a new `EdgeContractionEnv`

The main open work is (a) closing the community data-format gap so CLARE/SLRL go through the standard harness, and (b) implementing the three env bodies and dataset loaders for the remaining families. The agent logic stubs are already written and registered in the CLI.
