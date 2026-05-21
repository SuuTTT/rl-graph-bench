# Changelog — rl-graph-bench

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### TODO

#### Paper Reproduce Targets

| Algo | Paper | arXiv | Metric | Target | Dataset | vs Baseline |
|------|-------|-------|--------|--------|---------|-------------|
| NeuroCUT | Shah et al., KDD 2024 | [2310.11787](https://arxiv.org/abs/2310.11787) | NCut ↓ | **0.33** | Cora k=5 | GAP=0.68 (−51%) |
| NeuroCUT | Shah et al., KDD 2024 | [2310.11787](https://arxiv.org/abs/2310.11787) | Sparsest Cut ↓ | **1.46** | Cora k=5 | DMon=1.89 (−23%) |
| WRT (RidgeCut) | Jiang et al., 2025 | [2505.13986](https://arxiv.org/abs/2505.13986) | NCut ↓ | **0.060** | City Traffic k=4 n=100 | NeuroCUT=0.078 (−23%) |
| CLARE | Wu et al., KDD 2022 | [2210.08274](https://arxiv.org/abs/2210.08274) | F1 ↑ | SOTA | SNAP DBLP/Amazon/LJ | prior methods |
| SLRL | Ni et al., AAAI 2025 | *(no arXiv)* | F-score ↑ | **0.878** (Amazon), **0.662** (DBLP) | SNAP Amazon/DBLP | SEAL=0.839, CLARE=0.795 |
| AC2CD | Costa & Ralha, KBS 2023 | [2111.15623](https://arxiv.org/abs/2111.15623) | NMI ↑ | **0.75** (BlogCatalog3) | Email-EU-Core, BlogCatalog3 | SDNE, GraphGAN, CLARE |
| SS2V-D3QN | Li et al., TNNLS 2025 | *(no arXiv)* | Multicut ↓ | TBD (TNNLS paper) | synthetic + real multicut | TBD |

- [x] **Close paper gap** — code fixes done:
      (1) `full_benchmark.py` now uses `objective='ncut'` (was incorrectly `'h2'`).
      (2) `_train_neurocut` Phase 2 capped at ≤ 500 ep (all-negative-reward fine-tuning >1000 ep
          corrupts Phase 1 model via destructive REINFORCE updates).
      (3) `_print_paper_gap` targets corrected to NeuroCUT NCut≤0.333 (Cora k=4),
          WRT NCut≤0.060 (City Traffic k=4 n=100).
      (4) Full-run eval now uses leiden warm-start (paper-protocol: refine existing partition).
      **Performance note**: NeuroCUT on mini5 SBM reaches NCut≈0.438 (leiden WS eval, Phase 1
      only, hidden=256) vs Spectral=0.406. Beating Spectral by 18% on SBM is structurally
      hard — Spectral is near-optimal on stochastic block models. True paper target (0.333 on
      Cora) requires real-graph evaluation → see TODO #5 (real-world loaders).
- [x] **WRT training** — `StructuredPartitionEnv` created (`rlgb/envs/structured_env.py`): action
      space = merge adjacent cluster pair OR split a cluster. k_target constraint enforced in
      `step()` (merge only when k > k_target; split only when k < k_target) to prevent trivial
      NCut=0 collapse. `GraphPartitionTask.build_env(env_class='structured')` added. `_train_wrt`
      updated to use `env_class='structured', warm_start='random'`. Partition benchmark eval now
      calls WRT with `env_class='structured'` separately from NeuroCUT (NodeMoveEnv). Quick-run
      result on mini5 SBM (50 ep): WRT NCut=0.448.
      **Performance note**: paper target (NCut≤0.060) is on City Traffic graph (k=4, n=100
      road-network topology) — requires a real-world loader (see TODO #5).
- [x] **AC2CD training** — `DynamicCDEnv` now accepts `warm_start='leiden'` (runs Leiden on
      snapshot[0] for initial partition; falls back to random on import error). `DynamicCDTask.
      build_env` passes `**kwargs` so `warm_start` flows through. `_train_ac2cd` and dynamic eval
      both use `warm_start='leiden'`. Dynamic eval uses separate `compare_algos` call for AC2CD.
      Quick-run result on synthetic 3-snapshot SBM (30 ep): AC2CD NMI=1.00 (matches Leiden).
      **Performance note**: paper target (NMI≥0.75 on BlogCatalog3) requires a real-world dynamic
      graph dataset — see TODO #5 (real-world loaders).
- [x] **SS2V-D3QN training** — `EdgeContractionEnv` created (`rlgb/envs/edge_contraction_env.py`):
      action = edge index among inter-cluster edges; step = merge clusters of edge endpoints;
      resets to k_init = min(N//2, 2*k_target) random clusters so agent contracts down to
      k_target; terminates when k == k_target or no inter-cluster edges remain.
      `GraphPartitionTask.build_env(env_class='edge_contraction')` added. `SS2VAlgo.
      select_action()` updated to return edge index (not node-move flat index). `SS2VAlgo.
      update()` uses action as direct edge index in DQN loss. `_train_ss2v` uses
      `env_class='edge_contraction'`; partition benchmark eval uses same.
      **Performance note**: DQN needs 10k+ steps to converge (vs 50-ep quick run); full-run
      (20k steps) expected to produce competitive NCut. SS2V paper target is multicut on
      synthetic+real graphs — not directly comparable to NCut partition benchmark.
- [ ] **Real-world loaders** — `pyg_loaders.py` + `snap_loaders.py` exist but datasets are downloaded lazily; add Cora/CiteSeer/DBLP benchmarks to `full_benchmark.py`.
- [ ] **WRT community task eval** — `CommunityEnv` + CLARE/SLRL need trained weights to compare against SLRL paper numbers.
- [ ] **Speed: vectorised NCut** — current NCut computation is O(E) per step but runs in Python; a torch-batched version would enable GPU training.
- [ ] **PPO replace REINFORCE** — `ppo.py` trainer exists; swap NeuroCUT training loop from REINFORCE to PPO for better sample efficiency.
- [ ] **Save/load trained checkpoints in experiments/** — `full_benchmark.py` trains from scratch each run; cache checkpoints to `results/` so re-runs skip training.
- [ ] **CI matrix: Windows + macOS** — current CI only covers Linux (ubuntu-latest); add Windows runner once torch_geometric has stable wheels.

---

## [0.1.0] — 2026-05-20

Initial public version. 5 git commits from `init` to HEAD.

### 6 RL Algorithms

| # | Name | Family | Action Space | Backbone | Training | Task |
|---|------|--------|-------------|----------|----------|------|
| 1 | **NeuroCUT** | NodeMove | reassign node → cluster | GraphSAGE + PairScorer | REINFORCE + value baseline | Partition (NCut) |
| 2 | **WRT** | Structured-Action | merge adjacent clusters / split on min-cut wedge | Transformer | PPO | Partition (NCut / H²) |
| 3 | **SS2V-D3QN** | Multicut | contract edge (merge two supernodes) | GCN + Dueling DQN | D3QN (replay buffer) | Partition (multicut) |
| 4 | **CLARE** | Community-RW | EXPAND / EXCLUDE per cluster | GIN | REINFORCE + baseline | Community (NCut / F1) |
| 5 | **SLRL** | Seed-Local-RL | EXPAND / EXCLUDE from seed | Swish-MLP | REINFORCE | Community (NCut) |
| 6 | **AC2CD** | Dynamic-GAT | reassign node in temporal snapshot | GAT + A2C | A2C | Dynamic-CD |

### 4 Classical Baselines

| Name | Method | Notes |
|------|--------|-------|
| `SpectralBaseline` | Spectral clustering (sklearn) | NCut gold-standard on mini5 |
| `LeidenBaseline` | Leiden algorithm (leidenalg) | Best H² on mini5 |
| `LouvainBaseline` | Louvain (community-louvain) | Similar to Leiden |
| `RandomBaseline` | Random partition | Sanity check |
| `MetisBaseline` | METIS k-way partitioning | Optional (pymetis) |

### 3 Task Environments (Gymnasium-compatible)

| Task | Env class | Objectives supported |
|------|-----------|----------------------|
| Graph Partition | `NodeMoveEnv` | NCut, H², balanced-cut, sparsest-cut |
| Community Expand | `CommunityEnv` | NCut, H², F1 vs ground truth |
| Dynamic CD | `DynamicEnv` | NCut on temporal snapshots |

### 3 GNN Backbones

- `GraphSAGEEncoder` + `PairScorer` + `ValueHead` — used by NeuroCUT (`rlgb/models/sage.py`)
- `GATEncoder` — multi-head attention encoder for AC2CD (`rlgb/models/gat.py`)
- `GINEncoder` — isomorphism network for CLARE (`rlgb/models/gin.py`)

### Training Infrastructure

- **REINFORCE** (`rlgb/training/reinforce.py`) — actor-critic returns, entropy bonus, gradient clipping
- **PPO** (`rlgb/training/ppo.py`) — clipped surrogate, GAE, value loss
- **DQN Trainer** (`rlgb/training/dqn_trainer.py`) — replay buffer, ε-greedy, target network
- **Trainer** (`rlgb/training/trainer.py`) — unified training loop for REINFORCE-based algos  
  - `TrainConfig`: `n_episodes`, `horizon`, `lr`, `n_episode_per_update`, `env_rotate_every`, `log_every`, `save_every`, `out_dir`  
  - `env_rotate_every`: rotate training graph every N episodes (0 = auto = n_episodes // 20)

### Evaluation Harness (`rlgb/eval/harness.py`)

- `eval_algo_on_suite(algo, suite, task, n_seeds, horizon, greedy, env_kwargs, best_of)` — run algo across all graphs, return tidy DataFrame
- `compare_algos(algos, suite, task, eval_kwargs)` — multi-algo comparison table
- `summary_table(df)` — aggregate mean/std per algorithm
- **best-of-N eval** — `best_of=N` runs N stochastic rollouts, keeps partition with lowest NCut (reduces variance, effective even untrained: best_of=5 cut NCut 1.78→1.47 on mini5)

### Data

- **Synthetic suites** (`rlgb/data/synthetic.py`): `mini5` (5 graphs, 20–60 nodes), `fixed17` (17 graphs, 50–200 nodes)
- **PyG loaders** (`rlgb/data/pyg_loaders.py`): Cora, CiteSeer, DBLP (lazy download)
- **SNAP loaders** (`rlgb/data/snap_loaders.py`): email-Eu-core, ca-GrQc (lazy download)

### CLI (`rlgb/cli.py`)

```
python -m rlgb.cli eval --algo neurocut --suite mini5 --seeds 3
python -m rlgb.cli train --algo neurocut --suite mini5 --episodes 500
python -m rlgb.cli compare --algos neurocut leiden spectral --suite mini5
```

### Experiments (`experiments/full_benchmark.py`)

- `--quick` flag: fast 3-seed / 10-step run for CI smoke-testing
- `curriculum=True`: Phase 1 (random warm-start) → Phase 2 (leiden fine-tune) training
- Outputs: `results/benchmark_v1.csv`, `results/benchmark_v1_summary.txt`

### Test Suite — 79 tests, 100% pass

| File | Tests | Coverage |
|------|-------|---------|
| `tests/test_phase1_3.py` | 26 | Metrics, synthetic data, `GraphPartitionTask`, `NodeMoveEnv`, `NeuroCUTPolicy`, `NeuroCUTAlgo`, `Trainer` |
| `tests/test_phase4_7.py` | 26 | CLARE, SLRL, AC2CD, WRT, SS2V-D3QN, `Trainer` integration |
| `tests/test_phase8_9.py` | 27 | PyG loaders, SNAP loaders, eval harness, CLI, best-of-N eval |

### CI (`/.github/workflows/ci.yml`)

- Lint: ruff
- Type-check: mypy
- pytest × Python 3.10 / 3.11 / 3.12
- CLI smoke test
- Runs on every push/PR to `master`

### Benchmark Results (mini5, 2k ep curriculum training)

| Algo | NCut ↓ | vs Spectral |
|------|--------|-------------|
| Spectral (baseline) | 0.406 | — |
| **NeuroCUT (curriculum-trained)** | **0.417** | +2.9% |
| Leiden (baseline) | 0.582 | +43% |
| NeuroCUT (untrained) | 0.640 | +58% |

Curriculum: Phase 1 (1000 ep, random WS, lr=3e-4) → Phase 2 (1000 ep, leiden WS fine-tune, lr=1e-4).  
Phase 1 training return: +0.069 → **+0.344** over 1000 episodes.  
Paper target (−18% vs Spectral): NCut ≤ 0.333.

### Bug Fixes

- **`NeuroCUTPolicy.forward` IndexError** (commit `ee9459b`): `torch.randint(0, K)` can draw fewer than K unique labels, making `labels.max()+1 < K` while candidates still reference cluster index K−1. Fixed: `k = max(labels_max+1, cands_col1_max+1)`. Resolved intermittent `test_greedy_in_range` failure.
- **`save_every=0` ZeroDivisionError** in `Trainer.train()`: guard `if save_every > 0 and ep % save_every == 0`.
- **Random warm-start empty-cluster IndexError**: `k_target = max(labels.max()+1, obs["k"][0])` + `k_override` in `NeuroCUTAlgo.select_action`.

---

## [init] — 2026-05-19

Bare repository skeleton committed by user.
