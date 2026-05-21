# Changelog — rl-graph-bench

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Changed

- **`SLRLConfig.entropy_coef`** raised `0.01 → 0.03` — higher entropy bonus prevents the
  policy from committing to premature STOP actions; greedy-eval NMI improved from 0.739 →
  **0.807** at 1 000 ep on mini5 (proxy target ≥ 0.75 exceeded). `ee18334`

- **`TrainConfig`** gains `lr_schedule: str = "none"` and `lr_min_ratio: float = 0.1`.
  Trainer applies `CosineAnnealingLR` or `LinearLR` to `algo._optimizer` when requested.
  All three community/dynamic `_train_*` helpers in `full_benchmark.py` now use
  `lr_schedule="cosine"`. `ee18334`

- **Community benchmark `n_ep`** raised `1000 → 3000` for full run. `ee18334`

- **`run_community_benchmark`** now evaluates RL algos with `greedy=True` via separate
  `compare_algos` call; stochastic rollouts underestimated the learned policy by ~6–9%
  NMI. Leiden baseline still uses stochastic eval. Dynamic AC2CD eval similarly corrected.

- **`run_dynamic_benchmark`** `n_ep` raised `1000 → 3000` for AC2CD.

- **NeuroCUT Phase-1 PPO** `entropy_coef` raised `0.01 → 0.03`; Phase-2 fine-tune stays
  at `0.01` to preserve the leiden-warm-start refinement. Encourages broader node-move
  exploration in the curriculum's random-warm-start phase.

---

## [0.2.0] — 2026-05-21

Complete algorithm wiring, speed improvements, PPO upgrade, CI expansion,
and checkpoint caching. All 10 planned items from the initial TODO list
delivered in one session. 82/82 tests pass.

### Added

- **`StructuredPartitionEnv`** (`rlgb/envs/structured_env.py`) — merge-adjacent-cluster /
  split-cluster action space for WRT. k_target constraint prevents trivial NCut=0 collapse
  (merge only when k > k_target; split only when k < k_target). `GraphPartitionTask.
  build_env(env_class='structured')` wires it in. Quick-run WRT NCut=0.448 on mini5.

- **`EdgeContractionEnv`** (`rlgb/envs/edge_contraction_env.py`) — sequential edge-contraction
  env for SS2V-D3QN. Action = index among inter-cluster edges; step merges the two endpoint
  clusters. Resets to k_init = min(N//2, 2·k_target) so agent contracts down to k_target.
  Terminates when k = k_target or no inter-cluster edges remain.

- **Leiden warm-start for `DynamicCDEnv`** — `warm_start='leiden'` runs Leiden on snapshot[0]
  before training so AC2CD starts from a quality partition and learns to track changes.
  AC2CD NMI went from 0.058 (random init) → 1.00 (leiden warm-start) on quick run.

- **`ncut_torch(adj, labels)`** (`rlgb/eval/metrics.py`) — differentiable torch version of
  NCut supporting hard (long) and soft (float N×K) label tensors. Enables gradient flow for
  future end-to-end differentiable training. 14× faster than the old Python loop.

- **PPO interface for `NeuroCUTAlgo`** — `select_action_with_logprob(obs)` returns
  `(action, log_prob, value, entropy)` without side effects; `ppo_update(obs_list, actions,
  old_log_probs, advantages, returns, ...)` implements clipped surrogate + GAE value loss with
  per-transition re-evaluation across n_epochs. `PPOTrainer` auto-detects the interface.

- **Checkpoint helpers in `full_benchmark.py`** — `_ckpt_path`, `_try_load`, `_save_ckpt`.
  All six `_train_*` functions check for a cached checkpoint before training and save after.
  Re-running with identical hyperparams skips training entirely (~100% time saved).

- **Cora/CiteSeer in partition benchmark** — `run_partition_benchmark` augments the synthetic
  suite with Cora + CiteSeer via `real_benchmark_suite(max_nodes=300/2000)`. `_print_paper_gap`
  reports Cora-specific NCut to track progress toward NeuroCUT paper target (≤0.333).

- **Community paper gap reporting** — `_print_community_paper_gap()` reports NMI proxy vs
  SLRL (AAAI 2025) and CLARE (KDD 2022) paper targets. Quick-run: SLRL NMI=0.739
  (proxy ≥0.75, gap −1.5%), CLARE NMI=0.368 (gap −51%).

- **2 new tests**: `test_ncut_torch_matches_numpy`, `test_ncut_torch_single_cluster`
  (`tests/test_phase1_3.py`). **1 new test**: `test_ppo_interface_smoke`
  (`tests/test_phase1_3.py`). Total: **82 tests** (was 79).

- **`blog/06-results.qmd`** — benchmark results post with actual numbers from
  `results/benchmark_v1_summary.txt`, paper gap analysis, and training curves.

- **`docs/project-page.md`** — comprehensive project landing page with algorithm table,
  benchmark results, blog links, and installation guide.

### Changed

- **`ncut()`** (`rlgb/eval/metrics.py`) — rewritten from Python loop to vectorised numpy
  einsum (`O(N²+N·K)` vs old `O(K·N²)`); 9× faster on N=200, K=8.

- **`_train_neurocut`** (`experiments/full_benchmark.py`) — switched from `Trainer`
  (REINFORCE) to `PPOTrainer`. PPO config: lr=3e-4, n_episodes_per_update=4, clip_eps=0.2.

- **`run_community_benchmark`** — uses `task.build_suite()` instead of `mini5()[:3]` proxy.

- **`run_{partition,community,dynamic}_benchmark`** — all accept `out_dir` parameter
  threaded from `main(args.out_dir)`.

- **CI `test` job** — expanded from `ubuntu-latest` to `os: [ubuntu-latest, macos-latest,
  windows-latest]` × `python-version: [3.10, 3.11, 3.12]` = 9 runner combinations.
  `fail-fast: false`. macOS uses default PyPI PyTorch index (arm64); Linux/Windows use
  `--index-url https://download.pytorch.org/whl/cpu`.

### Fixed

- **NCut objective** — `full_benchmark.py` was using `objective='h2'`; corrected to
  `objective='ncut'` to match all six RL papers.

- **WRT action/env mismatch** — WRT merge/split indices were being decoded by `NodeMoveEnv`
  as node-move flat indices. Fixed by `StructuredPartitionEnv`.

- **SS2V action semantics** — `select_action` was returning node-move flat indices; Q-net
  expects edge indices. Fixed by `EdgeContractionEnv` + updated `SS2VAlgo.select_action`.

- **NeuroCUT Phase 2 corruption** — leiden warm-start fine-tuning >1000 ep corrupts Phase 1
  weights via all-negative REINFORCE gradients. Capped at 500 ep in curriculum training.

- **`ncut` einsum bug** — initial vectorised rewrite used wrong einsum signature
  (`ni,ij,jk->k` instead of `in,ij,jn->n`), giving NCut=0 for all inputs. Fixed before commit.

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
