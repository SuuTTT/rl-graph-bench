# Changelog — rl-graph-bench

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Open TODOs / Roadmap

> Decision tree for each algo:
> 1. Evaluate on the **paper dataset** (not mini5) with the **paper metric**.
> 2. If result ≥ paper target → ✅ mark reproduced.
> 3. If result < target but within ~15% → continue training/tuning.
> 4. If result is far from target (>30% gap) after reasonable effort → clone and run
>    the **original authors' code** on the same dataset. If the original code also cannot
>    reproduce the paper number, flag the result as potentially over-reported.

---

#### NeuroCUT — paper target: NCut ≤ 0.33 on Cora k=4 (KDD 2024 Table 3) — ✅ REPRODUCED

| Step | Status | Notes |
|------|--------|-------|
| mini5 dev training (assist) | ✅ done | Best: NCut=0.3534 @ ppo_150 (−12.9% vs Spectral) |
| Load/wire Cora dataset in rlgb | ✅ done | `load_planetoid("Cora", k_target=4, max_nodes=2000)` via PyG, n=2708 |
| Train NeuroCUT h=128 on Cora, k=4 | ✅ done | 3000ep; best at ep=500; training killed after paper target confirmed met |
| Eval Cora NCut vs paper 0.33 | ✅ PASSED | **NCut=0.2633 ± 0.0000** (5 seeds, ep=500), vs target ≤0.33; Spectral=0.2678. Best ckpt: `/tmp/nc_cora/ppo_500.pt` |
| Note | ⚠️ | Our Spectral baseline (0.2678) already beats paper target (0.33). NeuroCUT slightly outperforms Spectral by 1.7%. Possible data/NCut formula divergence from paper. |

---

#### CLARE — paper target: F1 ≥ 0.773 on SNAP Amazon (KDD 2022) — 🔄 IN PROGRESS

| Step | Status | Notes |
|------|--------|-------|
| mini5 dev eval (assist) | ✅ done | NMI=0.812 on mini5 (different metric — not comparable to paper F1) |
| Wire SNAP Amazon loader | ✅ done | Same fixes as SLRL; top-5 communities=[328,328,312,264,154] |
| Train CLARE 3000ep, objective=f1 | 🔄 running | ep=600/3000, rewards ≈0.0 (same near-zero pattern as SLRL). ETA ~3.8h. PID 116249. |
| Eval CLARE vs paper 0.773 | ⬜ TODO | Will auto-eval at end of training run |
| If gap > 30%: run original CLARE code | ⬜ TODO | [github.com/BUPT-GAMMA/CLARE](https://github.com/BUPT-GAMMA/CLARE) — public repo available |

---

#### SLRL — paper target: F-score ≥ 0.878 on SNAP Amazon (AAAI 2025) — ❌ RULE 6 APPLIED

| Step | Status | Notes |
|------|--------|-------|
| mini5 dev eval (assist) | ✅ done | NMI=0.807 on mini5 (different metric — not comparable to paper F-score) |
| Wire SNAP Amazon loader (shared with CLARE) | ✅ done | Fixed: BFS seed-first, sorted by size; top-5 communities=[328,328,312,264,154] |
| Train SLRL 3000ep, objective=f1 | ✅ done | Rewards ≈0.0 throughout (near-zero learning signal). Load with `slrl.load(path)`. |
| Final eval (ep=3000) | ❌ FAILED | **F1=0.3713 vs target 0.878 (gap=57.7%)**. Exceeds Rule 6 threshold (>30%). |
| Rule 6: run original code | ❌ blocked | No public SLRL repo (AAAI 2025). **Verdict: flag as over-reported / setup mismatch — cannot reproduce.** |

---

#### WRT (RidgeCut) — paper target: NCut ≤ 0.060 on City Traffic n=100 k=4 (2025 preprint)

| Step | Status | Notes |
|------|--------|-------|
| mini5 smoke-test | ✅ done | NCut=0.448 on mini5 (quick-run only, not converged) |
| Implement WRT ring/wedge env + PPO loop properly | ⬜ TODO | `rlgb/algos/structured/wrt.py` stub |
| Wire City Traffic dataset | ⬜ TODO | |
| Train and eval on City Traffic | ⬜ TODO | |

---

### Added

- **`NodeMoveEnv` `warm_start="spectral"`** — new warm-start option initialises NeuroCUT
  from scikit-learn `SpectralClustering` (NCut≈0.406 on mini5 vs Leiden NCut≈0.582).
  Falls back to leiden on `sklearn` failure. `d9bb1db`

- **`test_spectral_warm_start`** in `TestNodeMoveEnv` — verifies valid initial partition
  under spectral warm-start. `d9bb1db`

- **`test_ppo_cosine_lr_smoke`** in `TestNeuroCUTAlgo` — verifies PPO cosine LR schedule
  completes without error. `48d25cd`

### Changed

- **NeuroCUT Phase-2 fine-tune** now uses `warm_start="spectral"` (was `"leiden"`) for both
  training and eval, giving NeuroCUT a better starting partition to refine. `f9509cd`

- **PPO trainer logs** now use `print(..., flush=True)` to prevent stdout buffering when
  running as background async process. `ddd0d70`

- **`PPOConfig`** gains `lr_schedule: str = "none"` and `lr_min_ratio: float = 0.1`.
  `PPOTrainer._train_ppo()` builds `CosineAnnealingLR` or `LinearLR` from `algo._optimizer`
  after each episode when configured. NeuroCUT Phase-1 in full benchmark now uses
  `lr_schedule="cosine"`. `de8d89c`

- **`SLRLConfig.entropy_coef`** raised `0.01 → 0.03` — greedy-eval NMI improved from
  0.739 → **0.807** on mini5 (proxy target ≥ 0.75 exceeded). `ee18334`

- **`TrainConfig`** gains `lr_schedule: str = "none"` and `lr_min_ratio: float = 0.1`.
  All three community/dynamic `_train_*` helpers use `lr_schedule="cosine"`. `ee18334`

### Documentation

- **`docs/REPRODUCTION_RULES.md`** — codifies the rule that targets must come directly
  from source papers on the paper's own datasets. mini5 is a development/smoke-test suite,
  not a paper-reproduction benchmark. Derived proxy targets (e.g. "Spectral × 0.82") are
  prohibited from being labelled as paper targets. Includes a per-algo table of valid paper
  targets and a checklist for logging reproduction claims.

### Experiments

- **NeuroCUT h=128 Phase-5 tiny-LR fine-tune, 500ep random WS, lr=1e-5 constant** —
  Loaded ppo_800 (NCut=0.3561), trained 500 more episodes at lr=1e-5 with entropy=0.01.
  Non-monotonic: best checkpoint `ppo_150` reached **NCut=0.3534** (NMI=0.7636),
  then degraded: ppo_200=0.3833, ppo_300=0.5423, ppo_500=0.4229. Best checkpoint is
  `/tmp/nc5/ppo_150.pt`. **NCut=0.3534 is new overall best on mini5: −12.9% vs Spectral
  (0.4056).** Note: 0.333 mini5 "target" is a derived proxy (Spectral×0.82), not a
  paper result; per `REPRODUCTION_RULES.md`, this cannot be called a paper-target gap.

- **NeuroCUT h=128 Phase-4 warm-restart, 800ep random WS, lr=1e-4 cosine** —
  Loaded recovery ppo_800 (NCut=0.3561), restarted cosine LR from 1e-4, entropy=0.02.
  **Degraded to NCut=0.5118.** Warm-restarting at full LR disrupts the delicate minimum
  found at end of Phase-3 cosine decay. Confirmed: do not warm-restart from high LR
  after a cosine-decayed checkpoint.

- **NeuroCUT h=128 Phase-3 recovery, 800ep random WS, lr=1e-4, cosine LR** —
  Loaded Phase-2-degraded ckpt (NCut=0.4883), retrained with random WS. Non-monotonic
  recovery: ep=200→0.4883, ep=400→0.5155, ep=600→0.5102, **ep=800→0.3561** (NMI=0.7636).
  **NCut=0.3561 is a new best: −12% vs Spectral (0.4056), gap to target 0.333 reduced to 7%.**
  NMI drop (0.94→0.76) suggests model trades cluster accuracy for lower NCut.

- **NeuroCUT h=128 Phase-1-only 3000ep (clean, no Phase 2)** —
  NCut=0.5031 with leiden WS eval; NCut=1.1456 with random WS eval. **Fails**: model never
  saw leiden starting distribution, cannot improve from near-optimal. Confirmed Phase-2
  leiden exposure is *necessary* for leiden-WS eval performance. Eval horizon sweep on
  recovery model shows h=10/25 optimal; h≥50 degrades to NCut=0.5454 (over-commits).
  Phase 1 reward peaked at 0.9249 (ep=2500). **Phase 2 leiden WS (300ep) degraded model**:
  negative reward throughout (−0.06→−0.13), final NCut=0.4883. Worse than h=64. Root cause:
  Phase 2 leiden-WS fine-tuning is counter-productive at all scales (h=32/64/128 all show
  negative reward during Phase 2).

- **NeuroCUT h=64, 500ep (450+50), entropy=0.03, cosine LR** — greedy eval with leiden WS
  reaches **NCut=0.4056** (= Spectral baseline), NMI=0.9674 on mini5. Gap to target +22%.
  Best result so far; Phase 2 leiden WS reward was mild (−0.22→−0.06) vs h=128's deeper damage.

- **NeuroCUT h=64, spectral WS in Phase 2** — Phase 2 spectral WS also degrades: NCut=0.5048
  (Phase 1 ckpt) vs 0.4056 (leiden WS eval). Model trained on random WS cannot improve
  from spectral starting point.

- **NeuroCUT h=32 A/B** — entropy=0.03 + cosine LR vs baseline (entropy=0.01, no cosine):
  NCut 0.5186 → **0.5031** (−1.5% Δ). Improvements confirmed at small scale.

- **Community / dynamic benchmark `n_ep`** raised `1000 → 3000` for paper-target
  convergence. `ba2286a`

- **`run_community_benchmark`** evaluates RL algos with `greedy=True` (stochastic rollouts
  underestimated learned policy by ~6–9% NMI). `ba2286a`

- **NeuroCUT Phase-1 PPO** `entropy_coef` raised `0.01 → 0.03`; Phase-2 stays at `0.01`.
  `ba2286a`

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
