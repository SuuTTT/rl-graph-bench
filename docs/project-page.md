# RL Graph Bench — Project Page

> **A unified benchmark for reinforcement-learning graph-clustering algorithms.**  
> Six RL algorithms, three task families, one API — reproduced from scratch with
> 82 tests and cross-platform CI.

[![CI](https://github.com/your-org/rl-graph-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/rl-graph-bench/actions)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)]()
[![Platform](https://img.shields.io/badge/platform-linux%20|%20macOS%20|%20windows-lightgrey)]()

---

## Overview

**RL Graph Bench** provides re-implementations of six graph-clustering algorithms
that learn via reinforcement learning, together with a benchmark harness that
compares them against classical baselines (Spectral, Leiden, Louvain) on
consistent metrics.

| | |
|---|---|
| **Tasks** | Graph partition · Community detection · Dynamic community detection |
| **RL algorithms** | NeuroCUT · WRT · CLARE · SLRL · AC2CD · SS2V-D3QN |
| **Baselines** | Spectral · Leiden · Louvain · Random |
| **Metrics** | NCut ↓ · H² · NMI ↑ · ARI ↑ · Modularity-density ↑ |
| **Tests** | 82/82 · ubuntu / macOS / windows · Python 3.10–3.12 |

---

## Algorithms

| Algorithm | Paper | Task | Key Idea |
|-----------|-------|------|----------|
| **NeuroCUT** | [Shah et al., KDD 2024](https://arxiv.org/abs/2310.11787) | Partition | Node-move MDP; GraphSAGE encoder + PPO |
| **WRT** (RidgeCut) | [Jiang et al., 2025](https://arxiv.org/abs/2505.13986) | Partition | Merge/split cluster MDP; Wasserstein reward |
| **CLARE** | [Wu et al., KDD 2022](https://arxiv.org/abs/2210.08274) | Community | Community expansion; GIN encoder + REINFORCE |
| **SLRL** | Ni et al., AAAI 2025 | Community | Seed-based local expansion; GIN + actor-critic |
| **AC2CD** | [Costa & Ralha, KBS 2023](https://arxiv.org/abs/2111.15623) | Dynamic | GAT encoder; reward = NMI improvement across snapshots |
| **SS2V-D3QN** | Li et al., TNNLS 2025 | Partition | Edge contraction MDP; structure2vec + D3QN |

All algorithms share the `BaseAlgo` interface and are trained via `Trainer`
(REINFORCE) or `PPOTrainer` (PPO) in `src/rlgb/training/`.

---

## Benchmark Results (Quick Run)

Quick-run numbers: 30–50 training episodes on synthetic SBM graphs + Cora/CiteSeer.
See [blog post #6](blog/06-results.qmd) for analysis; run
`python experiments/full_benchmark.py --n_episodes 5000` for full-run results.

### Partition

| Algorithm | NCut ↓ | NMI ↑ | ARI ↑ |
|-----------|-------:|------:|------:|
| spectral | 0.599 | 0.752 | 0.728 |
| wrt | 0.840 | 0.657 | 0.624 |
| leiden | 1.380 | 0.711 | 0.649 |
| louvain | 1.414 | 0.696 | 0.639 |
| neurocut | 1.656 | 0.359 | 0.313 |
| ss2v_d3qn | 3.008 | 0.090 | 0.017 |
| random | 3.069 | 0.059 | −0.004 |

NeuroCUT at 2 000 ep curriculum training: **NCut ≈ 0.417** vs spectral 0.406 (gap −2.5%).

### Community Detection

| Algorithm | NCut ↓ | NMI ↑ | ARI ↑ |
|-----------|-------:|------:|------:|
| slrl | 0.448 | 0.739 | 0.743 |
| leiden | 0.655 | 0.863 | 0.822 |
| clare | 1.081 | 0.368 | 0.247 |

SLRL NMI = 0.739 vs paper target ≥ 0.75 (gap −1.5%) after only 30 ep.

### Dynamic Community Detection

| Algorithm | NCut ↓ | NMI ↑ | ARI ↑ |
|-----------|-------:|------:|------:|
| leiden | 0.416 | 1.000 | 1.000 |
| ac2cd | 1.254 | 0.476 | 0.481 |

---

## Installation

```bash
# Clone
git clone https://github.com/your-org/rl-graph-bench.git
cd rl-graph-bench

# Install (includes torch-geometric + leidenalg)
pip install -e ".[dev]"

# Verify
pytest --tb=short -q   # → 82 passed
```

**Optional GPU**: install the CUDA-enabled PyTorch wheel before the above step.

---

## Quick Start

```bash
# Run a quick benchmark (50 ep, ~5 min on CPU)
python experiments/full_benchmark.py --n_episodes 50 --out_dir results/quick_v1

# Full benchmark (5 000 ep, save/load checkpoints automatically)
python experiments/full_benchmark.py --n_episodes 5000 --out_dir results/full_v1

# Interactive dashboard (requires streamlit)
rlgb serve
# or: streamlit run dashboard/app.py -- --results_dir results/full_v1
```

Results are written to `out_dir/benchmark_v1.csv` (machine-readable) and
`benchmark_v1_summary.txt` (human-readable).

---

## Experiments

| Script | Purpose |
|--------|---------|
| [`experiments/full_benchmark.py`](experiments/full_benchmark.py) | Main benchmark runner; trains all 6 algos; saves checkpoints |
| [`experiments/eval_paper_benchmarks.py`](experiments/eval_paper_benchmarks.py) | Paper-target evaluation with Cora/CiteSeer |
| [`experiments/eval_multiseed.py`](experiments/eval_multiseed.py) | Multi-seed variance analysis |
| [`experiments/eval_neurocut_mini5.csv`](results/eval_neurocut_mini5.csv) | NeuroCUT training-curve data |
| [`experiments/generate_expert_trajectories.py`](experiments/generate_expert_trajectories.py) | Imitation learning: generate Leiden expert demos |
| [`experiments/fair_comparison.py`](experiments/fair_comparison.py) | Iso-time / iso-episode comparison across algos |

---

## Blog Posts

| # | Title | Date | Summary |
|---|-------|------|---------|
| 1 | [Landscape of RL Graph Clustering](blog/01-landscape.qmd) | 2025-06-01 | Survey/taxonomy of RL approaches 2022–2026 |
| 2 | [NeuroCUT Deep Dive](blog/02-neurocut.qmd) | 2025-06-08 | Node-move MDP, GraphSAGE + REINFORCE, curriculum training |
| 3 | [CLARE vs SLRL Head-to-Head](blog/03-community.qmd) | 2025-06-15 | GIN community expansion; why SLRL outperforms CLARE |
| 4 | [AC2CD and Dynamic Community Detection](blog/04-dynamic.qmd) | 2025-06-22 | GAT encoder, temporal snapshot rewards, Leiden warm-start |
| 5 | [Benchmark Design Philosophy](blog/05-benchmark.qmd) | 2025-06-29 | Unified API rationale, fair comparison principles |
| 6 | [Benchmark Results v1](blog/06-results.qmd) | 2026-05-21 | Actual numbers: partition, community, dynamic tasks |

---

## Dashboard

The Streamlit dashboard reads any `results/` directory and renders:

- Per-task leaderboard tables
- NCut / NMI bar charts (Plotly)
- Training-curve plots (if `per_episode` CSVs present)
- Algorithm filter + metric selector

```bash
streamlit run dashboard/app.py -- --results_dir results/quick_v1
```

---

## Project Structure

```
rl-graph-bench/
├── src/rlgb/
│   ├── algos/          # NeuroCUT, WRT, CLARE, SLRL, AC2CD, SS2V
│   ├── envs/           # NodeMoveEnv, StructuredPartitionEnv,
│   │                   #   EdgeContractionEnv, DynamicCDEnv, CommunityEnv
│   ├── eval/           # metrics.py (ncut, ncut_torch, h2, nmi, ari)
│   ├── tasks/          # GraphPartitionTask, CommunityTask, DynamicCDTask
│   ├── training/       # Trainer (REINFORCE), PPOTrainer
│   └── data/           # pyg_loaders, snap_loaders, graph_zoo
├── experiments/        # Benchmark runners and eval scripts
├── dashboard/          # Streamlit app
├── blog/               # Quarto blog (6 posts + index)
├── awesome/README.md   # Curated paper list with arXiv links
├── docs/               # Design docs, paper drafts, iteration logs
└── tests/              # 82 pytest tests
```

---

## Roadmap

- [ ] Real-world dataset loaders: City Traffic (WRT), BlogCatalog3 (AC2CD), SNAP DBLP/Amazon (SLRL/CLARE)
- [ ] Full 5 000-ep evaluation + paper gap closure
- [ ] Training-curve visualisations in dashboard
- [ ] Hyperparameter sweep infrastructure
- [ ] Paper draft with full results table

See [`CHANGELOG.md`](CHANGELOG.md) for the full version history.

---

## Citing

If you use RL Graph Bench in your work, please cite the relevant algorithm papers
listed in [`awesome/README.md`](awesome/README.md) as well as this repository.

---

## License

MIT
