# RL Graph Bench

**A unified benchmark for reinforcement-learning–based graph clustering.**

[![CI](https://github.com/your-org/rl-graph-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/rl-graph-bench/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

`rl-graph-bench` provides a clean, extensible framework for training and evaluating RL agents on three graph-clustering task families:

| Task | Environment | Objective |
|------|-------------|-----------|
| **Graph Partition** | `NodeMoveEnv` — reassign nodes to clusters | H², NCut, balanced-cut, sparsest-cut |
| **Community Expand** | `CommunityEnv` — seed-based exclude/expand | H², NCut, F1 (vs ground truth) |
| **Dynamic CD** | `DynamicCDEnv` — adapt partition as graph evolves | Δ Modularity density |

Six RL algorithms (+ classical baselines) are implemented with a shared `RLAgent` interface so you can swap any algo into any task with one line.

---

## Algorithms

### RL Methods

| Algo | Family | Task | Architecture | Trainer |
|------|--------|------|-------------|---------|
| `neurocut` | Node-move | Partition | GraphSAGE + REINFORCE | `Trainer` |
| `wrt` | Structured merge/split | Partition | Cluster Transformer + PPO | `PPOTrainer` |
| `ss2v_d3qn` | Edge contraction | Partition | DenseSAGE + Dueling D3QN | `DQNTrainer` |
| `clare` | Expand/Exclude | Community | GIN + REINFORCE | `Trainer` |
| `slrl` | Seed-based RL | Community | Linear Swish MLP | `Trainer` |
| `ac2cd` | Temporal adaptation | Dynamic | GAT + A2C | `Trainer` |

### Classical Baselines

| Baseline | Library | Notes |
|----------|---------|-------|
| `leiden` | `leidenalg` | RBConfiguration vertex partition |
| `louvain` | `igraph` | `community_multilevel` |
| `spectral` | `sklearn` | Precomputed affinity matrix |
| `random` | numpy | Random k-partition (lower bound) |
| `metis` | `pymetis` *(optional)* | k-way partitioning |

---

## Installation

```bash
# From workspace root
pip install -e "rl-graph-bench[dev]"

# Or minimal install
pip install -e rl-graph-bench/
```

**Required system packages** (Ubuntu/Debian):

```bash
apt-get install -y libmetis-dev    # for MetisBaseline (optional)
```

---

## Quick Start

### Train

```python
from rlgb.tasks.graph_partition import GraphPartitionTask
from rlgb.algos.node_move.neurocut import NeuroCUTAlgo
from rlgb.training.trainer import Trainer, TrainConfig
from rlgb.data.synthetic import mini5
import random

task  = GraphPartitionTask(objective="h2")
algo  = NeuroCUTAlgo()
suite = mini5()

trainer = Trainer(
    algo=algo,
    env_fn=lambda: task.build_env(random.choice(suite), horizon=10),
    config=TrainConfig(n_episodes=500),
)
trainer.train()
```

### Evaluate against baselines

```python
from rlgb.baselines.clustering import LeidenBaseline, SpectralBaseline
from rlgb.eval.harness import compare_algos, summary_table
from rlgb.data.synthetic import fixed17

suite = fixed17()
algos = [algo, LeidenBaseline(), SpectralBaseline()]
df  = compare_algos(algos, suite, task, n_seeds=3, horizon=10)
tbl = summary_table(df)
print(tbl[["ncut", "h2", "nmi"]])
```

### CLI

```bash
# Train
rlgb run --algo neurocut --dataset mini5 --steps 500

# Evaluate with checkpoint
rlgb eval --algo neurocut --checkpoint results/last.pt --dataset fixed17 --seeds 3

# Compare RL algo vs baselines
rlgb compare --algos neurocut,leiden,spectral --dataset mini5

# List all algos / datasets
rlgb list-algos
rlgb list-datasets

# Launch dashboard
rlgb serve
```

### PPO trainer (WRT)

```python
from rlgb.training.ppo import PPOTrainer, PPOConfig
from rlgb.algos.structured.wrt import WRTAlgo

trainer = PPOTrainer(
    algo=WRTAlgo(),
    env_fn=lambda: task.build_env(random.choice(suite), horizon=10),
    config=PPOConfig(n_episodes=1000, clip_eps=0.2, n_epochs=4),
)
trainer.train()
```

### DQN trainer (SS2V-D3QN)

```python
from rlgb.training.dqn_trainer import DQNTrainer, DQNConfig
from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo

trainer = DQNTrainer(
    algo=SS2VAlgo(),
    env_fn=lambda: task.build_env(random.choice(suite), horizon=10),
    config=DQNConfig(n_steps=20000, warmup_steps=500),
)
trainer.train()
```

---

## Benchmark Results (v0.3.0 — All P0 Targets Pass)

### Paper-reproduction targets

| Algorithm | Dataset | Metric | Paper Target | Our Result | Status |
|-----------|---------|--------|-------------|-----------|--------|
| NeuroCUT | Cora (k=4) | NCut ↓ | ≤ 0.33 | **0.2633** | ✅ |
| NeuroCUT | CiteSeer (k=4) | NCut ↓ | ≤ 0.20 | **0.0408** | ✅ |
| WRT | City Traffic (k=4, n=100) | NCut ↓ | ≤ 0.060 | **0.0581** | ✅ |
| CLARE | SNAP Amazon | F1 ↑ | ≥ 0.773 | **0.7956** | ✅ |
| SLRL | SNAP Amazon | F-score ↑ | ≥ 0.878 | **0.9050** | ✅ |
| SLRL | SNAP DBLP | F-score ↑ | ≥ 0.662 | **0.6922** | ✅ |
| AC2CD | BlogCatalog3 | NMI ↑ | ≥ 0.75 | **0.9541** | ✅ |
| SS2V-D3QN | mini5 SBM (proxy) | NCut ↓ | ≤ 0.55 (beats Leiden 0.5815) | **0.5391** | ✅ |

Reproduce any result with a single script: `python experiments/verify_<algo>.py`

### mini5 partition suite (NCut / H²)

| Algo | NCut ↓ | H² ↓ | NMI ↑ | Notes |
|------|--------|------|-------|-------|
| `spectral` | **0.406** | 3.96 | **0.97** | Classical baseline |
| `neurocut` (curriculum-trained¹) | **0.417** | 3.97 | — | −28% vs leiden, 2.9% behind spectral |
| `leiden` | 0.582 | **3.87** | 0.92 | Classical baseline |
| `louvain` | 0.58 | 3.88 | 0.90 | Classical baseline |
| `wrt` (untrained) | 0.50 | 4.02 | 0.83 | Improves with training |
| `ss2v_d3qn` (untrained) | 0.59 | 4.06 | 0.79 | Improves with training |
| `neurocut` (untrained) | 0.64 | 4.09 | 0.78 | Random weights |
| `random` | 1.98 | 4.79 | 0.08 | Reference |

¹ Curriculum: Phase 1 (1000 ep, random warm-start, lr=3e-4) → Phase 2 (1000 ep, leiden warm-start fine-tune, lr=1e-4).
Greedy eval from leiden warm-start. Paper target (−18% vs spectral): NCut ≤ 0.333 — requires ~5k ep on larger graphs.

### Community task (NCut / H²)

| Algo | NCut ↓ | H² ↓ | NMI ↑ |
|------|--------|------|-------|
| `spectral` | **0.41** | 3.96 | **0.97** |
| `leiden` | 0.66 | **3.44** | 0.86 |
| `clare` (untrained) | 0.43 | 3.65 | 0.77 |
| `slrl` (untrained) | 0.43 | 3.65 | 0.77 |

Untrained partition numbers from `experiments/full_benchmark.py --quick` (3 seeds, 10-step horizon).  
Trained NeuroCUT numbers: curriculum script in `experiments/full_benchmark.py` with `curriculum=True`.  
Full benchmark: `python experiments/full_benchmark.py`

---

## Repository Structure

```
rl-graph-bench/
├── rlgb/
│   ├── tasks/           # Task definitions (partition, community, dynamic)
│   ├── envs/            # Gymnasium environments
│   ├── algos/           # RL algorithms (neurocut, wrt, ss2v, clare, slrl, ac2cd)
│   ├── baselines/       # Classical baselines (leiden, louvain, spectral, random)
│   ├── models/          # Neural network modules (SAGE, GIN, GAT)
│   ├── training/        # Trainers (REINFORCE, PPO, DQN)
│   ├── eval/            # Metrics + evaluation harness
│   ├── data/            # Data loaders (synthetic, PyG, SNAP)
│   └── cli.py           # Typer CLI
├── dashboard/           # Streamlit dashboard
├── tests/               # pytest test suite (84 tests)
├── blog/                # Quarto blog posts
├── awesome/             # Curated paper/code list
└── experiments/         # Benchmark scripts
```

---

## Development

```bash
# Install dev dependencies
pip install -e "rl-graph-bench[dev]"

# Lint
ruff check rlgb/

# Type check
mypy rlgb/ --ignore-missing-imports

# Tests
pytest tests/ -v

# Full benchmark run (slow — ~30 min on CPU)
python experiments/full_benchmark.py
```

---

## Citation

If you use this benchmark, please cite the individual algorithm papers:

```bibtex
@misc{rl-graph-bench-2025,
  title  = {RL Graph Bench: A Unified Benchmark for RL-based Graph Clustering},
  year   = {2025},
  url    = {https://github.com/your-org/rl-graph-bench},
}
```

See [awesome/README.md](awesome/README.md) for full paper references for each algorithm.

---

## License

MIT License — see [LICENSE](LICENSE).
