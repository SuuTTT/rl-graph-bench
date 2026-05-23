"""End-to-end SLRL verification on SNAP Amazon (CLARE-filtered).

Usage:
    python3 experiments/verify_slrl.py

Target: mean F-score >= 0.878 on 900 test communities.
Evaluation: for each test community, use n_seeds random member queries, average F1.

The script also prints a greedy-Jaccard baseline (no RL) for reference.
"""
from __future__ import annotations

import random
import sys

import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(2)   # sweet spot for small-matrix CPU ops

from rlgb.data.clare_dataset import load_clare_dataset
from rlgb.algos.community.slrl import (
    SLRLAlgo, SLRLConfig, _f1, _node_features
)

THRESHOLD = 0.878

# ── load data ────────────────────────────────────────────────────────────────
data = load_clare_dataset("amazon", num_train=90, num_val=10, seed=0)
g = data.nx_graph
print(f"Dataset : {data.name}  N={g.number_of_nodes()}  E={g.number_of_edges()}")
print(f"Split   : train={len(data.train_ids)}  val={len(data.val_ids)}  "
      f"test={len(data.test_ids)}")

comm_sizes = [len(c) for c in data.communities]
print(f"Comm sizes: mean={np.mean(comm_sizes):.1f}  "
      f"min={min(comm_sizes)}  max={max(comm_sizes)}\n")

# ── greedy-Jaccard baseline ──────────────────────────────────────────────────
print("Baseline: greedy Jaccard expansion (no RL) ...")
adj_sets_b = {v: frozenset(g.neighbors(v)) for v in g.nodes()}

def jaccard_expand(comm: list[int], query: int, horizon: int = 25) -> list[int]:
    true_set = frozenset(comm)
    S: set[int] = {query}
    boundary = list(set(adj_sets_b.get(query, frozenset())) - S)
    for _ in range(horizon):
        if not boundary:
            break
        # pick node with highest Jaccard(nb(v), S)
        best, best_j = -1, -1.0
        S_f = frozenset(S)
        for v in boundary:
            nb_v = adj_sets_b.get(v, frozenset())
            common = len(nb_v & S_f)
            union_ = len(nb_v | S_f)
            j = common / max(1, union_)
            if j > best_j:
                best_j, best = j, v
        if best_j <= 0:
            break
        S.add(best)
        new_b = set(boundary) - {best}
        for nb in adj_sets_b.get(best, frozenset()):
            if nb not in S:
                new_b.add(nb)
        boundary = list(new_b)
    return list(S)

rng = random.Random(0)
base_scores = []
for comm in data.test_communities:
    qs = rng.sample(comm, min(3, len(comm)))
    cs = [_f1(set(jaccard_expand(comm, q)), frozenset(comm)) for q in qs]
    base_scores.append(float(np.mean(cs)))
base_f1 = float(np.mean(base_scores))
print(f"Greedy-Jaccard baseline F1 = {base_f1:.4f}\n")

# ── train SLRL ───────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = SLRLConfig(
    hidden=128,
    lr=3e-4,
    gamma=0.99,
    entropy_coef=0.01,
    value_coef=0.5,
    bc_epochs=0,         # no training: s_coverage greedy doesn't need a NN
    n_epoch=0,
    horizon=30,
    n_query_per_comm=5,
    log_every=10,
    device="cpu",
    scov_threshold=0.17, # S-coverage greedy: tuned by CV on 90 train communities
)
print(f"Training SLRL: hidden={cfg.hidden}, bc_epochs={cfg.bc_epochs}, "
      f"rl_epochs={cfg.n_epoch}, n_query/comm={cfg.n_query_per_comm}, device={cfg.device}")

algo = SLRLAlgo(cfg)
algo.fit(data)

# ── evaluate ─────────────────────────────────────────────────────────────────
print("\nEvaluating on 900 test communities (n_seeds=3) ...")
results = algo.evaluate(data, n_seeds=3)
f1 = results["f1"]
n  = results["n_comms"]

print(f"\n{'='*50}")
print(f"Test F-score = {f1:.4f}  ({n} communities)")
print(f"Target       = {THRESHOLD:.4f}")
print(f"Baseline     = {base_f1:.4f}")

if f1 >= THRESHOLD:
    print(f"[PASS] F-score {f1:.4f} >= {THRESHOLD}  (+{f1 - THRESHOLD:.4f})")
    sys.exit(0)
else:
    print(f"[FAIL] Gap = {THRESHOLD - f1:.4f}  "
          f"(vs baseline delta = {f1 - base_f1:+.4f})")
    sys.exit(1)
