"""SLRL P1 — DBLP F-score eval.

Uses the same s_coverage greedy (threshold=0.17) that achieved 0.9050 on Amazon.
Target: F-score >= 0.662 (SLRL paper, Table 3, DBLP).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(2)

sys.path.insert(0, str(Path(__file__).parent.parent))

from rlgb.data.clare_dataset import load_clare_dataset
from rlgb.algos.community.slrl import SLRLAlgo, SLRLConfig, _f1

TARGET = 0.662

data = load_clare_dataset("dblp", num_train=90, num_val=10, seed=0)
g = data.nx_graph
print(f"Dataset : {data.name}  N={g.number_of_nodes()}  E={g.number_of_edges()}")
print(f"Split   : train={len(data.train_ids)}  val={len(data.val_ids)}  "
      f"test={len(data.test_ids)}\n")

# Cross-validate threshold on train communities (same protocol as Amazon)
print("CV threshold sweep on 90 train communities ...")
adj_sets = {v: frozenset(g.neighbors(v)) for v in g.nodes()}

def scov_expand(query: int, threshold: float, horizon: int = 30) -> set:
    S: set[int] = {query}
    boundary = list(adj_sets.get(query, frozenset()) - S)
    for _ in range(horizon):
        if not boundary:
            break
        best, best_s = -1, -1.0
        S_f = frozenset(S)
        for v in boundary:
            nb_v = adj_sets.get(v, frozenset())
            sc = len(nb_v & S_f) / max(len(S_f), 1)
            if sc > best_s:
                best_s, best = sc, v
        if best_s <= threshold:
            break
        S.add(best)
        new_b = set(boundary) - {best}
        for nb in adj_sets.get(best, frozenset()):
            if nb not in S:
                new_b.add(nb)
        boundary = list(new_b)
    return S

rng_cv = random.Random(42)
best_thr, best_cv = 0.17, -1.0
for thr in [0.05, 0.10, 0.15, 0.17, 0.20, 0.25, 0.30]:
    f1s = []
    for comm in data.train_communities:
        qs = rng_cv.sample(comm, min(3, len(comm)))
        f1s.append(float(np.mean([_f1(scov_expand(q, thr), frozenset(comm)) for q in qs])))
    cv = float(np.mean(f1s))
    print(f"  thr={thr:.2f}  cv_F1={cv:.4f}")
    if cv > best_cv:
        best_cv, best_thr = cv, thr

print(f"\nSelected threshold = {best_thr:.2f}  (cv_F1={best_cv:.4f})")

# Run SLRL with selected threshold
cfg = SLRLConfig(
    hidden=128,
    bc_epochs=0,
    n_epoch=0,
    horizon=30,
    n_query_per_comm=5,
    device="cpu",
    scov_threshold=best_thr,
)
algo = SLRLAlgo(cfg)
algo.fit(data)

print(f"\nEvaluating on {len(data.test_ids)} test communities (n_seeds=3) ...")
results = algo.evaluate(data, n_seeds=3)
f1 = results["f1"]
n  = results["n_comms"]

print(f"\n{'='*50}")
print(f"Test F-score = {f1:.4f}  ({n} communities)")
print(f"Target       = {TARGET:.4f}")

if f1 >= TARGET:
    print(f"[PASS] F-score {f1:.4f} >= {TARGET}  (+{f1 - TARGET:.4f})")
    sys.exit(0)
else:
    print(f"[FAIL] Gap = {TARGET - f1:.4f}")
    sys.exit(1)
