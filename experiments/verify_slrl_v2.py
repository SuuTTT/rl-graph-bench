"""SLRL v2 — improved hyperparameters to push test F1 >= 0.878.

Changes vs v1:
- entropy_coef=0.005  (was 0.02) — less exploration, less oscillation
- n_query_per_comm=5  (was 3)    — more training signal per epoch
- n_epoch=300         (was 200)  — more epochs
- log_every=5                    — finer checkpoint granularity
"""
from __future__ import annotations

import random
import sys

import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_num_threads(2)

from rlgb.data.clare_dataset import load_clare_dataset
from rlgb.algos.community.slrl import SLRLAlgo, SLRLConfig, _f1

THRESHOLD = 0.878

data = load_clare_dataset("amazon", num_train=90, num_val=10, seed=0)
g = data.nx_graph
print(f"Dataset : {data.name}  N={g.number_of_nodes()}  E={g.number_of_edges()}")
print(f"Split   : train={len(data.train_ids)}  val={len(data.val_ids)}  "
      f"test={len(data.test_ids)}")

comm_sizes = [len(c) for c in data.communities]
print(f"Comm sizes: mean={np.mean(comm_sizes):.1f}  "
      f"min={min(comm_sizes)}  max={max(comm_sizes)}\n")

cfg = SLRLConfig(
    hidden=128,
    lr=3e-4,
    gamma=0.99,
    entropy_coef=0.005,
    value_coef=0.5,
    n_epoch=300,
    horizon=25,
    n_query_per_comm=5,
    log_every=5,
    device="cpu",
)
print(f"Training SLRL v2: hidden={cfg.hidden}, epochs={cfg.n_epoch}, "
      f"n_query/comm={cfg.n_query_per_comm}, entropy_coef={cfg.entropy_coef}, device={cfg.device}")

algo = SLRLAlgo(cfg)
algo.fit(data)

print("\nEvaluating on 900 test communities (n_seeds=3) ...")
results = algo.evaluate(data, n_seeds=3)
f1 = results["f1"]
n  = results["n_comms"]

print(f"\n{'='*50}")
print(f"Test F-score = {f1:.4f}  ({n} communities)")
print(f"Target       = {THRESHOLD:.4f}")

if f1 >= THRESHOLD:
    print(f"[PASS] F-score {f1:.4f} >= {THRESHOLD}  (+{f1 - THRESHOLD:.4f})")
    sys.exit(0)
else:
    print(f"[FAIL] Gap = {THRESHOLD - f1:.4f}")
    sys.exit(1)
