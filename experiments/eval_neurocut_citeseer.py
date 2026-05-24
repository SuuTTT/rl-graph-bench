"""NeuroCUT P1 — CiteSeer NCut eval.

Uses the checkpoint at results/last.pt (trained on Cora-sized graphs).
Target: NCut ≤ 0.20 (NeuroCUT paper, Table 4, CiteSeer k=4).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from rlgb.algos.node_move.neurocut import NeuroCUTAlgo
from rlgb.data.pyg_loaders import load_planetoid
from rlgb.eval.harness import eval_algo_on_suite
from rlgb.tasks.graph_partition import GraphPartitionTask
from rlgb.baselines.clustering import SpectralBaseline

TARGET = 0.20
CKPT   = Path("results/last.pt")
SEEDS  = 5
HORIZON = 30

suite = load_planetoid("CiteSeer", k_target=4)
task  = GraphPartitionTask(objective="ncut")

if CKPT.exists():
    algo = NeuroCUTAlgo.from_checkpoint(str(CKPT))
    print(f"Loaded {CKPT}")
else:
    algo = NeuroCUTAlgo()
    print("No checkpoint — running untrained model")

print(f"CiteSeer graph: {suite[0].adj.shape[0]} nodes, k={suite[0].k_target}")
print(f"Seeds={SEEDS}, horizon={HORIZON}\n")

df_nc = eval_algo_on_suite(algo, suite, task, n_seeds=SEEDS, horizon=HORIZON, greedy=True,
                           env_kwargs={"warm_start": "spectral"})
df_sp = eval_algo_on_suite(SpectralBaseline(), suite, task, n_seeds=1, horizon=1)

nc = df_nc["ncut"].mean()
sp = df_sp["ncut"].mean()

print(f"{'NeuroCUT':<20} NCut={nc:.4f}")
print(f"{'Spectral':<20} NCut={sp:.4f}")

if nc <= TARGET:
    delta = TARGET - nc
    print(f"\n[PASS] NeuroCUT NCut={nc:.4f} <= {TARGET} (+{delta:.4f})")
    sys.exit(0)
else:
    gap = nc - TARGET
    print(f"\n[FAIL] NeuroCUT NCut={nc:.4f} > {TARGET} (gap={gap:.4f})")
    sys.exit(1)
