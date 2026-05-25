"""NeuroCUT P2 — Cora Sparsest Cut eval.

Target: SparsestCut ≤ 1.46 (NeuroCUT paper, Table 4, Cora k=4).
Uses checkpoint results/last.pt (same checkpoint as P0 NCut=0.2633).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)

from rlgb.algos.node_move.neurocut import NeuroCUTAlgo
from rlgb.data.pyg_loaders import load_planetoid
from rlgb.eval.metrics import sparsest_cut, ncut
from rlgb.tasks.graph_partition import GraphPartitionTask
from rlgb.envs.node_move_env import NodeMoveEnv

TARGET = 1.46
CKPT   = Path("results/last.pt")
SEEDS  = 5
HORIZON = 30

suite = load_planetoid("Cora", k_target=4)
prob = suite[0]
task = GraphPartitionTask(objective="ncut")

print(f"Cora: {prob.adj.shape[0]} nodes  k={prob.k_target}")
print(f"CKPT: {CKPT}  exists={CKPT.exists()}")

# ── NeuroCUT ──────────────────────────────────────────────────────────────────
if CKPT.exists():
    algo = NeuroCUTAlgo.from_checkpoint(str(CKPT))
    print(f"Loaded checkpoint: {CKPT}")
else:
    algo = NeuroCUTAlgo()
    print("No checkpoint — running untrained model")

sc_vals, nc_vals = [], []
for seed in range(SEEDS):
    env = NodeMoveEnv(task=task, problem=prob, horizon=HORIZON, warm_start="spectral")
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        a = algo.select_action(obs, greedy=True)
        obs, _, term, trunc, _ = env.step(a)
        done = term or trunc
    labels = env.labels
    sc_vals.append(sparsest_cut(prob.adj, labels))
    nc_vals.append(ncut(prob.adj, labels))
    env.close()

sc_mean = float(np.mean(sc_vals))
nc_mean = float(np.mean(nc_vals))

# ── baselines ─────────────────────────────────────────────────────────────────
import sklearn.cluster as skc
import igraph as ig, leidenalg as la

def _spectral_labels(adj, k):
    sc = skc.SpectralClustering(n_clusters=k, affinity="precomputed", random_state=0)
    return sc.fit_predict(adj)

def _leiden_labels(adj, k):
    n = adj.shape[0]
    rows, cols = np.where(adj > 0)
    edges = [(int(u), int(v)) for u, v in zip(rows, cols) if u < v]
    g = ig.Graph(n=n, edges=edges)
    part = la.find_partition(g, la.ModularityVertexPartition, seed=0)
    raw = np.array(part.membership)
    # adjust to exactly k clusters via spectral if needed
    if raw.max() + 1 != k:
        raw = _spectral_labels(adj, k)
    return raw

sp_labels = _spectral_labels(prob.adj, prob.k_target)
ld_labels = _leiden_labels(prob.adj, prob.k_target)
sc_sp = sparsest_cut(prob.adj, sp_labels)
sc_ld = sparsest_cut(prob.adj, ld_labels)
nc_sp = ncut(prob.adj, sp_labels)
nc_ld = ncut(prob.adj, ld_labels)

print(f"\n{'Algorithm':<20} {'SparsestCut':>12}  {'NCut':>8}")
print("-" * 45)
print(f"{'NeuroCUT (RL)':<20} {sc_mean:>12.4f}  {nc_mean:>8.4f}")
print(f"{'Spectral':<20} {sc_sp:>12.4f}  {nc_sp:>8.4f}")
print(f"{'Leiden':<20} {sc_ld:>12.4f}  {nc_ld:>8.4f}")
print()

status = "PASS" if sc_mean <= TARGET else "FAIL"
sign   = "≤" if sc_mean <= TARGET else ">"
print(f"[{status}] NeuroCUT SparsestCut={sc_mean:.4f} {sign} target={TARGET}")
sys.exit(0 if sc_mean <= TARGET else 1)
