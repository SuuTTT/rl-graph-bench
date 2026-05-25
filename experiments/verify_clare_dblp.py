"""CLARE P1 — DBLP F1 eval.

Target: F1 >= 0.384 (CLARE paper, Table 2, DBLP).
Same Locator + Rewriter pipeline as verify_clare_full.py, but on DBLP data
from the bundled KDD2022CLARE dataset.
"""
from __future__ import annotations

import sys
import time
import random
import numpy as np

sys.path.insert(0, "/workspace/rl-graph-bench")

random.seed(42)
np.random.seed(42)

import torch
torch.manual_seed(42)


def _f1_pair(pred: list[int], true: list[int]) -> float:
    tp = len(set(pred) & set(true))
    if tp == 0:
        return 0.0
    prec = tp / len(pred)
    rec  = tp / len(true)
    return 2 * prec * rec / (prec + rec)


def bidirectional_avg_f1(pred_comms: list[list[int]],
                         true_comms: list[list[int]]) -> float:
    pred_f1 = np.array([
        max((_f1_pair(pc, tc) for tc in true_comms), default=0.0)
        for pc in pred_comms
    ])
    true_f1 = np.array([
        max((_f1_pair(pc, tc) for pc in pred_comms), default=0.0)
        for tc in true_comms
    ])
    return float((pred_f1.mean() + true_f1.mean()) / 2.0)


TARGET = 0.384

print("=" * 60)
print("CLARE P1 — DBLP verification")
print("=" * 60)

from rlgb.data.clare_dataset import load_clare_dataset
t0 = time.time()
print("Loading DBLP-1.90 dataset …")
data = load_clare_dataset("dblp", num_train=90, num_val=10, seed=0)
print(f"  {data.nx_graph.number_of_nodes()} nodes, "
      f"{data.nx_graph.number_of_edges()} edges, "
      f"{len(data.communities)} communities  ({time.time()-t0:.1f}s)")
print(f"  train={len(data.train_ids)}  val={len(data.val_ids)}"
      f"  test={len(data.test_ids)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# ── Phase 1: Locator ──────────────────────────────────────────────────────────
from rlgb.algos.community.clare_locator import CommunityLocator, LocatorConfig

lcfg = LocatorConfig(
    hidden_dim=64, output_dim=64, n_layers=2,
    margin=0.6, lr=1e-3, epochs=30, batch_size=256,
    subg_max_size=20, num_hop=2,
    num_pred=1000, comm_max_size=20,
    log_every=10, device=device,
)
locator = CommunityLocator(lcfg)

print("\n--- Phase 1: Training Locator ---")
t1 = time.time()
locator.fit(data)
print(f"Locator training: {time.time()-t1:.1f}s")

print("--- Predicting candidate communities ---")
t2 = time.time()
pred_comms = locator.predict(data)
locator_f1 = bidirectional_avg_f1(pred_comms, data.test_communities)
print(f"Locator F1={locator_f1:.4f}  ({time.time()-t2:.1f}s)")

feat_mat = locator.get_node_embeddings(data)
print(f"  feat_mat shape: {feat_mat.shape}")

# ── Phase 2: Rewriter ─────────────────────────────────────────────────────────
from rlgb.algos.community.clare_rewriter import CommunityRewriter, RewriterConfig

rcfg = RewriterConfig(
    agent_lr=1e-3,
    gamma=0.99,
    n_episode=10,
    n_epoch=1000,
    max_step=10,
    max_rewrite_step=4,
    cost_choice="f1",
    n_layers=2,
    comm_max_size=12,
    log_every=200,
)
rewriter = CommunityRewriter(rcfg)

print("\n--- Phase 2: Training Rewriter ---")
t4 = time.time()
rewriter.fit(data, feat_mat, pred_comms)
print(f"Rewriter training: {time.time()-t4:.1f}s")

print("--- Rewriting communities ---")
t5 = time.time()
refined_comms = rewriter.predict(pred_comms, data, feat_mat)
print(f"Rewriter inference: {time.time()-t5:.1f}s")

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n--- Final evaluation on test communities ---")
final_f1 = bidirectional_avg_f1(refined_comms, data.test_communities)

print(f"\n  Locator F1        = {locator_f1:.4f}")
print(f"  Full pipeline F1  = {final_f1:.4f}   (target >= {TARGET})")

status = "PASS" if final_f1 >= TARGET else "FAIL"
sign   = ">=" if final_f1 >= TARGET else "<"
print(f"\n[{status}] CLARE DBLP F1={final_f1:.4f} {sign} target={TARGET}")
sys.exit(0 if final_f1 >= TARGET else 1)
