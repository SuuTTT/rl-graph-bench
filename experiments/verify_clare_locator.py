#!/usr/bin/env python3
"""Verify native CLARE Locator achieves F1 >= 0.73 on the Amazon-1.90 dataset.

Mirrors the evaluation in KDD2022CLARE/Locator/matching.py:predict_community.
Candidate communities are matched to test communities greedily (Hungarian-style)
by maximum F1, then averaged.

Target: AvgF1 >= 0.73 (paper reports Locator-only = 0.7323).
"""
import sys
import time
import random
import numpy as np

# Make sure rlgb package is importable
sys.path.insert(0, "/workspace/rl-graph-bench")


def _f1_pair(pred: list[int], true: list[int]) -> float:
    tp = len(set(pred) & set(true))
    if tp == 0:
        return 0.0
    prec = tp / len(pred)
    rec  = tp / len(true)
    return 2 * prec * rec / (prec + rec)


def bidirectional_avg_f1(pred_comms: list[list[int]],
                         true_comms: list[list[int]]) -> float:
    """Exact replica of KDD2022CLARE/utils/metrics.py eval_scores F1.

    pred_side: for each predicted community, max F1 over all true communities.
    true_side: for each true community, max F1 over all predicted communities.
    result   = (mean_pred_side + mean_true_side) / 2
    """
    pred_f1 = np.array([
        max((_f1_pair(pc, tc) for tc in true_comms), default=0.0)
        for pc in pred_comms
    ])
    true_f1 = np.array([
        max((_f1_pair(pc, tc) for pc in pred_comms), default=0.0)
        for tc in true_comms
    ])
    return float((pred_f1.mean() + true_f1.mean()) / 2.0)


def main() -> None:
    print("=" * 60)
    print("CLARE Locator native verification")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    # ---- Data ----
    from rlgb.data.clare_dataset import load_clare_dataset
    t0 = time.time()
    print("Loading Amazon-1.90 dataset …")
    data = load_clare_dataset("amazon", num_train=90, num_val=10, seed=0)
    print(f"  {data.nx_graph.number_of_nodes()} nodes, "
          f"{data.nx_graph.number_of_edges()} edges, "
          f"{len(data.communities)} communities  ({time.time()-t0:.1f}s)")
    print(f"  train={len(data.train_ids)}  val={len(data.val_ids)}"
          f"  test={len(data.test_ids)}")

    # ---- Locator ----
    import torch
    from rlgb.algos.community.clare_locator import CommunityLocator, LocatorConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    cfg = LocatorConfig(
        hidden_dim=64, output_dim=64, n_layers=2,
        margin=0.6, lr=1e-3, epochs=30, batch_size=256,
        subg_max_size=20, num_hop=2,
        num_pred=1000, comm_max_size=20,
        log_every=5, device=device,
    )
    locator = CommunityLocator(cfg)

    print("\n--- Training Locator ---")
    t1 = time.time()
    locator.fit(data)
    print(f"Training time: {time.time()-t1:.1f}s")

    # ---- Predict ----
    print("\n--- Predicting candidate communities ---")
    t2 = time.time()
    pred_comms = locator.predict(data)
    print(f"Prediction time: {time.time()-t2:.1f}s")

    # ---- Evaluate: bidirectional AvgF1 on test communities ----
    print("\n--- Evaluating on test communities ---")
    print("  (bidirectional metric = same as eval_scores in KDD2022CLARE)")
    avg_f1 = bidirectional_avg_f1(pred_comms, data.test_communities)
    print(f"\n  AvgF1 (test) = {avg_f1:.4f}   (target >= 0.73)\n")

    target = 0.73
    if avg_f1 >= target:
        print(f"[PASS]  F1={avg_f1:.4f} >= {target}")
    else:
        gap = target - avg_f1
        print(f"[FAIL]  F1={avg_f1:.4f}  (gap {gap:.4f} below target {target})")
        sys.exit(1)


if __name__ == "__main__":
    main()
