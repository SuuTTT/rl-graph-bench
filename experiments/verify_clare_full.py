#!/usr/bin/env python3
"""Verify native CLARE full pipeline (Locator + Rewriter) achieves F1 >= 0.773.

Pipeline:
  1. Load Amazon-1.90 dataset (6926 nodes, 1000 communities, 90/10/900 split)
  2. Train Locator (30 epochs) → candidate communities
  3. Get per-node GCN embeddings from trained Locator → feat_mat for Rewriter
  4. Train Rewriter (1000 epochs, REINFORCE on training communities)
  5. Rewrite candidate communities → refined communities
  6. Evaluate bidirectional AvgF1 on 900 test communities

Target: AvgF1 >= 0.773 (paper reports 0.7895).
"""
import sys
import time
import random
import numpy as np

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
    print("CLARE full-pipeline verification (Locator + Rewriter)")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    import torch
    torch.manual_seed(42)

    # ---- Data ----
    from rlgb.data.clare_dataset import load_clare_dataset
    t0 = time.time()
    print("Loading Amazon-1.90 …")
    data = load_clare_dataset("amazon", num_train=90, num_val=10, seed=0)
    print(f"  {data.nx_graph.number_of_nodes()} nodes, "
          f"{data.nx_graph.number_of_edges()} edges, "
          f"{len(data.communities)} communities  ({time.time()-t0:.1f}s)")
    print(f"  train={len(data.train_ids)}  val={len(data.val_ids)}"
          f"  test={len(data.test_ids)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # ---- Phase 1: Locator ----
    from rlgb.algos.community.clare_locator import CommunityLocator, LocatorConfig

    lcfg = LocatorConfig(
        hidden_dim=64, output_dim=64, n_layers=2,
        margin=0.6, lr=1e-3, epochs=30, batch_size=256,
        subg_max_size=20, num_hop=2,
        num_pred=1000, comm_max_size=20,
        log_every=5, device=device,
    )
    locator = CommunityLocator(lcfg)

    print("\n--- Phase 1: Training Locator ---")
    t1 = time.time()
    locator.fit(data)
    print(f"Locator training: {time.time()-t1:.1f}s")

    print("--- Phase 1: Predicting candidate communities ---")
    t2 = time.time()
    pred_comms = locator.predict(data)
    locator_f1 = bidirectional_avg_f1(pred_comms, data.test_communities)
    print(f"Locator F1={locator_f1:.4f}  ({time.time()-t2:.1f}s)")

    # ---- Get per-node GCN embeddings for Rewriter ----
    print("\n--- Extracting per-node embeddings for Rewriter ---")
    t3 = time.time()
    feat_mat = locator.get_node_embeddings(data)   # (N, 64)
    print(f"  feat_mat shape: {feat_mat.shape}  ({time.time()-t3:.1f}s)")

    # ---- Phase 2: Rewriter ----
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

    print("--- Phase 2: Rewriting communities ---")
    t5 = time.time()
    refined_comms = rewriter.predict(pred_comms, data, feat_mat)
    print(f"Rewriter inference: {time.time()-t5:.1f}s")

    # ---- Evaluate ----
    print("\n--- Final evaluation on test communities ---")
    final_f1 = bidirectional_avg_f1(refined_comms, data.test_communities)

    print(f"\n  Locator F1        = {locator_f1:.4f}")
    print(f"  Full pipeline F1  = {final_f1:.4f}   (target >= 0.773)\n")

    target = 0.773
    if final_f1 >= target:
        print(f"[PASS]  F1={final_f1:.4f} >= {target}")
    else:
        gap = target - final_f1
        print(f"[FAIL]  F1={final_f1:.4f}  (gap {gap:.4f} below {target})")
        sys.exit(1)


if __name__ == "__main__":
    main()
