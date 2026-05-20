"""Community Expansion task — semi-supervised, F1/Jaccard/ONMI objectives.

Papers: CLARE (KDD 2022), SLRL (2025).

Benchmark suites
----------------
  partition (default) — fixed-k H² on synthetic graphs (reuses GraphPartition logic)
  snap_amazon / snap_dblp — SNAP community datasets (requires snap_loaders)
"""
from __future__ import annotations

import numpy as np

from rlgb.tasks.base import Problem, ClusteringTask
from rlgb.eval.metrics import h2, ncut, nmi, ari, compute_all


class CommunityExpandTask:
    """Task for semi-supervised community detection via expand/exclude RL.

    Primary metric: "f1" | "h2" | "nmi".
    Reward: -delta_h2 (default), or delta_f1 when gt communities are available.
    """

    name = "community"

    def __init__(
        self,
        objective: str = "h2",
        suite: list[Problem] | None = None,
    ) -> None:
        self.objective = objective
        self.primary_metric = objective
        self._suite = suite

    def build_suite(self, split: str = "test") -> list[Problem]:
        if self._suite is not None:
            return self._suite
        from rlgb.data.synthetic import mini5, fixed17
        return mini5() if split == "mini" else fixed17()

    def build_env(self, problem: "Problem", horizon: int = 10, seed: int = 0, **kwargs):
        from rlgb.envs.community_env import CommunityEnv
        return CommunityEnv(task=self, problem=problem, horizon=horizon, **kwargs)

    def reward(
        self,
        adj: np.ndarray,
        labels_before: np.ndarray,
        labels_after: np.ndarray,
        problem: Problem,
    ) -> float:
        """Reward = −Δ objective (positive = improvement)."""
        if self.objective == "h2":
            before = h2(adj, labels_before)
            after  = h2(adj, labels_after)
            return float(before - after)
        if self.objective == "ncut":
            before = ncut(adj, labels_before)
            after  = ncut(adj, labels_after)
            return float(before - after)
        if self.objective == "f1" and problem.gt_labels is not None:
            from rlgb.eval.metrics import f1_community
            before = _mean_best_f1(labels_before, problem.gt_labels)
            after  = _mean_best_f1(labels_after, problem.gt_labels)
            return float(after - before)
        # fallback
        return 0.0

    def evaluate(
        self, adj: np.ndarray, labels: np.ndarray, problem: Problem
    ) -> dict[str, float]:
        out = compute_all(adj, labels, gt_labels=problem.gt_labels)
        out["primary_metric"] = self.objective
        out["primary_value"]  = out.get(self.objective, float("nan"))
        return out


def _mean_best_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean F1 between predicted and ground-truth communities (best-match)."""
    pred_sets = {c: set(np.where(pred == c)[0].tolist()) for c in np.unique(pred)}
    gt_sets   = {c: set(np.where(gt == c)[0].tolist())   for c in np.unique(gt)}
    total = 0.0
    for g_set in gt_sets.values():
        best = 0.0
        for p_set in pred_sets.values():
            inter = len(g_set & p_set)
            if inter == 0:
                continue
            prec = inter / len(p_set)
            rec  = inter / len(g_set)
            f1   = 2 * prec * rec / (prec + rec)
            best = max(best, f1)
        total += best
    return total / max(len(gt_sets), 1)
