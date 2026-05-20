"""Graph Partition task — fixed-k cut/entropy minimisation.

Supported objectives
--------------------
  "h2"       – 2-level structural entropy (Li & Pan 2016)
  "ncut"     – Normalised Cut
  "balanced" – Balanced Cut (NCut / min cluster-volume balance)
  "sparsest" – Sparsest Cut

Benchmark suites included
--------------------------
  fixed17()  – canonical 17-graph eval suite
  mini5()    – 5-graph smoke-test suite
  lfr_suite(n_graphs, mu)  – LFR benchmark graphs
"""
from __future__ import annotations

import numpy as np

from rlgb.tasks.base import Problem, ClusteringTask
from rlgb.eval.metrics import h2, ncut, conductance, modularity, nmi, ari, compute_all
from rlgb.data.synthetic import fixed17, mini5, lfr


_OBJECTIVES = {"h2", "ncut", "balanced", "sparsest"}


class GraphPartitionTask:
    """Implements ClusteringTask for fixed-k graph partitioning.

    Parameters
    ----------
    objective : str
        Primary objective function used for reward and comparison.
    suite : list[Problem] | None
        Override eval suite (default: fixed17()).
    """

    name = "partition"

    def __init__(
        self,
        objective: str = "h2",
        suite: list[Problem] | None = None,
    ) -> None:
        if objective not in _OBJECTIVES:
            raise ValueError(f"objective must be one of {_OBJECTIVES}; got '{objective}'")
        self.objective = objective
        self.primary_metric = objective
        self._suite = suite

    # ── ClusteringTask protocol ───────────────────────────────────────────────

    def build_suite(self, split: str = "test") -> list[Problem]:
        if self._suite is not None:
            return self._suite
        if split == "mini":
            return mini5()
        return fixed17()

    def build_env(self, problem: Problem, horizon: int = 10, seed: int = 0, **kwargs):
        from rlgb.envs.node_move_env import NodeMoveEnv
        return NodeMoveEnv(task=self, problem=problem, horizon=horizon, **kwargs)

    def reward(
        self,
        adj: np.ndarray,
        labels_before: np.ndarray,
        labels_after: np.ndarray,
        problem: Problem,
    ) -> float:
        """Reward = −Δobjective (positive = improvement)."""
        before = self._objective_value(adj, labels_before)
        after  = self._objective_value(adj, labels_after)
        return float(before - after)  # positive when obj decreases

    def evaluate(
        self, adj: np.ndarray, labels: np.ndarray, problem: Problem
    ) -> dict[str, float]:
        out = compute_all(adj, labels, gt_labels=problem.gt_labels)
        out["primary_metric"] = self.objective
        out["primary_value"]  = out[self.objective]
        return out

    # ── helpers ──────────────────────────────────────────────────────────────

    def _objective_value(self, adj: np.ndarray, labels: np.ndarray) -> float:
        if self.objective == "h2":
            return h2(adj, labels)
        if self.objective == "ncut":
            return ncut(adj, labels)
        if self.objective == "balanced":
            return _balanced_cut(adj, labels)
        if self.objective == "sparsest":
            return _sparsest_cut(adj, labels)
        raise ValueError(self.objective)


def _balanced_cut(adj: np.ndarray, labels: np.ndarray) -> float:
    """Balanced Cut = max_cluster(cut/vol) — penalises unbalanced partitions."""
    adj = np.asarray(adj, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    deg = adj.sum(axis=1)
    worst = 0.0
    for c in np.unique(labels):
        mask = labels == c
        vol = float(deg[mask].sum())
        cut = float(adj[np.ix_(mask, ~mask)].sum())
        if vol > 0:
            worst = max(worst, cut / vol)
    return worst


def _sparsest_cut(adj: np.ndarray, labels: np.ndarray) -> float:
    """Sparsest Cut = max_cluster(cut / (|S| * |V-S|))."""
    adj = np.asarray(adj, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n = adj.shape[0]
    worst = 0.0
    for c in np.unique(labels):
        mask = labels == c
        s = int(mask.sum())
        cut = float(adj[np.ix_(mask, ~mask)].sum())
        denom = float(s * (n - s))
        if denom > 0:
            worst = max(worst, cut / denom)
    return worst
