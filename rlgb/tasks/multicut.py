"""Multicut (MCMP) task.

Objective: minimise total multicut cost = Σ_{(u,v) cut, w_{uv} > 0} w_{uv}
         + Σ_{(u,v) same cluster, w_{uv} < 0} |w_{uv}|

Equivalently: sum of |w_{uv}| for all edges whose assignment disagrees with
the sign of w.  Positive edges want to be in the same cluster; negative edges
want to be in different clusters.

The `Problem.adj` is the signed cost matrix (symmetric, float32).
"""
from __future__ import annotations

import numpy as np

from rlgb.tasks.base import Problem


def multicut_cost(adj: np.ndarray, labels: np.ndarray) -> float:
    """Total multicut cost for a signed graph partition.

    For each edge (u,v):
      - w > 0 and different cluster  → cost  +w  (want same, got different)
      - w < 0 and same cluster       → cost  +|w| (want different, got same)
    """
    adj = np.asarray(adj, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n = adj.shape[0]
    cost = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            w = adj[i, j]
            if w == 0.0:
                continue
            same = labels[i] == labels[j]
            if w > 0 and not same:
                cost += w
            elif w < 0 and same:
                cost += -w
    return cost


def multicut_cost_fast(adj: np.ndarray, labels: np.ndarray) -> float:
    """Vectorised multicut cost — O(n²) but numpy-fast."""
    adj = np.asarray(adj, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    same = (labels[:, None] == labels[None, :])   # (n, n) bool
    # positive edges in different clusters
    pos_cut = np.where((adj > 0) & ~same, adj, 0.0).sum() * 0.5
    # negative edges in same cluster
    neg_same = np.where((adj < 0) & same, -adj, 0.0).sum() * 0.5
    return float(pos_cut + neg_same)


class MulticutTask:
    """MCMP task — minimise signed multicut cost."""

    name = "multicut"
    primary_metric = "multicut_cost"

    def build_env(self, problem: Problem, horizon: int = 30, seed: int = 0,
                   env_class: str = "edge_contraction", **kwargs):
        if env_class == "edge_contraction":
            from rlgb.envs.edge_contraction_env import EdgeContractionEnv
            return EdgeContractionEnv(task=self, problem=problem, horizon=horizon,
                                      seed=seed, **kwargs)
        from rlgb.envs.node_move_env import NodeMoveEnv
        return NodeMoveEnv(task=self, problem=problem, horizon=horizon, seed=seed, **kwargs)

    def _cost_adj(self, adj: np.ndarray, problem: Problem) -> np.ndarray:
        """Return the signed cost matrix for a problem."""
        return problem.meta.get("cost_matrix", adj)

    def reward(
        self,
        adj: np.ndarray,
        labels_before: np.ndarray,
        labels_after: np.ndarray,
        problem: Problem,
    ) -> float:
        """Reward = decrease in multicut cost (positive = improvement)."""
        cost_adj = self._cost_adj(adj, problem)
        before = multicut_cost_fast(cost_adj, labels_before)
        after  = multicut_cost_fast(cost_adj, labels_after)
        return float(before - after)

    def evaluate(
        self, adj: np.ndarray, labels: np.ndarray, problem: Problem
    ) -> dict[str, float]:
        cost_adj = self._cost_adj(adj, problem)
        cost = multicut_cost_fast(cost_adj, labels)
        return {
            "multicut_cost": cost,
            "primary_metric": "multicut_cost",
            "primary_value": cost,
        }

    def build_suite(self, split: str = "test") -> list[Problem]:
        from rlgb.data.mcmp_instances import mcmp_test_suite
        flat: list[Problem] = []
        for probs in mcmp_test_suite().values():
            flat.extend(probs)
        return flat
