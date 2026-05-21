"""Dynamic CD task — modularity_density objective on temporal graphs."""
from __future__ import annotations

import numpy as np
from rlgb.tasks.base import Problem, ClusteringTask
from rlgb.eval.metrics import modularity_density, modularity, nmi, ari, compute_all


class DynamicCDTask:
    """Dynamic community detection task.

    Reward = -Δ modularity_density (higher is better so we negate delta).
    """

    name = "dynamic"
    primary_metric = "modularity_density"

    def __init__(self, suite: list[Problem] | None = None) -> None:
        self._suite = suite

    def build_suite(self, split: str = "test") -> list[Problem]:
        if self._suite:
            return self._suite
        return self._synthetic_dynamic_suite()

    def build_env(self, problem: "Problem", horizon: int = 10, seed: int = 0, **kwargs):
        from rlgb.envs.dynamic_env import DynamicCDEnv
        return DynamicCDEnv(task=self, problem=problem, horizon=horizon, seed=seed, **kwargs)

    def reward(self, adj, labels_before, labels_after, problem):
        before = modularity_density(adj, labels_before)
        after  = modularity_density(adj, labels_after)
        return float(after - before)   # positive when mod_density improves

    def evaluate(self, adj, labels, problem):
        out = compute_all(adj, labels, gt_labels=problem.gt_labels)
        out["primary_metric"] = self.primary_metric
        out["primary_value"]  = out["modularity_density"]
        return out

    def _synthetic_dynamic_suite(self) -> list[Problem]:
        """Simple dynamic suite: SBM with slowly shifting community structure."""
        from rlgb.data.synthetic import sbm
        from rlgb.tasks.base import Problem
        problems = []
        rng = np.random.default_rng(0)
        for i in range(3):
            adj0, labels, k = sbm(n=40, k=3, p_in=0.7, p_out=0.05, seed=i)
            # Create 3 snapshots by randomly flipping a few edges
            snaps = [adj0]
            cur = adj0.copy()
            for _ in range(2):
                mask = rng.random(cur.shape) < 0.05
                mask = np.tril(mask, -1)
                mask = mask + mask.T
                cur = np.clip(cur + mask - 2 * cur * mask, 0, 1)
                np.fill_diagonal(cur, 0)
                snaps.append(cur.astype(np.float32))
            problems.append(Problem(
                name=f"dynamic_sbm_{i}", adj=adj0, k_target=k,
                gt_labels=labels, family="sbm", task_type="dynamic",
                adj_snapshots=snaps,
            ))
        return problems
