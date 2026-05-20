"""Dynamic community detection environment (AC2CD task).

Models a dynamic graph as a sequence of temporal snapshots.
Each episode steps through snapshots, maintaining a running partition.
At each snapshot the agent can reassign nodes (NodeMove-style).

Observation: same Dict as NodeMoveEnv but adj is the CURRENT snapshot.
Action: Discrete(N*K) — node-to-cluster reassignment (same as NodeMoveEnv).

Reward: -Δ modularity_density (Li 2008) between consecutive snapshots.

Paper: AC2CD (2023) — GAT + A2C on temporal graphs.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rlgb.envs.base import ClusteringEnv
from rlgb.tasks.base import Problem, ClusteringTask
from rlgb.eval.metrics import modularity_density


class DynamicCDEnv(ClusteringEnv):
    """Temporal snapshot graph env for dynamic community detection.

    Parameters
    ----------
    task : ClusteringTask
    problem : Problem  — must have adj_snapshots list
    horizon : int      — max node-move steps per snapshot
    seed : int
    """

    def __init__(
        self,
        task: ClusteringTask,
        problem: Problem,
        horizon: int = 10,
        seed: int = 0,
    ) -> None:
        super().__init__(task=task, problem=problem, horizon=horizon, seed=seed)
        self._snapshots = (
            problem.adj_snapshots
            if problem.adj_snapshots
            else [problem.adj]
        )
        self._snap_idx = 0
        n = problem.n
        k = problem.k_target
        self.action_space = spaces.Discrete(n * k)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._snap_idx = 0
        self._step_count = 0
        self.adj = self._snapshots[0].copy()
        self.labels = self._rng.integers(0, self.problem.k_target,
                                         size=self.problem.n).astype(np.int32)
        return self._build_base_obs(), {}

    def step(self, action: int):
        n = self.problem.n
        k = self.problem.k_target
        node_idx  = int(action) // k
        clust_idx = int(action) % k

        old_labels = self.labels.copy()
        self.labels[node_idx] = clust_idx

        reward = self.task.reward(self.adj, old_labels, self.labels, self.problem)
        self._step_count += 1

        # Advance snapshot every horizon/n_snapshots steps
        if self._step_count % max(1, self.horizon // len(self._snapshots)) == 0:
            self._snap_idx = min(self._snap_idx + 1, len(self._snapshots) - 1)
            self.adj = self._snapshots[self._snap_idx].copy()

        terminated = self._step_count >= self.horizon
        return self._build_base_obs(), float(reward), terminated, False, {
            "step": self._step_count, "snap_idx": self._snap_idx
        }
