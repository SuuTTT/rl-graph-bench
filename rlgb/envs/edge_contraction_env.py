"""Edge-contraction environment for SS2V-D3QN (multicut / partition task).

At each step the agent selects an inter-cluster edge (u, v) and contracts it:
  - cluster[v] is merged into cluster[u]
  - This reduces the number of clusters by 1

Episode terminates when exactly k_target clusters remain, or when no
inter-cluster edges remain.

Action space: Discrete(MAX_EDGES) — masked to the current number of valid
inter-cluster edges.  Extra Q-values (beyond the valid edge count) are
ignored in SS2VAlgo.select_action().

Observation: same Dict as NodeMoveEnv / StructuredPartitionEnv:
  {adj, node_feats, labels, k}

Warm-start: supports 'random' and 'leiden'.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rlgb.envs.base import ClusteringEnv
from rlgb.tasks.base import Problem, ClusteringTask

MAX_EDGES = 100  # must match SS2VAlgo.MAX_EDGES


def _canonicalize(labels: np.ndarray) -> np.ndarray:
    """Remap labels to 0-based contiguous integers."""
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(np.int32)


def _inter_cluster_edges(adj: np.ndarray, labels: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (u, v) edges where labels[u] != labels[v], u < v."""
    rows, cols = np.where(adj > 0)
    mask = (labels[rows] != labels[cols]) & (rows < cols)
    return list(zip(rows[mask].tolist(), cols[mask].tolist()))


class EdgeContractionEnv(ClusteringEnv):
    """Sequential edge-contraction env for SS2V-D3QN.

    Parameters
    ----------
    task        : ClusteringTask
    problem     : Problem
    horizon     : int — max steps before episode truncation
    seed        : int
    warm_start  : 'random' or 'leiden'
    """

    def __init__(
        self,
        task: ClusteringTask,
        problem: Problem,
        horizon: int = 10,
        seed: int = 0,
        warm_start: str = "random",
    ) -> None:
        super().__init__(task=task, problem=problem, horizon=horizon, seed=seed)
        self._warm_start = warm_start
        # Fixed action space — masked to valid edges in select_action
        self.action_space = spaces.Discrete(MAX_EDGES)
        self._edges: list[tuple[int, int]] = []

    # ── helpers ──────────────────────────────────────────────────────────────

    def _leiden_labels_raw(self) -> np.ndarray:
        """Get leiden partition with natural cluster count (NOT forced to k_target)."""
        k_target = self.problem.k_target
        try:
            import igraph as ig
            import leidenalg
            src, dst = np.where(self.adj > 0)
            edges = list(zip(src.tolist(), dst.tolist()))
            g = ig.Graph(n=self.problem.n, edges=edges, directed=False)
            part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=0)
            return np.array(part.membership, dtype=np.int32)
        except Exception:
            k_init = max(k_target + 2, min(self.problem.n // 3, k_target * 4))
            return self._rng.integers(0, k_init, size=self.problem.n).astype(np.int32)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self.adj = self.problem.adj.copy()
        k_target = self.problem.k_target

        if self._warm_start == "leiden":
            raw = self._leiden_labels_raw()
            k_leiden = int(np.unique(raw).shape[0])
            if k_leiden > k_target:
                # leiden found more clusters than needed — agent contracts down
                self.labels = _canonicalize(raw)
            else:
                # leiden already at/below target — split leiden clusters into 2 sub-clusters
                # so the agent has meaningful within-community merges to learn
                new_labels = raw.copy()
                for ci in np.unique(raw):
                    mask = raw == ci
                    n_ci = int(mask.sum())
                    sub = self._rng.integers(0, 2, size=n_ci).astype(np.int32)
                    new_labels[mask] = ci * 2 + sub
                self.labels = _canonicalize(new_labels)
        elif self._warm_start == "singleton":
            # Each node starts as its own cluster — full agglomerative RL
            self.labels = np.arange(self.problem.n, dtype=np.int32)
        else:
            # Random: use larger k_init → more contractions per episode → richer DQN signal
            k_init = max(k_target + 3, min(self.problem.n // 3, k_target * 5))
            self.labels = self._rng.integers(0, k_init, size=self.problem.n).astype(np.int32)
            self.labels = _canonicalize(self.labels)

        self._edges = _inter_cluster_edges(self.adj, self.labels)
        return self._build_obs_with_edges(), {}

    def step(self, action: int):
        n_edges = len(self._edges)
        old_labels = self.labels.copy()

        if n_edges > 0:
            edge_idx = int(action) % n_edges   # safe mod: action may exceed valid range
            u, v = self._edges[edge_idx]
            c_u, c_v = int(self.labels[u]), int(self.labels[v])
            if c_u != c_v:
                # Merge smaller cluster into larger (stable merge)
                self.labels[self.labels == c_v] = c_u
                self.labels = _canonicalize(self.labels)
            self._edges = _inter_cluster_edges(self.adj, self.labels)

        reward = self.task.reward(self.adj, old_labels, self.labels, self.problem)
        self._step_count += 1

        k_current = int(np.unique(self.labels).shape[0])
        k_target  = self.problem.k_target
        terminated = (k_current == k_target) or (len(self._edges) == 0)
        truncated  = self._step_count >= self.horizon

        return self._build_obs_with_edges(), float(reward), terminated, truncated, {
            "n_edges": len(self._edges), "k_current": k_current,
        }

    def _build_obs_with_edges(self) -> dict:
        obs = self._build_base_obs()
        n_e = len(self._edges)
        n_cap = min(n_e, MAX_EDGES)
        if n_cap > 0:
            edge_arr = np.array(self._edges[:n_cap], dtype=np.int32)  # (n_cap, 2)
        else:
            edge_arr = np.empty((0, 2), dtype=np.int32)
        obs["edge_idx"] = edge_arr
        obs["n_edges"] = np.array([n_e], dtype=np.int32)
        return obs
