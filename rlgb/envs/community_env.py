"""Community expand/exclude environment for RL (CLARE/SLRL task).

Per-cluster EXPAND / EXCLUDE rewriter.
Each episode targets ONE cluster; outer trainer rotates cluster IDs.

Action space (Discrete)
-----------------------
  Actions 0 … |exclude|-1 : EXCLUDE member v from cluster
  Actions |exclude| … |all|-1 : EXPAND boundary node u into cluster
  Last action : STOP (explicit stop token, always legal)

Reward
------
  r_t = -(H²_after − H²_before)   [positive when H² decreases]
  Alternatively F1 against ground-truth communities (for semi-supervised).

Works with CommunityExpandTask which picks reward mode.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rlgb.envs.base import ClusteringEnv
from rlgb.tasks.base import Problem, ClusteringTask


class CommunityEnv(ClusteringEnv):
    """Community expand/exclude env.

    Parameters
    ----------
    task : ClusteringTask
    problem : Problem
    cluster_id : int
        Which cluster to rewrite each episode.
    horizon : int
    seed : int
    """

    MAX_CANDS = 200  # cap for obs array padding

    def __init__(
        self,
        task: ClusteringTask,
        problem: Problem,
        cluster_id: int = 0,
        horizon: int = 8,
        seed: int = 0,
    ) -> None:
        super().__init__(task=task, problem=problem, horizon=horizon, seed=seed)
        self.cluster_id = cluster_id
        # Action = index into [exclude_cands | expand_cands | stop]
        # We use Discrete(MAX_CANDS + 1) and mask illegals at select time
        self.action_space = spaces.Discrete(self.MAX_CANDS + 1)  # +1 for STOP
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "exclude_nodes": spaces.Box(0, problem.n, shape=(self.MAX_CANDS,), dtype=np.int32),
            "expand_nodes":  spaces.Box(0, problem.n, shape=(self.MAX_CANDS,), dtype=np.int32),
            "n_exclude": spaces.Box(0, self.MAX_CANDS, shape=(1,), dtype=np.int32),
            "n_expand":  spaces.Box(0, self.MAX_CANDS, shape=(1,), dtype=np.int32),
        })

    # ── gymnasium API ────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        # Warm start: Leiden or fall back to random
        self.labels = self._warm_start_labels()
        return self._build_obs(), {}

    def step(self, action: int):
        exc, exp = self._legal_candidates()
        n_exc = len(exc)
        n_exp = len(exp)
        total = n_exc + n_exp
        stop_action = total  # last action = STOP

        old_labels = self.labels.copy()
        if action >= total or action == stop_action:
            # STOP or out-of-range → no-op
            pass
        elif action < n_exc:
            # EXCLUDE: move member out to best neighbor cluster
            v = exc[action]
            best_c = self._best_neighbor_cluster(v, exclude=self.cluster_id)
            self.labels[v] = best_c
            self.labels = _canonicalize(self.labels)
        else:
            # EXPAND: bring boundary node into cluster
            u = exp[action - n_exc]
            self.labels[u] = self.cluster_id

        reward = self.task.reward(self.adj, old_labels, self.labels, self.problem)
        self._step_count += 1
        terminated = self._step_count >= self.horizon
        return self._build_obs(), float(reward), terminated, False, {"step": self._step_count}

    # ── candidate computation ────────────────────────────────────────────────

    def _legal_candidates(self) -> tuple[list[int], list[int]]:
        """(exclude_members, expand_boundary_nodes)."""
        n = self.problem.n
        members = np.where(self.labels == self.cluster_id)[0].tolist()
        if not members:
            return [], []
        members_set = set(members)
        # Boundary: adjacent to cluster but not in it
        adj_row = self.adj[members, :].any(axis=0)   # (N,) bool
        boundary = [u for u in range(n) if adj_row[u] and u not in members_set]
        return members, boundary[:self.MAX_CANDS // 2]

    def _best_neighbor_cluster(self, v: int, exclude: int) -> int:
        """Cluster with highest edge weight from v (excluding 'exclude')."""
        row = self.adj[v]
        k = int(self.labels.max()) + 1
        cluster_weight = np.zeros(k, dtype=np.float32)
        for u, w in enumerate(row):
            if w > 0 and self.labels[u] != exclude:
                cluster_weight[self.labels[u]] += w
        best = int(np.argmax(cluster_weight))
        return best if cluster_weight[best] > 0 else exclude

    def _warm_start_labels(self) -> np.ndarray:
        try:
            import igraph as ig
            import leidenalg as la
            g = ig.Graph.Weighted_Adjacency(self.adj.tolist(), mode="undirected")
            part = la.find_partition(g, la.ModularityVertexPartition,
                                     seed=int(self._rng.integers(2**31)))
            raw = np.array(part.membership, dtype=np.int32)
        except Exception:
            raw = self._rng.integers(0, self.problem.k_target,
                                     size=self.problem.n).astype(np.int32)
        return _adjust_to_k(raw, self.problem.k_target, self._rng)

    def _build_obs(self) -> dict:
        base = self._build_base_obs()
        exc, exp = self._legal_candidates()
        exc = exc[:self.MAX_CANDS]
        exp = exp[:self.MAX_CANDS]
        exc_arr = np.zeros(self.MAX_CANDS, dtype=np.int32)
        exp_arr = np.zeros(self.MAX_CANDS, dtype=np.int32)
        exc_arr[:len(exc)] = exc
        exp_arr[:len(exp)] = exp
        base["exclude_nodes"] = exc_arr
        base["expand_nodes"]  = exp_arr
        base["n_exclude"] = np.array([len(exc)], dtype=np.int32)
        base["n_expand"]  = np.array([len(exp)], dtype=np.int32)
        return base

    def set_cluster(self, cluster_id: int) -> None:
        """Change target cluster (call before reset for multi-cluster training)."""
        self.cluster_id = cluster_id


def _canonicalize(labels: np.ndarray) -> np.ndarray:
    mapping: dict[int, int] = {}
    out = np.empty_like(labels)
    ctr = 0
    for i, v in enumerate(labels):
        v = int(v)
        if v not in mapping:
            mapping[v] = ctr
            ctr += 1
        out[i] = mapping[v]
    return out


def _adjust_to_k(labels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    labels = labels.copy()
    while True:
        uniq, counts = np.unique(labels, return_counts=True)
        nc = len(uniq)
        if nc == k:
            break
        if nc > k:
            order = np.argsort(counts)
            c1, c2 = int(uniq[order[0]]), int(uniq[order[1]])
            labels[labels == c2] = c1
        else:
            largest = int(uniq[np.argmax(counts)])
            members = np.where(labels == largest)[0]
            if len(members) < 2:
                break
            new_c = int(labels.max()) + 1
            half = rng.choice(members, size=max(1, len(members) // 2), replace=False)
            labels[half] = new_c
    return labels
