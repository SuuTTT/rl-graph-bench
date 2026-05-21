"""StructuredPartitionEnv — merge/split cluster actions for WRT-family algos.

Action space
------------
  action < n_merge  → merge cluster pair self._pairs[action]
  action >= n_merge → split cluster (action - n_merge)

  The number of valid merge pairs changes at each step.  The action space is
  fixed at construction (K*(K+1)//2 max), but the algo is expected to only
  select valid indices.  Out-of-range actions are treated as no-ops.

Compatible algos : WRT (and any structured-action algo using the same encoding)
Compatible tasks : partition
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rlgb.envs.base import ClusteringEnv, _DEFAULT_N_FEAT
from rlgb.tasks.base import Problem, ClusteringTask


def _adjacent_cluster_pairs(
    adj: np.ndarray, labels: np.ndarray, k: int
) -> list[tuple[int, int]]:
    """Return (c_i, c_j) pairs with at least one cross-cluster edge, i < j."""
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    rows, cols = np.where(adj > 0)
    for i, j in zip(rows.tolist(), cols.tolist()):
        ci, cj = int(labels[i]), int(labels[j])
        if ci != cj:
            key = (min(ci, cj), max(ci, cj))
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return pairs


class StructuredPartitionEnv(ClusteringEnv):
    """Cluster-level merge/split environment for structured-action RL algos.

    Parameters
    ----------
    task : ClusteringTask
    problem : Problem
    horizon : int
    seed : int
    warm_start : "leiden" | "random" | "none"
    """

    def __init__(
        self,
        task: ClusteringTask,
        problem: Problem,
        horizon: int = 10,
        seed: int = 0,
        warm_start: str = "leiden",
    ) -> None:
        super().__init__(task=task, problem=problem, horizon=horizon, seed=seed)
        self._warm_start = warm_start
        k = problem.k_target
        # Max possible actions: K*(K-1)/2 merge pairs + K split = K*(K+1)/2
        max_actions = k * (k + 1) // 2
        self.action_space = spaces.Discrete(max(max_actions, 1))

        # Current merge pairs (recomputed after each step)
        self._pairs: list[tuple[int, int]] = []

    # ── gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self.labels = self._initial_labels()
        self._pairs = _adjacent_cluster_pairs(
            self.adj, self.labels, self.problem.k_target
        )
        obs = self._build_obs()
        return obs, {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        old_labels = self.labels.copy()
        n_merge = len(self._pairs)
        k_target = self.problem.k_target
        k_current = int(np.unique(self.labels).shape[0])

        if action < n_merge and k_current > k_target:
            # Merge only when we have more clusters than target (prevents k < k_target)
            c1, c2 = self._pairs[action]
            self.labels[self.labels == c2] = c1
            self.labels = _canonicalize(self.labels)
        elif action < n_merge + k_target and k_current < k_target:
            # Split only when we have fewer clusters than target (prevents k > k_target)
            c_split = action - n_merge
            members = np.where(self.labels == c_split)[0]
            if len(members) > 1:
                new_c = int(self.labels.max()) + 1
                half = members[: len(members) // 2]
                self.labels[half] = new_c
                self.labels = _canonicalize(self.labels)
        # else: no-op (k_current == k_target or action out of range)

        # Recompute pairs for next step
        self._pairs = _adjacent_cluster_pairs(
            self.adj, self.labels, self.problem.k_target
        )

        reward = self.task.reward(self.adj, old_labels, self.labels, self.problem)
        self._step_count += 1
        terminated = self._step_count >= self.horizon
        obs = self._build_obs()
        return obs, float(reward), terminated, False, {"step": self._step_count}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _initial_labels(self) -> np.ndarray:
        if self._warm_start == "leiden":
            lbl = self._leiden_warm_start()
        elif self._warm_start == "random":
            lbl = self._rng.integers(
                0, self.problem.k_target, size=self.problem.n
            ).astype(np.int32)
        else:
            lbl = np.zeros(self.problem.n, dtype=np.int32)
        lbl = _adjust_to_k(lbl, self.problem.k_target, self.adj, self._rng)
        return _canonicalize(lbl)

    def _leiden_warm_start(self) -> np.ndarray:
        try:
            import igraph as ig
            import leidenalg as la
            g = ig.Graph.Weighted_Adjacency(self.adj.tolist(), mode="undirected")
            part = la.find_partition(
                g, la.ModularityVertexPartition,
                seed=int(self._rng.integers(2**31)),
            )
            return np.array(part.membership, dtype=np.int32)
        except Exception:
            return self._rng.integers(
                0, self.problem.k_target, size=self.problem.n
            ).astype(np.int32)

    def _build_obs(self) -> dict:
        """Return obs dict compatible with WRT's select_action."""
        feats = self._node_features()
        return {
            "adj":        self.adj.astype(np.float32),
            "node_feats": feats.astype(np.float32),
            "labels":     self.labels.astype(np.int32),
            "k":          np.array([self.problem.k_target], dtype=np.int32),
        }


# ── label utilities (copied from node_move_env) ───────────────────────────────

def _canonicalize(labels: np.ndarray) -> np.ndarray:
    """Relabel clusters 0, 1, 2, … in order of first appearance."""
    mapping: dict[int, int] = {}
    out = np.empty_like(labels)
    counter = 0
    for i, c in enumerate(labels):
        c = int(c)
        if c not in mapping:
            mapping[c] = counter
            counter += 1
        out[i] = mapping[c]
    return out


def _adjust_to_k(
    labels: np.ndarray, k: int, adj: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Ensure labels uses exactly k distinct clusters (0..k-1).

    If |unique| > k: merge smallest clusters until k remain.
    If |unique| < k: split largest clusters until k reached.
    """
    labels = labels.copy()
    uniq = np.unique(labels)
    while len(uniq) > k:
        # Merge two smallest clusters
        counts = [(int((labels == c).sum()), int(c)) for c in uniq]
        counts.sort()
        c_small, c_big = counts[0][1], counts[1][1]
        labels[labels == c_small] = c_big
        uniq = np.unique(labels)
    while len(uniq) < k:
        # Split largest cluster
        counts = [(int((labels == c).sum()), int(c)) for c in uniq]
        counts.sort(reverse=True)
        c_big = counts[0][1]
        members = np.where(labels == c_big)[0]
        if len(members) < 2:
            break
        new_c = int(labels.max()) + 1
        labels[members[len(members) // 2 :]] = new_c
        uniq = np.unique(labels)
    return labels
