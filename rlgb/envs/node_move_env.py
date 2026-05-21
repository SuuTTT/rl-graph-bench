"""NodeMove environment — node-to-cluster reassignment RL.

Maps to:
  Tasks : GraphPartition
  Algos : A (NeuroCUT), B (WRT)

Action space
------------
  Discrete(N * K) encoded as action = node_idx * K + cluster_idx.
  Legal actions exclude self-reassignments (node already in that cluster).

Observation space (extends ClusteringEnv base)
----------------------------------------------
  Same Dict as base: adj, node_feats, labels, k.
  Extras:
    "candidates" : (M, 2) int32  – legal (node, cluster) pairs for this step

Reward
------
  r_t = -(H²_after - H²_before)   [or −ΔNCUT, configured by task]
  i.e. positive when moving a node reduces the objective.

Warm-start
----------
  Reset uses Leiden CPM → adjust-to-k as initial partition
  (same as rl-cluster-ops NeuroCutEnv).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rlgb.envs.base import ClusteringEnv, _DEFAULT_N_FEAT
from rlgb.tasks.base import Problem, ClusteringTask


class NodeMoveEnv(ClusteringEnv):
    """Node-to-cluster reassignment environment.

    Parameters
    ----------
    task : ClusteringTask
        Provides the reward function and problem.
    problem : Problem
        The graph instance for this episode.
    horizon : int
        Max move actions per episode.
    seed : int
    warm_start : "leiden" | "spectral" | "random" | "none"
        Initial partition strategy.
    leiden_labels : np.ndarray | None
        Pre-computed Leiden partition (avoids re-running at reset time).
    """

    def __init__(
        self,
        task: ClusteringTask,
        problem: Problem,
        horizon: int = 10,
        seed: int = 0,
        warm_start: str = "leiden",
        leiden_labels: np.ndarray | None = None,
    ) -> None:
        super().__init__(task=task, problem=problem, horizon=horizon, seed=seed)

        self._warm_start = warm_start
        self._leiden_labels = leiden_labels

        n = problem.n
        k = problem.k_target

        # Action = (node_idx, target_cluster) → flattened to Discrete(N*K)
        self.action_space = spaces.Discrete(n * k)

        # Extend obs with candidate array
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "candidates": spaces.Box(0, max(n, k), shape=(n * k, 2), dtype=np.int32),
        })

    # ── gymnasium API ────────────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0

        self.labels = self._initial_labels()
        obs = self._build_obs()
        return obs, {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        n = self.problem.n
        k = self.problem.k_target
        node_idx  = int(action) // k
        clust_idx = int(action) % k

        old_labels = self.labels.copy()
        self.labels[node_idx] = clust_idx
        self._canonicalize_labels()

        reward = self.task.reward(self.adj, old_labels, self.labels, self.problem)
        self._step_count += 1
        terminated = self._step_count >= self.horizon
        obs = self._build_obs()
        return obs, float(reward), terminated, False, {"step": self._step_count}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _initial_labels(self) -> np.ndarray:
        if self._leiden_labels is not None:
            lbl = self._leiden_labels.copy()
        elif self._warm_start == "leiden":
            lbl = self._leiden_warm_start()
        elif self._warm_start == "spectral":
            lbl = self._spectral_warm_start()
        elif self._warm_start == "random":
            lbl = self._rng.integers(0, self.problem.k_target,
                                     size=self.problem.n).astype(np.int32)
        else:
            lbl = np.zeros(self.problem.n, dtype=np.int32)

        lbl = _adjust_to_k(lbl, self.problem.k_target, self.adj, self._rng)
        return _canonicalize(lbl)

    def _leiden_warm_start(self) -> np.ndarray:
        try:
            import igraph as ig
            import leidenalg as la
            g = ig.Graph.Weighted_Adjacency(
                self.adj.tolist(), mode="undirected"
            )
            part = la.find_partition(
                g, la.ModularityVertexPartition,
                seed=int(self._rng.integers(2**31)),
            )
            return np.array(part.membership, dtype=np.int32)
        except Exception:
            return self._rng.integers(0, self.problem.k_target,
                                      size=self.problem.n).astype(np.int32)

    def _spectral_warm_start(self) -> np.ndarray:
        try:
            from sklearn.cluster import SpectralClustering
            sc = SpectralClustering(
                n_clusters=self.problem.k_target,
                affinity="precomputed",
                random_state=int(self._rng.integers(2**31)),
                n_init=5,
            )
            lbl = sc.fit_predict(self.adj).astype(np.int32)
            return lbl
        except Exception:
            return self._leiden_warm_start()

    def _canonicalize_labels(self) -> None:
        self.labels = _canonicalize(self.labels)

    def _legal_candidates(self) -> np.ndarray:
        """Return (M, 2) array of legal (node, cluster) pairs."""
        n = self.problem.n
        k = self.problem.k_target
        cands = []
        for node in range(n):
            for c in range(k):
                if self.labels[node] != c:
                    cands.append([node, c])
        if not cands:
            return np.zeros((0, 2), dtype=np.int32)
        return np.array(cands, dtype=np.int32)

    def _build_obs(self) -> dict:
        base = self._build_base_obs()
        cands = self._legal_candidates()
        # Pad to fixed shape (N*K, 2) for gymnasium Dict obs
        max_cands = self.problem.n * self.problem.k_target
        padded = np.zeros((max_cands, 2), dtype=np.int32)
        if len(cands):
            padded[:len(cands)] = cands
        base["candidates"] = padded
        base["n_candidates"] = np.array([len(cands)], dtype=np.int32)
        return base

    def legal_action_indices(self) -> list[int]:
        """Flat action indices for legal moves (used by NeuroCUT for masking)."""
        k = self.problem.k_target
        cands = self._legal_candidates()
        return [int(c[0]) * k + int(c[1]) for c in cands]


# ── utility functions ─────────────────────────────────────────────────────────

def _canonicalize(labels: np.ndarray) -> np.ndarray:
    """Relabel so cluster IDs are 0, 1, 2, … in first-appearance order."""
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


def _adjust_to_k(
    labels: np.ndarray,
    k_target: int,
    adj: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Merge smallest clusters or split largest until exactly k_target clusters."""
    labels = labels.copy()
    while True:
        uniq, counts = np.unique(labels, return_counts=True)
        n_clust = len(uniq)
        if n_clust == k_target:
            break
        if n_clust > k_target:
            # Merge two smallest
            order = np.argsort(counts)
            c1, c2 = int(uniq[order[0]]), int(uniq[order[1]])
            labels[labels == c2] = c1
        else:
            # Split largest cluster
            largest = int(uniq[np.argmax(counts)])
            members = np.where(labels == largest)[0]
            if len(members) < 2:
                break
            new_c = int(labels.max()) + 1
            half = rng.choice(members, size=max(1, len(members) // 2), replace=False)
            labels[half] = new_c
    return labels
