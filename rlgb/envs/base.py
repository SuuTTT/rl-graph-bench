"""Base gymnasium.Env for all clustering environments.

Observation space (Dict)
------------------------
  adj        : Box(N, N) float32  – dense adjacency (row-normalised also in extras)
  node_feats : Box(N, F) float32  – node feature matrix (degree, clustering, etc.)
  labels     : Box(N,) int32      – current cluster assignment (0-indexed)
  k          : Box(1,) int32      – target cluster count

The dense adjacency is always provided.  Individual envs may add task-specific
keys (e.g. "seed_node" for community-expand, "snapshot_idx" for dynamic-CD).

Action space
------------
  Defined by each concrete env subclass (Discrete or MultiDiscrete or Box).

Reward
------
  Always shaped so higher = better (negative cost improvement).
  The concrete env calls self.task.reward(adj, old_labels, new_labels, problem).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rlgb.tasks.base import Problem, ClusteringTask


_MAX_NODES = 1024  # obs-space upper bound (actual size varies per episode)
_DEFAULT_N_FEAT = 7  # degree-based 7-dim features


class ClusteringEnv(ABC, gym.Env):
    """Abstract base for all rl-graph-bench environments.

    Subclasses must implement:
      - _build_obs() -> dict
      - _legal_actions() -> list[int]
      - step(action) -> (obs, reward, terminated, truncated, info)
      - reset(seed, options) -> (obs, info)
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task: ClusteringTask,
        problem: Problem,
        horizon: int = 10,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.task = task
        self.problem = problem
        self.horizon = horizon
        self._rng = np.random.default_rng(seed)
        self._seed_int = seed

        n = problem.n
        f = _DEFAULT_N_FEAT

        self.observation_space = spaces.Dict({
            "adj":        spaces.Box(0.0, 1.0, shape=(n, n), dtype=np.float32),
            "node_feats": spaces.Box(-np.inf, np.inf, shape=(n, f), dtype=np.float32),
            "labels":     spaces.Box(0, problem.k_target, shape=(n,), dtype=np.int32),
            "k":          spaces.Box(1, problem.k_target, shape=(1,), dtype=np.int32),
        })
        # concrete envs set self.action_space in their __init__

        # Runtime state
        self.adj: np.ndarray = problem.adj.copy()
        self.labels: np.ndarray = np.zeros(n, dtype=np.int32)
        self._step_count: int = 0
        self._best_obj: float = float("inf")
        self._best_labels: np.ndarray = self.labels.copy()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _node_features(self) -> np.ndarray:
        """7-dim per-node feature: degree, clustering coeff, betweenness proxy,
        closeness proxy, k-core, intra-cluster degree ratio, cluster size ratio."""
        adj = self.adj
        n = adj.shape[0]

        # Optimize for large community detection graphs: bypass expensive O(N^3) matmul
        if self.problem.task_type == "community_expand" and n > 500:
            deg = adj.sum(axis=1)
            deg_n = deg / (deg.max() + 1e-9)
            pyg_data = self.problem.meta.get("pyg_data", None)
            if pyg_data is not None and hasattr(pyg_data, "x"):
                x = pyg_data.x.cpu().numpy() if hasattr(pyg_data.x, "cpu") else pyg_data.x
                if x.shape[1] < 7:
                    pad = np.zeros((n, 7 - x.shape[1]), dtype=np.float32)
                    return np.hstack([x, pad])
                return x[:, :7].astype(np.float32)
            
            # Simple O(N) fallback features to avoid A @ A matrix multiplication
            feats = np.zeros((n, 7), dtype=np.float32)
            feats[:, 0] = deg_n
            feats[:, 6] = 1.0 / n
            return feats

        deg = adj.sum(axis=1)                            # (N,)
        m2 = deg.sum()
        k = max(int(self.labels.max()) + 1, 1)

        # Cluster size
        sizes = np.bincount(self.labels, minlength=k).astype(np.float32)
        cluster_size_ratio = sizes[self.labels] / n      # (N,)

        # Intra-cluster degree — vectorized (no Python loop over nodes)
        same_cluster = (self.labels[:, None] == self.labels[None, :])  # (N, N) bool
        intra = (adj * same_cluster).sum(axis=1).astype(np.float32)    # (N,)
        intra_ratio = intra / (deg + 1e-9)               # (N,)

        # Triangle proxy — vectorized: tri[i] = sum_j adj[i,j] * (A²)[i,j]
        adj2 = adj @ adj                                 # (N, N), single matmul
        tri = (adj * adj2).sum(axis=1).astype(np.float32)  # (N,)
        vol = (deg * (deg - 1))
        cc = np.where(vol > 0, tri / (vol + 1e-9), 0.0)

        # Normalise
        deg_n  = deg  / (deg.max()  + 1e-9)
        cc_n   = cc   / (cc.max()   + 1e-9)
        intra_n = intra_ratio
        csr_n  = cluster_size_ratio

        # Pad/clip to 7 dims
        feats = np.stack([
            deg_n, cc_n, intra_n, csr_n,
            deg / (m2 + 1e-9),                           # global degree ratio
            (deg_n) ** 2,                                # degree-squared proxy
            np.ones(n, dtype=np.float32) * k / n,        # k/n global signal
        ], axis=1).astype(np.float32)
        return feats

    def _build_base_obs(self) -> dict[str, np.ndarray]:
        return {
            "adj":        self.adj,
            "node_feats": self._node_features(),
            "labels":     self.labels.copy(),
            "k":          np.array([self.problem.k_target], dtype=np.int32),
        }

    # ── gymnasium API ────────────────────────────────────────────────────────

    @abstractmethod
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        ...

    @abstractmethod
    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        ...

    def render(self) -> np.ndarray | None:
        """Return an RGB image of the graph with cluster colours (matplotlib)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            import networkx as nx

            G = nx.from_numpy_array(self.adj)
            k = max(int(self.labels.max()) + 1, 1)
            cmap = cm.get_cmap("tab10", k)
            colors = [cmap(int(self.labels[i])) for i in range(self.adj.shape[0])]
            fig, ax = plt.subplots(figsize=(5, 5))
            pos = nx.spring_layout(G, seed=self._seed_int)
            nx.draw(G, pos=pos, node_color=colors, node_size=80,
                    edge_color="#aaa", ax=ax, with_labels=False)
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return buf
        except Exception:
            return None

    def close(self) -> None:
        pass
