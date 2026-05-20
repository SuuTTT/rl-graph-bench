"""Classical graph-clustering baselines.

All baselines expose the same interface as RLAgent for drop-in use
with eval_algo_on_suite():

    baseline = LeidenBaseline(resolution=1.0)
    df = eval_algo_on_suite(baseline, suite, task, n_seeds=3, horizon=1)

Supported:
  - LeidenBaseline   — leidenalg modularity optimisation (igraph)
  - LouvainBaseline  — igraph community_multilevel (Louvain)
  - SpectralBaseline — sklearn SpectralClustering (fixed k)
  - RandomBaseline   — random partition (sanity lower-bound)
  - MetisBaseline    — METIS k-way partitioning (requires pymetis)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rlgb.algos.base import Transition


class _BaselineAgent:
    """Minimal RLAgent-compatible interface for non-RL baselines.

    Baselines compute their full partition in select_action() on the first
    call per episode, then cache it and serve NOOP actions for subsequent steps.
    """

    compatible_tasks: list[str] = ["partition", "community", "dynamic"]

    def select_action(self, obs: dict, greedy: bool = True) -> int:
        raise NotImplementedError

    def push_transition(self, t: Transition) -> None:
        pass

    def update(self) -> dict[str, float]:
        return {}

    def reset_episode(self) -> None:
        self._computed = False
        self._cached_labels: np.ndarray | None = None

    def save(self, path: str | Path) -> None:
        pass  # stateless

    def load(self, path: str | Path) -> None:
        pass  # stateless

    def _adj_to_igraph(self, adj: np.ndarray):
        import igraph as ig
        n = adj.shape[0]
        rows, cols = np.where(adj > 0)
        # upper triangle only to avoid duplicates
        edges = [(int(u), int(v)) for u, v in zip(rows, cols) if u < v]
        return ig.Graph(n=n, edges=edges)

    def _labels_to_action(self, labels: np.ndarray, obs: dict) -> int:
        """Store computed labels and return NOOP action 0."""
        self._cached_labels = labels.astype(np.int64)
        self._computed = True
        return 0

    def _inject_labels(self, obs: dict) -> dict:
        """Return obs with labels replaced by cached partition."""
        if self._cached_labels is not None:
            obs = dict(obs)
            obs["labels"] = self._cached_labels.astype(np.float32)
        return obs


# ── Leiden ────────────────────────────────────────────────────────────────────

class LeidenBaseline(_BaselineAgent):
    """Leiden algorithm (Traag et al. 2019) via leidenalg + igraph.

    Runs modularity-based community detection; for fixed-k tasks uses
    the resolution parameter to target approximately k clusters.

    Args:
        resolution: Resolution parameter (higher → more clusters).
        n_iterations: Number of Leiden iterations.
        seed: Random seed.
    """

    name = "leiden"

    def __init__(
        self,
        resolution: float = 1.0,
        n_iterations: int = 10,
        seed: int = 0,
    ) -> None:
        self._resolution    = resolution
        self._n_iterations  = n_iterations
        self._seed          = seed
        self._computed      = False
        self._cached_labels = None

    def select_action(self, obs: dict, greedy: bool = True) -> int:
        if not self._computed:
            import leidenalg
            g = self._adj_to_igraph(obs["adj"])
            part = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=self._resolution,
                n_iterations=self._n_iterations,
                seed=self._seed,
            )
            labels = np.zeros(len(obs["adj"]), dtype=np.int64)
            for cid, members in enumerate(part):
                for node in members:
                    labels[node] = cid
            return self._labels_to_action(labels, obs)
        return 0

    def reset_episode(self) -> None:
        self._computed = False
        self._cached_labels = None


# ── Louvain ───────────────────────────────────────────────────────────────────

class LouvainBaseline(_BaselineAgent):
    """Louvain algorithm via igraph community_multilevel.

    Args:
        seed: Random seed passed to igraph.
    """

    name = "louvain"

    def __init__(self, seed: int = 0) -> None:
        self._seed          = seed
        self._computed      = False
        self._cached_labels = None

    def select_action(self, obs: dict, greedy: bool = True) -> int:
        if not self._computed:
            g = self._adj_to_igraph(obs["adj"])
            part = g.community_multilevel()
            labels = np.zeros(len(obs["adj"]), dtype=np.int64)
            for cid, members in enumerate(part):
                for node in members:
                    labels[node] = cid
            return self._labels_to_action(labels, obs)
        return 0

    def reset_episode(self) -> None:
        self._computed = False
        self._cached_labels = None


# ── Spectral ──────────────────────────────────────────────────────────────────

class SpectralBaseline(_BaselineAgent):
    """Spectral Clustering (sklearn) — uses ground-truth k from obs.

    Args:
        n_clusters: Override cluster count (default: from obs["k"]).
        affinity: 'precomputed' uses adj directly; 'rbf' builds kernel.
        seed: Random seed.
    """

    name = "spectral"

    def __init__(
        self,
        n_clusters: int | None = None,
        affinity: str = "precomputed",
        seed: int = 0,
    ) -> None:
        self._n_clusters    = n_clusters
        self._affinity      = affinity
        self._seed          = seed
        self._computed      = False
        self._cached_labels = None

    def select_action(self, obs: dict, greedy: bool = True) -> int:
        if not self._computed:
            from sklearn.cluster import SpectralClustering
            adj = obs["adj"].astype(np.float64)
            k   = self._n_clusters or max(2, int(round(float(obs["k"][0]))))
            sc  = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                random_state=self._seed,
                n_init=5,
            )
            labels = sc.fit_predict(adj).astype(np.int64)
            return self._labels_to_action(labels, obs)
        return 0

    def reset_episode(self) -> None:
        self._computed = False
        self._cached_labels = None


# ── Random ────────────────────────────────────────────────────────────────────

class RandomBaseline(_BaselineAgent):
    """Uniformly random partition into k clusters (lower-bound sanity check).

    Args:
        seed: RNG seed.
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng           = np.random.default_rng(seed)
        self._computed      = False
        self._cached_labels = None

    def select_action(self, obs: dict, greedy: bool = True) -> int:
        if not self._computed:
            n = obs["adj"].shape[0]
            k = max(2, int(round(float(obs["k"][0]))))
            labels = self._rng.integers(0, k, size=n).astype(np.int64)
            return self._labels_to_action(labels, obs)
        return 0

    def reset_episode(self) -> None:
        self._computed = False
        self._cached_labels = None


# ── METIS (optional) ──────────────────────────────────────────────────────────

class MetisBaseline(_BaselineAgent):
    """METIS k-way graph partitioning via pymetis (optional dependency).

    Install:  pip install pymetis
    """

    name = "metis"

    def __init__(self, seed: int = 0) -> None:
        self._seed          = seed
        self._computed      = False
        self._cached_labels = None

    def select_action(self, obs: dict, greedy: bool = True) -> int:
        if not self._computed:
            try:
                import pymetis
            except ImportError as exc:
                raise ImportError("pymetis is required for MetisBaseline: pip install pymetis") from exc
            adj = obs["adj"]
            k   = max(2, int(round(float(obs["k"][0]))))
            n   = adj.shape[0]
            adjacency = [
                np.where(adj[i] > 0)[0].tolist() for i in range(n)
            ]
            _, parts = pymetis.part_graph(k, adjacency=adjacency)
            labels = np.array(parts, dtype=np.int64)
            return self._labels_to_action(labels, obs)
        return 0

    def reset_episode(self) -> None:
        self._computed = False
        self._cached_labels = None


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_BASELINES: dict[str, type] = {
    "leiden":   LeidenBaseline,
    "louvain":  LouvainBaseline,
    "spectral": SpectralBaseline,
    "random":   RandomBaseline,
    "metis":    MetisBaseline,
}


def get_baseline(name: str, **kwargs) -> _BaselineAgent:
    """Instantiate a baseline by name."""
    if name not in ALL_BASELINES:
        raise ValueError(f"Unknown baseline '{name}'. Choose from {list(ALL_BASELINES)}")
    return ALL_BASELINES[name](**kwargs)
