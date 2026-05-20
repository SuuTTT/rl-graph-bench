"""Base abstractions for clustering tasks.

A ClusteringTask owns:
  - a benchmark suite of Problem instances
  - objective functions (what to minimise)
  - evaluation protocol (which metrics to report)

All three concrete task types (GraphPartition, CommunityExpand, DynamicCD)
inherit from this base.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import numpy as np


@dataclass
class Problem:
    """One problem instance passed to an env or solver."""

    name: str
    adj: np.ndarray            # (N, N) symmetric float32
    k_target: int
    gt_labels: np.ndarray | None   # (N,) ground-truth, if available
    family: str                # "sbm" | "hier3" | "lfr" | "cora" | …
    task_type: str             # "partition" | "community_expand" | "dynamic_cd"
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.adj.shape[0]

    # Community-expand extras (ignored by other tasks)
    known_communities: list[list[int]] | None = None   # seed / training communities
    query_nodes: list[int] | None = None               # nodes to expand from

    # Dynamic-CD extras
    adj_snapshots: list[np.ndarray] | None = None  # ordered temporal snapshots


@dataclass
class EvalRecord:
    """One row in a results table."""

    problem_name: str
    algo_name: str
    seed: int
    n: int
    k_target: int
    k_got: int
    # Primary metrics (task-specific)
    primary_metric: str
    primary_value: float
    # Secondary metrics
    nmi: float | None
    ari: float | None
    f1: float | None
    h2: float | None
    ncut: float | None
    modularity: float | None
    modularity_density: float | None
    # Performance
    time_s: float
    n_steps: int
    status: str = "ok"


@runtime_checkable
class ClusteringTask(Protocol):
    """Protocol every task must satisfy."""

    name: str
    primary_metric: str       # "h2" | "ncut" | "f1" | "modularity_density"

    def build_suite(self, split: str = "test") -> list[Problem]:
        """Return list of Problem instances for eval."""
        ...

    def reward(
        self,
        adj: np.ndarray,
        labels_before: np.ndarray,
        labels_after: np.ndarray,
        problem: Problem,
    ) -> float:
        """Compute step reward (always shaped so higher = better)."""
        ...

    def evaluate(
        self, adj: np.ndarray, labels: np.ndarray, problem: Problem
    ) -> dict[str, float]:
        """Return dict of metric_name → value for a final partition."""
        ...
