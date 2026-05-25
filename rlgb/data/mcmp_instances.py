"""Multicut / MCMP instance generator.

Generates signed-cost ER and BA graphs matching the SS2V paper
(TNNLS 2025) distribution:
  - n ∈ {20, 40, 60}
  - ER: Erdős–Rényi G(n, p) with p = 0.3
  - BA: Barabási–Albert with m = 2 (avg degree ~4)
  - Edge costs w ∈ U[-1, +1]  (signed, float32)
  - Positive w → "same cluster", negative w → "different cluster"
  - Objective: minimise total multicut cost = Σ_{(u,v) cut} w_{uv}

The `Problem.adj` stores the cost matrix (signed).
`Problem.meta["is_mcmp"] = True` marks these as MCMP instances.
`Problem.k_target` is set to the number of positive components as a hint
  (not used as hard constraint by MCMP — the algo finds its own partition).
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from rlgb.tasks.base import Problem


def _make_signed_adj(G: nx.Graph, rng: np.random.Generator) -> np.ndarray:
    """Return signed cost matrix for graph G."""
    n = G.number_of_nodes()
    adj = np.zeros((n, n), dtype=np.float32)
    for u, v in G.edges():
        w = float(rng.uniform(-1.0, 1.0))
        adj[u, v] = w
        adj[v, u] = w
    return adj


def _positive_component_count(adj: np.ndarray) -> int:
    """Number of connected components in the subgraph of positive edges."""
    n = adj.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols = np.where(adj > 0)
    for u, v in zip(rows, cols):
        if u < v:
            G.add_edge(int(u), int(v))
    return max(1, nx.number_connected_components(G))


def er_mcmp(
    n: int,
    p: float = 0.3,
    n_instances: int = 50,
    seed_offset: int = 0,
) -> list[Problem]:
    """ER-MCMP instances: G(n, p) with signed costs."""
    problems = []
    rng = np.random.default_rng(seed_offset)
    for i in range(n_instances):
        G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(2**31)))
        if G.number_of_edges() == 0:
            G.add_edge(0, 1)
        adj = _make_signed_adj(G, rng)
        problems.append(Problem(
            name=f"er_n{n}_{i}",
            adj=np.abs(adj),  # full unsigned adj: SAGE sees all edges; wrapper filters action space
            k_target=1,       # episode ends when no positive inter-cluster edges remain (wrapper)
            gt_labels=None,
            family="er_mcmp",
            task_type="multicut",
            meta={"is_mcmp": True, "model": "er", "n": n, "p": p,
                  "cost_matrix": adj},
        ))
    return problems


def ba_mcmp(
    n: int,
    m: int = 2,
    n_instances: int = 50,
    seed_offset: int = 1000,
) -> list[Problem]:
    """BA-MCMP instances: Barabási–Albert G(n, m) with signed costs."""
    problems = []
    rng = np.random.default_rng(seed_offset)
    for i in range(n_instances):
        G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(2**31)))
        adj = _make_signed_adj(G, rng)
        problems.append(Problem(
            name=f"ba_n{n}_{i}",
            adj=np.abs(adj),  # full unsigned adj: SAGE sees all edges; wrapper filters action space
            k_target=1,       # episode ends when no positive inter-cluster edges remain (wrapper)
            gt_labels=None,
            family="ba_mcmp",
            task_type="multicut",
            meta={"is_mcmp": True, "model": "ba", "n": n, "m": m,
                  "cost_matrix": adj},
        ))
    return problems


def mcmp_train_suite(n: int = 40, n_instances: int = 50, seed_offset: int = 9999) -> list[Problem]:
    """Mixed ER+BA training instances at a given n."""
    er = er_mcmp(n, n_instances=n_instances // 2, seed_offset=seed_offset)
    ba = ba_mcmp(n, n_instances=n_instances // 2, seed_offset=seed_offset + 500)
    return er + ba


def mcmp_test_suite(sizes: tuple[int, ...] = (20, 40, 60)) -> dict[str, list[Problem]]:
    """9 test sets: {er,ba} × {20,40,60}.

    Returns dict mapping name -> list[Problem] (50 instances each).
    Reproducible with fixed seed_offset=5000.
    """
    suite: dict[str, list[Problem]] = {}
    for n in sizes:
        suite[f"er_n{n}"]  = er_mcmp(n, n_instances=50, seed_offset=5000 + n)
        suite[f"ba_n{n}"]  = ba_mcmp(n, n_instances=50, seed_offset=6000 + n)
    return suite
