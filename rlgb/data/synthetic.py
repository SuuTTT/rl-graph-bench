"""Synthetic graph generators.

All generators return (adj: np.ndarray, labels: np.ndarray, k: int).

Generators
----------
  sbm(n, k, p_in, p_out, seed)        – Stochastic Block Model
  lfr(n, mu, seed, ...)               – LFR benchmark
  hier3(cs, nc_inner, nc_outer, ...)  – 3-level hierarchy (ring of SBMs)
  hier2(n, k, seed)                   – 2-level hierarchy (SBM-of-SBMs)
  ring_cliques(clique_size, n_cliques) – ring of cliques
  ws_planted(n, k, ...)               – Watts-Strogatz with planted comms

Suites
------
  fixed17()  – 17 canonical eval graphs (same as rl-cluster-ops)
  mini5()    – 5-graph smoke-test suite
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import networkx as nx
import numpy as np

from rlgb.tasks.base import Problem


# ── SBM ──────────────────────────────────────────────────────────────────────

def sbm(
    n: int,
    k: int,
    p_in: float,
    p_out: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    sizes = [n // k] * k
    for i in range(n - sum(sizes)):
        sizes[i] += 1
    probs = np.full((k, k), p_out)
    np.fill_diagonal(probs, p_in)
    G = nx.stochastic_block_model(sizes, probs.tolist(), seed=int(rng.integers(2**31)))
    adj = nx.to_numpy_array(G).astype(np.float32)
    labels = np.repeat(np.arange(k), sizes).astype(np.int32)
    return adj, labels, k


# ── LFR ──────────────────────────────────────────────────────────────────────

def lfr(
    n: int = 200,
    mu: float = 0.3,
    tau1: float = 3.0,
    tau2: float = 1.5,
    average_degree: int = 8,
    min_community: int = 20,
    seed: int = 0,
    max_iters: int = 500,
) -> tuple[np.ndarray, np.ndarray, int]:
    for attempt in range(10):
        try:
            G = nx.LFR_benchmark_graph(
                n, tau1, tau2, mu,
                average_degree=average_degree,
                min_community=min_community,
                max_iters=max_iters,
                seed=int(seed + attempt * 1000),
            )
            comms = list({frozenset(G.nodes[v]["community"]) for v in G.nodes})
            comm_list = sorted(comms, key=lambda s: min(s))
            label_map = {v: i for i, c in enumerate(comm_list) for v in c}
            labels = np.array([label_map[v] for v in range(n)], dtype=np.int32)
            adj = nx.to_numpy_array(G, dtype=np.float32)
            return adj, labels, len(comms)
        except (nx.NetworkXError, nx.ExceededMaxIterations):
            continue
    raise RuntimeError(f"LFR failed to converge (seed={seed}, mu={mu})")


# ── Hier3 ─────────────────────────────────────────────────────────────────────

def hier3(
    clique_size: int = 5,
    nc_inner: int = 4,
    nc_outer: int = 4,
    p_intra: float = 0.7,
    p_inter: float = 0.05,
    p_global: float = 0.005,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """3-level hierarchy: nc_outer super-clusters, each with nc_inner sub-clusters
    of clique_size nodes.  k_target = nc_outer."""
    rng = np.random.default_rng(seed)
    n = clique_size * nc_inner * nc_outer
    adj = np.zeros((n, n), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int32)

    for oc in range(nc_outer):
        for ic in range(nc_inner):
            b = (oc * nc_inner + ic) * clique_size
            labels[b: b + clique_size] = oc
            # intra sub-cluster
            for i in range(b, b + clique_size):
                for j in range(i + 1, b + clique_size):
                    if rng.random() < p_intra:
                        adj[i, j] = adj[j, i] = 1.0
            # inter sub-cluster within super-cluster
            for ic2 in range(ic + 1, nc_inner):
                b2 = (oc * nc_inner + ic2) * clique_size
                for i in range(b, b + clique_size):
                    for j in range(b2, b2 + clique_size):
                        if rng.random() < p_inter:
                            adj[i, j] = adj[j, i] = 1.0
        # inter super-cluster (sparse global)
        for oc2 in range(oc + 1, nc_outer):
            b1_start = oc  * nc_inner * clique_size
            b2_start = oc2 * nc_inner * clique_size
            block_size = nc_inner * clique_size
            for i in range(b1_start, b1_start + block_size):
                for j in range(b2_start, b2_start + block_size):
                    if rng.random() < p_global:
                        adj[i, j] = adj[j, i] = 1.0

    return adj, labels, nc_outer


# ── Hier2 (SBM-of-SBMs) ──────────────────────────────────────────────────────

def hier2(
    n: int = 80,
    k: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """2-level hierarchy SBM."""
    rng = np.random.default_rng(seed)
    group_size = n // k
    adj = np.zeros((n, n), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int32)
    for c in range(k):
        b = c * group_size
        sz = group_size if c < k - 1 else n - b
        labels[b: b + sz] = c
        # dense intra
        for i in range(b, b + sz):
            for j in range(i + 1, b + sz):
                if rng.random() < 0.5:
                    adj[i, j] = adj[j, i] = 1.0
        # sparse inter
        for c2 in range(c + 1, k):
            b2 = c2 * group_size
            sz2 = group_size if c2 < k - 1 else n - b2
            for i in range(b, b + sz):
                for j in range(b2, b2 + sz2):
                    if rng.random() < 0.03:
                        adj[i, j] = adj[j, i] = 1.0
    return adj, labels, k


# ── Ring cliques ──────────────────────────────────────────────────────────────

def ring_cliques(
    clique_size: int = 5,
    n_cliques: int = 4,
) -> tuple[np.ndarray, np.ndarray, int]:
    n = clique_size * n_cliques
    adj = np.zeros((n, n), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int32)
    for c in range(n_cliques):
        b = c * clique_size
        labels[b: b + clique_size] = c
        for i in range(b, b + clique_size):
            for j in range(i + 1, b + clique_size):
                adj[i, j] = adj[j, i] = 1.0
        # ring connector: last node of c → first node of (c+1)%n_cliques
        nxt = (c + 1) % n_cliques
        adj[b + clique_size - 1, nxt * clique_size] = 1.0
        adj[nxt * clique_size, b + clique_size - 1] = 1.0
    return adj, labels, n_cliques


# ── Watts-Strogatz planted ────────────────────────────────────────────────────

def ws_planted(
    n: int = 120,
    k: int = 4,
    k_ws: int = 6,
    p_rewire: float = 0.1,
    p_inter: float = 0.03,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    block_size = n // k
    adj = np.zeros((n, n), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int32)
    for c in range(k):
        nodes = np.arange(c * block_size, (c + 1) * block_size)
        labels[nodes] = c
        WS = nx.watts_strogatz_graph(len(nodes), k_ws, p_rewire,
                                     seed=int(rng.integers(2**31)))
        for u, v in WS.edges():
            i, j = int(nodes[u]), int(nodes[v])
            adj[i, j] = adj[j, i] = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j] and rng.random() < p_inter:
                adj[i, j] = adj[j, i] = 1.0
    return adj, labels, k


# ── Canonical benchmark suites ────────────────────────────────────────────────

def _make_problem(
    name: str,
    adj: np.ndarray,
    k: int,
    labels: np.ndarray,
    family: str,
    task_type: str = "partition",
) -> Problem:
    return Problem(
        name=name, adj=adj, k_target=k, gt_labels=labels,
        family=family, task_type=task_type,
        meta={"n": int(adj.shape[0])},
    )


def mini5() -> list[Problem]:
    """5-graph smoke-test suite (fast, ~2 s total)."""
    G = nx.karate_club_graph()
    gt_karate = np.array(
        [1 if G.nodes[n]["club"] == "Officer" else 0 for n in G.nodes],
        dtype=np.int32,
    )
    suite = [
        _make_problem("karate", nx.to_numpy_array(G).astype(np.float32),
                      2, gt_karate, "real"),
    ]
    for seed in range(4):
        a, l, k = ring_cliques(5, 4) if seed == 0 else sbm(60, 3, 0.4, 0.04, seed=seed)
        fam = "ring" if seed == 0 else "sbm"
        suite.append(_make_problem(f"mini_{seed}", a, k, l, fam))
    return suite


def fixed17() -> list[Problem]:
    """Canonical 17-graph eval suite (mirrors rl-cluster-ops fixed-17)."""
    G = nx.karate_club_graph()
    gt_karate = np.array(
        [1 if G.nodes[n]["club"] == "Officer" else 0 for n in G.nodes],
        dtype=np.int32,
    )

    def sbm_gt(n: int, k: int) -> np.ndarray:
        labels = np.repeat(np.arange(k), n // k)
        rem = n - len(labels)
        return np.concatenate([labels, np.zeros(rem, dtype=int)]).astype(np.int32)

    def hier3_gt(cs: int, ni: int, no: int) -> np.ndarray:
        n = cs * ni * no
        lbl = np.zeros(n, dtype=np.int32)
        for oc in range(no):
            for ic in range(ni):
                b = (oc * ni + ic) * cs
                lbl[b: b + cs] = oc
        return lbl

    def ring_gt(cs: int, nc: int) -> np.ndarray:
        return np.repeat(np.arange(nc), cs).astype(np.int32)

    raw: list[tuple[str, np.ndarray, int, np.ndarray, str]] = [
        ("karate",       nx.to_numpy_array(G).astype(np.float32),        2, gt_karate,        "real"),
        ("ring_4x5",     ring_cliques(5, 4)[0],                          4, ring_gt(5, 4),    "ring"),
        ("ring_6x5",     ring_cliques(6, 5)[0],                          5, ring_gt(6, 5),    "ring"),
        ("sbm_100",      sbm(100, 4, 0.4,  0.03,  seed=99)[0],          4, sbm_gt(100, 4),   "sbm"),
        ("sbm_200",      sbm(200, 5, 0.35, 0.025, seed=99)[0],          5, sbm_gt(200, 5),   "sbm"),
    ]
    for s in range(5, 11):
        a, l, k = hier3(5, 4, 4, 0.7, 0.05, 0.005, seed=s)
        raw.append((f"hier3_s{s}", a, k, hier3_gt(5, 4, 4), "hier3"))
    for s in (8, 9):
        a, l, k = hier3(4, 3, 5, 0.7, 0.05, 0.005, seed=s)
        raw.append((f"hier3_b_s{s}", a, k, hier3_gt(4, 3, 5), "hier3"))
    for s in (8, 9):
        a, l, k = hier3(6, 3, 4, 0.7, 0.04, 0.005, seed=s)
        raw.append((f"hier3_c_s{s}", a, k, hier3_gt(6, 3, 4), "hier3"))
    raw.append(("hier2_n80_s8",  hier2(80, 4, seed=8)[0],  4, sbm_gt(80,  4), "hier2"))
    raw.append(("hier2_n90_s8",  hier2(90, 6, seed=8)[0],  6, sbm_gt(90,  6), "hier2"))

    return [_make_problem(n, a, k, l, f) for n, a, k, l, f in raw]
