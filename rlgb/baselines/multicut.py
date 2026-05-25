"""GAEC — Greedy Additive Edge Contraction baseline.

Greedy Additive Edge Contraction (Keuper et al., CVPR 2015) for the
Multicut / MCMP problem on signed graphs.

Algorithm:
  1. Start: each node is its own cluster.
  2. Repeatedly pick the edge with the highest (most positive) weight
     between two *different* clusters and merge those clusters.
  3. After a merge, update the edge weights between the new super-node and
     its neighbours using the sum rule.
  4. Stop when no positive inter-cluster edge remains.

This is an O(E log E) approximation for correlation clustering / multicut.
"""
from __future__ import annotations

from pathlib import Path

import heapq
import numpy as np

from rlgb.tasks.base import Problem


class GAECBaseline:
    """Greedy Additive Edge Contraction for signed-cost MCMP."""

    compatible_tasks = ["multicut"]

    def partition(self, adj: np.ndarray) -> np.ndarray:
        """Return cluster label array for a signed adjacency matrix.

        Parameters
        ----------
        adj : (n, n) symmetric float array of signed edge costs.

        Returns
        -------
        labels : (n,) int array of cluster ids.
        """
        adj = np.asarray(adj, dtype=np.float64)
        n = adj.shape[0]
        labels = np.arange(n, dtype=np.int64)

        # edge weights between clusters — stored as (cluster_i, cluster_j) → w_sum
        edge_w: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                w = adj[i, j]
                if w != 0.0:
                    edge_w[(i, j)] = w

        # max-heap via negation
        heap: list[tuple[float, int, int]] = []
        for (i, j), w in edge_w.items():
            if w > 0:
                heapq.heappush(heap, (-w, i, j))

        # canonical id for each original cluster id
        parent: dict[int, int] = {i: i for i in range(n)}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        while heap:
            neg_w, ci, cj = heapq.heappop(heap)
            ri, rj = find(ci), find(cj)
            if ri == rj:
                continue  # already merged
            w = -neg_w
            # re-check current weight between ri and rj
            key = (min(ri, rj), max(ri, rj))
            current_w = edge_w.get(key, 0.0)
            if current_w != w:
                # stale entry; the actual current weight may still be positive
                if current_w > 0:
                    heapq.heappush(heap, (-current_w, *key))
                continue
            if current_w <= 0:
                break  # no more positive edges

            # merge rj into ri
            parent[rj] = ri

            # update edge weights: for each neighbour of rj, add to ri
            keys_to_remove = []
            new_edges: list[tuple[tuple[int, int], float]] = []
            for (a, b), ew in list(edge_w.items()):
                ra, rb = find(a), find(b)
                if ra == rb:
                    keys_to_remove.append((a, b))
                elif (a, b) != (min(ra, rb), max(ra, rb)):
                    # need to rekey
                    keys_to_remove.append((a, b))
                    new_key = (min(ra, rb), max(ra, rb))
                    new_edges.append((new_key, ew))

            for k in keys_to_remove:
                edge_w.pop(k, None)
            for new_key, ew in new_edges:
                edge_w[new_key] = edge_w.get(new_key, 0.0) + ew
                if edge_w[new_key] > 0:
                    heapq.heappush(heap, (-edge_w[new_key], *new_key))

        # canonicalise labels
        root_to_label: dict[int, int] = {}
        out = np.empty(n, dtype=np.int32)
        c = 0
        for i in range(n):
            r = find(i)
            if r not in root_to_label:
                root_to_label[r] = c
                c += 1
            out[i] = root_to_label[r]
        return out

    def partition_problem(self, problem: Problem) -> np.ndarray:
        return self.partition(problem.adj)

    # ── RLAgent-compatible interface (stateless) ──────────────────────────────

    def select_action(self, obs: dict, greedy: bool = True) -> int:
        return 0

    def push_transition(self, t) -> None:
        pass

    def update(self) -> dict:
        return {}

    def reset_episode(self) -> None:
        pass

    def save(self, path: str | Path) -> None:
        pass

    def load(self, path: str | Path) -> None:
        pass
