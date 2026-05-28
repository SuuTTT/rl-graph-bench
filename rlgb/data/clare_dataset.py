"""CLARE-format community dataset loader.

Reads the filtered SNAP graph format used by KDD2022CLARE:
  {name}-1.90.cmty.txt   — one community per line, space-separated node IDs
  {name}-1.90.ungraph.txt — one undirected edge per line, space-separated

Files are looked up in ~/.rlgb_data/CLARE/{name}/ and, if absent, copied from
/tmp/KDD2022CLARE/dataset/{name}/ (the bundled repo data) before falling back to
a download error.

Bundled datasets: amazon, dblp, lj (and cross-dataset variants).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


_CACHE_ROOT = Path.home() / ".rlgb_data" / "CLARE"
_BUNDLED_ROOT = Path("/tmp/KDD2022CLARE/dataset")


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class CLAREGraphData:
    """All data needed by the Locator and Rewriter."""
    nx_graph: nx.Graph
    pyg_data: Data                   # x=(N,5) degree feats, edge_index=(2,E)
    communities: list[list[int]]     # all GT communities (remapped 0-based IDs)
    train_ids: list[int]             # indices into communities
    val_ids:   list[int]
    test_ids:  list[int]
    name: str = "unknown"

    @property
    def train_communities(self) -> list[list[int]]:
        return [self.communities[i] for i in self.train_ids]

    @property
    def val_communities(self) -> list[list[int]]:
        return [self.communities[i] for i in self.val_ids]

    @property
    def test_communities(self) -> list[list[int]]:
        return [self.communities[i] for i in self.test_ids]

    def to_problem_suite(self, split: str = "test") -> list[Problem]:
        """Convert SNAP communities to local subgraph Problem instances to prevent OOM and run 1000x faster."""
        from rlgb.tasks.base import Problem
        from torch_geometric.data import Data
        comms = {
            "train": self.train_communities,
            "val":   self.val_communities,
            "test":  self.test_communities,
        }[split]

        problems = []
        for i, comm in enumerate(comms):
            # Extract local 3-hop neighborhood around the community nodes
            local_nodes = set(comm)
            for _ in range(3):
                neighbors = set()
                for n in local_nodes:
                    neighbors.update(self.nx_graph.neighbors(n))
                local_nodes.update(neighbors)
            local_nodes = sorted(list(local_nodes))
            
            # Extract induced subgraph
            subg = self.nx_graph.subgraph(local_nodes)
            local_mapping = {orig: new for new, orig in enumerate(local_nodes)}
            adj_dense = nx.to_numpy_array(subg, dtype=np.float32)
            
            # Remap community and seed query node to local IDs
            local_comm = [local_mapping[n] for n in comm]
            q = comm[0]
            q_local = local_mapping[q]
            
            # Local binary ground-truth labels
            gt = np.zeros(len(local_nodes), dtype=np.int32)
            gt[local_comm] = 1

            # Slice PyG features for local nodes
            local_x = self.pyg_data.x[local_nodes]
            local_pyg_data = Data(x=local_x, edge_index=None)

            problems.append(Problem(
                name=f"{self.name}_{split}_{i}",
                adj=adj_dense,
                k_target=2,  # community vs non-community
                gt_labels=gt,
                family=f"snap_{self.name}",
                task_type="community_expand",
                known_communities=[[q_local]],
                query_nodes=[q_local],
                meta={"true_community": local_comm, "pyg_data": local_pyg_data}
            ))
        return problems



# ---------------------------------------------------------------------------
# Raw file loading (ported from KDD2022CLARE/utils/load_dataset.py)
# ---------------------------------------------------------------------------

def _ensure_files(name: str) -> Path:
    """Return directory containing the two {name}-1.90.*.txt files."""
    dest = _CACHE_ROOT / name
    dest.mkdir(parents=True, exist_ok=True)

    cmty_file = dest / f"{name}-1.90.cmty.txt"
    edge_file  = dest / f"{name}-1.90.ungraph.txt"

    if cmty_file.exists() and edge_file.exists():
        return dest

    # Try bundled KDD2022CLARE data
    bundled = _BUNDLED_ROOT / name
    if bundled.exists():
        for f in bundled.glob("*.txt"):
            shutil.copy(f, dest / f.name)
        if cmty_file.exists() and edge_file.exists():
            return dest

    raise FileNotFoundError(
        f"CLARE dataset '{name}' not found in {dest} or {bundled}.\n"
        "Clone FDUDSDE/KDD2022CLARE to /tmp/KDD2022CLARE or place the files "
        f"manually in {dest}/"
    )


def _load_raw(name: str) -> tuple[list[list[int]], list[list[int]]]:
    """Return (communities, edges) as lists of int lists."""
    d = _ensure_files(name)
    communities = [
        [int(n) for n in line.split()]
        for line in (d / f"{name}-1.90.cmty.txt").read_text().splitlines()
        if line.strip()
    ]
    edges = [
        [int(u), int(v)]
        for line in (d / f"{name}-1.90.ungraph.txt").read_text().splitlines()
        if line.strip()
        for u, v in [[int(x) for x in line.split()]]
        if u != v
    ]
    # Canonicalise direction
    edges = [[min(u, v), max(u, v)] for u, v in edges]
    return communities, edges


def _degree_features(nx_graph: nx.Graph, num_nodes: int,
                     normalize: bool = True) -> np.ndarray:
    """5-dim degree-based features matching the original CLARE implementation.

    Features per node: [degree, min_neighbour_deg, max_neighbour_deg,
                        mean_neighbour_deg, std_neighbour_deg]
    """
    deg = np.array([nx_graph.degree(n) for n in range(num_nodes)], dtype=np.float32)
    feat = np.zeros((num_nodes, 5), dtype=np.float32)
    feat[:, 0] = deg

    for node in range(num_nodes):
        nbrs = list(nx_graph.neighbors(node))
        if nbrs:
            nd = deg[nbrs]
            feat[node, 1] = nd.min()
            feat[node, 2] = nd.max()
            feat[node, 3] = nd.mean()
            feat[node, 4] = nd.std()

    if normalize:
        mu  = feat.mean(0, keepdims=True)
        std = feat.std(0,  keepdims=True)
        feat = (feat - mu) / (std + 1e-9)

    return feat


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_clare_dataset(
    name: str = "amazon",
    num_train: int = 90,
    num_val: int = 10,
    seed: int = 0,
) -> CLAREGraphData:
    """Load a CLARE-format dataset and return a CLAREGraphData.

    Splits communities randomly using `seed`.  Default split (90/10/900)
    matches the CLARE paper's Amazon evaluation protocol.
    """
    communities_raw, edges_raw = _load_raw(name)

    # Remap node IDs to 0-based
    all_nodes = sorted({n for com in communities_raw for n in com}
                       | {n for e in edges_raw for n in e})
    mapping = {orig: new for new, orig in enumerate(all_nodes)}
    num_nodes = len(all_nodes)

    edges  = [[mapping[u], mapping[v]] for u, v in edges_raw]
    communities = [[mapping[n] for n in com] for com in communities_raw]

    # Build graphs
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edges)

    edge_index = torch.tensor(
        [[u, v] for u, v in edges] + [[v, u] for u, v in edges],
        dtype=torch.long,
    ).t().contiguous()
    feats = torch.tensor(_degree_features(g, num_nodes), dtype=torch.float32)
    pyg_data = Data(x=feats, edge_index=edge_index)

    # Split communities
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(communities)).tolist()
    train_ids = idx[:num_train]
    val_ids   = idx[num_train: num_train + num_val]
    test_ids  = idx[num_train + num_val:]

    return CLAREGraphData(
        nx_graph=g,
        pyg_data=pyg_data,
        communities=communities,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        name=name,
    )
