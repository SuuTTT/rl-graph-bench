"""PyG-based loaders for standard graph clustering benchmarks.

Supported datasets:
  Planetoid:  Cora, CiteSeer, PubMed
  Amazon:     Computers, Photo
  DBLP (via torch_geometric.datasets)
  Coauthor:   CS, Physics

Each loader returns a list[Problem] ready for the benchmark harness.
Downloads are cached under ~/.rlgb_data/ by default.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from rlgb.tasks.base import Problem

_DEFAULT_ROOT = Path(os.environ.get("RLGB_DATA_DIR", Path.home() / ".rlgb_data"))


def _pyg_available() -> bool:
    try:
        import torch_geometric  # noqa: F401
        return True
    except ImportError:
        return False


def _adj_from_edge_index(edge_index: "torch.Tensor", n: int) -> np.ndarray:
    import torch
    adj = torch.zeros(n, n, dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj[edge_index[1], edge_index[0]] = 1.0  # ensure undirected
    return adj.numpy()


def load_planetoid(
    name: str = "Cora",
    root: Path | None = None,
    k_target: int | None = None,
    max_nodes: int = 2000,
) -> list[Problem]:
    """Load a Planetoid dataset (Cora/CiteSeer/PubMed) as a single Problem.

    Args:
        name: Dataset name ('Cora', 'CiteSeer', 'PubMed').
        root: Cache directory; defaults to ~/.rlgb_data.
        k_target: Override number of clusters; defaults to number of classes.
        max_nodes: Cap node count for quick iteration (uses LCC subgraph).

    Returns:
        List with one Problem.
    """
    if not _pyg_available():
        raise ImportError("torch_geometric is required for load_planetoid()")
    from torch_geometric.datasets import Planetoid
    root = root or _DEFAULT_ROOT
    dataset = Planetoid(root=str(root / "Planetoid"), name=name)
    data = dataset[0]

    n = int(data.num_nodes)
    labels = data.y.numpy()
    k = int(k_target or (labels.max() + 1))

    # Optionally subsample via BFS from highest-degree node
    if n > max_nodes:
        data, labels = _bfs_subsample(data, max_nodes)
        n = labels.shape[0]

    adj = _adj_from_edge_index(data.edge_index, n)
    return [Problem(
        name=name.lower(),
        adj=adj,
        k_target=k,
        gt_labels=labels,
        family="real",
        task_type="partition",
        meta={"source": "Planetoid", "n_classes": int(labels.max() + 1)},
    )]


def load_amazon(
    name: str = "Computers",
    root: Path | None = None,
    k_target: int | None = None,
    max_nodes: int = 2000,
) -> list[Problem]:
    """Load Amazon co-purchase graph (Computers or Photo)."""
    if not _pyg_available():
        raise ImportError("torch_geometric is required for load_amazon()")
    from torch_geometric.datasets import Amazon
    root = root or _DEFAULT_ROOT
    dataset = Amazon(root=str(root / "Amazon"), name=name)
    data = dataset[0]

    n = int(data.num_nodes)
    labels = data.y.numpy()
    k = int(k_target or (labels.max() + 1))

    if n > max_nodes:
        data, labels = _bfs_subsample(data, max_nodes)
        n = labels.shape[0]

    adj = _adj_from_edge_index(data.edge_index, n)
    return [Problem(
        name=f"amazon_{name.lower()}",
        adj=adj,
        k_target=k,
        gt_labels=labels,
        family="real",
        task_type="partition",
        meta={"source": "Amazon", "n_classes": int(labels.max() + 1)},
    )]


def load_coauthor(
    name: str = "CS",
    root: Path | None = None,
    k_target: int | None = None,
    max_nodes: int = 2000,
) -> list[Problem]:
    """Load Coauthor CS or Physics dataset."""
    if not _pyg_available():
        raise ImportError("torch_geometric is required for load_coauthor()")
    from torch_geometric.datasets import Coauthor
    root = root or _DEFAULT_ROOT
    dataset = Coauthor(root=str(root / "Coauthor"), name=name)
    data = dataset[0]

    n = int(data.num_nodes)
    labels = data.y.numpy()
    k = int(k_target or (labels.max() + 1))

    if n > max_nodes:
        data, labels = _bfs_subsample(data, max_nodes)
        n = labels.shape[0]

    adj = _adj_from_edge_index(data.edge_index, n)
    return [Problem(
        name=f"coauthor_{name.lower()}",
        adj=adj,
        k_target=k,
        gt_labels=labels,
        family="real",
        task_type="partition",
        meta={"source": "Coauthor", "n_classes": int(labels.max() + 1)},
    )]


def _bfs_subsample(data: "Data", max_nodes: int) -> tuple["Data", np.ndarray]:
    """BFS-subsample data to at most max_nodes nodes starting from highest degree."""
    import torch
    from torch_geometric.utils import subgraph

    n = int(data.num_nodes)
    edge_index = data.edge_index
    deg = torch.zeros(n, dtype=torch.long)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.long))

    start = int(deg.argmax().item())
    visited = [start]
    queue = [start]
    adj_list: dict[int, list[int]] = {i: [] for i in range(n)}
    for u, v in edge_index.t().tolist():
        adj_list[u].append(v)

    while queue and len(visited) < max_nodes:
        node = queue.pop(0)
        for nb in adj_list[node]:
            if nb not in set(visited):
                visited.append(nb)
                queue.append(nb)
                if len(visited) >= max_nodes:
                    break

    mask = torch.zeros(n, dtype=torch.bool)
    mask[visited] = True
    node_idx = torch.where(mask)[0]

    new_edge_index, _ = subgraph(node_idx, edge_index, relabel_nodes=True, num_nodes=n)
    labels = data.y[node_idx].numpy()

    # Rebuild minimal data object
    class _SubData:
        pass

    sub = _SubData()
    sub.num_nodes = len(visited)  # type: ignore[attr-defined]
    sub.edge_index = new_edge_index  # type: ignore[attr-defined]

    return sub, labels


def real_benchmark_suite(
    names: Sequence[str] | None = None,
    max_nodes: int = 500,
    root: Path | None = None,
) -> list[Problem]:
    """Load a mix of real-world graphs for the benchmark.

    Defaults to Cora + CiteSeer if torch_geometric is available,
    otherwise falls back to an empty list (synthetic data still usable).
    """
    if not _pyg_available():
        return []

    names = list(names or ["Cora", "CiteSeer"])
    probs: list[Problem] = []
    for name in names:
        try:
            if name in ("Cora", "CiteSeer", "PubMed"):
                probs.extend(load_planetoid(name, root=root, max_nodes=max_nodes))
            elif name in ("Computers", "Photo"):
                probs.extend(load_amazon(name, root=root, max_nodes=max_nodes))
            elif name in ("CS", "Physics"):
                probs.extend(load_coauthor(name, root=root, max_nodes=max_nodes))
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(f"Skipping {name}: {exc}", stacklevel=2)
    return probs
