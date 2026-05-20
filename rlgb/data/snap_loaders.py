"""SNAP-style community dataset loaders.

Supported formats:
  - SNAP ground-truth community files (e.g. com-DBLP, com-Amazon, com-LJ)
  - Edge-list + community-memberships file pairs

All loaders return list[Problem] with task_type="community" and
known_communities filled.  For partition tasks, gt_labels = community_id
of first membership (nodes with multiple memberships use the first one).

Downloads are NOT performed automatically — place the raw .gz or .txt files
in the expected cache directory ($RLGB_DATA_DIR or ~/.rlgb_data/SNAP/).
"""
from __future__ import annotations

import gzip
import os
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np

from rlgb.tasks.base import Problem

_DEFAULT_ROOT = Path(os.environ.get("RLGB_DATA_DIR", Path.home() / ".rlgb_data")) / "SNAP"

# Mapping from dataset alias to expected filenames
_SNAP_FILES: dict[str, tuple[str, str]] = {
    "dblp":   ("com-dblp.ungraph.txt",  "com-dblp.top5000.cmty.txt"),
    "amazon": ("com-amazon.ungraph.txt", "com-amazon.top5000.cmty.txt"),
    "lj":     ("com-lj.ungraph.txt",     "com-lj.top5000.cmty.txt"),
    "youtube":("com-youtube.ungraph.txt","com-youtube.top5000.cmty.txt"),
}

_SNAP_URLS: dict[str, str] = {
    "dblp":    "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
    "amazon":  "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz",
    "lj":      "https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz",
    "youtube": "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",
}


def _open_maybe_gz(path: Path):
    if path.suffix == ".gz":
        return gzip.open(str(path), "rt")
    return open(str(path), "r")


def _parse_edgelist(path: Path, max_nodes: int) -> tuple[dict[int, int], np.ndarray]:
    """Parse SNAP edge-list; remap node ids; return (id_map, adj).

    Caps the graph at max_nodes by taking the first max_nodes unique node ids
    encountered in the file.
    """
    edges: list[tuple[int, int]] = []
    seen_order: dict[int, int] = {}

    with _open_maybe_gz(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            for x in (u, v):
                if x not in seen_order:
                    if len(seen_order) >= max_nodes:
                        break
                    seen_order[x] = len(seen_order)
            else:
                if seen_order.get(u) is not None and seen_order.get(v) is not None:
                    edges.append((seen_order[u], seen_order[v]))
            if len(seen_order) >= max_nodes:
                break

    n = len(seen_order)
    adj = np.zeros((n, n), dtype=np.float32)
    for u, v in edges:
        if u < n and v < n:
            adj[u, v] = 1.0
            adj[v, u] = 1.0
    return seen_order, adj


def _parse_communities(path: Path, id_map: dict[int, int]) -> list[list[int]]:
    """Parse SNAP community file; return list of community node-id lists (remapped)."""
    communities: list[list[int]] = []
    with _open_maybe_gz(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            members = [id_map[int(x)] for x in line.split() if int(x) in id_map]
            if len(members) >= 2:
                communities.append(members)
    return communities


def _communities_to_labels(communities: list[list[int]], n: int) -> np.ndarray:
    """Assign each node its first community id; unlabelled nodes get id 0."""
    labels = np.zeros(n, dtype=np.int64)
    for cid, members in enumerate(communities):
        for node in members:
            if labels[node] == 0:
                labels[node] = cid + 1
    return labels


def load_snap(
    name: str,
    root: Path | None = None,
    max_nodes: int = 1000,
    k_target: int | None = None,
    top_k_communities: int = 20,
) -> list[Problem]:
    """Load a SNAP community dataset.

    Args:
        name: Dataset alias — one of 'dblp', 'amazon', 'lj', 'youtube'.
        root: Directory containing SNAP txt files; defaults to ~/.rlgb_data/SNAP/.
        max_nodes: Maximum nodes to load (early cutoff from edge-list).
        k_target: Override cluster count; defaults to top_k_communities.
        top_k_communities: How many top communities to keep.

    Returns:
        Single-element list[Problem].

    Raises:
        FileNotFoundError: If edge-list or community file is missing.
    """
    name = name.lower()
    if name not in _SNAP_FILES:
        raise ValueError(f"Unknown SNAP dataset: {name!r}. Choose from {list(_SNAP_FILES)}")

    root = root or _DEFAULT_ROOT
    edge_file, cmty_file = _SNAP_FILES[name]
    epath = root / edge_file
    cpath = root / cmty_file

    # Try .gz variants if plain text not found
    for candidate in [epath, Path(str(epath) + ".gz")]:
        if candidate.exists():
            epath = candidate
            break
    else:
        raise FileNotFoundError(
            f"Edge-list not found: {epath}\n"
            f"Download from {_SNAP_URLS.get(name, 'SNAP Stanford')} and place in {root}/"
        )
    for candidate in [cpath, Path(str(cpath) + ".gz")]:
        if candidate.exists():
            cpath = candidate
            break
    else:
        raise FileNotFoundError(
            f"Community file not found: {cpath}\nPlace in {root}/"
        )

    id_map, adj = _parse_edgelist(epath, max_nodes)
    n = len(id_map)
    communities = _parse_communities(cpath, id_map)[:top_k_communities]
    gt_labels = _communities_to_labels(communities, n)
    k = k_target or max(int(gt_labels.max()), 2)

    return [Problem(
        name=f"snap_{name}",
        adj=adj,
        k_target=k,
        gt_labels=gt_labels,
        family="real",
        task_type="community",
        known_communities=communities,
        meta={"source": "SNAP", "n_nodes": n, "url": _SNAP_URLS.get(name, "")},
    )]


def load_snap_from_files(
    edge_path: str | Path,
    community_path: str | Path,
    name: str = "custom",
    max_nodes: int = 1000,
    k_target: int | None = None,
    top_k_communities: int = 20,
) -> list[Problem]:
    """Generic loader for any SNAP-format edge-list + community-file pair."""
    epath = Path(edge_path)
    cpath = Path(community_path)
    if not epath.exists():
        raise FileNotFoundError(f"Edge file not found: {epath}")
    if not cpath.exists():
        raise FileNotFoundError(f"Community file not found: {cpath}")

    id_map, adj = _parse_edgelist(epath, max_nodes)
    n = len(id_map)
    communities = _parse_communities(cpath, id_map)[:top_k_communities]
    gt_labels = _communities_to_labels(communities, n)
    k = k_target or max(int(gt_labels.max()), 2)

    return [Problem(
        name=name,
        adj=adj,
        k_target=k,
        gt_labels=gt_labels,
        family="real",
        task_type="community",
        known_communities=communities,
        meta={"source": "custom", "n_nodes": n},
    )]
