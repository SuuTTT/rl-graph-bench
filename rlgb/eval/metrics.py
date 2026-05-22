"""Shared objective / metric functions.

Convention: lower is always better (cost functions).
Maximisation metrics (modularity, F1) are negated when used as objectives.

Public API
----------
  ncut(adj, labels)            -> float  (vectorised numpy, O(N²+N·K))
  ncut_torch(adj, labels)      -> torch.Tensor  (differentiable, GPU-compatible)
  h2(adj, labels)              -> float
  modularity(adj, labels)      -> float  (positive, higher=better — NOT negated)
  modularity_density(adj, labels) -> float
  conductance(adj, labels)     -> float  (lower=better)
  f1_community(pred, gt)       -> float  (higher=better)
  nmi(pred_labels, gt_labels)  -> float
  ari(pred_labels, gt_labels)  -> float
  compute_all(adj, labels, gt_labels?) -> dict[str, float]
"""
from __future__ import annotations

import numpy as np


# ── graph-partition metrics ──────────────────────────────────────────────────

def ncut(adj: np.ndarray, labels: np.ndarray) -> float:
    """Normalised Cut.  Sum over clusters of cut(S)/vol(S).

    Uses vectorised numpy matrix ops: O(N² + N·K) vs O(K·N²) for the
    naive loop.  ~3-5× faster on graphs with K ≥ 4.
    """
    adj = np.asarray(adj, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    N = adj.shape[0]
    if N == 0:
        return 0.0
    deg = adj.sum(axis=1)           # (N,)
    if deg.sum() == 0:
        return 0.0

    K = int(labels.max()) + 1
    # One-hot cluster membership  (N, K)
    mask = np.zeros((N, K), dtype=np.float64)
    mask[np.arange(N), labels] = 1.0

    vol = mask.T @ deg                           # (K,) — degree-sum per cluster
    within = np.einsum("in,ij,jn->n", mask, adj, mask)  # (K,) — within-cluster edge-weight
    cut = vol - within                           # (K,) — inter-cluster cut
    valid = vol > 1e-12
    if not valid.any():
        return 0.0
    return float((cut[valid] / vol[valid]).sum())


def ncut_torch(
    adj: "torch.Tensor",
    labels: "torch.Tensor",
) -> "torch.Tensor":
    """Differentiable Normalised Cut for torch GPU tensors.

    Enables gradient flow through a soft label tensor (one-hot or continuous).

    Args:
        adj:    (N, N) float tensor — dense adjacency.
        labels: (N,) long tensor OR (N, K) float tensor (soft assignment).

    Returns:
        Scalar tensor = NCut value (gradients flow through soft labels).
    """
    import torch
    N = adj.shape[0]
    if labels.dim() == 1:
        K = int(labels.max().item()) + 1
        mask = torch.zeros(N, K, dtype=adj.dtype, device=adj.device)
        mask.scatter_(1, labels.unsqueeze(1), 1.0)
    else:
        mask = labels.to(dtype=adj.dtype)   # (N, K) soft assignment

    deg = adj.sum(dim=1)                       # (N,)
    vol = mask.T @ deg                         # (K,)
    within = torch.einsum("in,ij,jn->n", mask, adj, mask)  # (K,) — within-cluster
    cut = vol - within                         # (K,)
    valid = vol > 1e-9
    if not valid.any():
        return adj.new_zeros(1).squeeze()
    return (cut[valid] / vol[valid]).sum()


def h2(adj: np.ndarray, labels: np.ndarray) -> float:
    """2-level structural entropy H²(G, P).

    H² = -Σ_j (g_j/2m)·log(vol_j/2m) - Σ_j Σ_{i∈Cj} (d_i/2m)·log(d_i/vol_j)

    Falls back to slow pure-numpy if glass is unavailable.
    """
    try:
        import sys
        if "/workspace/glass_shim" not in sys.path:
            sys.path.insert(0, "/workspace/glass_shim")
        from glass.seclust.entropy import structural_entropy
        return float(structural_entropy(adj, labels))
    except ImportError:
        pass
    # Pure-numpy fallback
    adj = np.asarray(adj, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    deg = adj.sum(axis=1)
    m2 = float(deg.sum())
    if m2 == 0:
        return 0.0
    clusters = np.unique(labels)
    h = 0.0
    for c in clusters:
        mask = labels == c
        vol_c = float(deg[mask].sum())
        if vol_c <= 0:
            continue
        g_c = float(adj[np.ix_(mask, ~mask)].sum())  # cut edges
        # Level-1 contribution: -(g_c/m2) * log(vol_c/m2)
        ratio = vol_c / m2
        if ratio > 0:
            h -= (g_c / m2) * np.log(ratio + 1e-12)
        # Level-2 contribution: -Σ_{i∈c} (d_i/m2) * log(d_i/vol_c)
        d_in = deg[mask]
        ratios = d_in / vol_c
        valid = ratios > 0
        if valid.any():
            h -= float(np.sum((d_in[valid] / m2) * np.log(ratios[valid] + 1e-12)))
    return float(h)


def conductance(adj: np.ndarray, labels: np.ndarray) -> float:
    """Mean cluster conductance (lower = better)."""
    adj = np.asarray(adj, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    deg = adj.sum(axis=1)
    total_vol = float(deg.sum())
    conds = []
    for c in np.unique(labels):
        mask = labels == c
        vol_s = float(deg[mask].sum())
        cut_s = float(adj[np.ix_(mask, ~mask)].sum())
        denom = min(vol_s, total_vol - vol_s)
        if denom > 0:
            conds.append(cut_s / denom)
    return float(np.mean(conds)) if conds else 0.0


def modularity(adj: np.ndarray, labels: np.ndarray) -> float:
    """Newman-Girvan modularity Q (higher = better, NOT negated here)."""
    adj = np.asarray(adj, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    m2 = float(adj.sum())
    if m2 == 0:
        return 0.0
    deg = adj.sum(axis=1)
    q = 0.0
    for c in np.unique(labels):
        mask = labels == c
        e_in = float(adj[np.ix_(mask, mask)].sum())
        a_in = float(deg[mask].sum())
        q += e_in / m2 - (a_in / m2) ** 2
    return float(q)


def modularity_density(adj: np.ndarray, labels: np.ndarray) -> float:
    """Modularity density (Li et al. 2008) — optimised by AC2CD.

    Q_ds = Σ_c [ (2m_c/m) - (n_c/n) · (2m_c + cut_c)/(m) ]
    where m_c = internal edges, cut_c = boundary edges.
    """
    adj = np.asarray(adj, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    m = float(adj.sum()) / 2  # undirected edge count
    n = adj.shape[0]
    if m == 0:
        return 0.0
    q_ds = 0.0
    for c in np.unique(labels):
        mask = labels == c
        n_c = int(mask.sum())
        m_c = float(adj[np.ix_(mask, mask)].sum()) / 2
        cut_c = float(adj[np.ix_(mask, ~mask)].sum())
        if n_c == 0:
            continue
        q_ds += (2 * m_c / m) - (n_c / n) * (2 * m_c + cut_c) / m
    return float(q_ds)


# ── community-detection metrics ──────────────────────────────────────────────

def f1_community(
    pred_community: list[int],
    gt_community: list[int],
) -> float:
    """F1 score between two community node sets."""
    pred_set = set(pred_community)
    gt_set   = set(gt_community)
    if not pred_set or not gt_set:
        return 0.0
    tp = len(pred_set & gt_set)
    p  = tp / len(pred_set)
    r  = tp / len(gt_set)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def mean_f1_communities(
    pred_communities: list[list[int]],
    gt_communities:   list[list[int]],
) -> float:
    """Best-match mean F1 across all GT communities."""
    if not pred_communities or not gt_communities:
        return 0.0
    scores = []
    for gt_c in gt_communities:
        best = max(f1_community(p, gt_c) for p in pred_communities)
        scores.append(best)
    return float(np.mean(scores))


def nmi(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(gt_labels, pred_labels,
                                                   average_method="arithmetic"))
    except Exception:
        return float("nan")


def ari(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    try:
        from sklearn.metrics import adjusted_rand_score
        return float(adjusted_rand_score(gt_labels, pred_labels))
    except Exception:
        return float("nan")


def acc(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    """Clustering accuracy via Hungarian assignment."""
    try:
        from scipy.optimize import linear_sum_assignment
        k = max(pred_labels.max(), gt_labels.max()) + 1
        mat = np.zeros((k, k), dtype=np.int64)
        for p, g in zip(pred_labels, gt_labels):
            mat[int(p), int(g)] += 1
        r, c = linear_sum_assignment(-mat)
        return float(mat[r, c].sum()) / len(pred_labels)
    except Exception:
        return float("nan")


# ── unified reporter ─────────────────────────────────────────────────────────

def compute_all(
    adj: np.ndarray,
    labels: np.ndarray,
    gt_labels: np.ndarray | None = None,
    gt_communities: list[list[int]] | None = None,
) -> dict[str, float]:
    """Compute all applicable metrics and return as a flat dict."""
    out: dict[str, float] = {}

    out["h2"]                = h2(adj, labels)
    out["ncut"]              = ncut(adj, labels)
    out["modularity"]        = modularity(adj, labels)
    out["modularity_density"] = modularity_density(adj, labels)
    out["conductance"]       = conductance(adj, labels)
    out["k_got"]             = float(len(np.unique(labels)))

    if gt_labels is not None:
        out["nmi"] = nmi(labels, gt_labels)
        out["ari"] = ari(labels, gt_labels)
        out["acc"] = acc(labels, gt_labels)

    if gt_communities is not None:
        pred_comms = [
            list(np.where(labels == c)[0]) for c in np.unique(labels)
        ]
        out["mean_f1"] = mean_f1_communities(pred_comms, gt_communities)
        out["f1"] = out["mean_f1"]  # alias used by SLRL/CLARE paper eval

    return out
