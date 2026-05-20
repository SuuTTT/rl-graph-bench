"""GraphSAGE-based policy for node-move clustering RL.

Architecture
------------
  Input : (N, F) node features + (N, N) dense adjacency
  Layer 1: SAGEConv(F → hidden) + ReLU
  Layer 2: SAGEConv(hidden → hidden) + ReLU
  Candidate scoring: for each (node, cluster) pair,
      score = MLP(h_node || mean_cluster_embedding)
  Value head: MLP(mean_graph_embedding → 1)

Works with torch_geometric for GPU-efficient ops.
Falls back to dense matrix-multiply SAGE if torch_geometric unavailable.

Adapted from rl-cluster-ops/src/policy/h2cut_policy.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAGEConfig:
    node_feat_dim: int = 7
    hidden: int = 64
    n_layers: int = 2
    pool: str = "mean"     # "mean" | "max" | "sum"
    dropout: float = 0.0


# ── SAGE layer (dense, no PyG dependency) ─────────────────────────────────────

class _DenseSAGELayer(nn.Module):
    """GraphSAGE message-passing with dense adjacency.

    h_v = Linear([h_v || mean_{u∈N(v)} h_u])
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # adj_norm: (N, N) row-normalised (D^{-1} A, no self-loops)
        agg = adj_norm @ x                          # (N, in_dim)
        out = self.lin(torch.cat([x, agg], dim=-1)) # (N, out_dim)
        return out


class _TryPyGSAGELayer(nn.Module):
    """SAGEConv via torch_geometric if available; else falls back to dense."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import SAGEConv
            self._pyg = SAGEConv(in_dim, out_dim, normalize=True)
            self._use_pyg = True
        except ImportError:
            self._dense = _DenseSAGELayer(in_dim, out_dim)
            self._use_pyg = False

    def forward(
        self,
        x: torch.Tensor,
        adj_or_edge_index,
        edge_weight=None,
    ) -> torch.Tensor:
        if self._use_pyg:
            # Convert dense (N, N) adj to sparse edge_index if needed
            if adj_or_edge_index.dtype != torch.long or adj_or_edge_index.dim() == 2 and adj_or_edge_index.shape[0] == adj_or_edge_index.shape[1]:
                edge_index = adj_or_edge_index.nonzero(as_tuple=False).t().contiguous().long()
                edge_weight = adj_or_edge_index[adj_or_edge_index != 0].float() if edge_weight is None else edge_weight
            else:
                edge_index = adj_or_edge_index
            return self._pyg(x, edge_index)
        # dense fallback: adj_or_edge_index is (N, N) dense adj
        adj = adj_or_edge_index.float()
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / deg
        return self._dense(x, adj_norm)


# ── GraphSAGE encoder ─────────────────────────────────────────────────────────

class GraphSAGEEncoder(nn.Module):
    """Multi-layer GraphSAGE that outputs (N, hidden) node embeddings."""

    def __init__(self, cfg: SAGEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dims = [cfg.node_feat_dim] + [cfg.hidden] * cfg.n_layers
        self.layers = nn.ModuleList([
            _TryPyGSAGELayer(dims[i], dims[i + 1])
            for i in range(cfg.n_layers)
        ])
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        adj_or_edge_index,
        edge_weight=None,
    ) -> torch.Tensor:
        """Return (N, hidden) node embeddings."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj_or_edge_index, edge_weight)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    @staticmethod
    def adj_to_edge_index(adj: torch.Tensor):
        """Convert dense (N, N) adj to PyG edge_index (2, E)."""
        idx = adj.nonzero(as_tuple=False).t().contiguous()
        return idx


# ── Scoring head: (node, cluster) pair scorer ────────────────────────────────

class PairScorer(nn.Module):
    """Score (node_embedding || cluster_mean_embedding) for candidate moves."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        node_emb: torch.Tensor,         # (M, H)  — candidate node embeddings
        cluster_emb: torch.Tensor,      # (M, H)  — mean embedding of target cluster
    ) -> torch.Tensor:
        return self.mlp(torch.cat([node_emb, cluster_emb], dim=-1)).squeeze(-1)  # (M,)


# ── Value head ────────────────────────────────────────────────────────────────

class ValueHead(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:  # (1,) → (1,)
        return self.mlp(graph_emb)


# ── NeuroCUT policy (full model) ──────────────────────────────────────────────

class NeuroCUTPolicy(nn.Module):
    """Full NeuroCUT policy: encoder + pair scorer + value head.

    forward(adj, node_feats, labels, candidates) → (logits, value)

    Parameters
    ----------
    candidates : (M, 2) int64 tensor of (node_idx, target_cluster) pairs.
    """

    def __init__(self, cfg: SAGEConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = SAGEConfig()
        self.cfg = cfg
        self.encoder = GraphSAGEEncoder(cfg)
        self.scorer  = PairScorer(cfg.hidden)
        self.value   = ValueHead(cfg.hidden)

    def forward(
        self,
        adj: torch.Tensor,           # (N, N) float32
        node_feats: torch.Tensor,    # (N, F) float32
        labels: torch.Tensor,        # (N,) int64
        candidates: torch.Tensor,    # (M, 2) int64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = adj.shape[0]
        k = int(labels.max().item()) + 1

        # Node embeddings
        h = self.encoder(node_feats, adj)  # (N, hidden)

        # Cluster mean embeddings
        # cluster_emb[c] = mean of h[nodes in cluster c]
        cluster_sum = torch.zeros(k, self.cfg.hidden, device=h.device)
        cluster_cnt = torch.zeros(k, 1, device=h.device)
        cluster_sum.index_add_(0, labels, h)
        cluster_cnt.index_add_(0, labels, torch.ones(N, 1, device=h.device))
        cluster_emb = cluster_sum / cluster_cnt.clamp(min=1.0)   # (K, hidden)

        # Score each (node, cluster) candidate
        if candidates.shape[0] == 0:
            # No legal moves: return uniform logits
            logits = torch.zeros(1, device=h.device)
            value  = self.value(h.mean(0, keepdim=True))
            return logits, value

        node_idx   = candidates[:, 0]    # (M,)
        clust_idx  = candidates[:, 1]    # (M,)
        node_emb   = h[node_idx]          # (M, hidden)
        clust_mean = cluster_emb[clust_idx]  # (M, hidden)
        logits = self.scorer(node_emb, clust_mean)  # (M,)

        # Graph-level value
        if self.cfg.pool == "max":
            g_emb, _ = h.max(0, keepdim=True)
        elif self.cfg.pool == "sum":
            g_emb = h.sum(0, keepdim=True)
        else:
            g_emb = h.mean(0, keepdim=True)
        value = self.value(g_emb)  # (1, 1)

        return logits, value

    def select_greedy(
        self,
        adj: torch.Tensor,
        node_feats: torch.Tensor,
        labels: torch.Tensor,
        candidates: torch.Tensor,
    ) -> int:
        """Return argmax action index (into candidates list)."""
        with torch.no_grad():
            logits, _ = self.forward(adj, node_feats, labels, candidates)
        return int(logits.argmax().item())
