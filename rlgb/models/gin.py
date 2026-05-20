"""GIN-based encoder for community expand/exclude RL (CLARE architecture).

Dense GIN (no PyG required). Input: (N, input_dim) node features + (N,N) adj.
Output: (N, hidden_dim) node embeddings for scoring EXPAND/EXCLUDE candidates.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GINLayer(nn.Module):
    """Dense GIN: h_v = MLP((1+ε)*h_v + Σ_{u∈N(v)} h_u)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        agg = adj_norm @ x
        return self.mlp((1 + self.eps) * x + agg)


class GINEncoder(nn.Module):
    """Multi-layer GIN encoder. Input dim padded to input_dim with zeros."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 64, n_layers: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        dims = [input_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList([GINLayer(dims[i], dims[i + 1]) for i in range(n_layers)])

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """x: (N, F), adj_norm: (N,N) row-normalised. Returns (N, hidden_dim)."""
        # Pad or truncate to input_dim
        N, F = x.shape
        if F < self.input_dim:
            x = torch.cat([x, torch.zeros(N, self.input_dim - F, device=x.device)], dim=1)
        elif F > self.input_dim:
            x = x[:, :self.input_dim]
        h = x
        for layer in self.layers:
            h = layer(h, adj_norm)
        return h

    @staticmethod
    def row_normalize(adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        return adj / deg


class CLARENet(nn.Module):
    """Full CLARE model: GIN encoder + ExcludeNet/ExpandNet scoring heads.

    Forward returns (exclude_logits, expand_logits) over ALL N nodes.
    Scoring feature = GIN embedding || membership_flag → (hidden+1)-dim.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = GINEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        score_dim = hidden_dim + 1
        self.exclude_net = nn.Sequential(
            nn.Linear(score_dim, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)
        )
        self.expand_net = nn.Sequential(
            nn.Linear(score_dim, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(
        self,
        adj: torch.Tensor,       # (N, N)
        node_feats: torch.Tensor, # (N, F)
        labels: torch.Tensor,    # (N,) int
        cluster_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (exclude_logits (N,), expand_logits (N,), value (1,))."""
        adj_norm = GINEncoder.row_normalize(adj)
        emb = self.encoder(node_feats, adj_norm)          # (N, hidden)
        membership = (labels == cluster_id).float().unsqueeze(1)  # (N, 1)
        feat = torch.cat([emb, membership], dim=1)        # (N, hidden+1)
        exc = self.exclude_net(feat).squeeze(-1)          # (N,)
        exp = self.expand_net(feat).squeeze(-1)           # (N,)
        val = self.value_head(emb.mean(0))                # (1,)
        return exc, exp, val
