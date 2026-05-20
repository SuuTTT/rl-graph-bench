"""GAT encoder for dynamic community detection (AC2CD architecture).

Multi-head graph attention over dense adjacency.
Used by DynamicEnv / AC2CD algo.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class GATLayer(nn.Module):
    """Single multi-head GAT layer on dense adjacency."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.a_tgt = nn.Parameter(torch.empty(n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_tgt.unsqueeze(0))
        self.lrelu = nn.LeakyReLU(0.2)
        self.act = _Swish()

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        N = h.shape[0]
        h_w = self.W(h).view(N, self.n_heads, self.head_dim)
        score_src = (h_w * self.a_src).sum(-1)   # (N, H)
        score_tgt = (h_w * self.a_tgt).sum(-1)
        e = self.lrelu(score_src.unsqueeze(1) + score_tgt.unsqueeze(0))  # (N, N, H)
        no_edge = (adj == 0) & (~torch.eye(N, dtype=torch.bool, device=h.device))
        e = e.masked_fill(no_edge.unsqueeze(2), float("-inf"))
        alpha = torch.softmax(e, dim=1).nan_to_num(0.0)   # (N, N, H)
        h_agg = torch.einsum("ijk,jkd->ikd", alpha, h_w)  # (N, H, D)
        return self.act(h_agg.reshape(N, -1))


class GATEncoder(nn.Module):
    """Stacked GAT layers. adj is (N,N) dense binary."""

    def __init__(self, in_dim: int = 7, hidden: int = 64, n_layers: int = 2, n_heads: int = 4) -> None:
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.layers = nn.ModuleList([GATLayer(dims[i], dims[i+1], n_heads) for i in range(n_layers)])

    def forward(self, feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = feats
        for layer in self.layers:
            h = layer(h, adj)
        return h
