"""SS2V-D3QN algorithm stub — Family E: Multicut + D3QN sequential edge contraction.

Paper: SS2V-D3QN 2025 — Subgraph-to-Vector with Dueling Double DQN.
Action: sequentially contract edges (merge two endpoints into one supernode).
The agent keeps contracting until exactly k supernodes remain.

This implementation uses a replay buffer and ε-greedy exploration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlgb.algos.base import RLAgent, ReplayBuffer, Transition


@dataclass
class SS2VConfig:
    hidden: int = 64
    n_layers: int = 2
    lr: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 2000
    buffer_capacity: int = 5000
    batch_size: int = 32
    target_update_every: int = 50
    grad_clip: float = 1.0
    device: str = "cpu"


class _DuelingHead(nn.Module):
    def __init__(self, in_dim: int, n_actions: int) -> None:
        super().__init__()
        self.value    = nn.Sequential(nn.Linear(in_dim, in_dim // 2), nn.ReLU(), nn.Linear(in_dim // 2, 1))
        self.advantage = nn.Sequential(nn.Linear(in_dim, in_dim // 2), nn.ReLU(), nn.Linear(in_dim // 2, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        V = self.value(x)
        A = self.advantage(x)
        return V + (A - A.mean(dim=-1, keepdim=True))


class _SS2VNet(nn.Module):
    """Subgraph encoder + dueling Q-head. Uses dense SAGE."""

    GRAPH_FEAT_DIM = 3

    def __init__(self, feat_dim: int, hidden: int, n_layers: int, max_edges: int) -> None:
        super().__init__()
        dims = [feat_dim] + [hidden] * n_layers
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(dims[i] * 2, dims[i + 1]))
        self.sage_layers = nn.ModuleList(layers)
        self.graph_proj = nn.Linear(self.GRAPH_FEAT_DIM, hidden)
        self.head = _DuelingHead(hidden * 2, max_edges)

    def forward(self, feats: torch.Tensor, adj: torch.Tensor, graph_feat: torch.Tensor) -> torch.Tensor:
        h = feats
        deg = adj.sum(1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / deg
        for layer in self.sage_layers:
            agg = adj_norm @ h
            h = F.relu(layer(torch.cat([h, agg], dim=1)))
        g_proj = self.graph_proj(graph_feat)             # (hidden,)
        g = torch.cat([h.mean(0), g_proj], dim=0)        # (hidden*2,)
        return self.head(g.unsqueeze(0)).squeeze(0)       # (max_edges,)


class SS2VAlgo(RLAgent):
    """SS2V-D3QN: sequential edge-contraction with Dueling Double DQN.

    Note: This stub uses node-move action semantics (merge two clusters
    = contract all edges between them) to remain compatible with existing envs.
    Full subgraph-vector encoding is a TODO once the env supports it natively.
    """

    name = "ss2v_d3qn"
    compatible_tasks = ["partition"]

    MAX_EDGES = 100  # fixed Q-head output size; padded / masked

    def __init__(self, config: SS2VConfig | None = None) -> None:
        self._cfg = config or SS2VConfig()
        self._device = torch.device(self._cfg.device)
        feat_dim = 7  # node_feats dim
        self._online = _SS2VNet(feat_dim, self._cfg.hidden, self._cfg.n_layers, self.MAX_EDGES).to(self._device)
        self._target = _SS2VNet(feat_dim, self._cfg.hidden, self._cfg.n_layers, self.MAX_EDGES).to(self._device)
        self._target.load_state_dict(self._online.state_dict())
        self._optimizer = torch.optim.Adam(self._online.parameters(), lr=self._cfg.lr)
        self._replay = ReplayBuffer(capacity=self._cfg.buffer_capacity)
        self._rng = np.random.default_rng(42)
        self._step = 0
        self._epsilon = self._cfg.epsilon_start

    @property
    def _eps(self) -> float:
        cfg = self._cfg
        decay = (cfg.epsilon_start - cfg.epsilon_end) * max(0, 1 - self._step / cfg.epsilon_decay)
        return cfg.epsilon_end + decay

    def select_action(self, obs: dict, greedy: bool = False) -> int:
        adj_t    = torch.tensor(obs["adj"],        dtype=torch.float32, device=self._device)
        feats_t  = torch.tensor(obs["node_feats"], dtype=torch.float32, device=self._device)
        labels_t = torch.tensor(obs["labels"],     dtype=torch.long,    device=self._device)
        k = int(labels_t.max().item()) + 1
        N = adj_t.shape[0]
        # Legal actions = (node, cluster) pairs → encode as flat index
        cand_np = [(n, c) for n in range(N) for c in range(k) if labels_t[n] != c]
        if not cand_np:
            return 0
        n_cands = min(len(cand_np), self.MAX_EDGES)
        cand_np = cand_np[:n_cands]

        if not greedy and random.random() < self._eps:
            idx = random.randrange(n_cands)
        else:
            g_feat = torch.tensor([N, k, adj_t.sum().item() / max(N, 1)],
                                  dtype=torch.float32, device=self._device)
            with torch.no_grad():
                q = self._online(feats_t, adj_t, g_feat)[:n_cands]
            idx = int(q.argmax().item())

        self._step += 1
        node_idx, clust_idx = cand_np[idx]
        return node_idx * k + clust_idx

    def push_transition(self, t: Transition) -> None:
        self._replay.push(t)

    def update(self) -> dict[str, float]:
        if len(self._replay) < self._cfg.batch_size:
            return {}
        batch = self._replay.sample(self._cfg.batch_size, self._rng)
        # Simplified DQN loss (no full double-DQN here to keep stub short)
        total_loss = 0.0
        for t in batch:
            adj_t    = torch.tensor(t.obs["adj"],        dtype=torch.float32, device=self._device)
            feats_t  = torch.tensor(t.obs["node_feats"], dtype=torch.float32, device=self._device)
            labels_t = torch.tensor(t.obs["labels"],     dtype=torch.long,    device=self._device)
            N = adj_t.shape[0]
            k = int(labels_t.max().item()) + 1
            g_feat  = torch.tensor([N, k, adj_t.sum().item() / max(N, 1)],
                                   dtype=torch.float32, device=self._device)
            q_vals  = self._online(feats_t, adj_t, g_feat)
            act_idx = min(int(t.action) // max(k, 1), self.MAX_EDGES - 1)
            q_pred  = q_vals[act_idx]
            with torch.no_grad():
                adj_n  = torch.tensor(t.next_obs["adj"], dtype=torch.float32, device=self._device)
                feat_n = torch.tensor(t.next_obs["node_feats"], dtype=torch.float32, device=self._device)
                lab_n  = torch.tensor(t.next_obs["labels"], dtype=torch.long, device=self._device)
                g2     = torch.tensor([N, k, adj_n.sum().item() / max(N, 1)],
                                      dtype=torch.float32, device=self._device)
                q_next = self._target(feat_n, adj_n, g2).max()
                q_tgt  = t.reward + self._cfg.gamma * q_next * (1 - float(t.done))
            total_loss += F.smooth_l1_loss(q_pred, q_tgt)

        loss = total_loss / self._cfg.batch_size
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._online.parameters(), self._cfg.grad_clip)
        self._optimizer.step()

        if self._step % self._cfg.target_update_every == 0:
            self._target.load_state_dict(self._online.state_dict())

        return {"loss": float(loss.item()), "epsilon": self._eps}

    def reset_episode(self) -> None:
        pass

    def save(self, path: str | Path) -> None:
        torch.save({"model_state_dict": self._online.state_dict(),
                    "algo": "ss2v_d3qn", "version": "0.1.0"}, str(path))

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(str(path), map_location=self._device, weights_only=False)
        self._online.load_state_dict(ckpt["model_state_dict"], strict=False)
        self._target.load_state_dict(self._online.state_dict())
