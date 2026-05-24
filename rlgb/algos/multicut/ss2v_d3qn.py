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
    """SAGE encoder + edge-level Q-values.

    Q-value for each candidate edge is computed from the embeddings of the
    two endpoint nodes, so the network can differentiate edges by their local
    structure rather than position.  This is critical for identifying which
    pairs of leiden sub-clusters belong to the same community.
    """

    GRAPH_FEAT_DIM = 3

    def __init__(self, feat_dim: int, hidden: int, n_layers: int, max_edges: int) -> None:
        super().__init__()
        self._max_edges = max_edges
        dims = [feat_dim] + [hidden] * n_layers
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(dims[i] * 2, dims[i + 1]))
        self.sage_layers = nn.ModuleList(layers)
        self.graph_proj = nn.Linear(self.GRAPH_FEAT_DIM, hidden)
        # Edge scorer: (h_u + h_v || h_u * h_v || g) → Q
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        feats: torch.Tensor,
        adj: torch.Tensor,
        graph_feat: torch.Tensor,
        edge_idx: torch.Tensor,     # (n_cands, 2) — node indices for each candidate
    ) -> torch.Tensor:
        h = feats
        deg = adj.sum(1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / deg
        for layer in self.sage_layers:
            agg = adj_norm @ h
            h = F.relu(layer(torch.cat([h, agg], dim=1)))

        g = self.graph_proj(graph_feat)  # (hidden,)

        n_cands = edge_idx.shape[0]
        if n_cands == 0:
            return torch.zeros(self._max_edges, device=feats.device)

        u_emb = h[edge_idx[:, 0]]                              # (n_cands, hidden)
        v_emb = h[edge_idx[:, 1]]                              # (n_cands, hidden)
        g_exp = g.unsqueeze(0).expand(n_cands, -1)             # (n_cands, hidden)
        ef    = torch.cat([u_emb + v_emb, u_emb * v_emb, g_exp], dim=1)  # (n_cands, hidden*3)
        q     = self.edge_scorer(ef).squeeze(1)                # (n_cands,)

        # Pad to fixed max_edges size (unused slots will be masked by caller)
        padded = torch.zeros(self._max_edges, device=feats.device)
        n = min(n_cands, self._max_edges)
        padded[:n] = q[:n]
        return padded


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
        """Select an inter-cluster edge to contract."""
        adj_np    = obs["adj"]
        labels_np = obs["labels"]
        N = len(labels_np)
        k = int(labels_np.max()) + 1

        # Use pre-computed n_edges if available, else compute from obs
        if "n_edges" in obs:
            n_edges = int(obs["n_edges"][0])
        else:
            rows, cols = np.where(adj_np > 0)
            inter_mask = (labels_np[rows] != labels_np[cols]) & (rows < cols)
            n_edges = int(inter_mask.sum())

        if n_edges == 0:
            return 0

        n_cands = min(n_edges, self.MAX_EDGES)

        if not greedy and random.random() < self._eps:
            idx = random.randrange(n_cands)
        else:
            adj_t   = torch.from_numpy(np.ascontiguousarray(adj_np, dtype=np.float32)).to(self._device)
            feats_t = torch.from_numpy(np.ascontiguousarray(obs["node_feats"], dtype=np.float32)).to(self._device)
            g_feat  = torch.tensor([N, k, float(adj_np.sum()) / max(N, 1)],
                                   dtype=torch.float32, device=self._device)
            if "edge_idx" in obs and obs["edge_idx"].shape[0] > 0:
                eidx_t = torch.from_numpy(
                    np.ascontiguousarray(obs["edge_idx"][:n_cands], dtype=np.int64)
                ).to(self._device)
            else:
                eidx_t = torch.zeros((0, 2), dtype=torch.int64, device=self._device)
            with torch.no_grad():
                q = self._online(feats_t, adj_t, g_feat, eidx_t)[:n_cands]
            idx = int(q.argmax().item())

        self._step += 1
        return idx

    def push_transition(self, t: Transition) -> None:
        self._replay.push(t)

    def update(self) -> dict[str, float]:
        if len(self._replay) < self._cfg.batch_size:
            return {}
        batch = self._replay.sample(self._cfg.batch_size, self._rng)
        total_loss = 0.0
        for t in batch:
            # Use numpy for cheap CPU metadata to avoid GPU→CPU syncs
            labels_np = t.obs["labels"]
            k         = int(labels_np.max()) + 1
            N         = len(labels_np)
            adj_sum   = float(np.sum(t.obs["adj"]))

            adj_t    = torch.from_numpy(np.ascontiguousarray(t.obs["adj"], dtype=np.float32)).to(self._device)
            feats_t  = torch.from_numpy(np.ascontiguousarray(t.obs["node_feats"], dtype=np.float32)).to(self._device)
            labels_t = torch.from_numpy(np.ascontiguousarray(labels_np, dtype=np.int64)).to(self._device)
            g_feat   = torch.tensor([N, k, adj_sum / max(N, 1)],
                                    dtype=torch.float32, device=self._device)
            def _eidx(ob: dict, n_valid: int) -> torch.Tensor:
                if "edge_idx" in ob and ob["edge_idx"].shape[0] > 0:
                    return torch.from_numpy(
                        np.ascontiguousarray(ob["edge_idx"][:n_valid], dtype=np.int64)
                    ).to(self._device)
                return torch.zeros((0, 2), dtype=torch.int64, device=self._device)

            n_valid_cur = int(t.obs.get("n_edges", [self.MAX_EDGES])[0])
            n_valid_cur = max(1, min(n_valid_cur, self.MAX_EDGES))
            eidx_cur    = _eidx(t.obs, n_valid_cur)
            q_vals  = self._online(feats_t, adj_t, g_feat, eidx_cur)
            act_idx = min(int(t.action), self.MAX_EDGES - 1)
            q_pred  = q_vals[act_idx]
            with torch.no_grad():
                lab_n_np = t.next_obs["labels"]
                k_n      = int(lab_n_np.max()) + 1
                adj_n_sum = float(np.sum(t.next_obs["adj"]))
                adj_n  = torch.from_numpy(np.ascontiguousarray(t.next_obs["adj"], dtype=np.float32)).to(self._device)
                feat_n = torch.from_numpy(np.ascontiguousarray(t.next_obs["node_feats"], dtype=np.float32)).to(self._device)
                g2     = torch.tensor([N, k_n, adj_n_sum / max(N, 1)],
                                      dtype=torch.float32, device=self._device)
                n_valid_next = int(t.next_obs.get("n_edges", [self.MAX_EDGES])[0])
                n_valid_next = max(1, min(n_valid_next, self.MAX_EDGES))
                eidx_next     = _eidx(t.next_obs, n_valid_next)
                q_online_next = self._online(feat_n, adj_n, g2, eidx_next)
                best_next_act = int(q_online_next[:n_valid_next].argmax().item())
                q_next = self._target(feat_n, adj_n, g2, eidx_next)[best_next_act]
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
