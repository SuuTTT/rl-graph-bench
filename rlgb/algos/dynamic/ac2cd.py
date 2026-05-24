"""AC2CD algorithm stub — Family D: Dynamic Graph + GAT + A2C.

Paper: AC2CD (2023) — Adaptive Community Change Detection.
Architecture: GAT encoder + Actor-Critic (A2C) on temporal snapshots.

This stub shares the same RLAgent interface. Uses GAT for node encoding
and the same node-move action space as NeuroCUT, making it compatible
with NodeMoveEnv / DynamicCDEnv.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlgb.algos.base import RLAgent, EpisodeBuffer, Transition
from rlgb.models.gat import GATEncoder
from rlgb.training.reinforce import REINFORCEConfig, compute_returns, reinforce_loss


@dataclass
class AC2CDConfig:
    node_feat_dim: int = 7
    hidden: int = 64
    n_layers: int = 2
    n_heads: int = 4
    lr: float = 1e-3
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 1.0
    device: str = "cpu"


class _AC2CDNet(nn.Module):
    """GAT encoder + actor (pair scorer) + critic (value head)."""

    def __init__(self, cfg: AC2CDConfig) -> None:
        super().__init__()
        self.encoder = GATEncoder(
            in_dim=cfg.node_feat_dim, hidden=cfg.hidden,
            n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        )
        self.actor = nn.Sequential(
            nn.Linear(cfg.hidden * 2, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden // 2), nn.ReLU(), nn.Linear(cfg.hidden // 2, 1)
        )

    def forward(self, adj, feats, labels, candidates):
        N = feats.shape[0]
        k = int(labels.max().item()) + 1
        h = self.encoder(feats, adj)   # (N, hidden)

        # Cluster mean embeddings
        c_sum = torch.zeros(k, h.shape[1], device=h.device)
        c_cnt = torch.zeros(k, 1, device=h.device)
        c_sum.index_add_(0, labels, h)
        c_cnt.index_add_(0, labels, torch.ones(N, 1, device=h.device))
        c_emb = c_sum / c_cnt.clamp(min=1.0)

        if candidates.shape[0] == 0:
            logits = torch.zeros(1, device=h.device)
            value  = self.critic(h.mean(0, keepdim=True))
            return logits, value

        n_idx = candidates[:, 0]
        c_idx = candidates[:, 1]
        pair  = torch.cat([h[n_idx], c_emb[c_idx]], dim=1)
        logits = self.actor(pair).squeeze(-1)
        value  = self.critic(h.mean(0, keepdim=True))
        return logits, value


class AC2CDAlgo(RLAgent):
    """AC2CD: Adaptive Community Change Detection (A2C variant, NodeMove actions)."""

    name = "ac2cd"
    compatible_tasks = ["dynamic", "partition"]

    def __init__(self, config: AC2CDConfig | None = None) -> None:
        self._cfg = config or AC2CDConfig()
        self._device = torch.device(self._cfg.device)
        self._net = _AC2CDNet(self._cfg).to(self._device)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._cfg.lr)
        self._rl_cfg = REINFORCEConfig(
            gamma=self._cfg.gamma, entropy_coef=self._cfg.entropy_coef,
            value_coef=self._cfg.value_coef, grad_clip=self._cfg.grad_clip,
        )
        self._ep_log_probs: list[torch.Tensor] = []
        self._ep_values:    list[torch.Tensor] = []
        self._ep_entropies: list[torch.Tensor] = []
        self._ep_rewards:   list[float] = []
        self._buffer = EpisodeBuffer()

    def select_action(self, obs: dict, greedy: bool = False) -> int:
        adj_np    = np.ascontiguousarray(obs["adj"],        dtype=np.float32)
        feats_np  = np.ascontiguousarray(obs["node_feats"], dtype=np.float32)
        adj_t    = torch.from_numpy(adj_np).to(self._device)
        feats_t  = torch.from_numpy(feats_np).to(self._device)
        labels_np = obs["labels"]          # numpy array — use on CPU to avoid GPU syncs
        labels_t  = torch.from_numpy(np.ascontiguousarray(labels_np, dtype=np.int64)).to(self._device)
        k = int(labels_np.max()) + 1
        N = adj_t.shape[0]
        # Vectorized candidate generation (use numpy to avoid 200 GPU→CPU syncs)
        if "candidates" in obs and "n_candidates" in obs:
            # Use precomputed candidates from NodeMoveEnv
            n_cands = int(obs["n_candidates"][0])
            cands_np = obs["candidates"][:n_cands]
        else:
            node_idx, clust_idx = np.meshgrid(np.arange(N), np.arange(k), indexing="ij")
            valid = labels_np[node_idx] != clust_idx
            cands_np = np.stack([node_idx[valid], clust_idx[valid]], axis=1)
        if len(cands_np) == 0:
            return 0
        cands = torch.tensor(cands_np, dtype=torch.long, device=self._device)

        if greedy:
            with torch.no_grad():
                logits, _ = self._net(adj_t, feats_t, labels_t, cands)
            act_i = int(logits.argmax().item())
        else:
            logits, value = self._net(adj_t, feats_t, labels_t, cands)
            dist  = torch.distributions.Categorical(logits=logits)
            act_t = dist.sample()
            self._ep_log_probs.append(dist.log_prob(act_t))
            self._ep_values.append(value.squeeze())
            self._ep_entropies.append(dist.entropy())
            act_i = int(act_t.item())

        node_idx, clust_idx = int(cands[act_i, 0]), int(cands[act_i, 1])
        return node_idx * k + clust_idx

    def push_transition(self, t: Transition) -> None:
        self._ep_rewards.append(t.reward)
        self._buffer.push(t)

    def update(self) -> dict[str, float]:
        if not self._ep_rewards or not self._ep_log_probs:
            return {}
        returns = compute_returns(self._ep_rewards, self._rl_cfg.gamma)
        loss, metrics = reinforce_loss(
            self._ep_log_probs, self._ep_values, self._ep_entropies,
            returns, self._rl_cfg, self._device,
        )
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._net.parameters(), self._rl_cfg.grad_clip)
        self._optimizer.step()
        self._ep_log_probs.clear(); self._ep_values.clear()
        self._ep_entropies.clear(); self._ep_rewards.clear()
        self._buffer.drain()
        return metrics

    def reset_episode(self) -> None:
        self._ep_log_probs.clear(); self._ep_values.clear()
        self._ep_entropies.clear(); self._ep_rewards.clear()

    def save(self, path: str | Path) -> None:
        torch.save({
            "model_state_dict": self._net.state_dict(),
            "config": vars(self._cfg),
            "algo": "ac2cd", "version": "0.1.0",
        }, str(path))

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(str(path), map_location=self._device, weights_only=False)
        self._net.load_state_dict(ckpt["model_state_dict"], strict=False)
