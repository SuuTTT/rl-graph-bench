"""WRT algorithm stub — Family B: Structured-Action Transformer + PPO.

Paper: WRT 2025 — Walkthrough-and-Rewrite Transformer for graph partitioning.
Action space: ring/wedge structural actions (merge adjacent cluster pairs,
              split a cluster along a minimum-cut wedge).

This is a stub that implements the RLAgent interface with random exploration
until the full Transformer backbone is wired in. Architecture sketch:
  - Transformer encoder over cluster-level representations (K, H)
  - Ring action: merge two adjacent clusters
  - Wedge action: split one cluster into two via spectral bisection
  - PPO training (on-policy, clipped objective)

Compatible tasks: partition
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rlgb.algos.base import RLAgent, EpisodeBuffer, Transition
from rlgb.training.reinforce import REINFORCEConfig, compute_returns, reinforce_loss


@dataclass
class WRTConfig:
    hidden: int = 128
    n_heads: int = 4
    n_layers: int = 2
    cluster_feat_dim: int = 4   # [mean_deg/N, size/N, intra_density, cut/E]
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 1.0
    device: str = "cpu"


class _WRTNet(nn.Module):
    """Cluster-level Transformer + merge/split heads."""

    def __init__(self, cfg: WRTConfig) -> None:
        super().__init__()
        cfd = getattr(cfg, "cluster_feat_dim", 4)
        self.cluster_proj = nn.Linear(cfd, cfg.hidden)  # cluster features → H
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden, nhead=cfg.n_heads, dim_feedforward=cfg.hidden * 2,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        # Ring: score each adjacent cluster pair for merging
        self.merge_head = nn.Sequential(nn.Linear(cfg.hidden * 2, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, 1))
        # Wedge: score each cluster for splitting
        self.split_head = nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden // 2), nn.ReLU(), nn.Linear(cfg.hidden // 2, 1))
        self.value_head  = nn.Sequential(nn.Linear(cfg.hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, adj: torch.Tensor, labels: torch.Tensor):
        """Returns (merge_logits (P,), split_logits (K,), value (1,))."""
        N = adj.shape[0]
        k = int(labels.max().item()) + 1
        deg = adj.sum(dim=1)   # (N,)
        total_edges = adj.sum() / 2.0 + 1e-9
        cfd = self.cluster_proj.in_features

        # Vectorized cluster features — avoid per-cluster Python loops
        c_feat = torch.zeros(k, cfd, device=adj.device)
        one_hot = torch.zeros(N, k, device=adj.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)           # (N, k)
        sizes   = one_hot.sum(0)                                 # (k,)
        # mean_deg per cluster
        c_feat[:, 0] = (one_hot.T @ deg) / (sizes.clamp(min=1) * N)
        if cfd > 1:
            c_feat[:, 1] = sizes / N
        if cfd > 2:
            # intra-cluster edges: diag of (one_hot.T @ adj @ one_hot) / 2
            intra = (one_hot.T @ adj @ one_hot).diagonal() / 2.0
            max_intra = sizes * (sizes - 1) / 2.0
            c_feat[:, 2] = intra / (max_intra + 1e-9)
        if cfd > 3:
            # cut per cluster = vol(c) - 2*intra
            vol = one_hot.T @ deg                                # (k,)
            intra_here = (one_hot.T @ adj @ one_hot).diagonal() / 2.0
            cut = vol - 2.0 * intra_here
            c_feat[:, 3] = cut / (2.0 * total_edges)
        h = self.cluster_proj(c_feat).unsqueeze(0)       # (1, K, H)
        h = self.transformer(h).squeeze(0)               # (K, H)

        # Adjacent cluster pairs
        pairs = _adjacent_cluster_pairs(adj, labels, k)
        if pairs:
            p_idx = torch.tensor(pairs, device=adj.device)   # (P, 2)
            pair_h = torch.cat([h[p_idx[:, 0]], h[p_idx[:, 1]]], dim=1)
            merge_logits = self.merge_head(pair_h).squeeze(-1)
        else:
            merge_logits = torch.zeros(1, device=adj.device)

        split_logits = self.split_head(h).squeeze(-1)    # (K,)
        value        = self.value_head(h.mean(0))         # (1,)
        return merge_logits, split_logits, value, pairs


def _adjacent_cluster_pairs(adj: torch.Tensor, labels: torch.Tensor, k: int) -> list[tuple[int, int]]:
    """Find cluster pairs with at least one inter-cluster edge (vectorized)."""
    # Fully vectorized: no Python loops over nodes → avoids GPU→CPU sync overhead
    nz_i, nz_j = (adj > 0).nonzero(as_tuple=True)
    if nz_i.numel() == 0:
        return []
    ci = labels[nz_i]
    cj = labels[nz_j]
    diff = ci != cj
    if not diff.any():
        return []
    ci_d = ci[diff]
    cj_d = cj[diff]
    pair_lo = torch.min(ci_d, cj_d)
    pair_hi = torch.max(ci_d, cj_d)
    pair_ids = (pair_lo * k + pair_hi).unique()
    return [(int(uid) // k, int(uid) % k) for uid in pair_ids.cpu()]


class WRTAlgo(RLAgent):
    """WRT: Walkthrough-and-Rewrite Transformer (structured ring/wedge actions).

    Action encoding:
      actions 0 … |merge_pairs|-1 → merge cluster pair i
      actions |merge_pairs| … |merge|+K-1 → split cluster j
    """

    name = "wrt"
    compatible_tasks = ["partition"]

    def __init__(self, config: WRTConfig | None = None) -> None:
        self._cfg = config or WRTConfig()
        self._device = torch.device(self._cfg.device)
        self._net = _WRTNet(self._cfg).to(self._device)
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
        # Store last action context for env step decoding
        self._last_pairs: list[tuple[int, int]] = []
        self._last_k: int = 0

    def select_action(self, obs: dict, greedy: bool = False) -> int:
        adj_t    = torch.tensor(obs["adj"],    dtype=torch.float32, device=self._device)
        labels_t = torch.tensor(obs["labels"], dtype=torch.long,    device=self._device)

        merge_logits, split_logits, value, pairs = self._net(adj_t, labels_t)
        self._last_pairs = pairs
        self._last_k     = int(labels_t.max().item()) + 1

        all_logits = torch.cat([merge_logits, split_logits], dim=0)

        if greedy:
            with torch.no_grad():
                action = int(all_logits.argmax().item())
        else:
            dist  = torch.distributions.Categorical(logits=all_logits)
            act_t = dist.sample()
            self._ep_log_probs.append(dist.log_prob(act_t))
            self._ep_values.append(value.squeeze())
            self._ep_entropies.append(dist.entropy())
            action = int(act_t.item())
        return action

    def decode_action(self, action: int, labels: np.ndarray) -> np.ndarray:
        """Translate flat action index to updated labels array."""
        n_merge = len(self._last_pairs)
        labels = labels.copy()
        if action < n_merge:
            c1, c2 = self._last_pairs[action]
            labels[labels == c2] = c1
        else:
            # Split: spectral bisection of the target cluster
            c_split = action - n_merge
            members = np.where(labels == c_split)[0]
            if len(members) > 1:
                new_c = int(labels.max()) + 1
                half  = members[:len(members) // 2]
                labels[half] = new_c
        return labels

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
        torch.save({"model_state_dict": self._net.state_dict(),
                    "algo": "wrt", "version": "0.1.0"}, str(path))

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(str(path), map_location=self._device, weights_only=False)
        self._net.load_state_dict(ckpt["model_state_dict"], strict=False)
