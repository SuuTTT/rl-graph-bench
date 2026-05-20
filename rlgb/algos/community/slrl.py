"""SLRL algorithm — Family C: Seed-based Local RL for community detection.

Seed-and-expand policy using Swish-MLP embeddings.
Paper: SLRL (Zhu et al., 2025).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlgb.algos.base import RLAgent, EpisodeBuffer, Transition
from rlgb.training.reinforce import REINFORCEConfig, compute_returns, reinforce_loss


class _Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class _LinearBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.residual = in_size == out_size
        self.f = nn.Sequential(_Swish(), nn.Linear(in_size, out_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.f(x)
        return z + x if self.residual else z


class SLRLNet(nn.Module):
    """Seed-expand policy: seed_emb + node_emb → (K+1,) logits (last=stop)."""

    def __init__(self, hidden: int = 64, n_feat: int = 1) -> None:
        super().__init__()
        self.seed_emb  = nn.Linear(n_feat, hidden, bias=False)
        self.node_emb  = nn.Linear(n_feat, hidden, bias=False)
        self.backbone  = nn.Sequential(
            _LinearBlock(hidden, hidden), _LinearBlock(hidden, hidden)
        )
        self.node_score = nn.Linear(hidden, 1, bias=False)
        self.stop_score = nn.Linear(hidden, 2, bias=True)
        nn.init.zeros_(self.node_score.weight)
        nn.init.zeros_(self.stop_score.weight)
        self.stop_score.bias.data[0] = 4.0  # prefer continue at start

        self.value_head = nn.Sequential(
            nn.Linear(hidden, 32), _Swish(), nn.Linear(32, 1)
        )

    def forward(
        self,
        seed_feat: torch.Tensor,   # (K,) scalar degree features
        cand_feat: torch.Tensor,   # (K,) candidate features
        comm_emb: torch.Tensor,    # (H,) mean community embedding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits (K+1,), value (1,))."""
        K = cand_feat.shape[0]
        sf = seed_feat.unsqueeze(1) if seed_feat.dim() == 1 else seed_feat   # (K,1)
        cf = cand_feat.unsqueeze(1) if cand_feat.dim() == 1 else cand_feat
        h = self.seed_emb(sf) + self.node_emb(cf)    # (K, H)
        h = self.backbone(h)
        node_logits = F.log_softmax(self.node_score(h).squeeze(1), dim=0)  # (K,)
        stop_log = F.log_softmax(
            self.stop_score(comm_emb.unsqueeze(0)), dim=1
        ).squeeze(0)   # (2,)
        logits = torch.cat([node_logits + stop_log[0], stop_log[1:]], dim=0)  # (K+1,)
        value  = self.value_head(comm_emb)  # (1,)
        return logits, value

    def embed(self, seed_feat: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        sf = seed_feat.unsqueeze(1) if seed_feat.dim() == 1 else seed_feat
        cf = cand_feat.unsqueeze(1) if cand_feat.dim() == 1 else cand_feat
        h = self.seed_emb(sf) + self.node_emb(cf)
        return self.backbone(h)


@dataclass
class SLRLConfig:
    hidden: int = 64
    n_feat: int = 1
    lr: float = 1e-3
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 1.0
    device: str = "cpu"


class SLRLAlgo(RLAgent):
    """SLRL RL agent for community detection (seed-expand policy).

    Uses degree as the single node feature (n_feat=1).
    Expects observations from CommunityEnv.
    """

    name = "slrl"
    compatible_tasks = ["community"]

    def __init__(self, config: SLRLConfig | None = None) -> None:
        self._cfg = config or SLRLConfig()
        self._device = torch.device(self._cfg.device)
        self._model = SLRLNet(hidden=self._cfg.hidden, n_feat=self._cfg.n_feat).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._cfg.lr)
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
        adj  = obs["adj"]
        deg  = torch.tensor(adj.sum(axis=1), dtype=torch.float32, device=self._device)
        deg  = deg / (deg.max().clamp(min=1.0))  # normalize to [0,1]

        n_exc = int(obs["n_exclude"][0])
        n_exp = int(obs["n_expand"][0])
        exc_nodes = obs["exclude_nodes"][:n_exc].tolist()
        exp_nodes = obs["expand_nodes"][:n_exp].tolist()
        all_cands = exc_nodes + exp_nodes
        K = len(all_cands)

        if K == 0:
            return K  # STOP

        cand_idx = torch.tensor(all_cands, dtype=torch.long, device=self._device)
        cand_feat = deg[cand_idx]      # (K,)
        # seed = cluster member with highest degree
        labels_t = torch.tensor(obs["labels"], dtype=torch.long, device=self._device)
        cluster_id = int(labels_t[exc_nodes[0]].item()) if exc_nodes else 0
        members = (labels_t == cluster_id).nonzero(as_tuple=True)[0]
        seed_node = int(members[deg[members].argmax()].item()) if len(members) else 0
        seed_feat = deg[seed_node].expand(K)   # broadcast seed feature

        # Mean community embedding for value + stop scoring
        emb_h = self._model.embed(seed_feat, cand_feat)  # (K, H)
        if len(members):
            mem_feat = deg[members]
            mem_emb = self._model.embed(mem_feat.expand_as(mem_feat), mem_feat)
            comm_emb = mem_emb.mean(0)
        else:
            comm_emb = emb_h.mean(0)

        if greedy:
            with torch.no_grad():
                logits, _ = self._model(seed_feat, cand_feat, comm_emb)
            action = int(logits.argmax().item())
        else:
            logits, value = self._model(seed_feat, cand_feat, comm_emb)
            dist = torch.distributions.Categorical(logits=logits)
            act_t = dist.sample()
            self._ep_log_probs.append(dist.log_prob(act_t))
            self._ep_values.append(value.squeeze())
            self._ep_entropies.append(dist.entropy())
            action = int(act_t.item())

        # Map K→STOP to env's stop token (total cands)
        return action if action < K else K

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
        nn.utils.clip_grad_norm_(self._model.parameters(), self._rl_cfg.grad_clip)
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
            "model_state_dict": self._model.state_dict(),
            "config": {"hidden": self._cfg.hidden, "n_feat": self._cfg.n_feat},
            "algo": "slrl", "version": "0.1.0",
        }, str(path))

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(str(path), map_location=self._device, weights_only=False)
        self._model.load_state_dict(ckpt["model_state_dict"], strict=False)
