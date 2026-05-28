"""CLARE algorithm — Family C: CommunityRW + GIN + REINFORCE.

Per-cluster EXPAND/EXCLUDE rewriting, trained via REINFORCE with baseline.

Paper: CLARE (Peng et al., KDD 2022)
Adaptation: H² objective instead of modularity (configurable).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlgb.algos.base import RLAgent, EpisodeBuffer, Transition
from rlgb.models.gin import CLARENet
from rlgb.training.reinforce import REINFORCEConfig, compute_returns, reinforce_loss


@dataclass
class CLAREConfig:
    input_dim: int = 64
    hidden_dim: int = 64
    hidden: int = 64  # alias for hidden_dim (convenience)
    lr: float = 1e-3
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 1.0
    device: str = "cpu"


class CLAREAlgo(RLAgent):
    """CLARE RL agent for community expand/exclude.

    Expects observations from CommunityEnv with keys:
      adj, node_feats, labels, exclude_nodes, expand_nodes, n_exclude, n_expand.
    """

    name = "clare"
    compatible_tasks = ["community"]

    def __init__(self, config: CLAREConfig | None = None) -> None:
        self._cfg = config or CLAREConfig()
        # Allow hidden= shorthand
        hidden = self._cfg.hidden if self._cfg.hidden != 64 else self._cfg.hidden_dim
        self._device = torch.device(self._cfg.device)
        self._model = CLARENet(
            input_dim=self._cfg.input_dim, hidden_dim=hidden
        ).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._cfg.lr)
        self._rl_cfg = REINFORCEConfig(
            gamma=self._cfg.gamma,
            entropy_coef=self._cfg.entropy_coef,
            value_coef=self._cfg.value_coef,
            grad_clip=self._cfg.grad_clip,
        )
        self._ep_log_probs: list[torch.Tensor] = []
        self._ep_values:    list[torch.Tensor] = []
        self._ep_entropies: list[torch.Tensor] = []
        self._ep_rewards:   list[float] = []
        self._buffer = EpisodeBuffer()

    def select_action(self, obs: dict, greedy: bool = False) -> int:
        adj_t    = torch.tensor(obs["adj"],        dtype=torch.float32, device=self._device)
        feats_t  = torch.tensor(obs["node_feats"], dtype=torch.float32, device=self._device)
        labels_t = torch.tensor(obs["labels"],     dtype=torch.long,    device=self._device)

        n_exc = int(obs["n_exclude"][0])
        n_exp = int(obs["n_expand"][0])
        exc_nodes = obs["exclude_nodes"][:n_exc].tolist()
        exp_nodes = obs["expand_nodes"][:n_exp].tolist()
        total = n_exc + n_exp

        if total == 0:
            if not greedy:
                zero = torch.zeros(1, device=self._device, requires_grad=True).squeeze()
                self._ep_log_probs.append(zero)
                self._ep_values.append(zero)
                self._ep_entropies.append(zero)
            return total  # STOP

        # Determine current cluster from labels of first exclude candidate
        cluster_id = int(labels_t[exc_nodes[0]].item()) if exc_nodes else 0

        exc_logits, exp_logits, value = self._model(adj_t, feats_t, labels_t, cluster_id)

        # Gather relevant logits
        cand_logits = []
        if exc_nodes:
            exc_idx = torch.tensor(exc_nodes, dtype=torch.long, device=self._device)
            cand_logits.append(exc_logits[exc_idx])
        if exp_nodes:
            exp_idx = torch.tensor(exp_nodes, dtype=torch.long, device=self._device)
            cand_logits.append(exp_logits[exp_idx])
        all_logits = torch.cat(cand_logits, dim=0)  # (total,)

        if greedy:
            with torch.no_grad():
                action = int(all_logits.argmax().item())
        else:
            dist = torch.distributions.Categorical(logits=all_logits)
            act_t = dist.sample()
            self._ep_log_probs.append(dist.log_prob(act_t))
            self._ep_values.append(value.squeeze())
            self._ep_entropies.append(dist.entropy())
            action = int(act_t.item())

        return action

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
            "config": {"input_dim": self._cfg.input_dim, "hidden_dim": self._cfg.hidden_dim},
            "algo": "clare", "version": "0.1.0",
        }, str(path))

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(str(path), map_location=self._device, weights_only=False)
        self._model.load_state_dict(ckpt["model_state_dict"], strict=False)
