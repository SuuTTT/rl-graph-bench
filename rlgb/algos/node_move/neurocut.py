"""NeuroCUT algorithm — Family A: NodeMove + GraphSAGE + REINFORCE.

Implementation faithful to:
  NeuroCUT: A Neural Approach for Robust Graph Partitioning
  Rishi Shah et al., KDD 2024, arXiv:2310.11787

Algorithm
---------
  1. Encode nodes with GraphSAGE (2 layers, hidden=64).
  2. Score each legal (node, cluster) pair via PairScorer.
  3. Sample action proportional to softmax scores (training) or argmax (eval).
  4. Update via REINFORCE with value-function baseline.
  5. Compatible tasks: GraphPartition (H², NCut, Balanced, Sparsest).

Compatible tasks : partition
Compatible envs  : NodeMoveEnv
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rlgb.algos.base import RLAgent, EpisodeBuffer, Transition
from rlgb.models.sage import NeuroCUTPolicy, SAGEConfig
from rlgb.training.reinforce import REINFORCEConfig, reinforce_loss, compute_returns


@dataclass
class NeuroCUTConfig:
    # Model
    node_feat_dim: int = 7
    hidden: int = 64
    n_layers: int = 2
    pool: str = "mean"
    dropout: float = 0.0
    # Training
    lr: float = 1e-3
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 1.0
    normalize_returns: bool = True
    # Device
    device: str = "cpu"


class NeuroCUTAlgo(RLAgent):
    """NeuroCUT RL agent.

    Can be used transductively (train per graph) or inductively (train on
    a suite, evaluate zero-shot on new graphs).

    Parameters
    ----------
    config : NeuroCUTConfig
    checkpoint : str | Path | None
        If given, loads pre-trained weights from this path at construction.

    Usage
    -----
        algo = NeuroCUTAlgo()
        trainer = Trainer(algo, env_fn, TrainConfig(n_episodes=300))
        trainer.train()
        df = trainer.eval(fixed17())
    """

    name = "neurocut"
    compatible_tasks = ["partition"]

    def __init__(
        self,
        config: NeuroCUTConfig | None = None,
        checkpoint: str | Path | None = None,
    ) -> None:
        self._cfg = config or NeuroCUTConfig()
        self._device = torch.device(self._cfg.device)

        sage_cfg = SAGEConfig(
            node_feat_dim=self._cfg.node_feat_dim,
            hidden=self._cfg.hidden,
            n_layers=self._cfg.n_layers,
            pool=self._cfg.pool,
            dropout=self._cfg.dropout,
        )
        self._policy = NeuroCUTPolicy(sage_cfg).to(self._device)
        self._optimizer = torch.optim.Adam(
            self._policy.parameters(), lr=self._cfg.lr
        )
        self._rl_cfg = REINFORCEConfig(
            gamma=self._cfg.gamma,
            entropy_coef=self._cfg.entropy_coef,
            value_coef=self._cfg.value_coef,
            grad_clip=self._cfg.grad_clip,
            normalize_returns=self._cfg.normalize_returns,
        )
        self._buffer = EpisodeBuffer()

        # Episode-level accumulators (filled by _episode_data hook)
        self._ep_log_probs:  list[torch.Tensor] = []
        self._ep_values:     list[torch.Tensor] = []
        self._ep_entropies:  list[torch.Tensor] = []
        self._ep_rewards:    list[float] = []

        if checkpoint is not None:
            self.load(checkpoint)

    # ── RLAgent protocol ──────────────────────────────────────────────────────

    def select_action(
        self, obs: dict[str, np.ndarray], greedy: bool = False
    ) -> int:
        """Sample a legal (node, cluster) action from the policy."""
        adj_t    = torch.tensor(obs["adj"],        dtype=torch.float32, device=self._device)
        feats_t  = torch.tensor(obs["node_feats"], dtype=torch.float32, device=self._device)
        labels_t = torch.tensor(obs["labels"],     dtype=torch.long,    device=self._device)
        k_target = max(int(labels_t.max().item()) + 1, int(round(float(obs["k"][0]))))

        # Extract actual candidates (exclude zero-padded rows)
        n_cands  = int(obs.get("n_candidates", [obs["candidates"].shape[0]])[0])
        cands_np = obs["candidates"][:n_cands]
        if n_cands == 0:
            return 0  # no-op
        cands_t  = torch.tensor(cands_np, dtype=torch.long, device=self._device)

        if greedy:
            with torch.no_grad():
                logits, _ = self._policy(adj_t, feats_t, labels_t, cands_t, k_override=k_target)
            idx = int(logits.argmax().item())
        else:
            self._policy.train()
            logits, value = self._policy(adj_t, feats_t, labels_t, cands_t, k_override=k_target)
            dist = torch.distributions.Categorical(logits=logits)
            idx_t = dist.sample()
            self._ep_log_probs.append(dist.log_prob(idx_t))
            self._ep_values.append(value.squeeze())
            self._ep_entropies.append(dist.entropy())
            idx = int(idx_t.item())

        # Convert candidate index → flat action index (node * K + cluster)
        node_idx, clust_idx = int(cands_np[idx, 0]), int(cands_np[idx, 1])
        flat_action = node_idx * k_target + clust_idx
        return flat_action

    def push_transition(self, t: Transition) -> None:
        """Record reward from env step (log_probs stored during select_action)."""
        self._ep_rewards.append(t.reward)
        self._buffer.push(t)

    def update(self) -> dict[str, float]:
        """Compute REINFORCE loss over accumulated episode data and step optimizer."""
        if not self._ep_rewards or not self._ep_log_probs:
            return {}

        returns = compute_returns(self._ep_rewards, self._rl_cfg.gamma)
        loss, metrics = reinforce_loss(
            self._ep_log_probs,
            self._ep_values,
            self._ep_entropies,
            returns,
            self._rl_cfg,
            self._device,
        )
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self._policy.parameters(), self._rl_cfg.grad_clip
        )
        self._optimizer.step()

        # Clear episode accumulators
        self._ep_log_probs.clear()
        self._ep_values.clear()
        self._ep_entropies.clear()
        self._ep_rewards.clear()
        self._buffer.drain()
        return metrics

    def reset_episode(self) -> None:
        self._ep_log_probs.clear()
        self._ep_values.clear()
        self._ep_entropies.clear()
        self._ep_rewards.clear()

    # ── PPO interface ─────────────────────────────────────────────────────────

    def select_action_with_logprob(
        self, obs: dict[str, np.ndarray], greedy: bool = False
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO collection interface: returns (flat_action, log_prob, value, entropy).

        Does NOT modify episode buffers — PPOTrainer manages its own rollout.
        """
        adj_t    = torch.tensor(obs["adj"],        dtype=torch.float32, device=self._device)
        feats_t  = torch.tensor(obs["node_feats"], dtype=torch.float32, device=self._device)
        labels_t = torch.tensor(obs["labels"],     dtype=torch.long,    device=self._device)
        k_target = max(int(labels_t.max().item()) + 1, int(round(float(obs["k"][0]))))
        n_cands  = int(obs.get("n_candidates", [obs["candidates"].shape[0]])[0])
        cands_np = obs["candidates"][:n_cands]

        if n_cands == 0:
            zero = next(self._policy.parameters()).new_zeros(1).squeeze()
            return 0, zero.detach(), zero.detach(), zero.detach()

        cands_t = torch.tensor(cands_np, dtype=torch.long, device=self._device)
        self._policy.train()
        logits, value = self._policy(adj_t, feats_t, labels_t, cands_t, k_override=k_target)
        dist = torch.distributions.Categorical(logits=logits)
        idx_t = logits.argmax() if greedy else dist.sample()
        log_prob = dist.log_prob(idx_t)
        entropy  = dist.entropy()

        idx       = int(idx_t.item())
        node_idx  = int(cands_np[idx, 0])
        clust_idx = int(cands_np[idx, 1])
        return node_idx * k_target + clust_idx, log_prob, value.squeeze(), entropy

    def _evaluate_action(
        self, obs: dict[str, np.ndarray], flat_action: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate policy at *obs* for *flat_action*; used in PPO epochs.

        Returns: (log_prob, value, entropy) with gradients intact.
        """
        adj_t    = torch.tensor(obs["adj"],        dtype=torch.float32, device=self._device)
        feats_t  = torch.tensor(obs["node_feats"], dtype=torch.float32, device=self._device)
        labels_t = torch.tensor(obs["labels"],     dtype=torch.long,    device=self._device)
        k_target = max(int(labels_t.max().item()) + 1, int(round(float(obs["k"][0]))))
        n_cands  = int(obs.get("n_candidates", [obs["candidates"].shape[0]])[0])
        cands_np = obs["candidates"][:n_cands]

        if n_cands == 0:
            zero = next(self._policy.parameters()).new_zeros(1)
            return zero.squeeze(), zero.squeeze(), zero.squeeze()

        cands_t = torch.tensor(cands_np, dtype=torch.long, device=self._device)
        self._policy.train()
        logits, value = self._policy(adj_t, feats_t, labels_t, cands_t, k_override=k_target)
        dist = torch.distributions.Categorical(logits=logits)

        # Decode flat_action → candidate index
        node_idx  = flat_action // k_target
        clust_idx = flat_action % k_target
        cand_idx  = 0  # fallback if action not found (e.g. env changed)
        for i in range(len(cands_np)):
            if int(cands_np[i, 0]) == node_idx and int(cands_np[i, 1]) == clust_idx:
                cand_idx = i
                break

        idx_t    = torch.tensor(cand_idx, dtype=torch.long, device=self._device)
        return dist.log_prob(idx_t), value.squeeze(), dist.entropy()

    def ppo_update(
        self,
        obs_list:      list[dict],
        actions:       list[int],
        old_log_probs: torch.Tensor,
        advantages:    torch.Tensor,
        returns:       torch.Tensor,
        clip_eps:      float = 0.2,
        n_epochs:      int   = 4,
        minibatch_size: int  = 32,
        value_coef:    float = 0.5,
        entropy_coef:  float = 0.01,
        grad_clip:     float = 1.0,
    ) -> dict[str, float]:
        """PPO clipped surrogate + value loss update.

        Processes each transition individually (graph sizes vary) and
        accumulates gradients over *minibatch_size* transitions before
        calling ``optimizer.step()``.
        """
        T = len(obs_list)
        if T == 0:
            return {}

        old_log_probs = old_log_probs.to(self._device).detach()
        advantages    = advantages.to(self._device)
        returns_t     = returns.to(self._device)

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            for start in range(0, T, minibatch_size):
                batch = slice(start, min(start + minibatch_size, T))
                batch_idx = range(*batch.indices(T))
                self._optimizer.zero_grad()
                p_loss = torch.zeros(1, device=self._device)
                v_loss = torch.zeros(1, device=self._device)
                h_sum  = torch.zeros(1, device=self._device)
                for i in batch_idx:
                    lp, val, ent = self._evaluate_action(obs_list[i], actions[i])
                    ratio  = torch.exp(lp - old_log_probs[i])
                    adv    = advantages[i]
                    surr   = torch.min(ratio * adv,
                                       ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
                    p_loss = p_loss - surr
                    v_loss = v_loss + 0.5 * (val - returns_t[i]) ** 2
                    h_sum  = h_sum  + ent
                n = max(len(list(batch_idx)), 1)
                loss = (p_loss / n
                        + value_coef  * v_loss / n
                        - entropy_coef * h_sum / n)
                loss.backward()
                nn.utils.clip_grad_norm_(self._policy.parameters(), grad_clip)
                self._optimizer.step()
                total_policy_loss += (p_loss / n).item()
                total_value_loss  += (v_loss / n).item()
                total_entropy     += (h_sum  / n).item()
                n_updates += 1

        return {
            "ppo/policy_loss": total_policy_loss / max(n_updates, 1),
            "ppo/value_loss":  total_value_loss  / max(n_updates, 1),
            "ppo/entropy":     total_entropy      / max(n_updates, 1),
        }

    def save(self, path: str | Path) -> None:
        torch.save({
            "policy_state_dict": self._policy.state_dict(),
            "policy_config": {
                "node_feat_dim": self._cfg.node_feat_dim,
                "hidden":        self._cfg.hidden,
                "n_layers":      self._cfg.n_layers,
                "pool":          self._cfg.pool,
            },
            "algo": "neurocut",
            "version": "0.1.0",
        }, str(path))

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(str(path), map_location=self._device, weights_only=False)
        # Support both new format and rl-cluster-ops legacy format
        key = "policy_state_dict" if "policy_state_dict" in ckpt else "state_dict"
        self._policy.load_state_dict(ckpt[key], strict=False)
        self._policy.eval()

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str = "cpu"
    ) -> "NeuroCUTAlgo":
        """Load a checkpoint and return a ready-to-eval NeuroCUTAlgo."""
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        pol_cfg = ckpt.get("policy_config", {})
        cfg = NeuroCUTConfig(
            node_feat_dim=pol_cfg.get("node_feat_dim", 7),
            hidden=pol_cfg.get("hidden", 64),
            n_layers=pol_cfg.get("n_layers", pol_cfg.get("n_gnn_layers", 2)),
            pool=pol_cfg.get("pool", "mean"),
            device=device,
        )
        algo = cls(config=cfg)
        key = "policy_state_dict" if "policy_state_dict" in ckpt else "state_dict"
        algo._policy.load_state_dict(ckpt[key], strict=False)
        algo._policy.eval()
        return algo
