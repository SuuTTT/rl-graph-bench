"""REINFORCE with baseline — shared training logic used by NodeMove, Community,
and other on-policy agents.

This module provides the pure-function `reinforce_update()` that any agent
can call.  It takes a list of completed-episode transitions, computes
discounted returns, policy gradient + value baseline + entropy bonus,
and returns gradients + metrics.

Agents own their own optimizer; they call `reinforce_update()` and then
call `optimizer.step()` themselves.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlgb.algos.base import Transition


@dataclass
class REINFORCEConfig:
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 1.0
    normalize_returns: bool = True


def compute_returns(rewards: list[float], gamma: float) -> list[float]:
    R = 0.0
    returns: list[float] = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    return returns


def reinforce_loss(
    log_probs: list[torch.Tensor],
    values: list[torch.Tensor],
    entropies: list[torch.Tensor],
    returns: list[float],
    cfg: REINFORCEConfig,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute REINFORCE-with-baseline loss.

    Returns (loss, metrics_dict).
    """
    T = len(log_probs)
    if T == 0:
        z = torch.zeros(1, device=device, requires_grad=True)
        return z.sum(), {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}

    ret_t = torch.tensor(returns, dtype=torch.float32, device=device)

    if cfg.normalize_returns and ret_t.std() > 1e-6:
        ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

    lp_t  = torch.stack(log_probs)          # (T,)
    val_t = torch.stack(values).squeeze(-1) # (T,)
    ent_t = torch.stack(entropies)          # (T,)

    advantages = (ret_t - val_t.detach())
    pg_loss    = -(lp_t * advantages).mean()
    v_loss     = F.mse_loss(val_t, ret_t)
    entropy    = ent_t.mean()

    loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy

    return loss, {
        "pg_loss":  float(pg_loss.item()),
        "v_loss":   float(v_loss.item()),
        "entropy":  float(entropy.item()),
        "loss":     float(loss.item()),
    }
