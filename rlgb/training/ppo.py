"""PPO Trainer — on-policy Proximal Policy Optimization loop.

Designed for structured-action algorithms like WRTAlgo.

Usage::

    from rlgb.training.ppo import PPOTrainer, PPOConfig
    from rlgb.algos.structured.wrt import WRTAlgo

    trainer = PPOTrainer(
        algo=WRTAlgo(),
        env_fn=lambda: task.build_env(problem, horizon=10),
        config=PPOConfig(n_episodes=1000),
    )
    trainer.train()
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from rlgb.algos.base import RLAgent, Transition


@dataclass
class PPOConfig:
    """Configuration for PPOTrainer.

    Args:
        n_episodes: Total training episodes.
        horizon: Max steps per episode.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_eps: PPO clip epsilon (ε).
        n_epochs: Number of PPO optimisation epochs per update.
        minibatch_size: Minibatch size within each PPO epoch.
        lr: Learning rate (overrides algo's internal lr when > 0).
        value_coef: Coefficient for value loss.
        entropy_coef: Coefficient for entropy bonus.
        grad_clip: Max gradient norm.
        n_episodes_per_update: Collect this many episodes before each PPO update.
        log_every: Print metrics every N episodes.
        save_every: Save checkpoint every N episodes (0 = only at end).
        out_dir: Directory for checkpoints and logs.
        device: Torch device string.
        seed: Master RNG seed.
    """
    n_episodes: int = 1000
    horizon: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    n_epochs: int = 4
    minibatch_size: int = 32
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 1.0
    n_episodes_per_update: int = 4
    log_every: int = 20
    save_every: int = 200
    out_dir: str = "results"
    device: str = "cpu"
    seed: int = 0
    lr_schedule: str = "none"   # "none" | "cosine" | "linear"
    lr_min_ratio: float = 0.1


class _Rollout:
    """Stores one collected rollout for PPO update."""
    __slots__ = ("obs_list", "actions", "log_probs", "values", "rewards",
                 "dones", "advantages", "returns")

    def __init__(self) -> None:
        self.obs_list:  list[dict]          = []
        self.actions:   list[int]           = []
        self.log_probs: list[torch.Tensor]  = []
        self.values:    list[torch.Tensor]  = []
        self.rewards:   list[float]         = []
        self.dones:     list[bool]          = []
        self.advantages: torch.Tensor | None = None
        self.returns:    torch.Tensor | None = None


def _compute_gae(
    rewards: list[float],
    values: list[torch.Tensor],
    dones: list[bool],
    gamma: float,
    lam: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised Advantage Estimation (Schulman 2015).

    Returns:
        advantages: (T,) tensor, normalised
        returns:    (T,) tensor (= advantages + values)
    """
    T = len(rewards)
    vals = torch.stack(values).detach().to(device)  # (T,)
    adv  = torch.zeros(T, device=device)
    last_gae = 0.0
    for t in reversed(range(T)):
        delta    = rewards[t] + (gamma * vals[t + 1].item() if t + 1 < T and not dones[t] else 0.0) - vals[t].item()
        last_gae = delta + gamma * lam * (0.0 if dones[t] else last_gae)
        adv[t]   = last_gae
    returns = adv + vals
    # Normalise advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, returns


class PPOTrainer:
    """On-policy PPO training loop.

    The algorithm's network must expose:
      - ``select_action_with_logprob(obs, greedy=False) → (action, log_prob, value, entropy)``
        OR the standard ``select_action`` API (in which case old log-probs are
        approximated; useful for REINFORCE-style algos switching to PPO).
    
    When the algo does not implement ``select_action_with_logprob``, PPOTrainer
    falls back to REINFORCE-style collection using the algo's own ``update()``
    method after every ``n_episodes_per_update`` episodes — making it safe to
    use with any RLAgent.

    Args:
        algo: RLAgent to train.
        env_fn: Zero-arg callable returning a fresh Gymnasium environment.
        config: PPOConfig hyperparameters.
    """

    def __init__(
        self,
        algo: RLAgent,
        env_fn: Callable,
        config: PPOConfig | None = None,
    ) -> None:
        self._algo   = algo
        self._env_fn = env_fn
        self._cfg    = config or PPOConfig()
        self._device = torch.device(self._cfg.device)
        Path(self._cfg.out_dir).mkdir(parents=True, exist_ok=True)
        self._rng = np.random.default_rng(self._cfg.seed)

        # Check if algo supports full PPO interface
        self._ppo_mode: bool = hasattr(algo, "select_action_with_logprob") and hasattr(algo, "ppo_update")

    # ── public API ────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop."""
        if self._ppo_mode:
            self._train_ppo()
        else:
            self._train_reinforce_fallback()

    def eval(
        self,
        suite: list | None = None,
        task=None,
        n_seeds: int = 3,
        horizon: int | None = None,
    ):
        """Evaluate the trained algo using the eval harness."""
        import pandas as pd
        if suite is None or task is None:
            return pd.DataFrame()
        from rlgb.eval.harness import eval_algo_on_suite
        return eval_algo_on_suite(
            self._algo, suite, task,
            n_seeds=n_seeds,
            horizon=horizon or self._cfg.horizon,
        )

    # ── REINFORCE fallback (works with any RLAgent) ────────────────────────────

    def _train_reinforce_fallback(self) -> None:
        """Collect episodes and call algo.update() — standard REINFORCE loop."""
        n_update = self._cfg.n_episodes_per_update
        running_reward: float = 0.0
        t0 = time.perf_counter()

        for ep in range(1, self._cfg.n_episodes + 1):
            env  = self._env_fn()
            obs, _ = env.reset(seed=int(self._rng.integers(0, 2**31)))
            self._algo.reset_episode()
            ep_reward = 0.0

            for _ in range(self._cfg.horizon):
                action = self._algo.select_action(obs, greedy=False)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                self._algo.push_transition(Transition(
                    obs=obs, action=action, reward=float(reward),
                    next_obs=next_obs, done=bool(terminated or truncated),
                ))
                obs         = next_obs
                ep_reward  += float(reward)
                if terminated or truncated:
                    break
            env.close()

            running_reward = 0.9 * running_reward + 0.1 * ep_reward

            if ep % n_update == 0:
                self._algo.update()

            if ep % self._cfg.log_every == 0:
                elapsed = time.perf_counter() - t0
                print(f"[PPO/REINFORCE] ep={ep}/{self._cfg.n_episodes}  "
                      f"reward={running_reward:.4f}  t={elapsed:.1f}s")

            if self._cfg.save_every > 0 and ep % self._cfg.save_every == 0:
                self._save(ep)

        self._save("last")

    # ── True PPO (requires algo.select_action_with_logprob + algo.ppo_update) ──

    def _train_ppo(self) -> None:
        """Full PPO loop with GAE, clipped objective, and value function loss."""
        import torch.optim.lr_scheduler as _lr_sched
        cfg = self._cfg
        n_update = cfg.n_episodes_per_update
        rollout  = _Rollout()
        running_reward = 0.0
        t0 = time.perf_counter()

        # Optional LR schedule
        _opt = getattr(self._algo, "_optimizer", None)
        if _opt and cfg.lr_schedule == "cosine":
            _scheduler = _lr_sched.CosineAnnealingLR(
                _opt, T_max=cfg.n_episodes,
                eta_min=cfg.lr * cfg.lr_min_ratio)
        elif _opt and cfg.lr_schedule == "linear":
            _scheduler = _lr_sched.LinearLR(
                _opt, start_factor=1.0,
                end_factor=cfg.lr_min_ratio,
                total_iters=cfg.n_episodes)
        else:
            _scheduler = None

        for ep in range(1, cfg.n_episodes + 1):
            env  = self._env_fn()
            obs, _ = env.reset(seed=int(self._rng.integers(0, 2**31)))
            self._algo.reset_episode()
            ep_reward = 0.0

            for _ in range(self._cfg.horizon):
                action, log_prob, value, _ = self._algo.select_action_with_logprob(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                rollout.obs_list.append(obs)
                rollout.actions.append(action)
                rollout.log_probs.append(log_prob.detach())
                rollout.values.append(value.detach().squeeze())
                rollout.rewards.append(float(reward))
                rollout.dones.append(bool(terminated or truncated))
                obs        = next_obs
                ep_reward += float(reward)
                if terminated or truncated:
                    break
            env.close()

            running_reward = 0.9 * running_reward + 0.1 * ep_reward

            if ep % n_update == 0 and rollout.obs_list:
                advs, rets = _compute_gae(
                    rollout.rewards, rollout.values, rollout.dones,
                    cfg.gamma, cfg.gae_lambda, self._device,
                )
                self._algo.ppo_update(
                    obs_list=rollout.obs_list,
                    actions=rollout.actions,
                    old_log_probs=torch.stack(rollout.log_probs),
                    advantages=advs,
                    returns=rets,
                    clip_eps=cfg.clip_eps,
                    n_epochs=cfg.n_epochs,
                    minibatch_size=cfg.minibatch_size,
                    value_coef=cfg.value_coef,
                    entropy_coef=cfg.entropy_coef,
                    grad_clip=cfg.grad_clip,
                )
                if _scheduler is not None:
                    _scheduler.step()
                rollout = _Rollout()

            if ep % cfg.log_every == 0:
                elapsed = time.perf_counter() - t0
                print(f"[PPO] ep={ep}/{cfg.n_episodes}  "
                      f"reward={running_reward:.4f}  t={elapsed:.1f}s")

            if cfg.save_every > 0 and ep % cfg.save_every == 0:
                self._save(ep)

        self._save("last")

    def _save(self, tag: str | int) -> None:
        p = Path(self._cfg.out_dir) / f"ppo_{tag}.pt"
        self._algo.save(str(p))
