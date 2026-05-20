"""DQN Trainer — off-policy Q-learning loop.

Designed for SS2VAlgo (Dueling Double DQN) and other value-based algos.

Unlike the on-policy Trainer, DQNTrainer:
  - Collects transitions step-by-step (not episode-by-episode)
  - Calls algo.update() every ``update_every`` environment steps
  - Uses ε-greedy exploration (managed by the algo's own _eps property)
  - Supports warm-up period before updates begin

Usage::

    from rlgb.training.dqn_trainer import DQNTrainer, DQNConfig
    from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo

    trainer = DQNTrainer(
        algo=SS2VAlgo(),
        env_fn=lambda: task.build_env(problem, horizon=10),
        config=DQNConfig(n_steps=20000),
    )
    trainer.train()
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from rlgb.algos.base import RLAgent, Transition


@dataclass
class DQNConfig:
    """Configuration for DQNTrainer.

    Args:
        n_steps: Total environment steps.
        horizon: Max steps per episode (also env's own episode length).
        warmup_steps: Steps to collect before calling update (fills replay buffer).
        update_every: Call algo.update() every N steps.
        log_every: Print metrics every N episodes.
        save_every: Save checkpoint every N episodes (0 = only at end).
        out_dir: Directory for checkpoints.
        seed: Master RNG seed.
    """
    n_steps: int = 20000
    horizon: int = 10
    warmup_steps: int = 500
    update_every: int = 4
    log_every: int = 50
    save_every: int = 500
    out_dir: str = "results"
    seed: int = 0


class DQNTrainer:
    """Off-policy DQN training loop.

    Args:
        algo: SS2VAlgo or any off-policy RLAgent with a replay buffer.
        env_fn: Zero-arg callable returning a fresh Gymnasium environment.
        config: DQNConfig hyperparameters.
    """

    def __init__(
        self,
        algo: RLAgent,
        env_fn: Callable,
        config: DQNConfig | None = None,
    ) -> None:
        self._algo   = algo
        self._env_fn = env_fn
        self._cfg    = config or DQNConfig()
        Path(self._cfg.out_dir).mkdir(parents=True, exist_ok=True)
        self._rng    = np.random.default_rng(self._cfg.seed)

    def train(self) -> None:
        """Run the full off-policy training loop."""
        total_steps = 0
        episode     = 0
        running_reward = 0.0
        t0 = time.perf_counter()

        while total_steps < self._cfg.n_steps:
            env  = self._env_fn()
            obs, _ = env.reset(seed=int(self._rng.integers(0, 2**31)))
            self._algo.reset_episode()
            ep_reward = 0.0
            episode  += 1

            for _ in range(self._cfg.horizon):
                greedy = total_steps < self._cfg.warmup_steps  # exploit only after warmup
                action  = self._algo.select_action(obs, greedy=False)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                self._algo.push_transition(Transition(
                    obs=obs, action=action, reward=float(reward),
                    next_obs=next_obs, done=bool(terminated or truncated),
                ))

                obs         = next_obs
                ep_reward  += float(reward)
                total_steps += 1

                if total_steps > self._cfg.warmup_steps and total_steps % self._cfg.update_every == 0:
                    self._algo.update()

                if terminated or truncated:
                    break

            env.close()
            running_reward = 0.9 * running_reward + 0.1 * ep_reward

            if episode % self._cfg.log_every == 0:
                elapsed = time.perf_counter() - t0
                eps_val = getattr(self._algo, "_eps", None)
                eps_str = f"  ε={eps_val:.3f}" if eps_val is not None else ""
                print(f"[DQN] step={total_steps}/{self._cfg.n_steps}  "
                      f"ep={episode}  reward={running_reward:.4f}{eps_str}  t={elapsed:.1f}s")

            if self._cfg.save_every > 0 and episode % self._cfg.save_every == 0:
                self._save(episode)

        self._save("last")

    def eval(
        self,
        suite: list | None = None,
        task=None,
        n_seeds: int = 3,
        horizon: int | None = None,
    ):
        """Evaluate using the eval harness."""
        import pandas as pd
        if suite is None or task is None:
            return pd.DataFrame()
        from rlgb.eval.harness import eval_algo_on_suite
        return eval_algo_on_suite(
            self._algo, suite, task,
            n_seeds=n_seeds,
            horizon=horizon or self._cfg.horizon,
        )

    def _save(self, tag: str | int) -> None:
        p = Path(self._cfg.out_dir) / f"dqn_{tag}.pt"
        self._algo.save(str(p))
