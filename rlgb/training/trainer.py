"""Unified REINFORCE trainer (on-policy, episode-based).

Works for any RLAgent that accumulates transitions via push_transition()
and calls update() at episode end.

Design
------
  - Collects full episodes, then calls algo.update() once per episode
    (on-policy: buffer is cleared after each update).
  - For transductive (per-graph) training: create one env per problem.
  - For inductive training: randomly sample graphs from a training suite,
    rotate the env at the start of each episode.
  - Logs a JSONL record every log_every episodes.
  - Checkpoints best-by-eval-metric to out_dir/best.pt every eval_every episodes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from rlgb.algos.base import RLAgent, Transition


@dataclass
class TrainConfig:
    """Hyperparameters controlling the training loop."""

    # Budget
    n_episodes: int = 500        # total episodes to train
    horizon: int = 10            # max steps per episode

    # Gradient
    gamma: float = 0.99
    lr: float = 1e-3
    grad_clip: float = 1.0

    # REINFORCE specifics
    n_episode_per_update: int = 4   # accumulate N episodes before one update
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    # Logging / checkpointing
    log_every: int = 20          # episodes
    eval_every: int = 100        # episodes (0 = no mid-training eval)
    save_every: int = 200        # episodes
    out_dir: str = "results/run"

    # Misc
    device: str = "cpu"
    seed: int = 0
    verbose: bool = True


class Trainer:
    """Generic on-policy trainer compatible with all REINFORCE-style agents.

    Parameters
    ----------
    algo : RLAgent
        Any RLAgent implementation (NeuroCUT, CLARE, SLRL, AC2CD, …).
    env_fn : Callable[[], gymnasium.Env]
        Zero-argument factory returning a fresh env.  For inductive training,
        this factory should randomly sample a graph from the training suite.
    config : TrainConfig

    Usage
    -----
        trainer = Trainer(algo, env_fn, config)
        trainer.train()
        df = trainer.eval(eval_suite)   # returns pd.DataFrame
    """

    def __init__(
        self,
        algo: RLAgent,
        env_fn: Callable,
        config: TrainConfig | None = None,
    ) -> None:
        self.algo = algo
        self.env_fn = env_fn
        self.cfg = config or TrainConfig()
        self._out = Path(self.cfg.out_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._log_path = self._out / "train_log.jsonl"
        self._rng = np.random.default_rng(self.cfg.seed)
        self._best_eval: float = float("inf")   # lower = better (primary metric)
        self._ep_returns: list[float] = []
        self._ep_lengths: list[int] = []

    # ── main loop ────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Run the full training loop."""
        t_start = time.time()
        env = self.env_fn()
        accumulated: list[list[Transition]] = []

        for ep in range(1, self.cfg.n_episodes + 1):
            # Rotate env for inductive training (env_fn may return a new graph)
            if ep > 1 and ep % max(1, self.cfg.n_episodes // 20) == 0:
                env.close()
                env = self.env_fn()

            episode = self._run_episode(env)
            accumulated.append(episode)

            ep_return = float(sum(t.reward for t in episode))
            self._ep_returns.append(ep_return)
            self._ep_lengths.append(len(episode))

            # Push to algo and update
            for t in episode:
                self.algo.push_transition(t)

            if len(accumulated) >= self.cfg.n_episode_per_update:
                metrics = self.algo.update()
                accumulated.clear()
            else:
                metrics = {}

            # Logging
            if ep % self.cfg.log_every == 0:
                self._log(ep, metrics, t_start)

            # Checkpointing (save_every=0 disables mid-training saves)
            if self.cfg.save_every > 0 and ep % self.cfg.save_every == 0:
                self.algo.save(self._out / "last.pt")

        env.close()
        self.algo.save(self._out / "last.pt")
        if self.cfg.verbose:
            print(f"Training complete ({self.cfg.n_episodes} episodes). "
                  f"Checkpoint → {self._out / 'last.pt'}")

    def _run_episode(self, env) -> list[Transition]:
        obs, _ = env.reset()
        episode: list[Transition] = []
        for _ in range(self.cfg.horizon):
            action = self.algo.select_action(obs, greedy=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode.append(Transition(
                obs=obs, action=action, reward=float(reward),
                next_obs=next_obs, done=bool(terminated or truncated), info=info,
            ))
            obs = next_obs
            if terminated or truncated:
                break
        return episode

    def _log(self, ep: int, metrics: dict, t_start: float) -> None:
        window = self._ep_returns[-self.cfg.log_every:]
        row = {
            "episode": ep,
            "mean_return": round(float(np.mean(window)), 5),
            "mean_len":    round(float(np.mean(self._ep_lengths[-self.cfg.log_every:])), 2),
            "elapsed_s":   round(time.time() - t_start, 1),
            **{k: round(float(v), 6) for k, v in metrics.items()},
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if self.cfg.verbose:
            print(
                f"  ep={ep:5d}  ret={row['mean_return']:+.4f}"
                f"  len={row['mean_len']:.1f}"
                f"  t={row['elapsed_s']:.0f}s",
                flush=True,
            )

    # ── evaluation ───────────────────────────────────────────────────────────

    def eval(
        self,
        suite: list | None = None,
        task=None,
        n_seeds: int = 3,
    ) -> "pd.DataFrame":  # type: ignore[name-defined]
        """Evaluate the trained algo on a problem suite.

        If suite is not provided, falls back to the training env's problem.
        Returns a pandas DataFrame with one row per (problem, seed).
        """
        import pandas as pd
        from rlgb.eval.harness import eval_algo_on_suite

        if suite is None:
            # Try to extract from env_fn
            env = self.env_fn()
            problem = getattr(env, "problem", None)
            env.close()
            suite = [problem] if problem is not None else []

        records = eval_algo_on_suite(
            algo=self.algo,
            suite=suite,
            task=task,
            n_seeds=n_seeds,
            horizon=self.cfg.horizon,
        )
        return pd.DataFrame(records)
