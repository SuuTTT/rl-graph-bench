"""Evaluation harness — run an algo on a benchmark suite and collect results.

Usage::

    from rlgb.eval.harness import eval_algo_on_suite
    from rlgb.algos.node_move.neurocut import NeuroCUTAlgo
    from rlgb.tasks.graph_partition import GraphPartitionTask
    from rlgb.data.synthetic import mini5

    task = GraphPartitionTask()
    algo = NeuroCUTAlgo()
    df = eval_algo_on_suite(algo, mini5(), task, n_seeds=2, horizon=5)
    print(df.groupby("problem")[["ncut","h2"]].mean())
"""
from __future__ import annotations

import copy
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from rlgb.tasks.base import Problem, EvalRecord, ClusteringTask
from rlgb.algos.base import RLAgent
from rlgb.eval.metrics import compute_all


def eval_algo_on_suite(
    algo: RLAgent,
    suite: list[Problem],
    task: ClusteringTask,
    n_seeds: int = 3,
    horizon: int = 10,
    greedy: bool = True,
    env_kwargs: dict[str, Any] | None = None,
    best_of: int = 1,
) -> pd.DataFrame:
    """Evaluate *algo* on every problem in *suite* across *n_seeds* random seeds.

    The algo is run *once per seed per problem* for *horizon* steps (pure inference,
    no training). The algo's weights are NOT modified.

    Args:
        algo: Trained (or untrained) RLAgent.
        suite: List of Problem instances to evaluate on.
        task: ClusteringTask providing the environment factory and metric names.
        n_seeds: Number of random seeds per problem.
        horizon: Number of steps to roll-out per episode.
        greedy: If True, run algo.select_action(..., greedy=True).
        env_kwargs: Extra kwargs forwarded to the env constructor.
        best_of: Run this many rollouts per (problem, seed) and keep the one
                 with the lowest NCut.  Values > 1 implement best-of-N selection
                 (stochastic policy only; ignored if greedy=True).

    Returns:
        pd.DataFrame with one row per (problem, seed) containing all metrics.
    """
    env_kwargs = env_kwargs or {}
    records: list[dict[str, Any]] = []

    for prob in suite:
        for seed in range(n_seeds):
            n_rollouts = best_of if (not greedy and best_of > 1) else 1
            best_labels: np.ndarray | None = None
            best_ncut = float("inf")
            best_elapsed = 0.0

            for rollout_idx in range(n_rollouts):
                rs = seed * 1000 + rollout_idx
                env = task.build_env(prob, horizon=horizon, seed=rs, **env_kwargs)
                obs, _ = env.reset(seed=rs)
                algo.reset_episode()

                start = time.perf_counter()
                for _ in range(horizon):
                    action = algo.select_action(obs, greedy=greedy)
                    obs, _reward, terminated, truncated, _info = env.step(action)
                    if terminated or truncated:
                        break
                elapsed = time.perf_counter() - start

                # Classical baselines may store their full partition directly
                # (bypassing the env's step-by-step label tracking).
                cached = getattr(algo, "_cached_labels", None)
                if cached is not None and len(cached) == prob.adj.shape[0]:
                    candidate = cached.astype(np.int64)
                else:
                    candidate = obs["labels"].astype(np.int64)

                if n_rollouts > 1:
                    from rlgb.eval.metrics import ncut
                    c_ncut = ncut(prob.adj, candidate)
                    if c_ncut < best_ncut:
                        best_ncut = c_ncut
                        best_labels = candidate
                        best_elapsed = elapsed
                else:
                    best_labels = candidate
                    best_elapsed = elapsed
                env.close()

            assert best_labels is not None
            metrics = compute_all(prob.adj, best_labels, prob.gt_labels,
                                             gt_communities=prob.known_communities)
            metrics["problem"] = prob.name
            metrics["seed"] = seed
            metrics["algo"] = algo.name
            metrics["task_type"] = prob.task_type
            metrics["n_nodes"] = prob.adj.shape[0]
            metrics["k_target"] = prob.k_target
            metrics["wall_sec"] = best_elapsed
            records.append(metrics)

    return pd.DataFrame(records)


def compare_algos(
    algos: list[RLAgent],
    suite: list[Problem],
    task: ClusteringTask,
    n_seeds: int = 3,
    horizon: int = 10,
    eval_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run eval_algo_on_suite for each algo and concatenate results."""
    kwargs = eval_kwargs or {}
    dfs = [eval_algo_on_suite(a, suite, task, n_seeds=n_seeds, horizon=horizon, **kwargs)
           for a in algos]
    return pd.concat(dfs, ignore_index=True)


def summary_table(df: pd.DataFrame, primary_metric: str = "ncut") -> pd.DataFrame:
    """Aggregate eval DataFrame into a mean±std summary table.

    Returns a DataFrame indexed by algo with columns for each metric.
    """
    numeric = df.select_dtypes(include="number").columns.tolist()
    grouped = df.groupby("algo")[numeric]
    mean = grouped.mean()
    std  = grouped.std().fillna(0.0)
    combined = mean.copy()
    for col in numeric:
        combined[col] = [
            f"{m:.4f}±{s:.4f}" for m, s in zip(mean[col], std[col])
        ]
    return combined
