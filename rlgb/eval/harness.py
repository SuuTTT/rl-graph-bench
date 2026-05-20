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

    Returns:
        pd.DataFrame with one row per (problem, seed) containing all metrics.
    """
    env_kwargs = env_kwargs or {}
    records: list[dict[str, Any]] = []

    for prob in suite:
        for seed in range(n_seeds):
            env = task.build_env(prob, horizon=horizon, seed=seed, **env_kwargs)
            obs, _ = env.reset(seed=seed)
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
                labels = cached.astype(np.int64)
            else:
                labels = obs["labels"].astype(np.int64)
            metrics = compute_all(prob.adj, labels, prob.gt_labels)
            metrics["problem"] = prob.name
            metrics["seed"] = seed
            metrics["algo"] = algo.name
            metrics["task_type"] = prob.task_type
            metrics["n_nodes"] = prob.adj.shape[0]
            metrics["k_target"] = prob.k_target
            metrics["wall_sec"] = elapsed
            records.append(metrics)
            env.close()

    return pd.DataFrame(records)


def compare_algos(
    algos: list[RLAgent],
    suite: list[Problem],
    task: ClusteringTask,
    n_seeds: int = 3,
    horizon: int = 10,
) -> pd.DataFrame:
    """Run eval_algo_on_suite for each algo and concatenate results."""
    dfs = [eval_algo_on_suite(a, suite, task, n_seeds=n_seeds, horizon=horizon)
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
