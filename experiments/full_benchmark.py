#!/usr/bin/env python3
"""Full benchmark — train all RL algos and compare against classical baselines.

Paper targets (from individual papers):
  NeuroCUT : NCut ↓18 % vs Spectral  →  target NCut ≤ 0.333 (Cora k=4)
  WRT      : NCut ↓23 % vs NeuroCUT  →  target NCut ≤ 0.060 (City Traffic k=4 n=100)
  SS2V-D3QN: Multicut obj ↓         →  TBD (TNNLS paper, no preprint)
  CLARE    : F1 = 0.773 on SNAP Amazon  (community task, semi-supervised)
  SLRL     : F-score = 0.878 on SNAP Amazon  (community task, semi-supervised)
  AC2CD    : NMI = 0.75 on BlogCatalog3  (dynamic task)

Usage::

    python experiments/full_benchmark.py          # full run (~30 min CPU)
    python experiments/full_benchmark.py --quick  # smoke run (~3 min CPU)

Results saved to:  results/benchmark_v1.csv
Summary printed to stdout and saved to results/benchmark_v1_summary.txt
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure rlgb is importable when run from experiments/
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlgb.data.synthetic import fixed17, mini5
from rlgb.tasks.graph_partition import GraphPartitionTask
from rlgb.tasks.community_expand import CommunityExpandTask
from rlgb.tasks.dynamic_cd import DynamicCDTask
from rlgb.eval.harness import eval_algo_on_suite, compare_algos, summary_table
from rlgb.baselines.clustering import LeidenBaseline, LouvainBaseline, SpectralBaseline, RandomBaseline


# ── Config ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Smoke run: short training, mini5 suite")
    p.add_argument("--seeds", type=int, default=3, help="Eval seeds per problem")
    p.add_argument("--out-dir", default="results", help="Output directory")
    p.add_argument("--horizon", type=int, default=10, help="Eval horizon (steps per episode)")
    return p.parse_args()


# ── Training helpers ──────────────────────────────────────────────────────────

def _train_neurocut(suite, task, n_episodes: int, hidden: int, horizon: int, seed: int,
                    curriculum: bool = False):
    from rlgb.algos.node_move.neurocut import NeuroCUTAlgo, NeuroCUTConfig
    from rlgb.training.trainer import Trainer, TrainConfig
    rng = random.Random(seed)
    algo = NeuroCUTAlgo(NeuroCUTConfig(hidden=hidden))

    if curriculum and n_episodes >= 500:
        # Phase 1 (90 %): random warm-start — learns to improve from any starting point.
        # Phase 2 (10 %, capped at 500 ep): leiden warm-start short fine-tune.
        # WARNING: Phase 2 must stay SHORT. All-negative reward signals over many
        # episodes corrupt the Phase 1 model (policy gradient pushes toward
        # "avoid all moves" → destructive eval behaviour).
        n1 = int(n_episodes * 0.9)
        n2 = min(n_episodes - n1, 500)
        env_fn1 = lambda: task.build_env(rng.choice(suite), horizon=horizon, warm_start='random')
        Trainer(algo=algo, env_fn=env_fn1,
                config=TrainConfig(n_episodes=n1, horizon=horizon, lr=3e-4,
                                   n_episode_per_update=8, log_every=n1 // 5 + 1,
                                   save_every=0, out_dir="/tmp/bench_ckpts", seed=seed)).train()
        if n2 > 0:
            rng2 = random.Random(seed + 1)
            env_fn2 = lambda: task.build_env(rng2.choice(suite), horizon=horizon, warm_start='leiden')
            Trainer(algo=algo, env_fn=env_fn2,
                    config=TrainConfig(n_episodes=n2, horizon=horizon, lr=1e-4,
                                       n_episode_per_update=8, log_every=n2 // 5 + 1,
                                       save_every=0, out_dir="/tmp/bench_ckpts", seed=seed + 1)).train()
    else:
        env_fn = lambda: task.build_env(rng.choice(suite), horizon=horizon)
        Trainer(algo=algo, env_fn=env_fn,
                config=TrainConfig(n_episodes=n_episodes, horizon=horizon, lr=3e-4,
                                   n_episode_per_update=4, log_every=n_episodes // 5 + 1,
                                   save_every=0, out_dir="/tmp/bench_ckpts", seed=seed)).train()
    return algo


def _train_wrt(suite, task, n_episodes: int, hidden: int, horizon: int, seed: int):
    from rlgb.algos.structured.wrt import WRTAlgo, WRTConfig
    from rlgb.training.ppo import PPOTrainer, PPOConfig
    rng = random.Random(seed)
    algo = WRTAlgo(WRTConfig(hidden=hidden))
    # Use StructuredPartitionEnv so merge/split actions are decoded correctly
    env_fn = lambda: task.build_env(rng.choice(suite), horizon=horizon,
                                    env_class='structured', warm_start='random')
    trainer = PPOTrainer(
        algo=algo, env_fn=env_fn,
        config=PPOConfig(n_episodes=n_episodes, horizon=horizon, lr=3e-4,
                         n_episodes_per_update=4, log_every=n_episodes // 5 + 1,
                         save_every=0, out_dir="/tmp/bench_ckpts", seed=seed),
    )
    trainer.train()
    return algo


def _train_ss2v(suite, task, n_steps: int, hidden: int, horizon: int, seed: int):
    from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo, SS2VConfig
    from rlgb.training.dqn_trainer import DQNTrainer, DQNConfig
    rng = random.Random(seed)
    algo = SS2VAlgo(SS2VConfig(hidden=hidden, epsilon_decay=n_steps // 2))
    env_fn = lambda: task.build_env(rng.choice(suite), horizon=horizon)
    warmup = min(500, n_steps // 4)
    trainer = DQNTrainer(
        algo=algo, env_fn=env_fn,
        config=DQNConfig(n_steps=n_steps, warmup_steps=warmup, horizon=horizon,
                         update_every=4, log_every=n_steps // (horizon * 5) + 1,
                         save_every=0, out_dir="/tmp/bench_ckpts", seed=seed),
    )
    trainer.train()
    return algo


def _train_ac2cd(suite, task, n_episodes: int, hidden: int, horizon: int, seed: int):
    from rlgb.algos.dynamic.ac2cd import AC2CDAlgo, AC2CDConfig
    from rlgb.training.trainer import Trainer, TrainConfig
    rng = random.Random(seed)
    algo = AC2CDAlgo(AC2CDConfig(hidden=hidden))
    # Warm-start from leiden: agent learns to track snapshot changes from good init
    env_fn = lambda: task.build_env(rng.choice(suite), horizon=horizon,
                                     warm_start="leiden")
    trainer = Trainer(
        algo=algo, env_fn=env_fn,
        config=TrainConfig(n_episodes=n_episodes, horizon=horizon, lr=3e-4,
                           n_episode_per_update=4, log_every=n_episodes // 5 + 1,
                           save_every=0, out_dir="/tmp/bench_ckpts", seed=seed),
    )
    trainer.train()
    return algo


def _train_clare(suite, task, n_episodes: int, hidden: int, horizon: int, seed: int):
    from rlgb.algos.community.clare import CLAREAlgo, CLAREConfig
    from rlgb.training.trainer import Trainer, TrainConfig
    rng = random.Random(seed)
    algo = CLAREAlgo(CLAREConfig(hidden=hidden))
    env_fn = lambda: task.build_env(rng.choice(suite), horizon=horizon)
    trainer = Trainer(
        algo=algo, env_fn=env_fn,
        config=TrainConfig(n_episodes=n_episodes, horizon=horizon, lr=3e-4,
                           n_episode_per_update=4, log_every=n_episodes // 5 + 1,
                           save_every=0, out_dir="/tmp/bench_ckpts", seed=seed),
    )
    trainer.train()
    return algo


def _train_slrl(suite, task, n_episodes: int, hidden: int, horizon: int, seed: int):
    from rlgb.algos.community.slrl import SLRLAlgo
    from rlgb.training.trainer import Trainer, TrainConfig
    rng = random.Random(seed)
    algo = SLRLAlgo()
    env_fn = lambda: task.build_env(rng.choice(suite), horizon=horizon)
    trainer = Trainer(
        algo=algo, env_fn=env_fn,
        config=TrainConfig(n_episodes=n_episodes, horizon=horizon, lr=3e-4,
                           n_episode_per_update=4, log_every=n_episodes // 5 + 1,
                           save_every=0, out_dir="/tmp/bench_ckpts", seed=seed),
    )
    trainer.train()
    return algo


# ── Benchmark tasks ───────────────────────────────────────────────────────────

def run_partition_benchmark(quick: bool, seeds: int, horizon: int) -> pd.DataFrame:
    """Graph partition: neurocut, wrt, ss2v vs leiden, louvain, spectral, random."""
    suite = mini5() if quick else fixed17()
    # NeuroCUT paper (KDD 2024) targets NCut; WRT paper (arXiv:2505.13986) also targets NCut.
    task  = GraphPartitionTask(objective="ncut")

    n_ep  = 50   if quick else 5000
    dqn_s = 500  if quick else 20000
    hid   = 32   if quick else 256

    print(f"\n{'='*60}")
    print(f"PARTITION BENCHMARK  |  suite={len(suite)} graphs  |  n_ep={n_ep}  |  hidden={hid}")
    print(f"{'='*60}")

    Path("/tmp/bench_ckpts").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print("[1/6] Training NeuroCUT …")
    neurocut = _train_neurocut(suite, task, n_ep, hid, horizon, seed=42,
                                curriculum=True)
    print(f"      done in {time.perf_counter()-t0:.1f}s")

    t1 = time.perf_counter()
    print("[2/6] Training WRT …")
    wrt = _train_wrt(suite, task, n_ep, hid, horizon, seed=42)
    print(f"      done in {time.perf_counter()-t1:.1f}s")

    t2 = time.perf_counter()
    print("[3/6] Training SS2V-D3QN …")
    ss2v = _train_ss2v(suite, task, dqn_s, hid, horizon, seed=42)
    print(f"      done in {time.perf_counter()-t2:.1f}s")

    algos = [neurocut, wrt, ss2v, LeidenBaseline(), LouvainBaseline(),
             SpectralBaseline(), RandomBaseline()]

    print(f"\n[4/6] Evaluating {len(algos)} algos × {len(suite)} graphs × {seeds} seeds …")
    best_n = 1 if quick else 5   # best-of-N stochastic rollouts for RL algos
    cls_algos = [LeidenBaseline(), LouvainBaseline(), SpectralBaseline(), RandomBaseline()]
    # NeuroCUT + SS2V use NodeMoveEnv; WRT uses StructuredPartitionEnv
    df_nc   = compare_algos([neurocut], suite, task, n_seeds=seeds, horizon=horizon,
                             eval_kwargs={"greedy": True, "best_of": best_n,
                                          "env_kwargs": {"warm_start": "leiden"}})
    df_wrt  = compare_algos([wrt],      suite, task, n_seeds=seeds, horizon=horizon,
                             eval_kwargs={"greedy": True, "best_of": best_n,
                                          "env_kwargs": {"env_class": "structured",
                                                         "warm_start": "leiden"}})
    df_ss2v = compare_algos([ss2v],     suite, task, n_seeds=seeds, horizon=horizon,
                             eval_kwargs={"greedy": True, "best_of": best_n,
                                          "env_kwargs": {"warm_start": "leiden"}})
    df_cls = compare_algos(cls_algos, suite, task, n_seeds=seeds, horizon=horizon)
    df = pd.concat([df_nc, df_wrt, df_ss2v, df_cls], ignore_index=True)
    df["benchmark"] = "partition"
    return df


def run_community_benchmark(quick: bool, seeds: int, horizon: int) -> pd.DataFrame:
    """Community expansion: clare, slrl vs leiden (adapts objective)."""
    task = CommunityExpandTask(objective="h2")
    # Use partition suite as a proxy (community env works on same graphs)
    suite = mini5()[:3] if quick else fixed17()[:8]

    n_ep = 30 if quick else 1000
    hid  = 32 if quick else 64

    print(f"\n{'='*60}")
    print(f"COMMUNITY BENCHMARK  |  suite={len(suite)} graphs  |  n_ep={n_ep}")
    print(f"{'='*60}")

    print("[1/3] Training CLARE …")
    clare = _train_clare(suite, task, n_ep, hid, horizon, seed=42)

    print("[2/3] Training SLRL …")
    slrl = _train_slrl(suite, task, n_ep, hid, horizon, seed=42)

    algos = [clare, slrl, LeidenBaseline()]
    print(f"[3/3] Evaluating {len(algos)} algos …")
    df = compare_algos(algos, suite, task, n_seeds=seeds, horizon=horizon)
    df["benchmark"] = "community"
    return df


def run_dynamic_benchmark(quick: bool, seeds: int, horizon: int) -> pd.DataFrame:
    """Dynamic CD: ac2cd vs leiden (evaluated per snapshot)."""
    task  = DynamicCDTask()
    suite = task.build_suite()  # synthetic temporal suite

    n_ep = 30 if quick else 1000
    hid  = 32 if quick else 64

    print(f"\n{'='*60}")
    print(f"DYNAMIC BENCHMARK  |  suite={len(suite)} graphs  |  n_ep={n_ep}")
    print(f"{'='*60}")

    print("[1/2] Training AC2CD …")
    ac2cd = _train_ac2cd(suite, task, n_ep, hid, horizon, seed=42)

    algos = [ac2cd, LeidenBaseline()]
    print("[2/2] Evaluating …")
    df_ac2cd = compare_algos([ac2cd], suite, task, n_seeds=seeds, horizon=horizon,
                              eval_kwargs={"env_kwargs": {"warm_start": "leiden"}})
    df_cls   = compare_algos([LeidenBaseline()], suite, task, n_seeds=seeds, horizon=horizon)
    df = pd.concat([df_ac2cd, df_cls], ignore_index=True)
    df["benchmark"] = "dynamic"
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"RL Graph Bench — Full Benchmark {'(QUICK)' if args.quick else '(FULL)'}")
    print(f"seeds={args.seeds}  horizon={args.horizon}  out={out_dir}")

    dfs = []

    try:
        df_part = run_partition_benchmark(args.quick, args.seeds, args.horizon)
        dfs.append(df_part)
    except Exception as exc:
        print(f"[WARN] Partition benchmark failed: {exc}")

    try:
        df_comm = run_community_benchmark(args.quick, args.seeds, args.horizon)
        dfs.append(df_comm)
    except Exception as exc:
        print(f"[WARN] Community benchmark failed: {exc}")

    try:
        df_dyn = run_dynamic_benchmark(args.quick, args.seeds, args.horizon)
        dfs.append(df_dyn)
    except Exception as exc:
        print(f"[WARN] Dynamic benchmark failed: {exc}")

    if not dfs:
        print("ERROR: All benchmarks failed.")
        sys.exit(1)

    all_df = pd.concat(dfs, ignore_index=True)

    # Save raw results
    csv_path = out_dir / "benchmark_v1.csv"
    all_df.to_csv(csv_path, index=False)
    print(f"\n✓ Raw results → {csv_path}")

    # Print summary tables per benchmark
    summary_lines = []
    for bname in all_df["benchmark"].unique():
        sub = all_df[all_df["benchmark"] == bname]
        tbl = summary_table(sub)
        metric_cols = [c for c in ["ncut", "h2", "nmi", "ari", "f1", "modularity_density"]
                       if c in tbl.columns]
        lines = [
            f"\n── {bname.upper()} ──",
            tbl[metric_cols].to_string(),
        ]
        for line in lines:
            print(line)
            summary_lines.append(line)

    # Paper-target gap analysis (partition only)
    if "partition" in all_df["benchmark"].values:
        _print_paper_gap(all_df[all_df["benchmark"] == "partition"])

    summary_path = out_dir / "benchmark_v1_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"\n✓ Summary → {summary_path}")


def _print_paper_gap(df: pd.DataFrame) -> None:
    """Print comparison vs paper-reported numbers (NCut objective only)."""
    print("\n── PAPER GAP ANALYSIS (partition, NCut objective) ──")
    # Targets from source papers (see docs/PAPER_TARGETS.md):
    #   NeuroCUT: NCut 0.33 on Cora k=4  (Shah et al., KDD 2024, arXiv:2310.11787)
    #   WRT:      NCut 0.060 City Traffic k=4 n=100  (Jiang et al. 2025, arXiv:2505.13986)
    #   SS2V-D3QN: TBD (TNNLS paper, no preprint)
    targets = {
        "neurocut":  {"ncut": 0.333},
        "wrt":       {"ncut": 0.060},
        "ss2v_d3qn": {},    # no public NCut target — multicut objective only
    }
    for algo_name, tgts in targets.items():
        sub = df[df["algo"] == algo_name]
        if sub.empty:
            continue
        if not tgts:
            print(f"  {algo_name:<14} (no NCut paper target — see TNNLS paper for multicut numbers)")
            continue
        for metric, target in tgts.items():
            if metric not in sub.columns:
                continue
            actual = sub[metric].mean()
            gap_pct = 100 * (actual - target) / (abs(target) + 1e-9)
            status = "✓ BEAT" if actual <= target else f"△ gap={gap_pct:+.1f}%"
            print(f"  {algo_name:<14} {metric}: actual={actual:.4f}  target≤{target}  {status}")


if __name__ == "__main__":
    main()
