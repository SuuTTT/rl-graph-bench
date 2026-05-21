"""Quick eval script for NeuroCUT checkpoints.

Usage:
    python experiments/eval_neurocut.py --ckpt /tmp/bench_ckpts/neurocut_5000ep_256h.pt
    python experiments/eval_neurocut.py --hidden 128 --n_seeds 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rlgb.algos.node_move.neurocut import NeuroCUTAlgo, NeuroCUTConfig
from rlgb.baselines.clustering import SpectralBaseline, LeidenBaseline
from rlgb.data.synthetic import mini5
from rlgb.eval.harness import eval_algo_on_suite
from rlgb.tasks.graph_partition import GraphPartitionTask


def main():
    parser = argparse.ArgumentParser(description="Eval NeuroCUT checkpoint vs baselines")
    parser.add_argument("--ckpt", default=None, help="Path to .pt checkpoint")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size (if no ckpt)")
    parser.add_argument("--n_seeds", type=int, default=5, help="Eval seeds")
    parser.add_argument("--horizon", type=int, default=25, help="Steps per episode")
    parser.add_argument("--warm_start", default="spectral",
                        choices=["spectral", "leiden", "random"],
                        help="Warm-start for eval")
    args = parser.parse_args()

    suite = mini5()
    task = GraphPartitionTask(objective="ncut")

    algo = NeuroCUTAlgo(NeuroCUTConfig(hidden=args.hidden))
    if args.ckpt:
        algo = NeuroCUTAlgo.from_checkpoint(args.ckpt)
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        print(f"No checkpoint — evaluating untrained NeuroCUT (h={args.hidden})")

    print(f"\nEval: {args.n_seeds} seeds × {len(suite)} graphs × horizon={args.horizon}")
    print(f"Warm-start: {args.warm_start}\n")

    df_nc = eval_algo_on_suite(algo, suite, task, n_seeds=args.n_seeds,
                                horizon=args.horizon, greedy=True,
                                env_kwargs={"warm_start": args.warm_start})
    df_sp = eval_algo_on_suite(SpectralBaseline(), suite, task, n_seeds=1, horizon=1)
    df_ld = eval_algo_on_suite(LeidenBaseline(), suite, task, n_seeds=1, horizon=1)

    nc_ncut = df_nc["ncut"].mean()
    sp_ncut = df_sp["ncut"].mean()
    ld_ncut = df_ld["ncut"].mean()

    print(f"{'Algorithm':<20} {'NCut':>8}  {'NMI':>8}")
    print("-" * 40)
    print(f"{'NeuroCUT (RL)':<20} {nc_ncut:>8.4f}  {df_nc['nmi'].mean():>8.4f}")
    print(f"{'Spectral':<20} {sp_ncut:>8.4f}  {df_sp['nmi'].mean():>8.4f}")
    print(f"{'Leiden':<20} {ld_ncut:>8.4f}  {df_ld['nmi'].mean():>8.4f}")
    print()

    target = 0.333
    if nc_ncut <= target:
        print(f"✓ PASSED: NCut={nc_ncut:.4f} ≤ target {target}")
    else:
        gap = 100 * (nc_ncut - target) / target
        print(f"✗ GAP: NCut={nc_ncut:.4f}, target={target}, gap={gap:+.0f}%")

    if nc_ncut < sp_ncut:
        print(f"✓ Beats Spectral by {100*(sp_ncut-nc_ncut)/sp_ncut:.1f}%")
    else:
        print(f"✗ Below Spectral (delta={(nc_ncut-sp_ncut):+.4f})")


if __name__ == "__main__":
    main()
