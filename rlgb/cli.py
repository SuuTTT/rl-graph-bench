"""rlgb CLI — entry point for training, evaluation, and the dashboard.

Usage examples::

    rlgb run  --algo neurocut --task partition --dataset mini5 --steps 500
    rlgb eval --algo neurocut --checkpoint results/best.pt --dataset mini5 --seeds 3
    rlgb compare --algos neurocut,ac2cd --task partition --dataset mini5
    rlgb serve
    rlgb list-algos
    rlgb list-datasets
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="rlgb",
    help="RL Graph Bench — unified RL graph-clustering benchmark.",
    add_completion=False,
    no_args_is_help=True,
)


# ── Registry helpers ──────────────────────────────────────────────────────────

def _get_task(task: str, objective: str = "h2"):
    if task == "partition":
        from rlgb.tasks.graph_partition import GraphPartitionTask
        return GraphPartitionTask(objective=objective)
    if task == "community":
        from rlgb.tasks.community_expand import CommunityExpandTask
        return CommunityExpandTask(objective=objective)
    if task == "dynamic":
        from rlgb.tasks.dynamic_cd import DynamicCDTask
        return DynamicCDTask()
    raise typer.BadParameter(f"Unknown task '{task}'. Choose: partition, community, dynamic")


def _get_algo(algo: str, hidden: int = 64, device: str = "cpu"):
    algo = algo.lower()
    if algo == "neurocut":
        from rlgb.algos.node_move.neurocut import NeuroCUTAlgo, NeuroCUTConfig
        return NeuroCUTAlgo(NeuroCUTConfig(hidden=hidden, device=device))
    if algo == "clare":
        from rlgb.algos.community.clare import CLAREAlgo, CLAREConfig
        return CLAREAlgo(CLAREConfig(hidden=hidden))
    if algo == "slrl":
        from rlgb.algos.community.slrl import SLRLAlgo
        return SLRLAlgo()
    if algo == "ac2cd":
        from rlgb.algos.dynamic.ac2cd import AC2CDAlgo, AC2CDConfig
        return AC2CDAlgo(AC2CDConfig(hidden=hidden, device=device))
    if algo == "wrt":
        from rlgb.algos.structured.wrt import WRTAlgo, WRTConfig
        return WRTAlgo(WRTConfig(hidden=hidden, device=device))
    if algo in ("ss2v", "ss2v_d3qn"):
        from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo, SS2VConfig
        return SS2VAlgo(SS2VConfig(hidden=hidden, device=device))
    raise typer.BadParameter(f"Unknown algo '{algo}'. Run `rlgb list-algos` for options.")


def _get_suite(dataset: str, task_obj) -> list:
    dataset = dataset.lower()
    if dataset == "mini5":
        from rlgb.data.synthetic import mini5
        return mini5()
    if dataset == "fixed17":
        from rlgb.data.synthetic import fixed17
        return fixed17()
    if dataset in ("cora", "citeseer", "pubmed"):
        from rlgb.data.pyg_loaders import load_planetoid
        return load_planetoid(dataset.capitalize(), max_nodes=500)
    if dataset in ("computers", "photo"):
        from rlgb.data.pyg_loaders import load_amazon
        return load_amazon(dataset.capitalize(), max_nodes=500)
    if dataset in ("cs", "physics"):
        from rlgb.data.pyg_loaders import load_coauthor
        return load_coauthor(dataset.upper(), max_nodes=500)
    # Fallback: task's own built-in suite
    return task_obj.build_suite()


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def run(
    algo: str = typer.Option("neurocut", help="Algorithm name (neurocut, clare, slrl, ac2cd, wrt, ss2v_d3qn)"),
    task: str = typer.Option("partition", help="Task type (partition, community, dynamic)"),
    objective: str = typer.Option("h2", help="Objective (h2, ncut, f1, modularity_density)"),
    dataset: str = typer.Option("mini5", help="Dataset (mini5, fixed17, cora, citeseer, computers, photo, cs)"),
    steps: int = typer.Option(500, help="Number of training episodes"),
    horizon: int = typer.Option(10, help="Steps per episode"),
    hidden: int = typer.Option(64, help="Hidden layer size"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    device: str = typer.Option("cpu", help="Torch device (cpu / cuda)"),
    out_dir: Path = typer.Option(Path("results"), help="Output directory for checkpoints and logs"),
    seed: int = typer.Option(0, help="Random seed"),
) -> None:
    """Train an RL algo on a graph clustering task."""
    import torch
    from rlgb.training.trainer import Trainer, TrainConfig

    task_obj = _get_task(task, objective)
    algo_obj = _get_algo(algo, hidden=hidden, device=device)
    suite    = _get_suite(dataset, task_obj)

    typer.echo(f"Training {algo} on {dataset} ({task}) for {steps} episodes …")

    import random as _random
    _rng = _random.Random(seed)

    def env_fn():
        prob = _rng.choice(suite)
        return task_obj.build_env(prob, horizon=horizon)

    cfg = TrainConfig(
        n_episodes=steps,
        horizon=horizon,
        lr=lr,
        device=device,
        seed=seed,
        out_dir=str(out_dir),
    )
    trainer = Trainer(algo=algo_obj, env_fn=env_fn, config=cfg)
    trainer.train()

    typer.echo(f"Training complete. Checkpoints saved to {out_dir}/")


@app.command()
def eval(
    algo: str = typer.Option("neurocut", help="Algorithm name"),
    task: str = typer.Option("partition", help="Task type"),
    objective: str = typer.Option("h2", help="Objective"),
    dataset: str = typer.Option("mini5", help="Dataset"),
    checkpoint: Optional[Path] = typer.Option(None, help="Path to .pt checkpoint"),
    seeds: int = typer.Option(3, help="Evaluation seeds"),
    horizon: int = typer.Option(10, help="Roll-out horizon"),
    hidden: int = typer.Option(64, help="Hidden layer size"),
) -> None:
    """Evaluate a trained algo on a benchmark suite."""
    from rlgb.eval.harness import eval_algo_on_suite, summary_table

    task_obj = _get_task(task, objective)
    algo_obj = _get_algo(algo, hidden=hidden)
    suite    = _get_suite(dataset, task_obj)

    if checkpoint is not None:
        algo_obj.load(str(checkpoint))
        typer.echo(f"Loaded checkpoint: {checkpoint}")

    typer.echo(f"Evaluating {algo} on {dataset} …")
    df = eval_algo_on_suite(algo_obj, suite, task_obj, n_seeds=seeds, horizon=horizon)
    tbl = summary_table(df, primary_metric=task_obj.primary_metric)
    typer.echo(tbl.to_string())
    out_csv = Path("results") / f"eval_{algo}_{dataset}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    typer.echo(f"\nFull results saved to {out_csv}")


@app.command()
def compare(
    algos: str = typer.Option("neurocut,ac2cd", help="Comma-separated algo names"),
    task: str = typer.Option("partition", help="Task type"),
    objective: str = typer.Option("h2", help="Objective"),
    dataset: str = typer.Option("mini5", help="Dataset"),
    seeds: int = typer.Option(2, help="Eval seeds per algo"),
    horizon: int = typer.Option(5, help="Roll-out horizon"),
    hidden: int = typer.Option(32, help="Hidden size"),
) -> None:
    """Compare multiple algos on the same suite."""
    from rlgb.eval.harness import compare_algos, summary_table

    task_obj  = _get_task(task, objective)
    suite     = _get_suite(dataset, task_obj)
    algo_list = [_get_algo(a.strip(), hidden=hidden) for a in algos.split(",")]

    typer.echo(f"Comparing {algos} on {dataset} …")
    df = compare_algos(algo_list, suite, task_obj, n_seeds=seeds, horizon=horizon)
    typer.echo(summary_table(df, primary_metric=task_obj.primary_metric).to_string())


@app.command()
def serve(
    port: int = typer.Option(8501, help="Streamlit server port"),
    results_dir: Path = typer.Option(Path("results"), help="Path to results directory"),
) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys
    from pathlib import Path as _Path

    dashboard = _Path(__file__).parent.parent / "dashboard" / "app.py"
    if not dashboard.exists():
        typer.echo(f"Dashboard not found at {dashboard}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Starting dashboard on http://localhost:{port}/ …")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard),
         "--server.port", str(port),
         "--",
         "--results_dir", str(results_dir)],
        check=False,
    )


@app.command(name="list-algos")
def list_algos() -> None:
    """List all available algorithms."""
    rows = [
        ("neurocut", "partition",  "NodeMove + GraphSAGE + REINFORCE"),
        ("clare",    "community",  "GIN + Exclude/Expand + REINFORCE"),
        ("slrl",     "community",  "Seed-based linear RL"),
        ("ac2cd",    "dynamic",    "GAT + Actor-Critic on temporal snapshots"),
        ("wrt",      "partition",  "Cluster-level Transformer + PPO merge/split"),
        ("ss2v_d3qn","partition",  "DenseSAGE + Dueling Double DQN"),
    ]
    header = f"{'ALGO':<14}{'TASK':<14}DESCRIPTION"
    typer.echo(header)
    typer.echo("-" * len(header))
    for name, task, desc in rows:
        typer.echo(f"{name:<14}{task:<14}{desc}")


@app.command(name="list-datasets")
def list_datasets() -> None:
    """List all available datasets."""
    rows = [
        ("mini5",    "synthetic", "5 small SBM graphs (smoke test)"),
        ("fixed17",  "synthetic", "17-graph canonical benchmark suite"),
        ("cora",     "real/PyG",  "Cora citation network (2708 nodes, 7 classes)"),
        ("citeseer", "real/PyG",  "CiteSeer citation network (3327 nodes, 6 classes)"),
        ("pubmed",   "real/PyG",  "PubMed citation network (19717 nodes, 3 classes)"),
        ("computers","real/PyG",  "Amazon Computers co-purchase (13752 nodes)"),
        ("photo",    "real/PyG",  "Amazon Photo co-purchase (7650 nodes)"),
        ("cs",       "real/PyG",  "Coauthor CS (18333 nodes)"),
        ("physics",  "real/PyG",  "Coauthor Physics (34493 nodes)"),
    ]
    header = f"{'DATASET':<14}{'TYPE':<14}DESCRIPTION"
    typer.echo(header)
    typer.echo("-" * len(header))
    for name, kind, desc in rows:
        typer.echo(f"{name:<14}{kind:<14}{desc}")


if __name__ == "__main__":
    app()
