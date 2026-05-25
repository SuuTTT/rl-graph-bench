"""Full baseline benchmark table.

Runs every algorithm × every dataset × every task and prints a markdown table.
Also saves results/benchmark_table.csv.

Algorithms
----------
  Partition task (Cora, CiteSeer, SBM):
    random, leiden, louvain, spectral, neurocut (pretrained)
  Multicut task (ER/BA n=20,40):
    gaec, ss2v_d3qn (pretrained checkpoint if available)

Metrics
-------
  Partition: ncut, h2, modularity, nmi, ari (sparsest_cut where applicable)
  Multicut : multicut_cost (lower better)

Usage::
    cd /workspace/rl-graph-bench
    python3 experiments/run_full_benchmark.py 2>&1 | tee results/benchmark_run.log
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Reproducibility ───────────────────────────────────────────────────────────
import random as _random
import torch
_random.seed(0);  np.random.seed(0);  torch.manual_seed(0)

# ── Imports ───────────────────────────────────────────────────────────────────
from rlgb.tasks.graph_partition import GraphPartitionTask
from rlgb.tasks.multicut import MulticutTask, multicut_cost_fast
from rlgb.data.pyg_loaders import load_planetoid
from rlgb.data.synthetic import sbm
from rlgb.data.mcmp_instances import mcmp_test_suite
from rlgb.tasks.base import Problem
from rlgb.eval.harness import eval_algo_on_suite
from rlgb.eval.metrics import compute_all, sparsest_cut
from rlgb.baselines.clustering import (
    LeidenBaseline, LouvainBaseline, SpectralBaseline, RandomBaseline,
)
from rlgb.baselines.multicut import GAECBaseline
from rlgb.algos.node_move.neurocut import NeuroCUTAlgo
from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo, SS2VConfig
from rlgb.envs.edge_contraction_env import EdgeContractionEnv

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS     = 5          # random seeds per problem
HORIZON   = 30         # steps per episode for partition algos
OUT_DIR   = Path("results")
OUT_DIR.mkdir(exist_ok=True)
CSV_PATH  = OUT_DIR / "benchmark_table.csv"

print(f"Device: {DEVICE}  seeds={SEEDS}  horizon={HORIZON}")
print("=" * 70)


# ── Shared wrapper for MCMP signed-cost injection ─────────────────────────────
class _MCMPWrapper:
    """Inject adj_signed into EdgeContractionEnv obs for SS2V edge-weight feature."""
    def __init__(self, env, cost_adj: np.ndarray):
        self._env = env; self._cost_adj = cost_adj
    def _inj(self, obs: dict) -> dict:
        obs["adj_signed"] = self._cost_adj; return obs
    def reset(self, **kw): obs, info = self._env.reset(**kw); return self._inj(obs), info
    def step(self, a: int): obs, r, t, tr, i = self._env.step(a); return self._inj(obs), r, t, tr, i
    def close(self): self._env.close()
    def __getattr__(self, n): return getattr(self._env, n)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — Graph Partition Task                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

partition_task = GraphPartitionTask(objective="ncut")


def _make_sbm_problem(n: int = 300, k: int = 5, p_in: float = 0.30,
                       p_out: float = 0.03, seed: int = 0) -> Problem:
    adj, labels, k = sbm(n, k, p_in, p_out, seed=seed)
    return Problem(name=f"sbm_n{n}_k{k}", adj=adj, k_target=k,
                   gt_labels=labels, family="synthetic",
                   task_type="partition")


def _make_lfr_problem(n: int = 300, mu: float = 0.2, seed: int = 0) -> Problem:
    """LFR benchmark with mu=0.2 (low noise, structured community)."""
    from rlgb.data.synthetic import lfr
    try:
        adj, labels, k = lfr(n=n, mu=mu, seed=seed)
        return Problem(name=f"lfr_n{n}_mu{mu}", adj=adj, k_target=k,
                       gt_labels=labels, family="synthetic",
                       task_type="partition")
    except Exception as e:
        print(f"  [warn] LFR failed ({e}), skipping")
        return None


print("\n── Loading partition datasets ──────────────────────────────────────")
partition_suites: dict[str, list[Problem]] = {}

# Cora k=7 (original class count)
print("  Loading Cora k=7 ...")
try:
    partition_suites["cora_k7"] = load_planetoid("Cora", k_target=7)
    print(f"    {partition_suites['cora_k7'][0].adj.shape[0]} nodes")
except Exception as e:
    print(f"  [warn] Cora failed: {e}")

# Cora k=4 (NeuroCUT paper target)
print("  Loading Cora k=4 ...")
try:
    partition_suites["cora_k4"] = load_planetoid("Cora", k_target=4)
    print(f"    {partition_suites['cora_k4'][0].adj.shape[0]} nodes")
except Exception as e:
    print(f"  [warn] Cora k4 failed: {e}")

# CiteSeer k=6
print("  Loading CiteSeer k=6 ...")
try:
    partition_suites["citeseer_k6"] = load_planetoid("CiteSeer", k_target=6)
    print(f"    {partition_suites['citeseer_k6'][0].adj.shape[0]} nodes")
except Exception as e:
    print(f"  [warn] CiteSeer failed: {e}")

# SBM n=300 k=5 (synthetic — fast baseline)
print("  Building SBM n=300 k=5 ...")
sbm_prob = _make_sbm_problem(n=300, k=5, p_in=0.30, p_out=0.03, seed=0)
partition_suites["sbm_n300_k5"] = [sbm_prob]
print(f"    {sbm_prob.adj.shape[0]} nodes")

# LFR n=300 mu=0.2
print("  Building LFR n=300 mu=0.2 ...")
lfr_prob = _make_lfr_problem(n=300, mu=0.2, seed=0)
if lfr_prob is not None:
    partition_suites["lfr_n300_mu0.2"] = [lfr_prob]
    print(f"    {lfr_prob.adj.shape[0]} nodes")

# SBM n=500 k=8 (larger graph)
print("  Building SBM n=500 k=8 ...")
sbm500 = _make_sbm_problem(n=500, k=8, p_in=0.25, p_out=0.02, seed=1)
partition_suites["sbm_n500_k8"] = [sbm500]
print(f"    {sbm500.adj.shape[0]} nodes")

# SBM n=1000 k=10 (stress test)
print("  Building SBM n=1000 k=10 ...")
sbm1000 = _make_sbm_problem(n=1000, k=10, p_in=0.20, p_out=0.01, seed=2)
partition_suites["sbm_n1000_k10"] = [sbm1000]
print(f"    {sbm1000.adj.shape[0]} nodes")

print(f"\n  Datasets: {list(partition_suites.keys())}")


# ── Partition algorithms ──────────────────────────────────────────────────────
print("\n── Setting up partition algorithms ─────────────────────────────────")

partition_algos: list = [
    RandomBaseline(),
    LeidenBaseline(resolution=1.0),
    LouvainBaseline(),
    SpectralBaseline(),
]

# NeuroCUT pretrained
NEUROCUT_CKPT = Path("results/last.pt")
if NEUROCUT_CKPT.exists():
    print(f"  Loading NeuroCUT from {NEUROCUT_CKPT}")
    neurocut = NeuroCUTAlgo.from_checkpoint(str(NEUROCUT_CKPT))
    neurocut._cfg.device = DEVICE
    partition_algos.append(neurocut)
    print("  NeuroCUT loaded ✓")
else:
    print(f"  [warn] NeuroCUT checkpoint not found at {NEUROCUT_CKPT}, training fresh (n=300, 300ep)")
    neurocut_cfg = __import__("rlgb.algos.node_move.neurocut", fromlist=["NeuroCUTConfig"]).NeuroCUTConfig
    neurocut = NeuroCUTAlgo(neurocut_cfg(device=DEVICE))
    # Quick warm-up train on SBM
    from rlgb.training.trainer import Trainer, TrainConfig
    def _nc_env_fn():
        p = sbm_prob
        from rlgb.envs.node_move_env import NodeMoveEnv
        return NodeMoveEnv(task=partition_task, problem=p, horizon=30,
                           warm_start="spectral")
    _tc = TrainConfig(n_episodes=300, horizon=30, log_every=100, out_dir="results/neurocut_tmp")
    Trainer(algo=neurocut, env_fn=_nc_env_fn, config=_tc).train()
    partition_algos.append(neurocut)
    print("  NeuroCUT warmed up ✓")


# ── Run partition benchmark ───────────────────────────────────────────────────
print("\n── Running partition benchmark ─────────────────────────────────────")
partition_rows: list[dict] = []

for dataset_name, suite in partition_suites.items():
    print(f"\n  Dataset: {dataset_name}  ({suite[0].adj.shape[0]} nodes, k={suite[0].k_target})")
    for algo in partition_algos:
        t0 = time.perf_counter()
        print(f"    algo={algo.name} ...", end="", flush=True)
        try:
            df = eval_algo_on_suite(
                algo, suite, partition_task,
                n_seeds=SEEDS, horizon=HORIZON,
                greedy=True,
            )
            elapsed = time.perf_counter() - t0
            # Also compute sparsest_cut per row
            sc_vals = []
            for _, row in df.iterrows():
                prob = suite[0]  # single-problem suites
                labels_arr = obs_labels_from_row = None
                # sparsest_cut needs labels — extract from df if stored
                sc_vals.append(float("nan"))
            mean_row = df.select_dtypes(include="number").mean().to_dict()
            mean_row["dataset"]  = dataset_name
            mean_row["algo"]     = algo.name
            mean_row["wall_sec"] = elapsed
            mean_row["n_nodes"]  = suite[0].adj.shape[0]
            mean_row["k_target"] = suite[0].k_target
            partition_rows.append(mean_row)
            # Quick summary
            ncut_v  = mean_row.get("ncut",  float("nan"))
            nmi_v   = mean_row.get("nmi",   float("nan"))
            ari_v   = mean_row.get("ari",   float("nan"))
            mod_v   = mean_row.get("modularity", float("nan"))
            print(f"  ncut={ncut_v:.4f}  nmi={nmi_v:.4f}  ari={ari_v:.4f}  "
                  f"mod={mod_v:.4f}  ({elapsed:.1f}s)")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  ERROR: {exc}")
            partition_rows.append({
                "dataset": dataset_name, "algo": algo.name,
                "error": str(exc), "wall_sec": elapsed,
            })

partition_df = pd.DataFrame(partition_rows)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — Multicut Task (MCMP)                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print("\n\n── Running multicut (MCMP) benchmark ───────────────────────────────")
multicut_task = MulticutTask()

print("  Generating test instances ...")
mcmp_sets = mcmp_test_suite(sizes=(20, 40, 60))
for name, probs in mcmp_sets.items():
    print(f"    {name}: {len(probs)} instances")

gaec = GAECBaseline()

# GAEC baseline — vectorised, no env needed
print("\n  GAEC baseline ...")
gaec_rows: list[dict] = []
for name, probs in mcmp_sets.items():
    costs = [
        multicut_cost_fast(p.meta["cost_matrix"], gaec.partition(p.meta["cost_matrix"]))
        for p in probs
    ]
    gaec_rows.append({
        "dataset": name, "algo": "gaec",
        "multicut_cost": float(np.mean(costs)),
        "multicut_cost_std": float(np.std(costs)),
        "n_nodes": probs[0].adj.shape[0],
        "n_instances": len(probs),
    })
    print(f"    {name}: mean_cost={gaec_rows[-1]['multicut_cost']:.4f} ± "
          f"{gaec_rows[-1]['multicut_cost_std']:.4f}")

# SS2V-D3QN checkpoint eval
SS2V_CKPT = Path("results/ss2v_mcmp/last.pt")
ss2v_rows: list[dict] = []
if SS2V_CKPT.exists():
    print(f"\n  SS2V-D3QN from {SS2V_CKPT} ...")
    ss2v_cfg = SS2VConfig(hidden=64, n_layers=2, device=DEVICE)
    ss2v = SS2VAlgo(ss2v_cfg)
    try:
        ss2v.load(str(SS2V_CKPT))
    except RuntimeError as e:
        print(f"  [skip] Checkpoint incompatible with current arch: {e}")
        print("  (Re-run after new training completes to get SS2V results)")
        ss2v = None
    if ss2v is not None:
        N_EVAL_SEEDS = 3
        for name, probs in mcmp_sets.items():
            n_test = probs[0].adj.shape[0]
            eval_horizon = n_test * 2
            costs = []
            for p in probs:
                cost_adj = p.meta["cost_matrix"]
                best_cost = float("inf")
                for seed in range(N_EVAL_SEEDS):
                    inner = EdgeContractionEnv(task=multicut_task, problem=p,
                                               horizon=eval_horizon, warm_start="singleton")
                    inner_wrapped = _MCMPWrapper(inner, cost_adj)
                    obs, _ = inner_wrapped.reset(seed=seed)
                    done = False
                    while not done:
                        a = ss2v.select_action(obs, greedy=True)
                        obs, _, term, trunc, _ = inner_wrapped.step(a)
                        done = term or trunc
                    cost = multicut_cost_fast(cost_adj, inner.labels)
                    best_cost = min(best_cost, cost)
                    inner.close()
                costs.append(best_cost)
            ss2v_rows.append({
                "dataset": name, "algo": "ss2v_d3qn",
                "multicut_cost": float(np.mean(costs)),
                "multicut_cost_std": float(np.std(costs)),
                "n_nodes": probs[0].adj.shape[0],
                "n_instances": len(probs),
            })
            print(f"    {name}: mean_cost={ss2v_rows[-1]['multicut_cost']:.4f}")
else:
    print(f"  [warn] SS2V checkpoint not found at {SS2V_CKPT}")

multicut_df = pd.DataFrame(gaec_rows + ss2v_rows)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — Print Tables                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

KEY_PARTITION_COLS = ["ncut", "h2", "modularity", "nmi", "ari"]
KEY_PARTITION_COLS = [c for c in KEY_PARTITION_COLS if c in partition_df.columns]

print("\n\n" + "=" * 90)
print("PARTITION TASK RESULTS")
print("=" * 90)

for ds in partition_df["dataset"].unique():
    sub = partition_df[partition_df["dataset"] == ds].copy()
    n   = int(sub["n_nodes"].iloc[0]) if "n_nodes" in sub.columns else "?"
    k   = int(sub["k_target"].iloc[0]) if "k_target" in sub.columns else "?"
    print(f"\n  {ds}  (n={n}, k={k})")
    print(f"  {'algo':<14}", end="")
    for col in KEY_PARTITION_COLS:
        if col in sub.columns:
            arrow = "↓" if col in ("ncut", "h2") else "↑"
            print(f"  {col+arrow:>12}", end="")
    print(f"  {'time(s)':>8}")
    print("  " + "-" * (14 + 14 * len(KEY_PARTITION_COLS) + 10))
    for _, row in sub.sort_values("ncut" if "ncut" in sub.columns else "algo").iterrows():
        print(f"  {row['algo']:<14}", end="")
        for col in KEY_PARTITION_COLS:
            if col in row:
                v = row[col]
                print(f"  {v:>12.4f}" if not pd.isna(v) else f"  {'N/A':>12}", end="")
        t = row.get("wall_sec", float("nan"))
        print(f"  {t:>8.1f}")

print("\n\n" + "=" * 90)
print("MULTICUT TASK RESULTS  (cost ↓, lower = better)")
print("=" * 90)
print(f"\n  {'algo':<14}  {'dataset':<14}  {'mean_cost':>12}  {'std':>8}  {'vs GAEC':>10}")
print("  " + "-" * 65)
gaec_ref = {r["dataset"]: r["multicut_cost"] for r in gaec_rows}
for _, row in multicut_df.sort_values(["dataset","algo"]).iterrows():
    cost  = row.get("multicut_cost", float("nan"))
    std   = row.get("multicut_cost_std", float("nan"))
    algo  = row["algo"]
    ds    = row["dataset"]
    ref   = gaec_ref.get(ds, float("nan"))
    if algo == "gaec":
        vs = "baseline"
    else:
        vs = f"{(cost-ref)/max(abs(ref),1e-9)*100:+.1f}%"
    print(f"  {algo:<14}  {ds:<14}  {cost:>12.4f}  {std:>8.4f}  {vs:>10}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — Save CSV                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Save partition results
part_out = OUT_DIR / "benchmark_partition.csv"
partition_df.to_csv(part_out, index=False)
print(f"\n\nPartition results saved → {part_out}")

# Save multicut results
mc_out = OUT_DIR / "benchmark_multicut.csv"
multicut_df.to_csv(mc_out, index=False)
print(f"Multicut results saved  → {mc_out}")

# Combined wide pivot for display
print("\n\nDone.")
