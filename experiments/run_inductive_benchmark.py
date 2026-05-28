#!/usr/bin/env python3
"""Inductive Generalization Benchmark Runner — rl-graph-bench.

Evaluates trained GNN-based RL policies zero-shot across graphs of varying scales
without any weight fine-tuning or training updates.
"""
from __future__ import annotations

import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR = OUT_DIR / "checkpoints"

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("=" * 80)
print(f"RL GRAPH BENCH — INDUCTIVE GENERALIZATION BENCHMARK")
print(f"Device: {DEVICE}  |  Out: {OUT_DIR}")
print("=" * 80)

# --- Imports ---
from rlgb.tasks.base import Problem
from rlgb.tasks.graph_partition import GraphPartitionTask
from rlgb.tasks.multicut import MulticutTask, multicut_cost_fast
from rlgb.eval.harness import eval_algo_on_suite
from rlgb.data.synthetic import sbm
from rlgb.data.pyg_loaders import load_planetoid
from rlgb.data.mcmp_instances import er_mcmp, ba_mcmp
from rlgb.baselines.clustering import SpectralBaseline, RandomBaseline
from rlgb.baselines.multicut import GAECBaseline
from rlgb.envs.edge_contraction_env import EdgeContractionEnv

# --- Signed wrapper for MCMP ---
class _MCMPWrapper:
    def __init__(self, env, cost_adj: np.ndarray):
        self._env = env
        self._cost_adj = cost_adj
        self._pos_edge_map = np.empty(0, dtype=np.int32)

    def _inj(self, obs: dict) -> dict:
        obs["adj_signed"] = self._cost_adj
        edge_idx = obs.get("edge_idx", np.empty((0, 2), dtype=np.int32))
        if edge_idx.shape[0] > 0:
            weights  = self._cost_adj[edge_idx[:, 0], edge_idx[:, 1]]
            pos_mask = weights > 0
            self._pos_edge_map = np.where(pos_mask)[0].astype(np.int32)
            obs["edge_idx"] = edge_idx[pos_mask]
        else:
            self._pos_edge_map = np.empty(0, dtype=np.int32)
        obs["n_edges"] = np.array([len(obs["edge_idx"])], dtype=np.int32)
        filtered = obs["edge_idx"]
        labels   = obs["labels"]
        if filtered.shape[0] > 0:
            cs   = np.zeros(filtered.shape[0], dtype=np.float32)
            seen: dict = {}
            for idx in range(len(filtered)):
                u, v  = int(filtered[idx, 0]), int(filtered[idx, 1])
                cu, cv = int(labels[u]), int(labels[v])
                key    = (min(cu, cv), max(cu, cv))
                if key not in seen:
                    seen[key] = float(
                        self._cost_adj[np.ix_(labels == cu, labels == cv)].sum()
                    )
                cs[idx] = seen[key]
            obs["cluster_sums"] = cs
        else:
            obs["cluster_sums"] = np.empty(0, dtype=np.float32)
        return obs

    def reset(self, **kw):
        obs, info = self._env.reset(**kw)
        return self._inj(obs), info

    def step(self, action: int):
        orig = (
            int(self._pos_edge_map[min(action, len(self._pos_edge_map) - 1)])
            if len(self._pos_edge_map) > 0
            else 0
        )
        obs, r, term, trunc, info = self._env.step(orig)
        obs = self._inj(obs)
        if obs["n_edges"][0] == 0:
            term = True
        elif len(obs["cluster_sums"]) > 0 and float(obs["cluster_sums"].max()) <= 0.0:
            term = True
        return obs, r, term, trunc, info

    def close(self):
        self._env.close()

    def __getattr__(self, n):
        return getattr(self._env, n)

eval_rows = []

# ==============================================================================
# TRACK 1: WRT ZERO-SHOT SCALE GENERALIZATION
# ==============================================================================
print("\n--- Track 1: Evaluating WRT Zero-Shot Scale Generalization ---")
from rlgb.algos.structured.wrt import WRTAlgo, WRTConfig
wrt_ckpt = CKPT_DIR / "wrt_sbm.pt"

if not wrt_ckpt.exists():
    print(f"  [error] WRT checkpoint {wrt_ckpt} not found! Skip.")
else:
    wrt_algo = WRTAlgo(WRTConfig(hidden=64, device=DEVICE))
    wrt_algo.load(str(wrt_ckpt))
    print("  WRT model loaded successfully ✓")
    
    partition_task = GraphPartitionTask(objective="ncut")
    wrt_scales = [50, 100, 200, 400]
    
    for n in wrt_scales:
        print(f"  Evaluating SBM scale N={n} (k=4) ...")
        adj, lab, k = sbm(n, k=4, p_in=0.25, p_out=0.02, seed=42)
        suite = [Problem(f"sbm_n{n}_k4", adj, k, lab, "sbm", "partition")]
        
        for algo in [RandomBaseline(), SpectralBaseline(), wrt_algo]:
            t0 = time.perf_counter()
            try:
                df = eval_algo_on_suite(algo, suite, partition_task, n_seeds=3, horizon=30, greedy=True)
                elapsed = time.perf_counter() - t0
                mean_row = df.select_dtypes(include="number").mean().to_dict()
                mean_row["track"] = "wrt_scale"
                mean_row["scale"] = n
                mean_row["algo"] = algo.name
                mean_row["wall_sec"] = elapsed
                eval_rows.append(mean_row)
                print(f"    algo={algo.name:<12} | ncut={mean_row.get('ncut', 0.0):.4f} | nmi={mean_row.get('nmi', 0.0):.4f} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"    algo={algo.name:<12} | [error] {e}")

# ==============================================================================
# TRACK 2: NEUROCUT ZERO-SHOT DOMAIN/SCALE GENERALIZATION
# ==============================================================================
print("\n--- Track 2: Evaluating NeuroCUT Zero-Shot Domain & Scale Generalization ---")
from rlgb.algos.node_move.neurocut import NeuroCUTAlgo, NeuroCUTConfig
neuro_ckpt = CKPT_DIR / "neurocut_cora.pt"

if not neuro_ckpt.exists():
    print(f"  [error] NeuroCUT checkpoint {neuro_ckpt} not found! Skip.")
else:
    neuro_algo = NeuroCUTAlgo(NeuroCUTConfig(hidden=128, device=DEVICE))
    neuro_algo.load(str(neuro_ckpt))
    print("  NeuroCUT model (Cora GNN) loaded successfully ✓")
    
    # 2.1 Scale evaluation on synthetic SBMs
    neuro_scales = [100, 300, 1000]
    for n in neuro_scales:
        print(f"  Evaluating SBM scale N={n} (k=4) ...")
        adj, lab, k = sbm(n, k=4, p_in=0.25, p_out=0.02, seed=42)
        suite = [Problem(f"sbm_n{n}_k4", adj, k, lab, "sbm", "partition")]
        
        for algo in [RandomBaseline(), SpectralBaseline(), neuro_algo]:
            t0 = time.perf_counter()
            try:
                df = eval_algo_on_suite(algo, suite, partition_task, n_seeds=3, horizon=30, greedy=True)
                elapsed = time.perf_counter() - t0
                mean_row = df.select_dtypes(include="number").mean().to_dict()
                mean_row["track"] = "neurocut_scale"
                mean_row["scale"] = n
                mean_row["algo"] = algo.name
                mean_row["wall_sec"] = elapsed
                eval_rows.append(mean_row)
                print(f"    algo={algo.name:<12} | ncut={mean_row.get('ncut', 0.0):.4f} | nmi={mean_row.get('nmi', 0.0):.4f} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"    algo={algo.name:<12} | [error] {e}")
                
    # 2.2 Cross-domain transfer on CiteSeer
    print("\n  Evaluating cross-domain transfer Cora → CiteSeer ...")
    try:
        citeseer_suite = load_planetoid("CiteSeer", k_target=4)
        for algo in [RandomBaseline(), SpectralBaseline(), neuro_algo]:
            t0 = time.perf_counter()
            try:
                df = eval_algo_on_suite(algo, citeseer_suite, partition_task, n_seeds=3, horizon=30, greedy=True)
                elapsed = time.perf_counter() - t0
                mean_row = df.select_dtypes(include="number").mean().to_dict()
                mean_row["track"] = "neurocut_cross_domain"
                mean_row["scale"] = 3327
                mean_row["algo"] = algo.name
                mean_row["wall_sec"] = elapsed
                eval_rows.append(mean_row)
                print(f"    algo={algo.name:<12} | ncut={mean_row.get('ncut', 0.0):.4f} | nmi={mean_row.get('nmi', 0.0):.4f} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"    algo={algo.name:<12} | [error] {e}")
    except Exception as e:
         print(f"    [error] CiteSeer loading failed: {e}")

# ==============================================================================
# TRACK 3: SS2V-D3QN ZERO-SHOT SCALE GENERALIZATION
# ==============================================================================
print("\n--- Track 3: Evaluating SS2V-D3QN Zero-Shot Scale Generalization (ER/BA Mixed) ---")
from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo, SS2VConfig
ss2v_ckpt = CKPT_DIR / "ss2v_mcmp.pt"

if not ss2v_ckpt.exists():
    print(f"  [error] SS2V checkpoint {ss2v_ckpt} not found! Skip.")
else:
    # Pure GNN (Paper Baseline)
    ss2v_algo_pure = SS2VAlgo(SS2VConfig(
        hidden=64, n_layers=2, device=DEVICE,
        hybrid=False
    ))
    ss2v_algo_pure.load(str(ss2v_ckpt))
    
    # Our Hybrid GNN (Contribution)
    ss2v_algo_hybrid = SS2VAlgo(SS2VConfig(
        hidden=64, n_layers=2, device=DEVICE,
        hybrid=True, hybrid_mode="top_k", hybrid_top_k=10
    ))
    ss2v_algo_hybrid.load(str(ss2v_ckpt))
    
    # Our Active MPC GNN (Track D1)
    ss2v_mpc_ckpt = CKPT_DIR / "ss2v_mpc_trained.pt"
    ss2v_algo_mpc = None
    if ss2v_mpc_ckpt.exists():
        ss2v_algo_mpc = SS2VAlgo(SS2VConfig(
            hidden=32, n_layers=2, device=DEVICE,
            hybrid=True, hybrid_mode="top_k", hybrid_top_k=5,
            mpc_planning=True, mpc_horizon=3, mpc_top_k=5
        ))
        ss2v_algo_mpc.load(str(ss2v_mpc_ckpt))
        print("  SS2V MPC model loaded successfully ✓")
    else:
        print("  SS2V MPC checkpoint not found, skipping MPC evaluation.")

    # Our Active MCTS GNN (Track F1)
    ss2v_mcts_ckpt = CKPT_DIR / "ss2v_mcts_trained.pt"
    ss2v_algo_mcts = None
    if ss2v_mcts_ckpt.exists():
        ss2v_algo_mcts = SS2VAlgo(SS2VConfig(
            hidden=32, n_layers=2, device=DEVICE,
            hybrid=True, hybrid_mode="top_k", hybrid_top_k=5,
            mcts_planning=True, mcts_simulations=10, mcts_cpuct=1.5
        ))
        ss2v_algo_mcts.load(str(ss2v_mcts_ckpt))
        print("  SS2V MCTS model loaded successfully ✓")
    else:
        print("  SS2V MCTS checkpoint not found, skipping MCTS evaluation.")
        
    mcmp_scales = [20, 40, 100, 200]
    gaec = GAECBaseline()
    
    for n in mcmp_scales:
        print(f"  Evaluating MCMP scale N={n} ...")
        n_inst = 10 if n >= 100 else 20
        er_probs = er_mcmp(n, n_instances=n_inst, seed_offset=5000 + n)
        ba_probs = ba_mcmp(n, n_instances=n_inst, seed_offset=6000 + n)
        
        for ds_name, probs in [("er", er_probs), ("ba", ba_probs)]:
            # GAEC
            costs_gaec = [multicut_cost_fast(p.meta["cost_matrix"], gaec.partition(p.meta["cost_matrix"])) for p in probs]
            eval_rows.append({
                "track": "ss2v_scale", "scale": n, "algo": "gaec",
                "dataset": f"{ds_name}_mcmp",
                "multicut_cost": float(np.mean(costs_gaec)), "wall_sec": 0.0,
            })
            print(f"    algo=gaec        | scale={n:<4} | ds={ds_name} | cost={eval_rows[-1]['multicut_cost']:.4f}")
            
            # SS2V Pure, Hybrid, MPC & MCTS side-by-side
            algo_configs = [
                ("ss2v_d3qn", ss2v_algo_pure),
                ("ss2v_d3qn_hybrid", ss2v_algo_hybrid)
            ]
            if ss2v_algo_mpc is not None:
                algo_configs.append(("ss2v_d3qn_mpc", ss2v_algo_mpc))
            if ss2v_algo_mcts is not None:
                algo_configs.append(("ss2v_d3qn_mcts", ss2v_algo_mcts))
                
            for algo_label, algo_obj in algo_configs:
                costs_ss2v = []
                t0 = time.perf_counter()
                for p in probs:
                    cost_adj = p.meta["cost_matrix"]
                    inner = EdgeContractionEnv(task=MulticutTask(), problem=p, horizon=n * 2, warm_start="singleton")
                    inner_wrapped = _MCMPWrapper(inner, cost_adj)
                    obs, _ = inner_wrapped.reset()
                    done = False
                    while not done:
                        a = algo_obj.select_action(obs, greedy=True)
                        obs, _, term, trunc, _ = inner_wrapped.step(a)
                        done = term or trunc
                    cost = multicut_cost_fast(cost_adj, inner.labels)
                    costs_ss2v.append(cost)
                    inner.close()
                elapsed = time.perf_counter() - t0
                eval_rows.append({
                    "track": "ss2v_scale", "scale": n, "algo": algo_label,
                    "dataset": f"{ds_name}_mcmp",
                    "multicut_cost": float(np.mean(costs_ss2v)), "wall_sec": elapsed,
                })
                print(f"    algo={algo_label:<16} | scale={n:<4} | ds={ds_name} | cost={eval_rows[-1]['multicut_cost']:.4f} ({elapsed:.1f}s)")

# ==============================================================================
# SECTION 4: SAVE & COMPILE SUMMARY REPORT
# ==============================================================================
print("\n--- Compile & Save Results ---")
df_full = pd.DataFrame(eval_rows)

# Save CSV
csv_out = OUT_DIR / "inductive_benchmark.csv"
df_full.to_csv(csv_out, index=False)
print(f"  Raw CSV saved → {csv_out}")

# Build Markdown Report
report = []
report.append("# Inductive Generalization Benchmark Report — rl-graph-bench\n")
report.append(f"_Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')} | Device: {DEVICE}_\n")
report.append("This report presents the zero-shot scale and domain transfer capabilities of trained GNN-based RL policies compared to standard classical baselines.\n")

# WRT Generalization Table
report.append("## 1. WRT Zero-Shot Scale Generalization")
report.append("WRT trained on $N=300$, evaluated zero-shot at scales from $N=50$ to $N=400$.\n")
sub_wrt = df_full[df_full["track"] == "wrt_scale"]
if not sub_wrt.empty:
    report.append("| Scale (N) | Algorithm | NCut ↓ | NMI ↑ | ARI ↑ | Time (s) |")
    report.append("|-----------|-----------|--------|-------|-------|----------|")
    for _, r in sub_wrt.sort_values(["scale", "ncut"]).iterrows():
        report.append(f"| {r['scale']} | **{r['algo']}** | {r.get('ncut', 0.0):.4f} | {r.get('nmi', 0.0):.4f} | {r.get('ari', 0.0):.4f} | {r['wall_sec']:.1f} |")
report.append("\n")

# NeuroCUT Generalization Table
report.append("## 2. NeuroCUT Zero-Shot scale & Domain Generalization")
report.append("NeuroCUT GNN policy trained on Cora ($N=2708$), evaluated zero-shot across synthetic scales and domain-transferred to CiteSeer ($N=3327$).\n")
sub_neuro = df_full[df_full["track"].isin(["neurocut_scale", "neurocut_cross_domain"])]
if not sub_neuro.empty:
    report.append("| Scale (N) | Domain / Type | Algorithm | NCut ↓ | NMI ↑ | ARI ↑ | Time (s) |")
    report.append("|-----------|---------------|-----------|--------|-------|-------|----------|")
    for _, r in sub_neuro.sort_values(["scale", "ncut"]).iterrows():
        domain_type = "CiteSeer (Cross-Domain)" if r["scale"] == 3327 else f"SBM (Scale N={r['scale']})"
        report.append(f"| {r['scale']} | {domain_type} | **{r['algo']}** | {r.get('ncut', 0.0):.4f} | {r.get('nmi', 0.0):.4f} | {r.get('ari', 0.0):.4f} | {r['wall_sec']:.1f} |")
report.append("\n")

# SS2V Generalization Table
report.append("## 3. SS2V-D3QN Zero-Shot Scale Generalization (ER/BA Mixed)")
report.append("SS2V edge contraction policy trained on $N=40$, evaluated zero-shot across ER/BA scales from $N=20$ to $N=200$.\n")
sub_ss2v = df_full[df_full["track"] == "ss2v_scale"]
if not sub_ss2v.empty:
    report.append("| Scale (N) | Dataset | Algorithm | Mean Cost ↓ | Time (s) |")
    report.append("|-----------|---------|-----------|-------------|----------|")
    for _, r in sub_ss2v.sort_values(["scale", "dataset", "multicut_cost"]).iterrows():
        report.append(f"| {r['scale']} | {r['dataset']} | **{r['algo']}** | {r['multicut_cost']:.4f} | {r['wall_sec']:.1f} |")
report.append("\n")

summary_md = OUT_DIR / "inductive_benchmark_summary.md"
summary_md.write_text("\n".join(report))
print(f"  Summary Markdown saved → {summary_md}")

print("\n" + "=" * 80)
print("INDUCTIVE GENERALIZATION BENCHMARK COMPLETED SUCCESSFULLY!")
print("=" * 80)
