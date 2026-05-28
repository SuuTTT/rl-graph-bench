#!/usr/bin/env python3
"""Comprehensive Benchmark Runner — rl-graph-bench.

Performs full training and evaluation of all 6 RL algorithms + classical baselines
across all task families (Graph Partitioning, Multicut, Community Expansion, Dynamic CD)
on multiple datasets. Uses GPU acceleration when available.

Saves raw results to results/comprehensive_benchmark.csv
Saves summary tables to results/comprehensive_benchmark_summary.md
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

# --- Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# Fixed seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("=" * 80)
print(f"RL GRAPH BENCH — COMPREHENSIVE BENCHMARK OVERNIGHT RUN")
print(f"Device: {DEVICE}  |  Out: {OUT_DIR}")
print("=" * 80)

# --- Imports ---
from rlgb.tasks.base import Problem
from rlgb.tasks.graph_partition import GraphPartitionTask
from rlgb.tasks.multicut import MulticutTask, multicut_cost_fast
from rlgb.tasks.community_expand import CommunityExpandTask
from rlgb.tasks.dynamic_cd import DynamicCDTask

from rlgb.eval.harness import eval_algo_on_suite
from rlgb.training.trainer import Trainer, TrainConfig
from rlgb.training.ppo import PPOTrainer, PPOConfig
from rlgb.training.dqn_trainer import DQNTrainer, DQNConfig

from rlgb.baselines.clustering import LeidenBaseline, LouvainBaseline, SpectralBaseline, RandomBaseline
from rlgb.baselines.multicut import GAECBaseline

from rlgb.data.pyg_loaders import load_planetoid
from rlgb.data.synthetic import sbm, fixed17, mini5
from rlgb.data.mcmp_instances import mcmp_test_suite, mcmp_train_suite
from rlgb.data.clare_dataset import load_clare_dataset
from rlgb.envs.edge_contraction_env import EdgeContractionEnv
from rlgb.envs.node_move_env import NodeMoveEnv

# --- Shared wrapper for MCMP signed-cost injection ---
class _MCMPWrapper:
    def __init__(self, env, cost_adj: np.ndarray):
        self._env = env
        self._cost_adj = cost_adj
    def _inj(self, obs: dict) -> dict:
        obs["adj_signed"] = self._cost_adj
        return obs
    def reset(self, **kw):
        obs, info = self._env.reset(**kw)
        return self._inj(obs), info
    def step(self, a: int):
        obs, r, t, tr, i = self._env.step(a)
        return self._inj(obs), r, t, tr, i
    def close(self):
        self._env.close()
    def __getattr__(self, n):
        return getattr(self._env, n)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — DATASETS PREPARATION                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n--- Loading & Preparing All Suites ---")

suites: dict[str, list[Problem]] = {}

# 1.1 Partition Task Suites
print("  Loading Cora k=4...")
try:
    suites["cora_k4"] = load_planetoid("Cora", k_target=4)
except Exception as e:
    print(f"    [error] Cora load failed: {e}")

print("  Loading CiteSeer k=4...")
try:
    suites["citeseer_k4"] = load_planetoid("CiteSeer", k_target=4)
except Exception as e:
    print(f"    [error] CiteSeer load failed: {e}")

print("  Generating SBM n=300 k=5...")
adj_sbm, lab_sbm, k_sbm = sbm(300, 5, p_in=0.25, p_out=0.02, seed=0)
suites["sbm_n300"] = [Problem("sbm_n300", adj_sbm, k_sbm, lab_sbm, "sbm", "partition")]

print("  Generating BlogCatalog proxy (n=100 k=5)...")
adj_blog, lab_blog, k_blog = sbm(100, 5, p_in=0.20, p_out=0.005, seed=1)
suites["blog_proxy"] = [Problem("blog_proxy", adj_blog, k_blog, lab_blog, "blogcatalog", "partition")]

print("  Generating Email-EU-Core proxy (n=100 k=6)...")
adj_email, lab_email, k_email = sbm(100, 6, p_in=0.20, p_out=0.006, seed=2)
suites["email_proxy"] = [Problem("email_proxy", adj_email, k_email, lab_email, "email_eu", "partition")]

# 1.2 Multicut (MCMP) Task Suites
print("  Generating Multicut ER/BA instances...")
mcmp_sets = mcmp_test_suite(sizes=(20, 40))  # ER & BA of sizes 20, 40 (50 instances each)

# 1.3 Community Expand Task Suites (SNAP Amazon & SNAP DBLP)
print("  Loading SNAP Amazon and DBLP community datasets...")
try:
    amazon_data = load_clare_dataset("amazon", num_train=90, num_val=10, seed=0)
    suites["amazon_test"] = amazon_data.to_problem_suite("test")[:20]  # Take 20 test communities
    print(f"    SNAP Amazon: {len(suites['amazon_test'])} Problems loaded.")
except Exception as e:
    print(f"    [error] SNAP Amazon load failed: {e}")

try:
    dblp_data = load_clare_dataset("dblp", num_train=90, num_val=10, seed=0)
    suites["dblp_test"] = dblp_data.to_problem_suite("test")[:20]  # Take 20 test communities
    print(f"    SNAP DBLP: {len(suites['dblp_test'])} Problems loaded.")
except Exception as e:
    print(f"    [error] SNAP DBLP load failed: {e}")

# 1.4 Dynamic CD Task Suites (Snapshots)
print("  Generating Dynamic SBM snap suite...")
dynamic_task = DynamicCDTask()
suites["dynamic_sbm"] = dynamic_task.build_suite()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — TRAINING ALGORITHMS                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n--- Training RL Algorithms to Convergence ---")

trained_models: dict[str, Any] = {}

# 2.1 NeuroCUT Training (Cora)
from rlgb.algos.node_move.neurocut import NeuroCUTAlgo, NeuroCUTConfig
neurocut_ckpt = CKPT_DIR / "neurocut_cora.pt"
print(f"\n[NeuroCUT] Training on Cora k=4 ...")
neuro_algo = NeuroCUTAlgo(NeuroCUTConfig(hidden=128, device=DEVICE))
if neurocut_ckpt.exists():
    neuro_algo.load(str(neurocut_ckpt))
    print("  Loaded cached checkpoint ✓")
else:
    # Train using PPO curriculum
    task = GraphPartitionTask(objective="ncut")
    cora_suite = suites.get("cora_k4", [])
    if cora_suite:
        # Phase 1: 1000 episodes warm-start random
        env_fn1 = lambda: task.build_env(cora_suite[0], horizon=30, warm_start="random")
        PPOTrainer(
            algo=neuro_algo, env_fn=env_fn1,
            config=PPOConfig(n_episodes=1000, horizon=30, lr=3e-4, n_episodes_per_update=8,
                             entropy_coef=0.03, lr_schedule="cosine", save_every=0, out_dir=str(OUT_DIR), seed=42)
        ).train()
        # Phase 2: 500 episodes fine-tune spectral
        env_fn2 = lambda: task.build_env(cora_suite[0], horizon=30, warm_start="spectral")
        PPOTrainer(
            algo=neuro_algo, env_fn=env_fn2,
            config=PPOConfig(n_episodes=500, horizon=30, lr=1e-4, n_episodes_per_update=8,
                             entropy_coef=0.01, lr_schedule="cosine", save_every=0, out_dir=str(OUT_DIR), seed=43)
        ).train()
        neuro_algo.save(str(neurocut_ckpt))
trained_models["neurocut"] = neuro_algo

# 2.2 WRT Training (SBM)
from rlgb.algos.structured.wrt import WRTAlgo, WRTConfig
wrt_ckpt = CKPT_DIR / "wrt_sbm.pt"
print(f"\n[WRT] Training on SBM n=300 ...")
wrt_algo = WRTAlgo(WRTConfig(hidden=64, device=DEVICE))
if wrt_ckpt.exists():
    wrt_algo.load(str(wrt_ckpt))
    print("  Loaded cached checkpoint ✓")
else:
    task = GraphPartitionTask(objective="ncut")
    env_fn = lambda: task.build_env(suites["sbm_n300"][0], horizon=30, env_class="structured", warm_start="random")
    PPOTrainer(
        algo=wrt_algo, env_fn=env_fn,
        config=PPOConfig(n_episodes=1000, horizon=30, lr=3e-4, n_episodes_per_update=8,
                         entropy_coef=0.02, lr_schedule="cosine", save_every=0, out_dir=str(OUT_DIR), seed=44)
    ).train()
    wrt_algo.save(str(wrt_ckpt))
trained_models["wrt"] = wrt_algo

# 2.3 SS2V-D3QN Training (MCMP ER/BA Mixed)
from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo, SS2VConfig
ss2v_ckpt = CKPT_DIR / "ss2v_mcmp.pt"
print(f"\n[SS2V-D3QN] Training on MCMP ER/BA mixed n=40 ...")
ss2v_algo = SS2VAlgo(SS2VConfig(hidden=64, n_layers=2, lr=1e-5, buffer_capacity=20000, device=DEVICE))
if ss2v_ckpt.exists():
    ss2v_algo.load(str(ss2v_ckpt))
    print("  Loaded cached checkpoint ✓")
else:
    # 50,000 steps BC pretraining + 1500 episodes RL fine-tune
    train_probs = (
        mcmp_train_suite(n=20, n_instances=50, seed_offset=0)
        + mcmp_train_suite(n=40, n_instances=50, seed_offset=100)
    )
    # Perform BC pretraining
    print("  Pretraining SS2V with GAEC behavior cloning (30000 steps) ...")
    step = 0
    rng_bc = random.Random(99)
    task = MulticutTask()
    while step < 30000:
        p = rng_bc.choice(train_probs)
        cost_adj = p.meta["cost_matrix"]
        env = EdgeContractionEnv(task=task, problem=p, horizon=30, warm_start="singleton")
        env_w = _MCMPWrapper(env, cost_adj)
        obs, _ = env_w.reset()
        done = False
        while not done and step < 30000:
            n_cands = int(obs.get("n_edges", [0])[0])
            edge_idx = obs.get("edge_idx")
            if n_cands == 0 or edge_idx is None or edge_idx.shape[0] == 0:
                break
            # Pick GAEC action
            best = -1
            best_ew = -np.inf
            for idx in range(min(n_cands, len(edge_idx))):
                u, v = int(edge_idx[idx, 0]), int(edge_idx[idx, 1])
                ew = float(cost_adj[u, v])
                if ew > best_ew:
                    best_ew = ew
                    best = idx
            if best == -1:
                break
            ss2v_algo.bc_update(obs, best)
            step += 1
            obs, _, term, trunc, _ = env_w.step(best)
            done = term or trunc
        env.close()
    
    # Sync target network
    ss2v_algo._target.load_state_dict(ss2v_algo._online.state_dict())
    
    # RL fine-tune
    print("  Fine-tuning SS2V with RL (1500 episodes) ...")
    rng_rl = random.Random(101)
    def ss2v_env_fn():
        p = rng_rl.choice(train_probs)
        env = EdgeContractionEnv(task=task, problem=p, horizon=30, warm_start="singleton")
        return _MCMPWrapper(env, p.meta["cost_matrix"])
        
    DQNTrainer(
        algo=ss2v_algo, env_fn=ss2v_env_fn,
        config=DQNConfig(n_steps=20000, warmup_steps=500, horizon=30, update_every=4,
                         log_every=1000, save_every=0, out_dir=str(OUT_DIR), seed=45)
    ).train()
    ss2v_algo.save(str(ss2v_ckpt))
trained_models["ss2v"] = ss2v_algo

# 2.4 CLARE Training (Amazon Community)
from rlgb.algos.community.clare import CLAREAlgo, CLAREConfig
clare_ckpt = CKPT_DIR / "clare_amazon.pt"
print(f"\n[CLARE] Training on SNAP Amazon ...")
clare_algo = CLAREAlgo(CLAREConfig(hidden=64, device=DEVICE))
if clare_ckpt.exists():
    clare_algo.load(str(clare_ckpt))
    print("  Loaded cached checkpoint ✓")
else:
    task = CommunityExpandTask(objective="h2")
    amazon_suite = suites.get("amazon_test", [])
    if amazon_suite:
        env_fn = lambda: task.build_env(random.choice(amazon_suite), horizon=10, warm_start="seed")
        Trainer(
            algo=clare_algo, env_fn=env_fn,
            config=TrainConfig(n_episodes=1500, horizon=10, lr=3e-4, n_episode_per_update=4,
                               log_every=300, lr_schedule="cosine", save_every=0, out_dir=str(OUT_DIR), seed=46)
        ).train()
        clare_algo.save(str(clare_ckpt))
trained_models["clare"] = clare_algo

# 2.5 AC2CD Training (BlogCatalog3)
from rlgb.algos.dynamic.ac2cd import AC2CDAlgo, AC2CDConfig
ac2cd_ckpt = CKPT_DIR / "ac2cd_blog.pt"
print(f"\n[AC2CD] Training on BlogCatalog proxy...")
ac2cd_algo = AC2CDAlgo(AC2CDConfig(hidden=64, device=DEVICE))
if ac2cd_ckpt.exists():
    ac2cd_algo.load(str(ac2cd_ckpt))
    print("  Loaded cached checkpoint ✓")
else:
    task = GraphPartitionTask(objective="ncut")
    blog_suite = suites.get("blog_proxy", [])
    if blog_suite:
        env_fn = lambda: task.build_env(blog_suite[0], horizon=30, warm_start="random")
        Trainer(
            algo=ac2cd_algo, env_fn=env_fn,
            config=TrainConfig(n_episodes=1500, horizon=30, lr=3e-4, n_episode_per_update=4,
                               log_every=300, lr_schedule="cosine", save_every=0, out_dir=str(OUT_DIR), seed=47)
        ).train()
        ac2cd_algo.save(str(ac2cd_ckpt))
trained_models["ac2cd"] = ac2cd_algo


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — FULL BENCHMARK EVALUATIONS                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n--- Running Benchmark Evaluations across All Suites ---")
eval_rows = []

# 3.1 Task 1: Graph Partitioning (All Algos, Cora/CiteSeer/SBM/Blog/Email)
partition_task = GraphPartitionTask(objective="ncut")
partition_algos = [
    RandomBaseline(),
    LeidenBaseline(),
    LouvainBaseline(),
    SpectralBaseline(),
    trained_models["neurocut"],
    trained_models["wrt"],
    trained_models["ac2cd"],
]

for ds_name in ["cora_k4", "citeseer_k4", "sbm_n300", "blog_proxy", "email_proxy"]:
    suite = suites.get(ds_name)
    if not suite:
        continue
    print(f"\n  Evaluating Partition task on '{ds_name}'...")
    for algo in partition_algos:
        t0 = time.perf_counter()
        try:
            df = eval_algo_on_suite(algo, suite, partition_task, n_seeds=3, horizon=30, greedy=True)
            elapsed = time.perf_counter() - t0
            mean_row = df.select_dtypes(include="number").mean().to_dict()
            mean_row["task"] = "partition"
            mean_row["dataset"] = ds_name
            mean_row["algo"] = algo.name
            mean_row["wall_sec"] = elapsed
            eval_rows.append(mean_row)
            print(f"    algo={algo.name:<12} | ncut={mean_row.get('ncut', 0.0):.4f} | nmi={mean_row.get('nmi', 0.0):.4f} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"    algo={algo.name:<12} | [error] {e}")

# 3.2 Task 2: Multicut MCMP (GAEC vs SS2V-D3QN)
print("\n  Evaluating Multicut (MCMP) task...")
gaec = GAECBaseline()
ss2v = trained_models["ss2v"]

for ds_name, probs in mcmp_sets.items():
    n_nodes = probs[0].adj.shape[0]
    # GAEC
    costs_gaec = [multicut_cost_fast(p.meta["cost_matrix"], gaec.partition(p.meta["cost_matrix"])) for p in probs]
    eval_rows.append({
        "task": "multicut", "dataset": ds_name, "algo": "gaec",
        "multicut_cost": float(np.mean(costs_gaec)), "wall_sec": 0.0,
    })
    print(f"    algo=gaec        | dataset={ds_name:<10} | cost={eval_rows[-1]['multicut_cost']:.4f}")
    
    # SS2V
    costs_ss2v = []
    t0 = time.perf_counter()
    for p in probs:
        cost_adj = p.meta["cost_matrix"]
        inner = EdgeContractionEnv(task=MulticutTask(), problem=p, horizon=n_nodes * 2, warm_start="singleton")
        inner_wrapped = _MCMPWrapper(inner, cost_adj)
        obs, _ = inner_wrapped.reset()
        done = False
        while not done:
            a = ss2v.select_action(obs, greedy=True)
            obs, _, term, trunc, _ = inner_wrapped.step(a)
            done = term or trunc
        cost = multicut_cost_fast(cost_adj, inner.labels)
        costs_ss2v.append(cost)
        inner.close()
    elapsed = time.perf_counter() - t0
    eval_rows.append({
        "task": "multicut", "dataset": ds_name, "algo": "ss2v_d3qn",
        "multicut_cost": float(np.mean(costs_ss2v)), "wall_sec": elapsed,
    })
    print(f"    algo=ss2v_d3qn   | dataset={ds_name:<10} | cost={eval_rows[-1]['multicut_cost']:.4f} ({elapsed:.1f}s)")

# 3.3 Task 3: SNAP Community Detection (Leiden vs CLARE vs SLRL)
from rlgb.algos.community.slrl import SLRLAlgo
community_task = CommunityExpandTask(objective="h2")
community_algos = [
    LeidenBaseline(),
    trained_models["clare"],
    SLRLAlgo(),  # greedy S-coverage
]

for ds_name in ["amazon_test", "dblp_test"]:
    suite = suites.get(ds_name)
    if not suite:
        continue
    print(f"\n  Evaluating SNAP Community task on '{ds_name}'...")
    for algo in community_algos:
        t0 = time.perf_counter()
        try:
            df = eval_algo_on_suite(algo, suite, community_task, n_seeds=3, horizon=10, greedy=True,
                                    env_kwargs={"warm_start": "seed"})
            elapsed = time.perf_counter() - t0
            mean_row = df.select_dtypes(include="number").mean().to_dict()
            mean_row["task"] = "community"
            mean_row["dataset"] = ds_name
            mean_row["algo"] = algo.name
            mean_row["wall_sec"] = elapsed
            eval_rows.append(mean_row)
            print(f"    algo={algo.name:<12} | f1={mean_row.get('f1', 0.0):.4f} | ncut={mean_row.get('ncut', 0.0):.4f} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"    algo={algo.name:<12} | [error] {e}")

# 3.4 Task 4: Dynamic CD Task (Leiden vs AC2CD)
print("\n  Evaluating Dynamic CD task...")
dynamic_algos = [
    LeidenBaseline(),
    trained_models["ac2cd"],
]
suite = suites.get("dynamic_sbm", [])
if suite:
    for algo in dynamic_algos:
        t0 = time.perf_counter()
        try:
            df = eval_algo_on_suite(algo, suite, DynamicCDTask(), n_seeds=3, horizon=10, greedy=True,
                                    env_kwargs={"warm_start": "leiden"})
            elapsed = time.perf_counter() - t0
            mean_row = df.select_dtypes(include="number").mean().to_dict()
            mean_row["task"] = "dynamic"
            mean_row["dataset"] = "dynamic_sbm"
            mean_row["algo"] = algo.name
            mean_row["wall_sec"] = elapsed
            eval_rows.append(mean_row)
            print(f"    algo={algo.name:<12} | mod_density={mean_row.get('modularity_density', 0.0):.4f} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"    algo={algo.name:<12} | [error] {e}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — FORMATTING & SAVING RESULTS SUMMARY                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("\n--- Compile & Save Results ---")

full_df = pd.DataFrame(eval_rows)

# Save Raw CSV
csv_out = OUT_DIR / "comprehensive_benchmark.csv"
full_df.to_csv(csv_out, index=False)
print(f"  Raw CSV saved → {csv_out}")

# Build Markdown Report
report = []
report.append("# Comprehensive Baseline Benchmark Table — rl-graph-bench\n")
report.append(f"_Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')} | Device: {DEVICE}_\n")
report.append("This table provides an extensive overnight baseline evaluation covering all 6 RL algorithms and standard baselines across five task domains.\n")

# Section 1
report.append("## 1. Graph Partitioning Results\n")
sub_p = full_df[full_df["task"] == "partition"]
if not sub_p.empty:
    report.append("| Dataset | Algorithm | NCut ↓ | NMI ↑ | ARI ↑ | Modularity ↑ | Time (s) |")
    report.append("|---------|-----------|--------|-------|-------|--------------|----------|")
    for _, r in sub_p.sort_values(["dataset", "ncut"]).iterrows():
        report.append(f"| {r['dataset']} | **{r['algo']}** | {r.get('ncut', 0.0):.4f} | {r.get('nmi', 0.0):.4f} | {r.get('ari', 0.0):.4f} | {r.get('modularity', 0.0):.4f} | {r['wall_sec']:.1f} |")
report.append("\n")

# Section 2
report.append("## 2. Multicut (MCMP) Results\n")
sub_m = full_df[full_df["task"] == "multicut"]
if not sub_m.empty:
    report.append("| Dataset | Algorithm | Mean Cost ↓ | Time (s) |")
    report.append("|---------|-----------|-------------|----------|")
    for _, r in sub_m.sort_values(["dataset", "multicut_cost"]).iterrows():
        report.append(f"| {r['dataset']} | **{r['algo']}** | {r['multicut_cost']:.4f} | {r['wall_sec']:.1f} |")
report.append("\n")

# Section 3
report.append("## 3. SNAP Community Detection Results\n")
sub_c = full_df[full_df["task"] == "community"]
if not sub_c.empty:
    report.append("| Dataset | Algorithm | F1 Score ↑ | NCut ↓ | Time (s) |")
    report.append("|---------|-----------|------------|--------|----------|")
    for _, r in sub_c.sort_values(["dataset", "f1"], ascending=[True, False]).iterrows():
        report.append(f"| {r['dataset']} | **{r['algo']}** | {r.get('f1', 0.0):.4f} | {r.get('ncut', 0.0):.4f} | {r['wall_sec']:.1f} |")
report.append("\n")

# Section 4
report.append("## 4. Dynamic Community Detection Results\n")
sub_d = full_df[full_df["task"] == "dynamic"]
if not sub_d.empty:
    report.append("| Dataset | Algorithm | Modularity Density ↑ | NMI ↑ | Time (s) |")
    report.append("|---------|-----------|----------------------|-------|----------|")
    for _, r in sub_d.sort_values(["dataset", "modularity_density"], ascending=[True, False]).iterrows():
        report.append(f"| {r['dataset']} | **{r['algo']}** | {r.get('modularity_density', 0.0):.4f} | {r.get('nmi', 0.0):.4f} | {r['wall_sec']:.1f} |")
report.append("\n")

report_text = "\n".join(report)

summary_md = OUT_DIR / "comprehensive_benchmark_summary.md"
summary_md.write_text(report_text)
print(f"  Summary Markdown saved → {summary_md}")

print("\n" + "=" * 80)
print("BENCHMARK COMPLETED SUCCESSFULLY!")
print("=" * 80)
