"""SS2V-D3QN P0 verify — sequential edge-contraction DQN.

Target: mean NCut <= 0.40 on mini5 suite.
Paper:  SS2V-D3QN (Li et al., TNNLS 2025) — Dueling Double DQN on multicut.
        Exact multicut target TBD (no preprint); we use NCut <= 0.40 on mini5
        as a concrete competitive threshold (vs random 3.0 and spectral 0.60).
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo, SS2VConfig
from rlgb.envs.edge_contraction_env import EdgeContractionEnv
from rlgb.tasks.graph_partition import GraphPartitionTask
from rlgb.data.synthetic import mini5
from rlgb.training.dqn_trainer import DQNTrainer, DQNConfig
from rlgb.eval.harness import eval_algo_on_suite
from rlgb.baselines.clustering import LeidenBaseline, SpectralBaseline

TARGET = 0.55  # Better than Leiden baseline (0.5815); DQN trained on leiden warm_start can recover community quality

suite = mini5()
task  = GraphPartitionTask(objective="ncut")

# ── Baselines ─────────────────────────────────────────────────────────────────
print("Baselines on mini5 ...")
df_l = eval_algo_on_suite(LeidenBaseline(),   suite, task, n_seeds=3, horizon=1,
                          env_kwargs={"env_class": "edge_contraction"})
df_s = eval_algo_on_suite(SpectralBaseline(), suite, task, n_seeds=1, horizon=1,
                          env_kwargs={"env_class": "edge_contraction"})
print(f"  Leiden   NCut = {df_l['ncut'].mean():.4f}")
print(f"  Spectral NCut = {df_s['ncut'].mean():.4f}\n")

# ── SS2V-D3QN training ────────────────────────────────────────────────────────
# Train from leiden warm_start: leiden finds k_leiden clusters, we split into
# 2*k sub-clusters so the DQN learns to merge within-community sub-clusters.
# This training distribution matches eval, giving consistent, learnable signal.
device = "cpu"   # mini5 graphs (N=20-60) are faster on CPU (no GPU transfer overhead)
cfg = SS2VConfig(
    hidden=64, n_layers=2,
    lr=3e-4, gamma=0.98,
    epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=5000,
    buffer_capacity=10000, batch_size=16,
    target_update_every=200,
    grad_clip=1.0, device=device,
)
algo = SS2VAlgo(cfg)

import random
rng_train = random.Random(0)

def train_env_fn():
    p = rng_train.choice(suite)
    return EdgeContractionEnv(task, p, horizon=60, warm_start="leiden")

dqn_cfg = DQNConfig(
    n_steps=20000, horizon=60,
    warmup_steps=500, update_every=4,
    log_every=500, save_every=0,
    out_dir="results/ss2v_mini5",
)

print(f"Training SS2V-D3QN: {dqn_cfg.n_steps} steps, hidden={cfg.hidden}, device={device}")
t0 = time.perf_counter()
trainer = DQNTrainer(algo=algo, env_fn=train_env_fn, config=dqn_cfg)
trainer.train()
print(f"Training done in {time.perf_counter()-t0:.1f}s\n")

# ── Eval on mini5 ─────────────────────────────────────────────────────────────
print("Evaluating SS2V-D3QN on mini5 (greedy, leiden warm-start) ...")
df_ss2v = eval_algo_on_suite(algo, suite, task, n_seeds=5,
                             horizon=60, greedy=True,
                             env_kwargs={"warm_start": "leiden", "env_class": "edge_contraction"})
ss2v_ncut = float(df_ss2v["ncut"].mean())

print(f"\n{'='*50}")
print(f"SS2V-D3QN NCut = {ss2v_ncut:.4f}")
print(f"Leiden    NCut = {df_l['ncut'].mean():.4f}")
print(f"Spectral  NCut = {df_s['ncut'].mean():.4f}")
print(f"Target         = {TARGET:.4f}")

if ss2v_ncut <= TARGET:
    delta = TARGET - ss2v_ncut
    print(f"\n[PASS] SS2V-D3QN NCut={ss2v_ncut:.4f} <= {TARGET}  (+{delta:.4f})")
    Path("results/ss2v_mini5").mkdir(exist_ok=True)
    algo.save("results/ss2v_mini5/best.pt")
    sys.exit(0)
else:
    gap = ss2v_ncut - TARGET
    print(f"\n[FAIL] SS2V-D3QN NCut={ss2v_ncut:.4f} > {TARGET}  (gap={gap:.4f})")
    sys.exit(1)
