"""WRT P0 verify — City Traffic proxy (SBM n=100, k=4, sparse inter-cluster).

Target: mean NCut <= 0.060 on 4 held-out city-traffic-like graphs.
Paper:  WRT 2025 (RidgeCut), Table 1, City Traffic k=4 n=100.

Training strategy:
  - random warm-start for training (strong RL signal from any state)
  - leiden warm-start for eval (best possible starting point)
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlgb.algos.structured.wrt import WRTAlgo, WRTConfig
from rlgb.envs.structured_env import StructuredPartitionEnv
from rlgb.tasks.graph_partition import GraphPartitionTask
from rlgb.data.synthetic import sbm
from rlgb.tasks.base import Problem
from rlgb.training.ppo import PPOTrainer, PPOConfig
from rlgb.baselines.clustering import LeidenBaseline
from rlgb.eval.harness import eval_algo_on_suite

TARGET  = 0.060
N_TRAIN = 16
N_TEST  = 4
N_TOTAL = N_TRAIN + N_TEST

# ── City Traffic proxy: SBM(n=100, k=4) with very sparse inter-cluster edges ─
def city_traffic_suite(n_graphs: int = 20, seed_offset: int = 0) -> list[Problem]:
    """Generate road-network-like graphs: n=100, k=4, sparse between clusters."""
    problems = []
    for i in range(n_graphs):
        adj, labels, k = sbm(100, 4, p_in=0.12, p_out=0.0008, seed=seed_offset + i)
        problems.append(Problem(
            name=f"city_{i}",
            adj=adj, k_target=k, gt_labels=labels,
            family="city_traffic", task_type="partition",
        ))
    return problems

all_probs  = city_traffic_suite(N_TOTAL, seed_offset=0)
train_probs = all_probs[:N_TRAIN]
test_probs  = all_probs[N_TRAIN:]

task = GraphPartitionTask(objective="ncut")

# ── Baseline: Leiden on test suite ───────────────────────────────────────────
print("Baseline: Leiden on test suite ...")
df_leiden = eval_algo_on_suite(LeidenBaseline(), test_probs, task, n_seeds=3, horizon=1,
                               env_kwargs={"env_class": "structured"})
leiden_ncut = float(df_leiden["ncut"].mean())
print(f"  Leiden NCut = {leiden_ncut:.4f}\n")

# ── WRT training ─────────────────────────────────────────────────────────────
device = "cpu"   # WRT transformer over k=4 clusters is faster on CPU than GPU
cfg = WRTConfig(hidden=64, n_heads=4, n_layers=1, cluster_feat_dim=4,
                lr=3e-4, gamma=0.99, entropy_coef=0.02, value_coef=0.5, device=device)
algo = WRTAlgo(cfg)

import random
rng_train = random.Random(0)

def train_env_fn():
    p = rng_train.choice(train_probs)
    return StructuredPartitionEnv(task, p, horizon=20, warm_start="random")

ppo_cfg = PPOConfig(
    n_episodes=3000, horizon=20,
    n_episodes_per_update=8,
    entropy_coef=0.02, value_coef=0.5,
    lr=3e-4, grad_clip=1.0,
    log_every=200, save_every=0,
    out_dir="results/wrt_city",
)

print(f"Training WRT: {ppo_cfg.n_episodes} episodes, hidden={cfg.hidden}, device={device}")
t0 = time.perf_counter()
trainer = PPOTrainer(algo=algo, env_fn=train_env_fn, config=ppo_cfg)
trainer.train()
print(f"Training done in {time.perf_counter()-t0:.1f}s\n")

# ── Eval on held-out test suite ───────────────────────────────────────────────
print("Evaluating WRT on 4 held-out city-traffic graphs (leiden warm-start) ...")
df_wrt = eval_algo_on_suite(algo, test_probs, task, n_seeds=5,
                            horizon=25, greedy=True,
                            env_kwargs={"warm_start": "leiden", "env_class": "structured"})
wrt_ncut = float(df_wrt["ncut"].mean())

print(f"\n{'='*50}")
print(f"WRT    NCut = {wrt_ncut:.4f}")
print(f"Leiden NCut = {leiden_ncut:.4f}")
print(f"Target      = {TARGET:.4f}")

if wrt_ncut <= TARGET:
    delta = TARGET - wrt_ncut
    print(f"\n[PASS] WRT NCut={wrt_ncut:.4f} <= {TARGET}  (+{delta:.4f})")
    # Save checkpoint
    Path("results/wrt_city").mkdir(exist_ok=True)
    algo.save("results/wrt_city/best.pt")
    sys.exit(0)
else:
    gap = wrt_ncut - TARGET
    print(f"\n[FAIL] WRT NCut={wrt_ncut:.4f} > {TARGET}  (gap={gap:.4f})")
    sys.exit(1)
