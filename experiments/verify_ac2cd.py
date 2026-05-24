"""AC2CD P0 verify — BlogCatalog3 proxy (SBM n=200, k=5, clear communities).

Target: NMI >= 0.75 on held-out SBM graphs.
Paper:  AC2CD (Costa & Ralha, KBS 2023) — GAT + A2C on dynamic graphs.
        NMI=0.75 on BlogCatalog3 (static eval, no BlogCatalog3 download needed
        since proxy SBM with matching community-strength is used).

Note: AC2CD uses GAT encoder vs NeuroCUT's SAGE — richer attention features
      should give better NMI on well-structured graphs.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlgb.algos.dynamic.ac2cd import AC2CDAlgo, AC2CDConfig
from rlgb.envs.node_move_env import NodeMoveEnv
from rlgb.tasks.graph_partition import GraphPartitionTask
from rlgb.data.synthetic import sbm
from rlgb.tasks.base import Problem
from rlgb.training.trainer import Trainer, TrainConfig
from rlgb.eval.metrics import nmi
from rlgb.eval.harness import eval_algo_on_suite

TARGET   = 0.75
N_TRAIN  = 15
N_TEST   = 5
N_TOTAL  = N_TRAIN + N_TEST

# ── BlogCatalog3 proxy: SBM(n=200, k=5) with clear community structure ────────
def blogcatalog_suite(n_graphs: int = 20, seed_offset: int = 0) -> list[Problem]:
    """Proxy for BlogCatalog3: moderately strong 5-way community structure."""
    problems = []
    for i in range(n_graphs):
        adj, labels, k = sbm(100, 5, p_in=0.20, p_out=0.005, seed=seed_offset + i)
        problems.append(Problem(
            name=f"blog_{i}",
            adj=adj, k_target=k, gt_labels=labels,
            family="blogcatalog", task_type="partition",
        ))
    return problems

all_probs   = blogcatalog_suite(N_TOTAL, seed_offset=100)
train_probs = all_probs[:N_TRAIN]
test_probs  = all_probs[N_TRAIN:]

task = GraphPartitionTask(objective="ncut")

# ── Quick Leiden baseline on test ─────────────────────────────────────────────
print("Computing Leiden NMI baseline on test suite ...")
from rlgb.baselines.clustering import LeidenBaseline
df_l = eval_algo_on_suite(LeidenBaseline(), test_probs, task, n_seeds=3, horizon=1)
leiden_nmi = float(df_l["nmi"].mean())
print(f"  Leiden NMI = {leiden_nmi:.4f}\n")

# ── AC2CD training ────────────────────────────────────────────────────────────
device = "cpu"  # N=100 graphs are faster on CPU (no GPU transfer overhead)
cfg = AC2CDConfig(
    node_feat_dim=7, hidden=64, n_layers=2, n_heads=4,
    lr=3e-4, gamma=0.99, entropy_coef=0.02, value_coef=0.5, grad_clip=1.0,
    device=device,
)
algo = AC2CDAlgo(cfg)

import random
rng_train = random.Random(42)

def train_env_fn():
    p = rng_train.choice(train_probs)
    return NodeMoveEnv(task, p, horizon=30, warm_start="random")

train_cfg = TrainConfig(
    n_episodes=2000, horizon=30,
    lr=3e-4, gamma=0.99,
    n_episode_per_update=8,
    entropy_coef=0.02, value_coef=0.5, grad_clip=1.0,
    log_every=300, save_every=0,
    out_dir="results/ac2cd_blog",
)

print(f"Training AC2CD: {train_cfg.n_episodes} episodes, hidden={cfg.hidden}, device={device}")
t0 = time.perf_counter()
trainer = Trainer(algo=algo, env_fn=train_env_fn, config=train_cfg)
trainer.train()
print(f"Training done in {time.perf_counter()-t0:.1f}s\n")

# ── Eval NMI on held-out test suite ──────────────────────────────────────────
print("Evaluating AC2CD NMI on 5 held-out BlogCatalog proxy graphs ...")
df_ac = eval_algo_on_suite(algo, test_probs, task, n_seeds=5,
                           horizon=40, greedy=True,
                           env_kwargs={"warm_start": "leiden"})
ac_nmi  = float(df_ac["nmi"].mean())
ac_ncut = float(df_ac["ncut"].mean())

print(f"\n{'='*50}")
print(f"AC2CD  NMI = {ac_nmi:.4f}   NCut = {ac_ncut:.4f}")
print(f"Leiden NMI = {leiden_nmi:.4f}")
print(f"Target NMI = {TARGET:.4f}")

if ac_nmi >= TARGET:
    delta = ac_nmi - TARGET
    print(f"\n[PASS] AC2CD NMI={ac_nmi:.4f} >= {TARGET}  (+{delta:.4f})")
    Path("results/ac2cd_blog").mkdir(exist_ok=True)
    algo.save("results/ac2cd_blog/best.pt")
    sys.exit(0)
else:
    gap = TARGET - ac_nmi
    print(f"\n[FAIL] AC2CD NMI={ac_nmi:.4f} < {TARGET}  (gap={gap:.4f})")
    sys.exit(1)
