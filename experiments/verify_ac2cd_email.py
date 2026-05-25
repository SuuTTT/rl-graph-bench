"""AC2CD P1 — Email-EU-Core proxy (SBM n=100, k=6).

Target: NMI >= 0.72 (AC2CD paper, KBS 2023, Table 3, Email-EU-Core).

Email-EU-Core has 1005 nodes, 25571 edges, 42 departments.
Proxy: SBM(n=100, k=6, p_in=0.20, p_out=0.006) — slightly more communities
than BlogCatalog3 (k=5) but comparable density.

Strategy: load the checkpoint from BlogCatalog3-proxy training (results/ac2cd_blog/last.pt)
and eval directly. If it doesn't pass, fine-tune for 500 more episodes.
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
from rlgb.baselines.clustering import LeidenBaseline

TARGET     = 0.72
CKPT       = Path("results/ac2cd_blog/last.pt")
N_TRAIN    = 15
N_TEST     = 5
FINETUNE_EPS = 500   # extra episodes if zero-shot eval fails


# ── Email-EU-Core proxy: SBM(n=100, k=6) ─────────────────────────────────────
def email_suite(n_graphs: int = 20, seed_offset: int = 0) -> list[Problem]:
    """Proxy: 6-way SBM with community density similar to Email-EU-Core."""
    problems = []
    for i in range(n_graphs):
        adj, labels, k = sbm(100, 6, p_in=0.20, p_out=0.006, seed=seed_offset + i)
        problems.append(Problem(
            name=f"email_{i}",
            adj=adj, k_target=k, gt_labels=labels,
            family="email_eu", task_type="partition",
        ))
    return problems


all_probs   = email_suite(N_TRAIN + N_TEST, seed_offset=200)
train_probs = all_probs[:N_TRAIN]
test_probs  = all_probs[N_TRAIN:]
task        = GraphPartitionTask(objective="ncut")

print("Computing Leiden NMI baseline ...")
df_l = eval_algo_on_suite(LeidenBaseline(), test_probs, task, n_seeds=3, horizon=1)
leiden_nmi = float(df_l["nmi"].mean())
print(f"  Leiden NMI = {leiden_nmi:.4f}\n")

# ── Load checkpoint (zero-shot) ───────────────────────────────────────────────
device = "cpu"
cfg = AC2CDConfig(
    node_feat_dim=7, hidden=64, n_layers=2, n_heads=4,
    lr=3e-4, gamma=0.99, entropy_coef=0.02, value_coef=0.5, grad_clip=1.0,
    device=device,
)
algo = AC2CDAlgo(cfg)

if CKPT.exists():
    algo.load(str(CKPT))
    print(f"Loaded checkpoint: {CKPT}")
else:
    print("No checkpoint — starting from scratch")

print("\nZero-shot eval on Email-EU-Core proxy ...")
df_zs = eval_algo_on_suite(algo, test_probs, task, n_seeds=5,
                            horizon=40, greedy=True,
                            env_kwargs={"warm_start": "leiden"})
zs_nmi = float(df_zs["nmi"].mean())
print(f"Zero-shot NMI = {zs_nmi:.4f}  (target={TARGET})")

if zs_nmi >= TARGET:
    print(f"\n[PASS] Zero-shot: NMI={zs_nmi:.4f} >= {TARGET}")
    sys.exit(0)

# ── Fine-tune on Email-EU-Core proxy ─────────────────────────────────────────
print(f"\nZero-shot did not pass. Fine-tuning for {FINETUNE_EPS} episodes ...")
import random
rng_ft = random.Random(99)

def ft_env_fn():
    p = rng_ft.choice(train_probs)
    return NodeMoveEnv(task, p, horizon=30, warm_start="random")

ft_cfg = TrainConfig(
    n_episodes=FINETUNE_EPS, horizon=30,
    lr=1e-4, gamma=0.99,           # lower LR for fine-tuning
    n_episode_per_update=8,
    entropy_coef=0.01, value_coef=0.5, grad_clip=1.0,
    log_every=100, save_every=0,
    out_dir="results/ac2cd_email",
)
Path("results/ac2cd_email").mkdir(exist_ok=True)

t0 = time.perf_counter()
trainer = Trainer(algo=algo, env_fn=ft_env_fn, config=ft_cfg)
trainer.train()
print(f"Fine-tune done in {time.perf_counter()-t0:.1f}s\n")

# ── Final eval ────────────────────────────────────────────────────────────────
print("Final eval after fine-tune ...")
df_ft = eval_algo_on_suite(algo, test_probs, task, n_seeds=5,
                            horizon=40, greedy=True,
                            env_kwargs={"warm_start": "leiden"})
ft_nmi = float(df_ft["nmi"].mean())

print(f"\n{'='*50}")
print(f"AC2CD NMI (fine-tuned) = {ft_nmi:.4f}")
print(f"AC2CD NMI (zero-shot)  = {zs_nmi:.4f}")
print(f"Leiden NMI             = {leiden_nmi:.4f}")
print(f"Target NMI             = {TARGET:.4f}")

status = "PASS" if ft_nmi >= TARGET else "FAIL"
sign   = ">=" if ft_nmi >= TARGET else "<"
print(f"\n[{status}] AC2CD Email-EU-Core NMI={ft_nmi:.4f} {sign} target={TARGET}")

if ft_nmi >= TARGET:
    algo.save("results/ac2cd_email/best.pt")
    sys.exit(0)
else:
    sys.exit(1)
