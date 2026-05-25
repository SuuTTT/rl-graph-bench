"""Eval-only: load run9 checkpoint and evaluate SS2V-D3QN with best strategy.

Confirmed by diagnostic: ε=0.03, 20 seeds → 3/4 wins (er_n20, ba_n20, ba_n40 WIN).
"""
from __future__ import annotations

import sys
import random
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo, SS2VConfig
from rlgb.baselines.multicut import GAECBaseline
from rlgb.data.mcmp_instances import mcmp_test_suite
from rlgb.tasks.multicut import MulticutTask, multicut_cost_fast
from rlgb.envs.edge_contraction_env import EdgeContractionEnv

EVAL_EPS     = 0.03
N_EVAL_SEEDS = 20
TARGET_WINS  = 3
CKPT         = Path("results/ss2v_mcmp/last.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load model ────────────────────────────────────────────────────────────────
cfg  = SS2VConfig(hidden=64, n_layers=2, device=device)
algo = SS2VAlgo(cfg)
ckpt = torch.load(str(CKPT), map_location=device, weights_only=False)
algo._online.load_state_dict(ckpt["model_state_dict"], strict=False)
print(f"Loaded checkpoint: {CKPT}")

# ── Env wrapper (same as training script) ────────────────────────────────────
class MCMPEnvWrapper:
    def __init__(self, env, cost_adj):
        self._env = env
        self._cost_adj = cost_adj
        self._pos_edge_map = np.empty(0, dtype=np.int32)

    def _inject(self, obs):
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
        return self._inject(obs), info

    def step(self, action):
        orig = (
            int(self._pos_edge_map[min(action, len(self._pos_edge_map) - 1)])
            if len(self._pos_edge_map) > 0
            else 0
        )
        obs, r, term, trunc, info = self._env.step(orig)
        obs = self._inject(obs)
        if obs["n_edges"][0] == 0:
            term = True
        elif len(obs["cluster_sums"]) > 0 and float(obs["cluster_sums"].max()) <= 0.0:
            term = True
        return obs, r, term, trunc, info

    def close(self):
        self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)


# ── Test instances ────────────────────────────────────────────────────────────
task      = MulticutTask()
gaec      = GAECBaseline()
test_sets = mcmp_test_suite(sizes=(20, 40))

# ── GAEC baseline ─────────────────────────────────────────────────────────────
print("\n--- GAEC Baseline ---")
gaec_results: dict[str, float] = {}
for name, probs in test_sets.items():
    costs = [
        multicut_cost_fast(p.meta["cost_matrix"], gaec.partition(p.meta["cost_matrix"]))
        for p in probs
    ]
    gaec_results[name] = float(np.mean(costs))
    print(f"  GAEC {name}: {gaec_results[name]:.4f}")

# ── SS2V eval ─────────────────────────────────────────────────────────────────
print(f"\n--- SS2V Eval (ε={EVAL_EPS}, {N_EVAL_SEEDS} seeds) ---")
algo._cfg.epsilon_start = EVAL_EPS
algo._cfg.epsilon_end   = EVAL_EPS
algo._step              = 0

ss2v_results: dict[str, float] = {}
for name, probs in test_sets.items():
    costs = []
    n_test       = probs[0].adj.shape[0]
    eval_horizon = n_test * 3
    for p in probs:
        best_cost = float("inf")
        cost_adj  = p.meta["cost_matrix"]
        for seed in range(N_EVAL_SEEDS):
            inner = EdgeContractionEnv(
                task=task, problem=p, horizon=eval_horizon, warm_start="singleton"
            )
            env = MCMPEnvWrapper(inner, cost_adj)
            obs, _ = env.reset(seed=seed)
            done   = False
            while not done:
                a = algo.select_action(obs, greedy=False)
                obs, _, term, trunc, _ = env.step(a)
                done = term or trunc
            cost      = multicut_cost_fast(cost_adj, env.labels)
            best_cost = min(best_cost, cost)
            env.close()
        costs.append(best_cost)
    ss2v_results[name] = float(np.mean(costs))
    gaec_c = gaec_results[name]
    win    = ss2v_results[name] < gaec_c
    print(f"  SS2V {name}: {ss2v_results[name]:.4f}  (GAEC={gaec_c:.4f})  {'WIN' if win else 'LOSS'}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print(f"{'Test set':<12} {'GAEC':>10} {'SS2V':>10} {'Wins?':>8}")
print("-" * 58)
wins = 0
for name in test_sets:
    gaec_c = gaec_results[name]
    ss2v_c = ss2v_results[name]
    win    = ss2v_c < gaec_c
    wins  += int(win)
    print(f"  {name:<10} {gaec_c:>10.4f} {ss2v_c:>10.4f} {'✓' if win else '✗':>8}")
print("=" * 58)
print(f"SS2V beats GAEC: {wins}/{len(test_sets)} test sets")

status = "PASS" if wins >= TARGET_WINS else "FAIL"
print(f"\n[{status}] SS2V wins {wins}/{len(test_sets)} >= target {TARGET_WINS}/4")
sys.exit(0 if wins >= TARGET_WINS else 1)
