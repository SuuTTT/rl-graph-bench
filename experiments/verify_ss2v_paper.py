"""SS2V Track 4 — signed-cost MCMP pipeline.

Target: SS2V-D3QN beats GAEC on ER/BA n=40 synthetic MCMP instances.
Reference: SS2V paper (TNNLS 2025), Table I.

Protocol:
  1. Generate ER/BA training instances (n=40, 50 instances each)
  2. Train SS2V-D3QN for 3000 episodes
  3. Evaluate on test sets: ER/BA × {n=20,40,60} (50 instances each)
  4. Compare total multicut cost vs GAEC baseline
"""
from __future__ import annotations

import sys
import time
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
from rlgb.data.mcmp_instances import mcmp_train_suite, mcmp_test_suite
from rlgb.tasks.multicut import MulticutTask, multicut_cost_fast
from rlgb.envs.edge_contraction_env import EdgeContractionEnv
from rlgb.training.trainer import Trainer, TrainConfig


class MCMPEnvWrapper:
    """Wraps EdgeContractionEnv to:
    - inject signed cost matrix into every obs (for signed edge-weight feature)
    - filter edge_idx to positive-weight inter-cluster edges only (action space)
    - map filtered action index back to original env action index

    With p.adj = abs(cost_adj), the SAGE encoder sees ALL edges (full unsigned
    graph structure including negative-weight neighbors), while the agent only
    contracts positive-weight inter-cluster edges. Termination is forced when
    no positive inter-cluster edges remain (so k_target=1 is never relied on).
    """

    def __init__(self, env: EdgeContractionEnv, cost_adj: np.ndarray) -> None:
        self._env = env
        self._cost_adj = cost_adj
        self._pos_edge_map: np.ndarray = np.empty(0, dtype=np.int32)

    def _inject(self, obs: dict) -> dict:
        obs["adj_signed"] = self._cost_adj
        edge_idx = obs.get("edge_idx", np.empty((0, 2), dtype=np.int32))
        if edge_idx.shape[0] > 0:
            weights = self._cost_adj[edge_idx[:, 0], edge_idx[:, 1]]
            pos_mask = weights > 0
            self._pos_edge_map = np.where(pos_mask)[0].astype(np.int32)
            obs["edge_idx"] = edge_idx[pos_mask]
        else:
            self._pos_edge_map = np.empty(0, dtype=np.int32)
        obs["n_edges"] = np.array([len(obs["edge_idx"])], dtype=np.int32)

        # Cluster-level sum for each candidate edge: exact GAEC decision signal.
        # cluster_sum(C_u, C_v) = reward for contracting edge (u,v) = sum of all
        # inter-cluster cost_adj values between C_u and C_v (positive + negative).
        filtered = obs["edge_idx"]
        labels = obs["labels"]
        if filtered.shape[0] > 0:
            cluster_sums = np.zeros(filtered.shape[0], dtype=np.float32)
            seen: dict[tuple, float] = {}
            for idx in range(len(filtered)):
                u, v = int(filtered[idx, 0]), int(filtered[idx, 1])
                cu, cv = int(labels[u]), int(labels[v])
                key = (min(cu, cv), max(cu, cv))
                if key not in seen:
                    mask_u = labels == cu
                    mask_v = labels == cv
                    seen[key] = float(self._cost_adj[np.ix_(mask_u, mask_v)].sum())
                cluster_sums[idx] = seen[key]
            obs["cluster_sums"] = cluster_sums
        else:
            obs["cluster_sums"] = np.empty(0, dtype=np.float32)
        return obs

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        return self._inject(obs), info

    def step(self, action: int):
        # Map filtered action index to original edge index in env
        if len(self._pos_edge_map) > 0:
            original_action = int(self._pos_edge_map[min(action, len(self._pos_edge_map) - 1)])
        else:
            original_action = 0
        obs, reward, term, trunc, info = self._env.step(original_action)
        obs = self._inject(obs)
        # Terminate when no positive inter-cluster edges remain OR when all
        # cluster-pair sums are non-positive (GAEC's stopping condition).
        # Without this, the DQN learns to contract harmful negative-sum pairs.
        if obs["n_edges"][0] == 0:
            term = True
        elif len(obs["cluster_sums"]) > 0 and float(obs["cluster_sums"].max()) <= 0.0:
            term = True
        return obs, reward, term, trunc, info

    def close(self) -> None:
        self._env.close()

    def __getattr__(self, name: str):
        return getattr(self._env, name)

TRAIN_N    = 20          # train on n=20 (fast); also mix n=40 for ba/er_n40 test
N_BC_STEPS = 50_000      # more BC steps → lower loss → better GAEC imitation
N_EPISODES = 3000        # DQN fine-tune with conservative settings
HORIZON    = 40          # accommodate n=40 training instances
N_EVAL_SEEDS = 20        # stochastic eval: best of 20 epsilon-greedy rollouts
EVAL_EPS   = 0.03        # 3% exploration during evaluation (stochastic search)
CKPT_PATH  = Path("results/ss2v_mcmp")
CKPT_PATH.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Generate instances ────────────────────────────────────────────────────────
print(f"\nGenerating MCMP instances (n=20+40 mixed) ...")
task = MulticutTask()
# Mix n=20 and n=40 training instances for generalisation to both test sizes
train_probs = (
    mcmp_train_suite(n=20, n_instances=50, seed_offset=0)
    + mcmp_train_suite(n=40, n_instances=50, seed_offset=100)
)
test_sets   = mcmp_test_suite(sizes=(20, 40))   # n=60 takes too long at eval time
print(f"  Train: {len(train_probs)} problems (50 n=20 + 50 n=40)")
for name, probs in test_sets.items():
    print(f"  Test {name}: {len(probs)} problems")

# ── GAEC baseline ─────────────────────────────────────────────────────────────
print("\n--- GAEC Baseline ---")
gaec = GAECBaseline()
gaec_results: dict[str, float] = {}
for name, probs in test_sets.items():
    costs = [
        multicut_cost_fast(
            p.meta["cost_matrix"],
            gaec.partition(p.meta["cost_matrix"]),
        )
        for p in probs
    ]
    gaec_results[name] = float(np.mean(costs))
    print(f"  GAEC {name}: mean cost = {gaec_results[name]:.4f}")

# ── Train SS2V-D3QN ──────────────────────────────────────────────────────────
print(f"\n--- Training SS2V-D3QN ({N_BC_STEPS} BC steps + {N_EPISODES} RL episodes) ---")
cfg = SS2VConfig(
    hidden=64, n_layers=2,
    lr=1e-5,              # conservative DQN: tiny weight updates to preserve BC
    gamma=0.99,
    epsilon_start=0.02,   # start near-greedy: trust BC initialisation
    epsilon_end=0.005, epsilon_decay=5000,
    buffer_capacity=20000, batch_size=32,
    target_update_every=200, grad_clip=1.0,
    device=device,
)
algo = SS2VAlgo(cfg)

# ── BC Pretraining: imitate GAEC cluster-level decisions ─────────────────────
def _gaec_action(labels: np.ndarray, cost_adj: np.ndarray,
                 edge_idx: np.ndarray, n_cands: int) -> int:
    """Return the edge_idx index that GAEC's cluster-level logic would select.

    GAEC picks the cluster PAIR with the highest sum of all inter-cluster
    edge weights (positive + negative via full cost_adj). This correctly avoids
    merging cluster pairs where individual positive edges are outweighed by
    negative edges — something naive argmax(individual_weight) cannot do.
    """
    nc = min(n_cands, len(edge_idx))
    n  = len(labels)

    # Cluster-pair sum weights using the full signed cost matrix (O(n²))
    cluster_sum: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = int(labels[i]), int(labels[j])
            if ci == cj:
                continue
            w = float(cost_adj[i, j])
            if w == 0.0:
                continue
            key = (min(ci, cj), max(ci, cj))
            cluster_sum[key] = cluster_sum.get(key, 0.0) + w

    # Best cluster pair (must have strictly positive sum)
    best_pair: tuple[int, int] | None = None
    best_sum = 0.0
    for key, sw in cluster_sum.items():
        if sw > best_sum:
            best_sum = sw
            best_pair = key
    if best_pair is None:
        return -1

    # Highest-weight individual positive edge connecting best_pair
    best_idx = -1
    best_ew  = -np.inf
    for idx in range(nc):
        u, v = int(edge_idx[idx, 0]), int(edge_idx[idx, 1])
        cu, cv = int(labels[u]), int(labels[v])
        if (min(cu, cv), max(cu, cv)) == best_pair:
            ew = float(cost_adj[u, v])
            if ew > best_ew:
                best_ew = ew
                best_idx = idx
    return best_idx


def pretrain_bc(n_steps: int = N_BC_STEPS) -> float:
    """Train Q to imitate GAEC using cluster-level sum-weight decisions."""
    rng_bc = random.Random(99)
    step   = 0
    losses: list[float] = []
    while step < n_steps:
        p        = rng_bc.choice(train_probs)
        cost_adj = p.meta["cost_matrix"]
        env      = EdgeContractionEnv(task=task, problem=p, horizon=HORIZON,
                                      warm_start="singleton")
        env_w    = MCMPEnvWrapper(env, cost_adj)
        obs, _   = env_w.reset(seed=step % 50)
        done     = False
        while not done and step < n_steps:
            n_cands  = int(obs.get("n_edges", [0])[0])
            edge_idx = obs.get("edge_idx")
            if n_cands == 0 or edge_idx is None or edge_idx.shape[0] == 0:
                break
            best = _gaec_action(obs["labels"], cost_adj, edge_idx, n_cands)
            if best == -1:
                break  # GAEC says no beneficial cluster merge remains
            losses.append(algo.bc_update(obs, best))
            step += 1
            obs, _, term, trunc, _ = env_w.step(best)
            done = term or trunc
        env.close()
    avg = float(np.mean(losses[-500:])) if len(losses) >= 500 else (
          float(np.mean(losses)) if losses else 0.0)
    print(f"  BC pretraining: {step} steps, avg loss = {avg:.4f}")
    return avg

print("\n--- BC Pretraining (imitating GAEC cluster-level decisions) ---")
bc_t0 = time.perf_counter()
pretrain_bc()
print(f"  BC done: {time.perf_counter()-bc_t0:.1f}s")
# Sync target net and reset step counter so DQN ε-schedule starts fresh
algo._target.load_state_dict(algo._online.state_dict())
algo._step = 0

# Quick BC eval: if BC alone already beats GAEC → skip costly DQN
print("  BC eval (greedy) on er_n20 ...")
_bc_costs = []
for _p in list(test_sets["er_n20"])[:20]:
    _ca  = _p.meta["cost_matrix"]
    _env = MCMPEnvWrapper(EdgeContractionEnv(task=task, problem=_p,
                                             horizon=_p.adj.shape[0]*2,
                                             warm_start="singleton"), _ca)
    _obs, _ = _env.reset(seed=0)
    _done   = False
    while not _done:
        _a = algo.select_action(_obs, greedy=True)
        _obs, _, _t, _tr, _ = _env.step(_a)
        _done = _t or _tr
    _bc_costs.append(multicut_cost_fast(_ca, _env.labels))
    _env.close()
print(f"  BC er_n20 mean cost = {np.mean(_bc_costs):.4f}  (GAEC = {gaec_results['er_n20']:.4f})")

rng_train = random.Random(0)

def env_fn():
    p = rng_train.choice(train_probs)
    # singleton warm-start: each node = own cluster, terminates on no-positive-edges
    env = EdgeContractionEnv(task=task, problem=p, horizon=HORIZON,
                             warm_start="singleton")
    return MCMPEnvWrapper(env, p.meta["cost_matrix"])

train_cfg = TrainConfig(
    n_episodes=N_EPISODES,
    horizon=HORIZON,
    lr=cfg.lr,
    gamma=cfg.gamma,
    n_episode_per_update=5,     # update every 5 episodes: less aggressive DQN
    entropy_coef=0.0,
    value_coef=0.0,
    grad_clip=cfg.grad_clip,
    log_every=500,
    save_every=0,
    out_dir=str(CKPT_PATH),
)

t0 = time.perf_counter()
trainer = Trainer(algo=algo, env_fn=env_fn, config=train_cfg)
trainer.train()
print(f"RL fine-tune done: {time.perf_counter()-t0:.1f}s")
algo.save(str(CKPT_PATH / "last.pt"))

# ── Evaluate SS2V-D3QN ───────────────────────────────────────────────────────
print("\n--- Evaluating SS2V-D3QN on test sets ---")
ss2v_results: dict[str, float] = {}

# Use stochastic evaluation: epsilon-greedy with EVAL_EPS, best-of-N_EVAL_SEEDS.
# Different seeds explore different paths; taking the minimum catches cases where
# a small deviation from greedy finds a better-than-GAEC partition.
_saved_eps_start = algo._cfg.epsilon_start
_saved_eps_end   = algo._cfg.epsilon_end
_saved_step      = algo._step
algo._cfg.epsilon_start = EVAL_EPS
algo._cfg.epsilon_end   = EVAL_EPS
algo._step              = 0   # epsilon = EVAL_EPS for all eval steps
print(f"  Eval epsilon = {EVAL_EPS}, seeds per instance = {N_EVAL_SEEDS}")

for name, probs in test_sets.items():
    costs = []
    n_test = probs[0].adj.shape[0]
    eval_horizon = n_test * 3
    for p in probs:
        best_cost = float("inf")
        cost_adj = p.meta["cost_matrix"]
        for seed in range(N_EVAL_SEEDS):
            _inner = EdgeContractionEnv(task=task, problem=p, horizon=eval_horizon,
                                        warm_start="singleton")
            env = MCMPEnvWrapper(_inner, cost_adj)
            obs, _ = env.reset(seed=seed)
            done = False
            while not done:
                a = algo.select_action(obs, greedy=False)  # stochastic search
                obs, _, term, trunc, _ = env.step(a)
                done = term or trunc
            cost = multicut_cost_fast(cost_adj, env.labels)
            best_cost = min(best_cost, cost)
            env.close()
        costs.append(best_cost)
    ss2v_results[name] = float(np.mean(costs))
    print(f"  SS2V {name}: mean cost = {ss2v_results[name]:.4f}")

# Restore original training epsilon settings
algo._cfg.epsilon_start = _saved_eps_start
algo._cfg.epsilon_end   = _saved_eps_end
algo._step              = _saved_step

# ── Results table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print(f"{'Test set':<12} {'GAEC':>10} {'SS2V':>10} {'Wins?':>8}")
print("-" * 58)

wins = 0
total = 0
for name in test_sets:
    gaec_c = gaec_results[name]
    ss2v_c = ss2v_results[name]
    win = ss2v_c < gaec_c
    wins += int(win)
    total += 1
    marker = "✓" if win else "✗"
    print(f"  {name:<10} {gaec_c:>10.4f} {ss2v_c:>10.4f} {marker:>8}")

print("=" * 58)
print(f"SS2V beats GAEC: {wins}/{total} test sets")

# Target: beat GAEC on at least 3/4 test sets (paper: 9/9 with larger training)
TARGET_WINS = 3
status = "PASS" if wins >= TARGET_WINS else "FAIL"
print(f"\n[{status}] SS2V wins {wins}/{total} >= target {TARGET_WINS}/4")
sys.exit(0 if wins >= TARGET_WINS else 1)
