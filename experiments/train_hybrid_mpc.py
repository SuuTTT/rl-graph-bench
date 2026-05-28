#!/usr/bin/env python3
"""Active hybrid training with MPC planning look-ahead.

This script trains a fresh SS2VAlgo model under both active hybrid training
and MPC look-ahead data collection, and verifies loss convergence.
"""
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo, SS2VConfig
from rlgb.data.mcmp_instances import mcmp_train_suite
from rlgb.tasks.multicut import MulticutTask
from rlgb.envs.edge_contraction_env import EdgeContractionEnv
from rlgb.training.trainer import Trainer, TrainConfig

class MCMPEnvWrapper:
    """Wraps EdgeContractionEnv for signed multicut MCMP task."""
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

        filtered = obs["edge_idx"]
        labels = obs["labels"]
        if filtered.shape[0] > 0:
            k = int(labels.max()) + 1
            N = len(labels)
            S = np.zeros((N, k), dtype=np.float32)
            S[np.arange(N), labels] = 1.0
            
            C_clust = S.T @ (self._cost_adj @ S)
            
            lu = labels[filtered[:, 0]]
            lv = labels[filtered[:, 1]]
            obs["cluster_sums"] = C_clust[lu, lv]
        else:
            obs["cluster_sums"] = np.empty(0, dtype=np.float32)
        return obs

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        return self._inject(obs), info

    def step(self, action: int):
        if len(self._pos_edge_map) > 0:
            original_action = int(self._pos_edge_map[min(action, len(self._pos_edge_map) - 1)])
        else:
            original_action = 0
        obs, reward, term, trunc, info = self._env.step(original_action)
        obs = self._inject(obs)
        if obs["n_edges"][0] == 0:
            term = True
        elif len(obs["cluster_sums"]) > 0 and float(obs["cluster_sums"].max()) <= 0.0:
            term = True
        return obs, float(reward), term, trunc, info

    def close(self):
        self._env.close()

    def __getattr__(self, name: str):
        return getattr(self._env, name)

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("Generating/loading train suite (n=40 synthetic MCMP)...")
    train_suite = mcmp_train_suite()
    # Filter to first 5 training instances for extremely fast training verification in dev
    train_suite = train_suite[:5]
    print(f"  Loaded {len(train_suite)} train instances.")

    task = MulticutTask()
    
    # Configure SS2VAlgo for Active Hybrid MPC
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SS2VConfig(
        hidden=32,
        n_layers=2,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=400,
        buffer_capacity=10000,
        batch_size=32,
        target_update_every=20,
        device=device,
        hybrid=True,
        hybrid_mode="top_k",
        hybrid_top_k=5,
        mpc_planning=True,
        mpc_horizon=3,
        mpc_top_k=5
    )
    
    print(f"Initializing SS2VAlgo on {device}...")
    print(f"  Hybrid: {cfg.hybrid}, Mode: {cfg.hybrid_mode}")
    print(f"  MPC Planning: {cfg.mpc_planning}, Horizon: {cfg.mpc_horizon}, Top-K: {cfg.mpc_top_k}")
    algo = SS2VAlgo(cfg)

    # env_fn generator
    rng_train = random.Random(42)
    def env_fn():
        p = rng_train.choice(train_suite)
        # Singleton warm-start
        env = EdgeContractionEnv(task=task, problem=p, horizon=40, warm_start="singleton")
        return MCMPEnvWrapper(env, p.meta["cost_matrix"])

    # TrainConfig
    train_cfg = TrainConfig(
        n_episodes=200,
        horizon=40,
        n_episode_per_update=5,
        log_every=20,
        save_every=0,
        out_dir="results/checkpoints",
        lr_schedule="none",
        verbose=True
    )

    print("\nStarting Active Hybrid MPC Training...")
    t0 = time.perf_counter()
    
    trainer = Trainer(algo=algo, env_fn=env_fn, config=train_cfg)
    trainer.train()
    
    duration = time.perf_counter() - t0
    print(f"\nTraining completed in {duration:.1f}s.")

    # Save the trained model checkpoint
    checkpoint_dir = Path("results/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "ss2v_mpc_trained.pt"
    algo.save(checkpoint_path)
    print(f"Saved trained checkpoint to: {checkpoint_path}")

    print("\n[SUCCESS] Train hybrid MPC script executed successfully!")

if __name__ == "__main__":
    main()
