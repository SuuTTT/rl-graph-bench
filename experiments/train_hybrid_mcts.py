#!/usr/bin/env python3
"""Active hybrid training under AlphaZero-style Monte Carlo Tree Search (MCTS).

Trains a fresh GNN model using active MCTS data collection and PUCT prioritization,
and saves the trained checkpoint.
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
from experiments.train_hybrid_mpc import MCMPEnvWrapper

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("Generating/loading train suite (n=40 synthetic MCMP)...")
    train_suite = mcmp_train_suite()
    # Filter to first 5 training instances for rapid development training
    train_suite = train_suite[:5]
    print(f"  Loaded {len(train_suite)} train instances.")

    task = MulticutTask()
    
    # Configure SS2VAlgo for Active MCTS
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
        mcts_planning=True,
        mcts_simulations=15,  # 15 simulations during training for balanced speed/rigor
        mcts_cpuct=1.5
    )
    
    print(f"Initializing SS2VAlgo on {device}...")
    print(f"  Hybrid: {cfg.hybrid}, Mode: {cfg.hybrid_mode}")
    print(f"  MCTS Planning: {cfg.mcts_planning}, Simulations: {cfg.mcts_simulations}, CPUCT: {cfg.mcts_cpuct}")
    algo = SS2VAlgo(cfg)

    # env_fn generator
    rng_train = random.Random(42)
    def env_fn():
        p = rng_train.choice(train_suite)
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

    print("\nStarting Active MCTS Training...")
    t0 = time.perf_counter()
    
    trainer = Trainer(algo=algo, env_fn=env_fn, config=train_cfg)
    trainer.train()
    
    duration = time.perf_counter() - t0
    print(f"\nTraining completed in {duration:.1f}s.")

    # Save the trained model checkpoint
    checkpoint_dir = Path("results/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "ss2v_mcts_trained.pt"
    algo.save(checkpoint_path)
    print(f"Saved trained checkpoint to: {checkpoint_path}")

    print("\n[SUCCESS] Train hybrid MCTS script executed successfully!")

if __name__ == "__main__":
    main()
