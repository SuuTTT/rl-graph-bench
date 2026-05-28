#!/usr/bin/env python3
"""Validate online active training of SLRL using the standard Trainer."""
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rlgb.data.clare_dataset import load_clare_dataset
from rlgb.algos.community.slrl import SLRLAlgo, SLRLConfig
from rlgb.tasks.community_expand import CommunityExpandTask
from rlgb.training.trainer import Trainer, TrainConfig

def main():
    print("Loading SNAP Amazon dataset...")
    # Load small subset for rapid training verification
    data = load_clare_dataset("amazon", num_train=10, num_val=2, seed=42)
    
    print("\nConverting SNAP communities to standard Problem instances...")
    train_suite = data.to_problem_suite("train")[:5]
    val_suite = data.to_problem_suite("val")[:2]
    print(f"  Train Suite has {len(train_suite)} Problems.")
    print(f"  Val Suite has {len(val_suite)} Problems.")
    
    task = CommunityExpandTask(objective="h2")
    
    # Initialize SLRLAlgo with neural network active mode (scov_threshold = 0.0)
    cfg = SLRLConfig(
        hidden=32,
        lr=1e-3,
        gamma=0.99,
        entropy_coef=0.01,
        value_coef=0.5,
        device="cpu",  # CPU is faster for these single tiny graphs
        scov_threshold=0.0
    )
    algo = SLRLAlgo(config=cfg)
    
    # Define env_fn to randomly choose one training problem
    def env_fn():
        p = random.choice(train_suite)
        return task.build_env(p, horizon=10, warm_start="seed")
        
    print("\nRunning online active SLRL training using standard Trainer...")
    # Train for 50 episodes
    train_cfg = TrainConfig(
        n_episodes=50,
        horizon=10,
        n_episode_per_update=4,
        log_every=10,
        save_every=0,
        out_dir="results/active_slrl_test",
        lr_schedule="none",
        verbose=True
    )
    
    trainer = Trainer(algo=algo, env_fn=env_fn, config=train_cfg)
    trainer.train()
    
    print("\nEvaluating trained SLRL policy on validation suite...")
    val_df = trainer.eval(suite=val_suite, task=task, n_seeds=1)
    print("Evaluation results:")
    print(val_df[["problem", "wall_sec", "h2", "ncut", "f1"]])
    
    print("\n[SUCCESS] Active SLRL training verification run completed successfully!")

if __name__ == "__main__":
    main()
