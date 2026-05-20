"""rl-graph-bench: unified RL benchmark for graph clustering.

Three task tracks
-----------------
  GraphPartition    – fixed-k cut optimisation (NCut, H², Balanced, Sparsest)
  CommunityExpand   – semi-supervised community search (F1, Jaccard, ONMI)
  DynamicCD         – dynamic graph community detection (modularity density)

Five algorithm families
-----------------------
  A  NodeMove       – NeuroCUT / WRT: node-to-cluster reassignment
  B  StructuredAct  – WRT ring/wedge constrained partitioning
  C  CommunityRW    – CLARE / SLRL add-remove-stop rewriting
  D  DynamicAC      – AC2CD GAT actor-critic on temporal snapshots
  E  Multicut       – SS2V-D3QN sequential edge contraction

Quick-start
-----------
  from rlgb.tasks.graph_partition import GraphPartitionTask
  from rlgb.algos.node_move import NeuroCUTAlgo
  from rlgb.training.trainer import Trainer, TrainConfig

  task    = GraphPartitionTask(objective="h2")
  algo    = NeuroCUTAlgo(hidden=64)
  trainer = Trainer(algo=algo, task=task, config=TrainConfig(n_steps=100_000))
  trainer.train()
  df      = trainer.eval()
  print(df.to_string())
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
