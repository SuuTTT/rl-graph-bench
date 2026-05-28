# Tasks — Track F1: AlphaZero-Style Monte Carlo Tree Search (MCTS) with GNN Priors for Graph Partitioning

- `[x]` **F1.1: MCTS Engine Implementation in `SS2VAlgo`**
  - `[x]` Add `mcts_planning: bool = False`, `mcts_simulations: int = 30`, and `mcts_cpuct: float = 1.5` to `SS2VConfig` in `rlgb/algos/multicut/ss2v_d3qn.py`.
  - `[x]` Implement the private nested `_MCTSNode` class inside `SS2VAlgo` to manage tree statistics (N, W, Q, P, children).
  - `[x]` Implement `_mcts_search(self, obs, simulations)` PUCT selection, analytical expansion, GNN evaluation, and backup.
  - `[x]` Integrate `_mcts_search` into `select_action` and update DQN target bootstrapping to align with MCTS search values.
  - `[x]` Run basic tests to verify MCTS tree search functions without error (26/26 unit tests passed).

- `[x]` **F1.2: Active MCTS Training & Verification**
  - `[x]` Create [train_hybrid_mcts.py](file:///workspace/rl-graph-bench/experiments/train_hybrid_mcts.py) to train a model under active MCTS data collection.
  - `[x]` Verify training convergence and running rewards (achieved near-optimal return of `-0.1141` at episode 100).
  - `[x]` Save the MCTS-trained model checkpoint to `results/checkpoints/ss2v_mcts_trained.pt`.

- `[x]` **F1.3: Upgraded Comparative Inductive Benchmark**
  - `[x]` Integrate MCTS evaluation into [run_inductive_benchmark.py](file:///workspace/rl-graph-bench/experiments/run_inductive_benchmark.py).
  - `[x]` Run a comprehensive zero-shot scale generalization benchmark comparing GAEC, Pure, Hybrid, MPC, and MCTS (AlphaZero MCTS GNN achieved first place at scale N=100 and N=200!).
  - `[x]` Compile the final reports and LaTeX tables in the repository (documented in `inductive_benchmark_summary.md` and `walkthrough.md`).
