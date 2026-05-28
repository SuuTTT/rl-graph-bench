# Implementation Plan: Track F1 AlphaZero-Style Monte Carlo Tree Search (MCTS) with GNN Priors for Graph Partitioning

This plan details the design, technical modifications, and execution steps for **Track F1: AlphaZero-Style Monte Carlo Tree Search (MCTS) with GNN Priors for Combinatorial Graph Partitioning** under the `v0.6.0` milestone.

---

## 1. Goal Description

While our MPC look-ahead planner yields significant improvements, look-ahead rollouts are limited by a fixed greedy rollout heuristic (GAEC). To achieve state-of-the-art combinatorial partitioning performance, we propose replacing look-ahead rollouts with **AlphaZero-Style Monte Carlo Tree Search (MCTS)**. 

By leveraging our deterministic **analytical Graph World Model**, we can build an exact search tree on the fly without any neural dynamics approximation errors. The GNN online $Q(s, a)$ network acts as a **policy prior** (guiding PUCT selection), and the target net acts as a **value estimator** at leaf boundaries. This represents a highly sophisticated, publication-worthy synthesis of Deep Graph-RL and Monte Carlo Tree Search.

---

## 2. Technical Design

```mermaid
graph TD
    s_0[Current State s_0] -->|PUCT Selection| Leaf[Select Leaf Node s_l]
    Leaf -->|Analytical Expansion| Next[Expand s_l using Graph World Model]
    Next -->|GNN Prior Evaluation| Prior[Prior P(s, a) = Softmax(Q_online)]
    Next -->|Terminal Value Estimate| Val[Value V(s_l) = Max(Q_target)]
    Val -->|Backup| Backprop[Backpropagate V up search path]
    Backprop -->|Iterate| s_0
    s_0 -->|Search Complete| best_a[Execute Best Action: Max visit count N(s_0, a)]
```

### A. PUCT Selection Formula
During tree traversal from the root, we select the child action $a$ that maximizes the PUCT (Predictor Upper Confidence Bound applied to Trees) score:
$$U(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$
where:
* $Q(s, a)$ is the running average search value of action $a$ from state $s$.
* $N(s, a)$ is the visit count of child edge $a$.
* $P(s, a)$ is the policy prior, computed as a softmax over the GNN Q-values of candidate edges in state $s$:
  $$P(s, a) = \text{Softmax}\left(\frac{Q_{\text{gnn}}(s, a)}{\tau}\right)$$
* $c_{\text{puct}}$ is a search exploration constant (e.g., 1.5).

### B. Analytical Expansion & Backup
* **Expansion**: When tree traversal reaches a leaf node $s_l$, we expand it by querying the Graph World Model (`_simulate_contraction`) for all valid positive edge contractions, obtaining the immediate reward $r_l$ and next observations.
* **Evaluation**: We evaluate the leaf state value using the target GNN value estimate:
  $$V(s_l) = \max_{a'} Q_{\text{target}}(s_l, a')$$
* **Backup**: We backpropagate the value up the traversal path. For each state-action pair $(s, a)$ along the path, we update:
  $$N(s, a) \leftarrow N(s, a) + 1$$
  $$W(s, a) \leftarrow W(s, a) + V(s_l)$$
  $$Q(s, a) \leftarrow \frac{W(s, a)}{N(s, a)}$$

### C. Search-Guided Execution & Target Updates
* **Action Selection**: After running $M$ simulations (e.g., $M=30$), the agent selects the action with the maximum visit count $N(\text{root}, a)$ to step the real environment.
* **MCTS-Adapted Q-learning Targets**: GNN Q-values are trained to bootstrap from the search values $Q_{\text{mcts}}(\text{root}, a)$, driving GNN representation learning to closely align with the tree search.

---

## 3. Proposed Changes

### Component 1: `SS2VAlgo` MCTS Integration

#### [MODIFY] [ss2v_d3qn.py](file:///workspace/rl-graph-bench/rlgb/algos/multicut/ss2v_d3qn.py)
* **Add MCTS Config Parameters**: Add `mcts_planning: bool = False`, `mcts_simulations: int = 30`, and `mcts_cpuct: float = 1.5` to `SS2VConfig`.
* **Tree Node Structure**: Define a private nested class `_MCTSNode` containing visit counts, total values, prior probabilities, child links, and observation states.
* **MCTS Search Engine**: Implement `_mcts_search(self, obs, simulations)` which runs PUCT selection, analytical expansion, GNN evaluation, and value backpropagation.
* **Update `select_action`**: If `mcts_planning` is True, use `_mcts_search` to select the action.
* **Search-Guided Q-Updates**: Update DQN training targets to align GNN Q-values with MCTS search values.

---

## 4. Verification Plan

### Automated Experiments
1. **MCTS Active Training**: Create `experiments/train_hybrid_mcts.py` to train a GNN under active MCTS data collection and verify convergence.
2. **Upgraded Inductive Benchmark**: Add MCTS evaluations to `experiments/run_inductive_benchmark.py` and run a comprehensive comparative zero-shot scale benchmark:
   - Compare `gaec`, `ss2v_d3qn` (pure), `ss2v_d3qn_hybrid` (one-step), `ss2v_d3qn_mpc` (MPC-guided), and `ss2v_d3qn_mcts` (MCTS-guided).
3. **LaTeX & Markdown Reports**: Compile MCTS benchmark results into LaTeX tables.
