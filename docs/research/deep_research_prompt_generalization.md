# Deep Research Prompt: Generalizing Co-Adapted TD-MPC Graph Solvers

Copy the prompt below to run a deep research task analyzing the scientific significance, generalization pathways, and advanced baselines for this framework.

---

## [DEEP RESEARCH PROMPT START]

### Role and Objective
You are an elite Principal AI Researcher specializing in Graph Machine Learning and Neural Combinatorial Optimization. Your task is to conduct a highly rigorous, literature-grounded deep research analysis on generalizing a novel RL/Control framework for discrete graph optimization.

### Background on the Current Framework
We have successfully built and validated a framework for **Signed Multicut (MCMP)** utilizing:
1. **Analytical Graph World Model**: Receding-horizon MPC look-ahead planner ($H$-horizons, $K$-branching) utilizing the exact, non-differentiable graph transition operator (sequential edge contraction) rather than a learned neural dynamics model.
2. **Co-Adapted Target updates**: Setting Q-learning bootstrap targets based directly on actions selected by the look-ahead MPC planner to align value representations with the planning tree.
3. **Planner Distillation**: Direct training of a GNN policy head from look-ahead planner actions.

### Core Research Questions to Explore

#### 1. Scientific Significance & Generalization Pathways
* **Is Signed Multicut via sequential edge contraction too narrow?** How can we generalize this problem formulation into a unified **Discrete Graph MDP (G-MDP)** framework?
* **What other NP-hard graph partitioning and clustering problems can be formulated under this exact dynamics + co-adapted neural search paradigm?** Detail formulations for:
  * Correlation Clustering (Max-Agree / Min-Disagree)
  * Generalized Graph Partitioning (METIS-style balanced min-cut)
  * Max-Cut / Min-Cut
  * Community Detection (Modularization optimization)
* **What are the generalized graph transition operators** (e.g. node swapping, edge addition/deletion, label propagation transitions) that can serve as exact-dynamics world models for these generalized G-MDPs?

#### 2. Advanced & Unassailable Baselines
To make a generalized G-MDP paper acceptable to top-tier venues (NeurIPS, ICLR, ICML), what state-of-the-art classical and neural baselines must we evaluate against? Research and contrast:
* **Classical Exact & Semidefinite Solvers**:
  * Semidefinite Programming (SDP) relaxations (Goemans-Williamson, etc.)
  * Integer Linear Programming (ILP) solvers (Gurobi, SCIP)
  * METIS / KaHIP (for balanced graph partitioning)
* **Neural-Algorithmic Solvers**:
  * ECORD (Edge Contraction for Max-Cut)
  * CLARE (Contrastive Learning Assisted Refinement)
  * S2V-DQN / GCG / NeuroAlgorithmic MCTS variants

#### 3. Formal Scientific Novelty & Contribution Synthesis
* How does this unified **"Exact Graph Dynamics + Co-Adapted Neural Search"** paradigm compare theoretically to:
  * **MuZero / Reanalyse**: Contrast exact analytical graph operators with MuZero's learned latent dynamics.
  * **AlphaZero / Expert Iteration**: Contrast receding-horizon MPC with AlphaZero's full Monte Carlo Tree Search (MCTS) under game rules.
* Synthesize a proposed title, abstract, and draft 4 key core contributions for a conference submission based on this generalized paradigm.

### Expected Output Structure
1. **Critical Review & Viability Analysis**: A detailed evaluation of whether generalizing the problem formulation is a high-impact idea for a top-tier ML conference.
2. **The Unified G-MDP Mathematical Framework**: Formal state, action, transition, and reward formulations for at least 3 generalized combinatorial graph problems.
3. **Comprehensive Baselines Map**: A structured comparison table of classical mathematical programming, heuristics, and neural solvers.
4. **Conference-Level Contribution Draft**: Propose a strong title, abstract, and contribution checklist designed to wow reviewers.

## [DEEP RESEARCH PROMPT END]
