# Deep Research Prompt: Temporal Difference Model Predictive Control (TD-MPC) & Analytical Graph World Models

Use this prompt in a deep research assistant (e.g., Gemini Deep Research, o1, or Perplexity) to generate a comprehensive, publication-grade literature review and technical analysis before we begin implementing our advanced combinatorial planning architecture.

---

### **System Instruction**
You are an elite academic researcher specializing in deep reinforcement learning (RL) for combinatorial optimization, model-based RL (MBRL), and model predictive control (MPC) on graph structures. Perform an exhaustive literature review, technical comparison, and mathematical synthesis based on the query below. Provide rigorous definitions, pseudocode, and analytical formulations.

---

### **Research Query & Core Tasks**

#### **1. Core Literature Search & Theoretical Foundations**
Perform a deep search on the following key paradigms and compile a summary of their core equations, dynamics representation, and planning algorithms:
* **TD-MPC & TD-MPC2** (Hansen et al., ICML 2022 / 2024): Analyze how joint learning of representations, analytical or learned dynamics ($s_{t+1} = f(s_t, a_t)$), reward models ($R(s, a)$), and terminal value functions ($V(s)$) is achieved using temporal difference (TD) learning. Focus on how planning (via the Cross-Entropy Method, CEM, or Model Predictive Path Integral, MPPI) is executed over these latent trajectories.
* **Model-Based RL with Deterministic/Analytical World Models**: Research cases where the dynamics are not learned but are algebraically or physically exact (deterministic transition models). Analyze how exact dynamics transitions are integrated with learned GNN value/Q estimators.
* **RL + Search for Combinatorial Optimization**: Search for high-impact papers (such as AlphaGo/AlphaZero, MuZero, and recent works at NeurIPS/ICML) that combine sequential decision-making on graphs (e.g., TSP, MIS, Multicut) with look-ahead search or rollouts. Focus specifically on how planning-guided transitions are used to fill replay buffers and update target values.

#### **2. Technical & Mathematical Synthesis**
* **Co-Adapted Target Formulation**: In standard Q-learning, Q-values bootstrap from the next state's max online Q-value. In planning-guided Q-learning (such as MPC or MCTS), how should bootstrap targets be adjusted to flow through the search-guided action pathways? Detail the equations for $n$-step TD learning under planning guidance.
* **Analytical Graph Dynamics representation**: Formulate how sequential graph contractions (merging supernodes, canonical label reindexing, and edge cost summation) can be modeled as a deterministic, exact, non-differentiable transition world model. 
* **State vs. Action representation in Contraction Dynamics**: Explore how the state of a graph contraction environment can be represented (adjacency, features, label mapping) to allow rapid CPU/GPU analytical simulation of contractions during multi-step look-ahead planning.

#### **3. Algorithmic Recommendations**
* Detail the pseudocode for a **Model Predictive Control (MPC) Rollout Search** of horizon $H$ designed for discrete graph edge contractions.
* Propose how to balance GNN Q-value predictions (which capture global spatial features) with GAEC-style local signed cost reductions over the look-ahead horizon $H$. Recommend optimal normalization strategies (e.g. z-score, rank-percentile) to prevent metric scale mismatch.
* Outline potential pitfalls of multi-step planning in discrete graph contraction (e.g., action masking changes, contraction of negative edges, state representation growth) and how to systematically mitigate them.

---

### **Desired Output Format**
Compile your findings into a rigorous, publication-grade academic report containing:
1. **Theoretical Review**: Comparative table of TD-MPC, classical MPC, and RL-guided search.
2. **Mathematical Formulations**: Exact latex equations for analytical graph transitions, horizon returns, and co-adapted TD update targets.
3. **Algorithmic Blueprints**: Comprehensive pseudocode for MPC rollout search and active training loops.
4. **Architectural Guidelines**: Best practices for implementing deterministic world models for combinatorial optimization.
