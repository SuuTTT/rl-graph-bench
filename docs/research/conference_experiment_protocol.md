# Rigorous Peer-Review Experimental Protocol for Co-Adapted TD-MPC Graph Solvers

This document establishes the official publication-grade experimental protocol to evaluate the **Analytical Graph World Model + Co-Adapted TD-MPC** framework for Signed Multicut Combinatorial Optimization (MCMP). This protocol is designed to meet the rigorous standards of top-tier machine learning conferences (NeurIPS, ICLR, ICML).

---

## 1. Problem Formulation (Discrete Graph Contraction MDP)

We formulate the Signed Multicut problem (MCMP) as a sequential edge contraction Markov Decision Process (MDP) over a signed graph $G = (V, E, W)$, where $W \in \mathbb{R}^{|V| \times |V|}$ represents the symmetric signed cost matrix of edge affinities.

A state partition $\mathcal{C}$ decomposes $V$ into disjoint subsets (communities) representing a multicut solution. The MDP operates as follows:

1. **State Space $\mathcal{S}$**:
   Each state $s_t = (G_t, \mathcal{C}_t)$ comprises the active contracted graph $G_t$ and the node partition mapping $\mathcal{C}_t: V \to \{0, \dots, k-1\}$. At $t=0$, we initialize the system with the singleton partition $\mathcal{C}_0(i) = i$ for all $i \in V$, where each vertex is its own cluster ($k = |V|$).

2. **Action Space $\mathcal{A}(s_t)$**:
   An action $a_t$ selects an active candidate edge $e = (u, v) \in E(G_t)$ to contract. The action vocabulary is restricted to inter-cluster edges where the cost weight is strictly positive:
   $$\mathcal{A}(s_t) = \left\{ (u, v) \in E(G_t) \mid \mathcal{C}_t(u) \neq \mathcal{C}_t(v) \land W_{\mathcal{C}_t(u), \mathcal{C}_t(v)} > 0 \right\}$$
   Restricting actions to positive edges is a mathematically justified search space reduction, as contracting negative edges (representing high affinity for separation) increases multicut costs.

3. **Deterministic Graph Transition Dynamics $T(s_t, a_t)$**:
   Contracting the chosen edge $e = (u, v)$ merges cluster $\mathcal{C}_t(u)$ and cluster $\mathcal{C}_t(v)$ into a single supernode. Node partition labels are updated:
   $$\mathcal{C}_{t+1}(i) = \begin{cases} 
      \mathcal{C}_t(u) & \text{if } \mathcal{C}_t(i) == \mathcal{C}_t(v) \\
      \mathcal{C}_t(i) & \text{otherwise}
   \end{cases}$$
   The node partition indices are then canonicalized to remain contiguous in $[0, k-2]$. This exact, analytical graph contraction operator serves as our exact-dynamics world model.

4. **Reward Function $R(s_t, a_t)$**:
   The immediate reward is the net sum of edge weights between the two merging clusters:
   $$R(s_t, a_t) = \sum_{x \in \mathcal{C}_t(u)} \sum_{y \in \mathcal{C}_t(v)} W_{x, y}$$
   Cumulative rewards over an episode correspond directly to the total cost reduction (community separation objective) achieved.

---

## 2. Multi-Dimensional Performance Metrics Suite

To provide rigorous evidence of control performance and model characteristics, we track five quantitative metrics:

### A. Primary Optimization Margin (GAEC cost gap)
We measure the percentage improvement of our agent's final partition cost compared to the standard classical GAEC solver:
$$\text{Gap}_{\text{GAEC}}(\%) = \frac{\text{Cost}_{\text{Agent}} - \text{Cost}_{\text{GAEC}}}{|\text{Cost}_{\text{GAEC}}|} \times 100$$
A negative gap denotes that the agent outperforms GAEC (lower cost).

### B. Decision Wall-Clock Time
* **Inference Overhead**: Average duration (milliseconds) to compute Q-values and search actions per decision step.
* **Episode Latency**: Cumulative wall-clock time (seconds) to solve a full graph instance.

### C. Search-Value Calibration (Alignment)
We evaluate how well the GNN Q-head $Q_\theta(s, a)$ aligns with the rollout planning scores. We compute:
* **Calibration MSE**: The mean squared error between the model's Q-values and actual roll-out rewards.
* **Pearson Correlation $r$**: The linear correlation between Q-head outputs and look-ahead rollout returns. A high $r$ ($> 0.8$) proves that GNN representations have co-adapted to the planning tree space.

### D. Inductive Out-of-Distribution Generalization Ratio
We evaluate models trained on small graphs ($N \le 40$) directly on massive graphs ($N=300$ and $N=500$), and measure the generalization performance ratio:
$$\text{GenRatio} = \frac{\text{Cost}_{\text{Agent}}(N)}{\text{Cost}_{\text{GAEC}}(N)}$$

### E. Non-Parametric Statistical Significance (Wilcoxon Signed-Rank Test)
To guarantee that the GNN agent's improvements are not due to random graph sampling noise, we execute a two-sided Wilcoxon signed-rank test over paired cost samples on the evaluation suites. We reject the null hypothesis (that GAEC and our agent perform identically) if the p-value is strictly less than the alpha threshold:
$$p < 0.05$$

---

## 3. Dataset Specifications

We establish a benchmarking protocol utilizing two distinct graph generation topologies:

1. **Erdős-Rényi (ER) Signed Multicut Suite**:
   * Topology: Random graphs $G(N, p)$ with $p \in \{0.15, 0.25\}$.
   * Signed Weights: Edge weights sampled uniformly from $[-1, 1]$.
   * Evaluation Scales: $N \in \{40, 100, 300, 500\}$.
   * Instances: 5 unique, seeded instances per scale to compute statistical significance bounds.

2. **Barabási-Albert (BA) Signed Multicut Suite**:
   * Topology: Scale-free graphs generated via preferential attachment with parameter $m \in \{2, 4\}$.
   * Signed Weights: Positive and negative edge weights sampled from Uniform $[-1, 1]$.
   * Evaluation Scales: $N \in \{40, 100, 300, 500\}$.
   * Instances: 5 unique, seeded instances per scale.

---

## 4. Baseline Algorithms & Targets

To prove the superiority of the proposed co-adapted TD-MPC framework, we evaluate it against four state-of-the-art baselines:

| Baseline Algorithm | Category | Description | Target Performance Margin |
|--------------------|----------|-------------|----------------------------|
| **GAEC** | Classical Approximation | Greedy Additive Edge Contraction. Contracts the edge with maximum cluster-sum at every step. | Beat GAEC cost by $\ge 1\%$ on ER graphs; maintain tight margins on BA. |
| **Pure SS2V-D3QN** | Deep RL Heuristic | Graph Neural Network trained with standard Double Dueling DQN updates; no planning rollouts at test time. | Ours beats Pure by $\ge 30\%$ cost reduction across all scales. |
| **One-Step Hybrid SS2V** | Deep RL Hybrid | Fuses GNN Q-values with 1-step immediate cluster weights; no multi-step rollout look-ahead. | Ours beats Hybrid by $\ge 5\%$ cost reduction on ER graphs. |
| **Standard DQN MPC** | Deep RL Ablation | Double DQN GNN trained with standard DQN target updates but evaluated using $H=3, K=5$ look-ahead search. | Ours beats Standard DQN MPC by $\ge 25\%$ cost reduction, proving co-adapted updates. |

---

## 5. Statistical Hypotheses for Peer-Review

For our paper to be accepted at a top conference, our experiments must validate three key hypotheses:

* **Hypothesis 1 (Search-Value Co-Adaptation)**: Value calibration correlation $r$ will be significantly higher ($r \ge 0.85$) for the co-adapted update agent compared to standard DQN ($r < 0.60$), proving that target updating under the planning operator aligns GNN representations with look-ahead trajectories.
* **Hypothesis 2 (Zero-Shot Inductive Generalization)**: The MPC look-ahead GNN model maintains stable multicut costs on massive scales ($N=500$) without structural collapse, whereas pure GNNs suffer from severe spatial distribution shift.
* **Hypothesis 3 (Statistical Significance)**: The improvements of our Co-Adapted TD-MPC model over the GAEC baseline are statistically significant under a two-sided Wilcoxon test with $p < 0.05$.
