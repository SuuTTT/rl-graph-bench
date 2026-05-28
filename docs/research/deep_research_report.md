# Preliminary Deep Research Report: Model Predictive Control (MPC) & Deterministic Graph World Models for Reinforcement Learning

This report compiles academic literature findings, mathematical formulations, and architectural recommendations for combining **Temporal Difference Model Predictive Control (TD-MPC)**, **Graph Neural Networks (GNNs)**, and **Model Predictive Control (MPC)** on discrete graph combinatorial optimization problems.

---

## 1. Comparative Analysis of Theoretical Paradigms

To understand how our proposed framework positions itself within the state-of-the-art DRL and planning literature, we compare three core paradigms:

| Feature / Dimension | Classical TD-MPC (ICML 2022)[1][2] | Classical Discrete MPC (e.g. CEM/MPPI) | Our Proposed Graph MPC World Model |
| :--- | :--- | :--- | :--- |
| **Dynamics Representation** | Learned **Task-Oriented Latent Dynamics (TOLD)**[1][3] | Known continuous/discrete algebraic system | **Exact Analytical Deterministic Graph Dynamics** |
| **Transition Function** | Neural Network: $s_{t+1} = f(s_t, a_t)$ | Hand-crafted mathematical ODE/difference eq | Deterministic algebraic contraction: $G_{t+1} = T(G_t, a_t)$ |
| **Terminal Value Estimation** | Learned Model-Free terminal value $V(s)$[1] | Hand-crafted terminal cost or none | **Learned Dueling Double DQN $Q(G, \cdot)$** |
| **Search/Planning Space** | Latent continuous state-space[1] | Action trajectory sequences | Filtered Discrete Candidate Tree ($K$-width) |
| **Data Collection Guidance** | MPPI local trajectory optimization[1][2] | Direct system execution | **Search-guided look-ahead rollouts (MPC-style)** |
| **Differentiability** | Fully differentiable (learned networks) | Often non-differentiable / discrete | Non-differentiable algebraic graph transformations |

### Core Synergy:
In continuous continuous domains (like MuJoCo physics control), learning a learned latent world model (TD-MPC) is necessary because the raw environment state is highly complex, continuous, and noisy[1][3]. 
In contrast, **sequential graph edge contraction** is mathematically discrete and deterministic. Since the algebraic rules for contracting an edge are exact and known, we possess an **Exact, Zero-Error Graph World Model** analytically. This allows us to perform Model Predictive Control (MPC) without any dynamics approximation errors.

---

## 2. Mathematical Synthesis

### A. Analytical Graph Transition Dynamics
Let a signed graph state at step $t$ be represented as $G_t = (\mathcal{V}_t, \mathcal{E}_t, W_t, L_t)$, where:
- $\mathcal{V}_t$ is the set of supernodes (clusters).
- $\mathcal{E}_t$ is the set of active candidate edges between different clusters.
- $W_t: \mathcal{E}_t \to \mathbb{R}$ is the cluster-level sum of signed edge weights.
- $L_t: \mathcal{V}_{\text{orig}} \to \mathcal{V}_t$ is the mapping from original nodes to their current supernode clusters.

An action $a_t = (u, v) \in \mathcal{E}_t$ contracts the edge between supernodes $u$ and $v$. The analyticalDynamics transition model $T(G_t, a_t) = G_{t+1}$ is computed algebraically:
1. **Label Mapping Update**:
   $$L_{t+1}(i) = \begin{cases} L_t(u) & \text{if } L_t(i) == L_t(v) \\ L_t(i) & \text{otherwise} \end{cases} \quad \forall i \in \mathcal{V}_{\text{orig}}$$
2. **Canonical Label Reindexing**:
   $$L_{t+1} \gets \operatorname{canonicalize}(L_{t+1})$$
3. **Signed Adjacency Update**:
   The next cluster-level cost matrix $A^{\text{cost}}_{t+1}$ is computed by summing the weights of neighboring clusters:
   $$A^{\text{cost}}_{t+1}[\bar{u}, \bar{w}] = \sum_{i \in \mathcal{C}(\bar{u}), j \in \mathcal{C}(\bar{w})} A^{\text{cost}}_{\text{orig}}[i, j]$$
   where $\mathcal{C}(\bar{u})$ represents the set of original nodes mapped to canonical supernode $\bar{u}$ under $L_{t+1}$.

Because this transition is exact, the immediate reward $R(G_t, a_t)$ (multicut cost reduction) is also computed analytically without environment interaction:
$$R(G_t, a_t) = \operatorname{Cost}(G_t) - \operatorname{Cost}(G_{t+1})$$

---

### B. MPC Horizon Return and Target updates

#### **1. Horizon Return Calculation**
Given a current state $s_t$, a candidate action $e_0 \in \mathcal{E}_t$, and a look-ahead horizon $H$, the cumulative planning return $R_{\text{horizon}}(e_0)$ is computed by simulating $H$ steps into the future:
$$R_{\text{horizon}}(e_0) = r_t(e_0) + \sum_{\tau=1}^{H-1} \gamma^{\tau} R\left(G_{t+\tau}, \pi_{\text{rollout}}(G_{t+\tau})\right) + \gamma^H \max_{a} Q\left(G_{t+H}, a; \theta\right)$$

where:
- $\pi_{\text{rollout}}$ is a fast rollout policy (e.g. classical GAEC or a greedy local relaxation).
- $Q(G_{t+H}, a; \theta)$ is the GNN's online value estimator, serving as the terminal value function at the planning frontier.

#### **2. Co-Adapted Target updates**
In standard Q-learning, the value bootstrap target uses the online argmax of next-state Q-values:
$$y_t = r_t + \gamma Q_{\text{target}}\left(s_{t+1}, \operatorname{argmax}_{a'} Q(s_{t+1}, a'; \theta); \theta^-\right)$$

Under **Actor-Guided Co-Adaptation**, the Q-network must learn value approximations that align with the planning search. We update the Q-value target to bootstrap from the optimal hybrid action $a^*_{\text{hybrid}}$ chosen by the hybrid selection policy in the next state:
$$a^*_{\text{hybrid}} = \operatorname{hybrid\_policy}\left(Q(s_{t+1}, \cdot; \theta), W(s_{t+1})\right)$$
$$y_t = r_t + \gamma Q_{\text{target}}\left(s_{t+1}, a^*_{\text{hybrid}}; \theta^-\right)$$

---

## 3. Algorithmic Pseudocode

### Algorithm 1: MPC Look-Ahead Rollout Action Selection
```python
def select_action_mpc(obs, config, online_net):
    # 1. Filter candidates using GNN online Q-values
    feats = obs["node_feats"]
    adj = obs["adj"]
    edge_idx = obs["edge_idx"]
    n_cands = len(edge_idx)
    
    q_values = online_net(feats, adj, edge_idx)[:n_cands]
    
    # Prune search space to top-K structurally sound edges
    q_sorted_idx = argsort(q_values, descending=True)
    top_candidates = q_sorted_idx[:config.hybrid_top_k]
    
    best_action_idx = -1
    best_horizon_return = -inf
    
    # 2. Run MPC planning for each candidate in parallel or loop
    for cand_idx in top_candidates:
        # Simulate candidate action first step analytically
        sim_state, r0 = simulate_contraction(obs, cand_idx)
        cumulative_return = r0
        
        # Rollout for H-1 subsequent steps using fast greedy policy
        for tau in range(1, config.mpc_horizon):
            if sim_state.is_terminal():
                break
            # Rollout action selection: greedy max cluster sum
            rollout_action = select_greedy_action(sim_state)
            sim_state, r_tau = simulate_contraction(sim_state, rollout_action)
            cumulative_return += (config.gamma ** tau) * r_tau
            
        # Add terminal value estimate at the planning frontier
        if not sim_state.is_terminal():
            terminal_q = online_net(sim_state.feats, sim_state.adj, sim_state.edge_idx)
            terminal_val = max(terminal_q)
            cumulative_return += (config.gamma ** config.mpc_horizon) * terminal_val
            
        if cumulative_return > best_horizon_return:
            best_horizon_return = cumulative_return
            best_action_idx = cand_idx
            
    return best_action_idx
```

---

## 4. Key Architectural Recommendations for Implementation

1. **Avoid CPU-GPU Bottlenecks**: Simulation of contractions involves graph manipulation. If done on the CPU, frequent device-to-host and host-to-device transfers will bottleneck the training loop. We must represent graph states compactly in Numpy/CPU array operations during rollouts, and only query the GPU for the GNN's terminal value estimation at the end of the horizon.
2. **Horizon Tuning**: A look-ahead horizon of $H=3$ is recommended. Horizon lengths $>5$ lead to exponential growth in the candidate trees and increased simulation overhead, while $H=1$ collapses to immediate one-step hybrid selection.
3. **Bootstrap Normalization**: Because terminal Q-values and rollout rewards have different mathematical ranges, we recommend z-score normalization on both before blending them, preventing the terminal GNN estimator from dominating or being ignored.
