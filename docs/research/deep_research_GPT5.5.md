Below is the English version with a reference section appended.

---

# Temporal Difference Model Predictive Control and Analytical Graph World Models

## Abstract and Core Judgment

The central idea of TD-MPC is not to “learn a model that reconstructs the environment as faithfully as possible.” Rather, it is to “learn the quantities most relevant for control.” TD-MPC jointly learns an encoder, latent dynamics model, reward model, terminal value function, and policy prior in a latent space. At inference time, it performs only short-horizon local trajectory optimization, then uses the learned value function to estimate the long-horizon tail return. TD-MPC is built around Task-Oriented Latent Dynamics, while TD-MPC2 further frames the world model as a control-centric implicit model trained through joint-embedding prediction, reward prediction, and temporal-difference learning, without observation reconstruction [1–4].

For your target problem—graph-contraction-based combinatorial planning—the most important transfer is not to directly copy TD-MPC’s learned world model. Instead, the right transfer is the **planning–value closed loop**. The learned latent dynamics should be replaced by an **analytical, exact, non-differentiable graph transition operator**. In other words, for graph contraction, the more natural paradigm is:

[
\text{analytical dynamics} + \text{learned graph representation/value/Q} + \text{short-horizon planning}.
]

This is closer to AlphaZero-style systems, which use exact game rules with learned policy/value networks, and to neural-guided search methods for combinatorial optimization, where the problem transition is exact and only the value/prior/search guidance is learned [5–12].

Therefore, for discrete edge contraction, multi-step look-ahead, and the fusion of global GNN value estimates with local GAEC-style heuristics, the recommended architecture is not a literal “discrete TD-MPC clone.” It is better described as:

[
\boxed{
\text{Analytical Graph World Model}
+
\text{Search-Conditioned Q Learning}
+
\text{Planner Distribution Distillation}
}
]

Deployment should use receding-horizon MPC or beam-rollout search. Training targets should not fall back to ordinary one-step max-Q targets. Instead, bootstrapping should be aligned with the search-guided root action or root action distribution. If an actor is introduced, it should serve as a planning prior, candidate generator, or distillation target—not as the sole behavior policy used for bootstrapping. This is consistent with concerns raised by LOOP and later TD-M(PC)(^2)-style analyses about actor divergence and structural mismatch between the planner’s behavior distribution and the learned policy prior [13–15].

---

# 1. Theoretical Review

## 1.1 Core Literature Lineage

TD-MPC combines model predictive control with temporal-difference learning. The short-horizon part of the decision problem is handled by latent-space planning, while the long-horizon tail is handled by a learned value function. TD-MPC2 extends this idea to more scalable and robust implicit world models, adding policy priors, discrete regression for reward/value modeling, Q ensembles, SimNorm, and action masking [1–4].

The shared principle is:

[
\textbf{planning handles local optimization, while TD learning handles long-horizon value extension.}
]

Observation reconstruction is not the main objective.

Classical MPC is a broader receding-horizon control framework. At each step, it solves a finite-horizon optimal control problem using a model, executes the first action, then replans at the next state. In continuous control, sampling-based optimizers such as MPPI and CEM are common. TD-MPC2 explicitly uses MPPI in latent space, while CEM is a general black-box trajectory-distribution update method widely used in stochastic and combinatorial optimization [16–19].

For discrete graph contraction, the receding-horizon MPC idea still applies, but the optimizer should change. Instead of Gaussian action-sequence sampling, a better fit is beam search, categorical rollout, elite resampling over discrete candidates, or tree search over dynamically changing edge sets.

AlphaZero, AlphaGo Zero, and MuZero provide another major line of work: **search-guided learning**. AlphaZero assumes known game rules, so the transition model is exact. MuZero learns a model that preserves only the quantities needed for planning: reward, policy, and value, rather than reconstructing observations. Expert Iteration generalizes the idea as: search produces a stronger expert; the network generalizes the expert’s decisions. Reanalyse further shows that old replay data can be replanned to generate improved training targets [5–8].

In graph combinatorial optimization, similar ideas appear in many forms. S2V-DQN formulates combinatorial optimization as a sequential construction MDP and uses graph embeddings to choose greedy actions. Abe et al. redesign the AlphaGo Zero framework for NP-hard graph problems and use MCTS with policy/value networks at test time. Li et al. use GCNs to predict node membership in optimal solutions and combine this with guided tree search. SGBS combines neural policies with rollout simulation in fixed-width tree search. ECORD accelerates MaxCut search by running the expensive GNN only once as preprocessing, then relying on a fast search procedure [9–12].

---

## 1.2 Comparative Table

| Paradigm                         | Dynamics Model                                                                                                             | Planner                                           | Learning Signal                                                               | Replay / Old Data Usage                                                       | Direct Relevance to Graph Contraction                                                          |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Classical MPC                    | Known or analytical model; receding-horizon optimization                                                                   | MPPI, CEM, sampling optimizers, numerical solvers | Usually not replay-centric; often online optimization                         | Mainly used for state estimation and replanning                               | If graph transitions are exact, the contraction operator can directly serve as the world model |
| TD-MPC                           | Learned latent dynamics + reward + terminal value; no reconstruction                                                       | Latent-space MPPI + policy prior                  | TOLD model and value trained jointly with TD learning                         | Samples length-(H) trajectories from replay                                   | Keep the “short planning + long value” structure, but graph dynamics need not be learned       |
| TD-MPC2                          | Control-centric implicit world model; joint embedding, reward prediction, TD learning, Q ensemble, SimNorm, action masking | MPPI + policy prior + warm start                  | Multi-step model objective + entropy-regularized policy prior; EMA target Q   | Replay repeatedly trains world model and planner-guided behavior              | Most useful idea is planner-conditioned value learning, not learned dynamics itself            |
| AlphaZero / Expert Iteration     | Exact rules / analytical simulator                                                                                         | MCTS / tree search                                | Search-improved policy + outcome/value supervision                            | Self-play trajectories fill replay; search distribution becomes policy target | Closest match to exact graph dynamics + learned value/prior                                    |
| MuZero / Reanalyse               | Learned model useful for planning, not necessarily reconstructive                                                          | MCTS + learned dynamics                           | Reward/value/policy joint training; replanning old data generates new targets | Old trajectories can be reanalysed using improved planner                     | If graph dynamics are exact, skip model learning but keep reanalyse-style target refresh       |
| Neural-guided graph optimization | Usually exact problem transition; environment rules are not learned                                                        | MCTS, tree search, beam search, rollout search    | Policy/value/Q/probability estimates                                          | Search generates improved candidates or labels                                | Very close to analytical graph transition + GNN critic/prior                                   |

---

## 1.3 Unified Abstraction for Your Setting

These methods can be reduced to three layers:

1. **Exact executable transition**
   In AlphaZero, this is the game rule. In graph contraction, this is the edge-contraction operator.

2. **Generalizable evaluator**
   A value function, Q function, or candidate-action prior.

3. **Test-time planner with extra compute**
   MCTS, beam search, MPPI/CEM, or simulation-guided rollout search.

TD-MPC’s contribution is to couple layers 2 and 3 under temporal-difference learning. Your graph contraction problem naturally removes the need to learn layer 1, because the transition is analytically available.

---

# 2. Mathematical Synthesis and Co-Adapted Targets

## 2.1 From Ordinary Q Learning to Search-Conditioned Q Learning

Standard one-step Q-learning is usually written as:

[
y_t^{\text{DQN}}
================

r_t
+
\gamma
\max_{a'}
\bar Q(x_{t+1}, a').
]

This target implicitly assumes that deployment also uses the one-step greedy policy:

[
a_t = \arg\max_a Q(x_t,a).
]

However, in TD-MPC, LOOP, AlphaZero, MuZero, and Reanalyse-like systems, the executed action is often not the raw greedy Q action. Instead, it is induced by a finite-horizon planner:

[
\pi_H
\quad \text{or} \quad
a_H^\star(x).
]

If training uses ordinary max-Q but deployment uses planner-selected actions, there is structural mismatch between training and control. This is related to actor divergence in LOOP and planner-policy mismatch in TD-M(PC)(^2)-style analyses [13–15].

Define an exact or target model (T), a target value function (\bar V), and a candidate action sequence:

[
\mathbf a_{0:H-1} = (a_0,\dots,a_{H-1}).
]

The finite-horizon return is:

[
\mathcal G_H(x_t,\mathbf a_{0:H-1})
===================================

\sum_{h=0}^{H-1}
\gamma^h r(x_h,a_h)
+
\gamma^H \bar V(x_H),
]

with transition:

[
x_{h+1}=T(x_h,a_h).
]

The planner returns:

[
\mathbf a_H^\star(x_t)
======================

\arg\max_{\mathbf a_{0:H-1}}
\mathcal G_H(x_t,\mathbf a_{0:H-1}),
]

and the executed root action is:

[
a_H^\star(x_t)
==============

\left[\mathbf a_H^\star(x_t)\right]_0.
]

This is the root action actually deployed by MPC/search.

Under this definition, the co-adapted (n)-step TD target should be:

[
y_t^{(n,H)}
===========

\sum_{k=0}^{n-1}
\gamma^k r_{t+k}
+
\gamma^n
\bar Q
\left(
x_{t+n},
a_H^\star(x_{t+n})
\right).
]

If the planner outputs a root action distribution (\rho_H(\cdot \mid x)), then a soft target is:

[
y_t^{(n,H,\rho)}
================

\sum_{k=0}^{n-1}
\gamma^k r_{t+k}
+
\gamma^n
\sum_a
\rho_H(a\mid x_{t+n})
\bar Q(x_{t+n},a).
]

These equations are a synthesis rather than a direct quotation from a single paper. They combine the TD-MPC idea of “short planning plus terminal value,” the AlphaZero/MuZero idea of search-improved root actions, and the Reanalyse idea of refreshing old data with improved planning targets.

The main principle is:

[
\boxed{
\text{If deployment uses planning, bootstrapping should also follow the planned continuation.}
}
]

---

## 2.2 Planner Distribution and Distillation Objective

If the planner outputs a root distribution (\rho_H(a\mid x)), it can be used as a distillation target for a learned actor or proposal prior:

[
\mathcal L_{\text{distill}}
===========================

*

\mathbb E_{x\sim \mathcal D}
\sum_a
\rho_H(a\mid x)
\log \pi_\theta(a\mid x).
]

This corresponds to the AlphaZero / Expert Iteration idea: search produces a stronger policy, and the network learns to imitate/generalize it.

However, the actor should not completely replace the planner. If distillation is too weak, planner and network become disconnected systems. If it is too strong, the actor may collapse to the planner’s current distribution and reduce future exploration. The best engineering role for (\pi_\theta) is usually as:

[
\text{candidate generator} + \text{planning prior} + \text{amortized proposal model}.
]

The actual deployed policy should remain planner-based.

---

## 2.3 Replay Usage Under Planning Guidance

In AlphaZero-like systems, self-play trajectories enter replay, and the training labels are usually the search-improved policy distribution plus the final outcome. In MuZero/Reanalyse, old trajectories can be replanned to produce improved targets.

The general lesson is:

[
\boxed{
\text{Search should not be only an inference-time add-on. It can be a target generator.}
}
]

For graph contraction, replay should store:

[
(x_t, a_t, r_t, x_{t+1}, \mathcal A(x_t), \rho_H(\cdot\mid x_t)).
]

Here:

* (x_t) is the current contraction state.
* (a_t) is the executed contraction.
* (r_t) is the contraction reward.
* (x_{t+1}) is the analytically contracted next state.
* (\mathcal A(x_t)) is the dynamic action mask.
* (\rho_H) is the root planner distribution.

This allows the critic, prior, and value targets to remain aligned with the planner.

---

# 3. Analytical Graph World Model

## 3.1 Exact Graph-Contraction Transition

For a graph-contraction environment, define the state as:

[
x_t = (G_t, M_t, \Phi_t),
]

where:

[
G_t=(V_t,E_t,c_t)
]

is the current contracted graph, (M_t) maps original nodes to current supernodes, and (\Phi_t) contains aggregated features used by the GNN.

An action:

[
a_t \in \mathcal A(x_t)
]

corresponds to a legal current edge:

[
a_t=(i,j)\in E_t,
]

meaning: merge supernodes (i) and (j).

The dynamics are:

[
x_{t+1}
=======

T_{\text{contract}}(x_t,a_t).
]

These dynamics are:

[
\text{analytical},\quad
\text{deterministic},\quad
\text{exact},\quad
\text{usually non-differentiable}.
]

They involve equivalence-class merging, sparse edge-weight aggregation, discrete relabeling, and action-mask updates.

Given a set of contracted edges (S\subseteq E), define a contraction map:

[
f:V\to V',
]

where two original nodes map to the same new node if and only if they are connected in ((V,S)). The contracted edge set is:

[
E'
==

{f(u)f(v): f(u)\neq f(v),\ uv\in E}.
]

The contracted edge weight is the sum over all parallel edges induced by the merge:

[
c'_{ij}
=======

\sum_{uv\in E:\ f(u)=i,\ f(v)=j}
c_{uv}.
]

This parallel-edge summation is the core analytical update rule for the graph world model.

A sparse-matrix version can be written as:

[
A'
==

## K_S^\top A K_S

\operatorname{diag}(K_S^\top A K_S),
]

where (K_S) is the contraction matrix. This is especially useful for GPU-batched rollout simulation [20].

For sequential single-edge contraction, each action (a_t=(u_t,v_t)) is a special case of this update. Internally, the transition performs:

[
\text{vertex quotient merge}
+
\text{parallel edge summation}
+
\text{self-loop deletion}.
]

For caching, batched inference, and transposition tables, a canonical relabeling operator is useful:

[
\kappa_t: V_{t+1}\to {1,\dots,|V_{t+1}|}.
]

The full transition can be written as:

[
x_{t+1}
=======

\operatorname{Canon}
\left(
T_{\text{contract}}(x_t,a_t)
\right).
]

The purpose of (\operatorname{Canon}) is engineering stability rather than mathematical necessity. Two equivalent contracted graphs with different node labels should be treated as the same search state. This is compatible with GNN permutation invariance; canonical relabeling mainly serves caching, hashing, and duplicate-state detection.

---

## 3.2 Reward and GAEC-Style Local Cost Reduction

In minimum-cost multicut, negative edge weights favor separating nodes, while positive edge weights favor joining them. GAEC iteratively contracts large positive edges until no positive edge remains.

Thus, for current graph (G_t) and contraction action (a_t=(i,j)), a natural local gain is:

[
g_{\text{GAEC}}(x_t,a_t)
========================

c_t(i,j).
]

If:

[
c_t(i,j)>0,
]

then the action is locally favorable in the GAEC sense. If:

[
c_t(i,j)\le 0,
]

then the action should either be excluded from the candidate set or allowed only when deeper search predicts a sufficiently large long-term benefit.

If the environment reward should be a strict telescoping objective improvement, define a global objective (F(x)) and set:

[
r_t = F(x_t)-F(x_{t+1}).
]

Then every full trajectory satisfies:

[
\sum_{t=0}^{T-1} r_t
====================

F(x_0)-F(x_T).
]

This aligns training exactly with the final optimization objective. The practical issue is that some global objectives (F) may be expensive to evaluate at intermediate contraction states.

A practical compromise is:

* train the critic on long-horizon return;
* use (g_{\text{GAEC}}) as a cheap one-step heuristic during planning;
* combine the local heuristic with GNN-based global (Q_\theta) or (V_\theta).

---

## 3.3 State and Action Representation

For the GNN, the main input should be the **current contracted graph**, not the entire original graph plus contraction history as a large monolithic object.

A useful state representation contains:

* sparse adjacency or edge list of (G_t);
* edge weights (c_t);
* supernode sizes;
* internal accumulated edge weight;
* positive and negative external boundary volume;
* aggregated original-node features;
* optional structural statistics such as degree, triangle counts, and cut-boundary statistics.

The mapping from original nodes to supernodes, (M_t), should be stored as auxiliary metadata. It is important for reconstructing solutions and updating features, but it need not always be passed directly through the GNN.

The action should not be represented only as an edge ID. It should be represented by an edge-level feature vector:

[
\psi(x_t,a=(i,j))
=================

[
c_t(i,j),
|C_i|,
|C_j|,
\deg_t^+(i),
\deg_t^-(i),
\deg_t^+(j),
\deg_t^-(j),
\text{bridge/triangle stats},
\text{boundary features}
].
]

This lets the Q network use both global graph context and local GAEC-style information.

A practical architecture is:

[
\text{GNN}(G_t)\to h_i,h_j
]

followed by an edge MLP:

[
Q_\theta(x_t,(i,j))
===================

\operatorname{MLP}
\left(
h_i,h_j,\psi(x_t,(i,j))
\right).
]

For efficiency, the planner should avoid running a full GNN at every simulated node if possible. One efficient pattern is:

[
\text{encode graph once}
+
\text{score many edges cheaply}
+
\text{run deeper rollout only for top-}K\text{ candidates}.
]

This is similar in spirit to ECORD, which avoids repeatedly invoking an expensive GNN inside every search step [12].

---

# 4. Algorithmic Blueprints

## 4.1 MPC Rollout Search for Discrete Graph Contraction

The following is an engineering-oriented blueprint for receding-horizon graph-contraction planning. It is not a line-by-line discretization of TD-MPC. Rather, it combines MPC-style rolling optimization, exact graph transitions, search-conditioned value estimation, and normalized local/global scoring.

```text
Algorithm: Graph-Contraction MPC Rollout Search

Inputs:
    x0                current contraction state
    H                 planning horizon
    B                 beam width
    K                 per-state candidate cap
    Q_target          target Q or terminal value network
    Prior             learned edge prior / proposal network
    ExactStep         exact contraction transition
    CandidateSet      returns valid edges under current action mask
    FuseScore         combines normalized local/global scores

Procedure:
    Beam ← {(x0, [], 0)}
        # each item is (state, action_sequence, cumulative_score)

    for h in {0, ..., H-1}:
        NewBeam ← ∅

        for (x, seq, score) in Beam:
            A ← CandidateSet(x)

            if A is empty:
                add (x, seq, score) to NewBeam
                continue

            # global scores from GNN/Q/prior
            q[a] ← Q_target.root_score(x, a) for a in A
            p[a] ← Prior(x, a) for a in A

            # cheap local contraction gain
            g[a] ← local_gain_GAEC(x, a) for a in A

            # prune to top-K by prior or preliminary hybrid score
            C ← topK(A, using p or preliminary fusion)

            for a in C:
                x_next, reward ← ExactStep(x, a)

                s_step ← FuseScore(q[a], g[a], p[a], x, a)

                add (
                    x_next,
                    seq + [a],
                    score + γ^h * reward + bonus(s_step)
                ) to NewBeam

        Beam ← keep_top_B(NewBeam)

    for each (xH, seq, score) in Beam:
        score ← score + γ^H * V_target(xH)

    seq* ← argmax_seq score

    return first action of seq*
           and optionally root distribution over first actions
```

The term `bonus(s_step)` can be zero. In that case, the actual rollout return is determined only by analytical rewards and terminal value. Alternatively, the fused score can be used as a tie-breaker or pruning signal.

This algorithm is usually better suited to graph contraction than directly applying MPPI/CEM to a changing discrete action vocabulary. In graph contraction, the legal edge set changes after every contraction. A fixed-index action-sequence distribution is therefore unnatural.

---

## 4.2 Search-Conditioned Training Loop

The key training objective is not to learn graph dynamics. The graph dynamics are exact. The goal is to learn:

[
\text{value} + \text{Q function} + \text{search prior}
]

under the analytical dynamics, while keeping TD targets aligned with the planner.

```text
Algorithm: Search-Conditioned Training with Exact Graph Dynamics

Initialize replay buffer D
Initialize Qθ, Vθ, Priorθ
Initialize slowly updated target networks Q̄, V̄, Prior̄

repeat:

    # Data collection
    observe x_t

    a_t, ρ_t ← Graph-Contraction MPC Rollout Search(
                    x_t,
                    H,
                    B,
                    K,
                    Q̄/V̄,
                    Prior̄,
                    ExactStep,
                    CandidateSet
                )

    execute a_t in environment or exact simulator

    store (
        x_t,
        a_t,
        r_t,
        x_{t+1},
        mask_t,
        ρ_t
    ) in D

    # Learning
    sample mini-batch trajectories from D

    for each sampled transition index t:

        compute n-step search-conditioned target:

            y_t =
                Σ_{k=0}^{n-1} γ^k r_{t+k}
                +
                γ^n Σ_a ρ̄_H(a | x_{t+n}) Q̄(x_{t+n}, a)

        critic loss:

            L_Q = (Qθ(x_t, a_t) - y_t)^2

        optional state-value loss:

            L_V =
                (
                    Vθ(x_t)
                    -
                    Σ_a ρ_t(a | x_t) Q̄(x_t,a)
                )^2

        planner distillation loss:

            L_π =
                - Σ_a ρ_t(a | x_t) log Priorθ(a | x_t)

        optional conservative / behavior regularization:

            L_reg to reduce OOD overestimation

    update θ using:

        L = L_Q + λ_V L_V + λ_π L_π + λ_reg L_reg

    update target networks by EMA
```

The important points are:

1. The critic continuation action comes from the target planner, not from raw (\arg\max_a Q).
2. Replay should store the root search distribution and the action mask.
3. If an actor/prior exists, it should be distilled from the planner distribution.

---

# 5. Fusing GNN Q Values with GAEC Local Heuristics

The safest way to combine global GNN value estimates and local GAEC-style gains is not direct addition. Their scales are usually incompatible. Instead, normalize them inside the current candidate action set.

Let:

[
\mathcal A(x)
]

be the legal action set at state (x). Let:

[
q(a)
]

be the GNN/Q score, and:

[
g(a)
]

be the local GAEC gain.

## 5.1 Robust Z-Score Fusion

Use robust statistics over the current legal action set:

[
\tilde q(a)
===========

\frac{
q(a)-\operatorname{median}*{b\in\mathcal A(x)}q(b)
}{
\operatorname{MAD}*{b\in\mathcal A(x)}q(b)+\varepsilon
},
]

[
\tilde g(a)
===========

\frac{
g(a)-\operatorname{median}*{b\in\mathcal A(x)}g(b)
}{
\operatorname{MAD}*{b\in\mathcal A(x)}g(b)+\varepsilon
}.
]

Then fuse:

[
S(a)
====

\lambda_h \tilde q(a)
+
(1-\lambda_h)\tilde g(a).
]

## 5.2 Rank-Percentile Fusion

Alternatively, use rank percentiles:

[
u_q(a)
======

\operatorname{PctRank}_{\mathcal A(x)}(q(a)),
]

[
u_g(a)
======

\operatorname{PctRank}_{\mathcal A(x)}(g(a)).
]

Then:

[
S(a)
====

\lambda_h u_q(a)
+
(1-\lambda_h)u_g(a).
]

Rank fusion is often more stable when Q values are poorly calibrated or when edge-weight magnitudes vary strongly across graph instances.

## 5.3 Depth-Dependent Fusion

Let (\lambda_h) depend on rollout depth (h).

A reasonable schedule is:

[
\lambda_0 > \lambda_1 > \cdots > \lambda_{H-1}.
]

Near the root, use a larger (\lambda_h), because the GNN sees the real current state and can make a better global judgment. Deeper in rollout, simulated states become more distribution-shifted, so the learned Q estimate may become less reliable. The analytical local heuristic does not drift, so it can be trusted more for deeper pruning.

---

# 6. Architectural Guidelines and Risks

## 6.1 Recommended Implementation Priority

Build the first version without learned dynamics.

The minimal strong version should contain:

[
\text{exact contraction operator}
+
\text{GNN encoder}
+
\text{edge-level Q head}
+
\text{GAEC local score}
+
\text{beam-rollout planner}
+
\text{search-conditioned TD target}.
]

Once this loop works, additional components can be added:

* planner distribution distillation;
* reanalyse-style target refresh;
* conservative Q regularization;
* improved candidate priors;
* batched GPU rollout;
* transposition-table caching.

This system is conceptually closer to an AlphaZero / Expert Iteration / TD-MPC hybrid than to a standard model-based RL system that redundantly learns deterministic rules.

---

## 6.2 State Cache and Simulator Representation

Maintain two representations:

1. **Simulator representation**

   * union-find;
   * supernode membership;
   * sparse edge-merge structure;
   * dynamic legal-action set;
   * fast contraction updates.

2. **Network representation**

   * canonical COO/CSR graph tensor;
   * edge features;
   * supernode features;
   * action mask;
   * optional graph hash.

For many parallel rollouts, the RAMA-style sparse-matrix contraction formula is worth borrowing:

[
A'
==

## K^\top A K

\operatorname{diag}(K^\top A K).
]

This converts graph contraction into sparse matrix operations, making GPU-batched rollout more feasible [20].

---

## 6.3 Action Masking

Training should explicitly store and use the dynamic action mask.

TD-MPC2 uses zero-padding and action masking for multi-task, multi-action-space settings. Graph contraction has a related issue: every contraction changes the legal action set. If action masks are ignored, Q targets, policy distillation, and planner distributions will leak probability mass or value estimates onto invalid edges.

Therefore:

[
\rho_H(a\mid x)=0
\quad
\text{for all}
\quad
a\notin \mathcal A(x).
]

The same masking must be applied in:

* Q target computation;
* prior logits;
* planner expansion;
* distillation loss;
* replay reanalysis.

---

## 6.4 Main Risks and Mitigations

### Risk 1: Training–Deployment Mismatch

If planner-generated behavior fills replay but the bootstrap target follows raw greedy Q or an undistilled actor, the system may develop overestimation and distribution drift.

Mitigation:

[
\text{Use search-conditioned TD targets.}
]

Also distill planner distributions into the actor/prior when an actor is used.

---

### Risk 2: Premature Negative-Edge Contraction

In multicut-style problems, positive edges tend to favor joining, while negative edges tend to favor cutting. If the planner contracts strongly negative edges too early because of erroneous long-term Q estimates, the state may become irreversibly damaged.

Mitigation:

Use a two-level filter:

1. Hard-filter strongly negative edges below a threshold.
2. Allow them only if deeper search predicts a sufficiently large long-term advantage.

This sacrifices some optimality but can strongly improve training stability.

---

### Risk 3: Duplicate States and Search Explosion

Different contraction orders can lead to the same quotient graph. Without canonical relabeling and a transposition table, the planner will evaluate the same state repeatedly.

Mitigation:

Use:

[
\operatorname{Canon}(G)
]

plus graph hashing. Cache:

[
(\text{canonical state hash}, \text{root action})
]

and reuse Q/value/planner evaluations.

---

### Risk 4: Normalization Mismatch

Global Q values, local edge gains, and prior logits have different magnitudes and distributions. They also change with graph size, remaining node count, candidate count, and rollout depth.

Mitigation:

Normalize per state over the legal candidate set:

[
\mathcal A(x).
]

Use robust z-score or rank-percentile normalization rather than global fixed statistics.

---

# 7. Limitations and Open Questions

I did not find a standard, widely adopted paper specifically titled or framed as “TD-MPC for discrete graph contraction with an exact non-differentiable world model.” The co-adapted target equations, normalization strategy, and rollout-search pseudocode above are therefore a unified design proposal synthesized from TD-MPC/TD-MPC2, AlphaZero/MuZero/Reanalyse, LOOP, and neural-guided graph optimization literature.

A second limitation is that TD-M(PC)(^2)-style work on structural policy mismatch and persistent value overestimation is highly relevant, but some of the available material is still preprint/project-page level. It should be treated as a strong empirical clue rather than settled consensus.

The overall architectural judgment remains robust:

[
\boxed{
\text{Use an analytical world model. Learn value, Q, priors, search distillation, and target alignment.}
}
]

---

# References

[1] Nicklas Hansen et al. **Temporal Difference Learning for Model Predictive Control**. ICML 2022.
[https://proceedings.mlr.press/v162/hansen22a.html](https://proceedings.mlr.press/v162/hansen22a.html)

[2] TD-MPC project page. **Temporal Difference Learning for Model Predictive Control**.
[https://www.nicklashansen.com/td-mpc/](https://www.nicklashansen.com/td-mpc/)

[3] Nicklas Hansen et al. **TD-MPC2: Scalable, Robust World Models for Continuous Control**. arXiv, 2023/2024.
[https://arxiv.org/html/2310.16828v2](https://arxiv.org/html/2310.16828v2)

[4] TD-MPC2 project page.
[https://www.tdmpc2.com/](https://www.tdmpc2.com/)

[5] David Silver et al. **Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm**. AlphaZero. arXiv, 2017.
[https://arxiv.org/abs/1712.01815](https://arxiv.org/abs/1712.01815)

[6] Julian Schrittwieser et al. **Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model**. MuZero. arXiv, 2019.
[https://arxiv.org/abs/1911.08265](https://arxiv.org/abs/1911.08265)

[7] Anthony et al. **Thinking Fast and Slow with Deep Learning and Tree Search**. Expert Iteration. arXiv, 2017.
[https://arxiv.org/abs/1705.08439](https://arxiv.org/abs/1705.08439)

[8] MuZero Reanalyse / related OpenReview material.
[https://openreview.net/forum?id=HKtsGW-lNbw](https://openreview.net/forum?id=HKtsGW-lNbw)

[9] Elias Khalil et al. **Learning Combinatorial Optimization Algorithms over Graphs**. S2V-DQN. NeurIPS 2017 / arXiv.
[https://arxiv.org/abs/1704.01665](https://arxiv.org/abs/1704.01665)

[10] Kenshin Abe et al. **Solving NP-Hard Problems on Graphs with Extended AlphaGo Zero**. OpenReview.
[https://openreview.net/forum?id=0_ao8yS2eBw](https://openreview.net/forum?id=0_ao8yS2eBw)

[11] Li et al. **Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search**. NeurIPS 2018.
[https://proceedings.neurips.cc/paper/2018/hash/8d3bba7425e7c98c50f52ca1b52d3735-Abstract.html](https://proceedings.neurips.cc/paper/2018/hash/8d3bba7425e7c98c50f52ca1b52d3735-Abstract.html)

[12] ECORD / neural-guided large-scale combinatorial search. OpenReview.
[https://openreview.net/forum?id=olQbo52II9](https://openreview.net/forum?id=olQbo52II9)

[13] Sikchi et al. **Learning Off-Policy with Online Planning**. LOOP.
[https://publications.ri.cmu.edu/learning-off-policy-with-online-planning](https://publications.ri.cmu.edu/learning-off-policy-with-online-planning)

[14] TD-M(PC)(^2): **Improving Temporal Difference MPC Through Policy Constraint**. Project page.
[https://darthutopian.github.io/tdmpc_square/](https://darthutopian.github.io/tdmpc_square/)

[15] TD-M(PC)(^2) arXiv preprint.
[https://arxiv.org/html/2502.03550v1](https://arxiv.org/html/2502.03550v1)

[16] Stanford ASL. **Model Predictive Control Lecture Notes**.
[https://stanfordasl.github.io/aa203/sp2223/pdfs/lecture/lecture_11.pdf](https://stanfordasl.github.io/aa203/sp2223/pdfs/lecture/lecture_11.pdf)

[17] MathWorks. **What Is Model Predictive Control?**
[https://www.mathworks.com/help/mpc/gs/what-is-mpc.html](https://www.mathworks.com/help/mpc/gs/what-is-mpc.html)

[18] Williams et al. **Model Predictive Path Integral Control**. arXiv.
[https://arxiv.org/abs/1509.01149](https://arxiv.org/abs/1509.01149)

[19] Rubinstein and Kroese. **The Cross-Entropy Method / Cross-Entropy Optimization Tutorial**.
[https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf](https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf)

[20] Abbas et al. **RAMA: A Rapid Multicut Algorithm on GPU**. CVPR 2022.
[https://openaccess.thecvf.com/content/CVPR2022/papers/Abbas_RAMA_A_Rapid_Multicut_Algorithm_on_GPU_CVPR_2022_paper.pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Abbas_RAMA_A_Rapid_Multicut_Algorithm_on_GPU_CVPR_2022_paper.pdf)

[21] IJCAI 2021 multicut / graph contraction reference.
[https://www.ijcai.org/proceedings/2021/595](https://www.ijcai.org/proceedings/2021/595)

[22] SoRB: **Search on the Replay Buffer**. arXiv.
[https://arxiv.org/abs/1906.05253](https://arxiv.org/abs/1906.05253)

[23] Recent multicut reinforcement-learning / edge-based GNN reference. arXiv, 2026.
[https://arxiv.org/html/2605.13673v1](https://arxiv.org/html/2605.13673v1)
