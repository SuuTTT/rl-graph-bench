# Academic Empirical Validation & Ablation Report — rl-graph-bench

_Generated at: 2026-05-27 15:08:46 | Device: cuda_

This report documents the rigorous publication-ready academic ablation studies for the hybrid spatial-relaxation Graph Reinforcement Learning paradigm paired with Model Predictive Control (MPC) and a deterministic analytical Graph World Model.

## 1. Study 1: MPC Horizon ($H$) & Candidate Branching ($K$) Sensitivity Study
Evaluates the optimization-computation scaling trade-offs of the GNN-guided look-ahead planner across different parameters.

### LaTeX booktabs Code:
```latex
\begin{table}[t]
\caption{MPC Look-Ahead Sensitivity Study. Evaluates structural multicut cost and wall-clock execution times across search horizons $H$ and branching factors $K$ on $N=40$ ER and BA synthetic scales.}
\label{sens_grid}
\begin{tabular}{lccccc}
\toprule
H & K & ER Cost & ER Time (s) & BA Cost & BA Time (s) \\
\midrule
1 & 1 & 28.9803 & 3.6157 & 4.5917 & 3.1662 \\
1 & 3 & 28.4303 & 9.1655 & 3.1864 & 4.7460 \\
1 & 5 & 27.7718 & 21.6768 & 3.2400 & 18.2724 \\
2 & 1 & 28.9803 & 3.4911 & 4.5917 & 1.4270 \\
2 & 3 & 28.1686 & 6.9061 & 3.3919 & 10.5966 \\
2 & 5 & 29.0365 & 20.8823 & 3.2952 & 16.2973 \\
3 & 1 & 28.9803 & 2.0652 & 4.5917 & 1.0956 \\
3 & 3 & 27.7408 & 4.5723 & 3.3919 & 3.6060 \\
3 & 5 & 28.7959 & 0.8638 & 3.3041 & 0.7431 \\
\bottomrule
\end{tabular}
\end{table}

```

### Markdown Table:
| Horizon (H) | Branching (K) | ER Cost ↓ | ER Time (s) | BA Cost ↓ | BA Time (s) |
|-------------|---------------|-----------|-------------|-----------|-------------|
| 1 | 1 | 28.9803 | 3.6 | 4.5917 | 3.2 |
| 1 | 3 | 28.4303 | 9.2 | 3.1864 | 4.7 |
| 1 | 5 | 27.7718 | 21.7 | 3.2400 | 18.3 |
| 2 | 1 | 28.9803 | 3.5 | 4.5917 | 1.4 |
| 2 | 3 | 28.1686 | 6.9 | 3.3919 | 10.6 |
| 2 | 5 | 29.0365 | 20.9 | 3.2952 | 16.3 |
| 3 | 1 | 28.9803 | 2.1 | 4.5917 | 1.1 |
| 3 | 3 | 27.7408 | 4.6 | 3.3919 | 3.6 |
| 3 | 5 | 28.7959 | 0.9 | 3.3041 | 0.7 |


## 2. Study 2: Co-Adaptation DQN Target Update Ablation Study
Validates whether training representations under co-adapted updates aligns network Q-values with planned MPC look-ahead search, compared to standard Double DQN bootstrapping targets.

### LaTeX booktabs Code:
```latex
\begin{table}[t]
\caption{Co-Adaptation Ablation Analysis. Demonstrates the performance impact of training GNN representations under co-adapted target updates versus standard double Q-learning target updates, evaluated under identical $H=3, K=5$ look-ahead search.}
\label{coadaptation_ablation}
\begin{tabular}{lcc}
\toprule
Model & ER Cost & BA Cost \\
\midrule
GAEC Baseline & 25.7464 & 3.3526 \\
Standard Double DQN GNN & 54.1590 & 6.4853 \\
Co-Adapted Active MPC GNN & 28.7959 & 3.3041 \\
\bottomrule
\end{tabular}
\end{table}

```

### Markdown Table:
| Model | ER Cost ↓ | BA Cost ↓ |
|-------|-----------|-----------|
| GAEC Baseline | 25.7464 | 3.3526 |
| Standard Double DQN GNN | 54.1590 | 6.4853 |
| Co-Adapted Active MPC GNN | 28.7959 | 3.3041 |


## 3. Study 3: Extreme Out-of-Distribution Scale Generalization ($N=300, 500$)
Tests the limits of GNN generalization. Our look-ahead active GNN is evaluated zero-shot at scales up to $N=500$ (over 12x the training scale $N=40$).

### LaTeX booktabs Code:
```latex
\begin{table}[t]
\caption{Extreme Out-of-Distribution Scale Generalization. zero-shot evaluation performance and search-computation time scaling on massive synthetic scales $N=300$ and $N=500$.}
\label{extreme_scale}
\begin{tabular}{lcccc}
\toprule
Scale & Dataset & Algorithm & Cost & Time (s) \\
\midrule
300 & er & GAEC & 2725.3967 & 0.0000 \\
300 & er & SS2V (Pure Paper) & 3307.3956 & 5.6651 \\
300 & er & SS2V (One-Step Hybrid) & 3290.9767 & 3.6687 \\
300 & er & SS2V (Active MPC) & 3289.0047 & 13.4958 \\
300 & ba & GAEC & 21.9251 & 0.0000 \\
300 & ba & SS2V (Pure Paper) & 106.4017 & 2.6803 \\
300 & ba & SS2V (One-Step Hybrid) & 106.1544 & 2.1003 \\
300 & ba & SS2V (Active MPC) & 106.8017 & 14.0066 \\
500 & er & GAEC & 7974.4160 & 0.0000 \\
500 & er & SS2V (Pure Paper) & 9290.0747 & 25.6293 \\
500 & er & SS2V (One-Step Hybrid) & 9273.4790 & 28.8947 \\
500 & er & SS2V (Active MPC) & 9276.0312 & 147.8252 \\
500 & ba & GAEC & 28.1614 & 0.0000 \\
500 & ba & SS2V (Pure Paper) & 191.9738 & 26.6768 \\
500 & ba & SS2V (One-Step Hybrid) & 197.3618 & 20.1262 \\
500 & ba & SS2V (Active MPC) & 195.4576 & 131.2103 \\
\bottomrule
\end{tabular}
\end{table}

```

### Markdown Table:
| Scale (N) | Dataset | Algorithm | Mean Cost ↓ | Time (s) |
|-----------|---------|-----------|-------------|----------|
| 300 | er | **GAEC** | 2725.3967 | 0.0 |
| 300 | er | **SS2V (Pure Paper)** | 3307.3956 | 5.7 |
| 300 | er | **SS2V (One-Step Hybrid)** | 3290.9767 | 3.7 |
| 300 | er | **SS2V (Active MPC)** | 3289.0047 | 13.5 |
| 300 | ba | **GAEC** | 21.9251 | 0.0 |
| 300 | ba | **SS2V (Pure Paper)** | 106.4017 | 2.7 |
| 300 | ba | **SS2V (One-Step Hybrid)** | 106.1544 | 2.1 |
| 300 | ba | **SS2V (Active MPC)** | 106.8017 | 14.0 |
| 500 | er | **GAEC** | 7974.4160 | 0.0 |
| 500 | er | **SS2V (Pure Paper)** | 9290.0747 | 25.6 |
| 500 | er | **SS2V (One-Step Hybrid)** | 9273.4790 | 28.9 |
| 500 | er | **SS2V (Active MPC)** | 9276.0312 | 147.8 |
| 500 | ba | **GAEC** | 28.1614 | 0.0 |
| 500 | ba | **SS2V (Pure Paper)** | 191.9738 | 26.7 |
| 500 | ba | **SS2V (One-Step Hybrid)** | 197.3618 | 20.1 |
| 500 | ba | **SS2V (Active MPC)** | 195.4576 | 131.2 |

