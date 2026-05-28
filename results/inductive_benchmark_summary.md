# Inductive Generalization Benchmark Report — rl-graph-bench

_Generated at: 2026-05-27 18:16:36 | Device: cuda_

This report presents the zero-shot scale and domain transfer capabilities of trained GNN-based RL policies compared to standard classical baselines.

## 1. WRT Zero-Shot Scale Generalization
WRT trained on $N=300$, evaluated zero-shot at scales from $N=50$ to $N=400$.

| Scale (N) | Algorithm | NCut ↓ | NMI ↑ | ARI ↑ | Time (s) |
|-----------|-----------|--------|-------|-------|----------|
| 50 | **spectral** | 0.4963 | 0.7306 | 0.6778 | 0.2 |
| 50 | **wrt** | 0.6769 | 0.7662 | 0.7386 | 3.1 |
| 50 | **random** | 3.2401 | 0.0647 | -0.0142 | 0.9 |
| 100 | **wrt** | 0.8773 | 1.0000 | 1.0000 | 4.1 |
| 100 | **spectral** | 0.8816 | 0.9491 | 0.9473 | 0.2 |
| 100 | **random** | 3.1116 | 0.0270 | -0.0084 | 0.1 |
| 200 | **spectral** | 0.8142 | 1.0000 | 1.0000 | 4.2 |
| 200 | **wrt** | 0.8142 | 1.0000 | 1.0000 | 9.2 |
| 200 | **random** | 3.0250 | 0.0211 | 0.0042 | 4.0 |
| 400 | **spectral** | 0.7812 | 1.0000 | 1.0000 | 10.2 |
| 400 | **wrt** | 0.7947 | 0.9899 | 0.9933 | 10.4 |
| 400 | **random** | 3.0064 | 0.0093 | 0.0015 | 5.9 |


## 2. NeuroCUT Zero-Shot scale & Domain Generalization
NeuroCUT GNN policy trained on Cora ($N=2708$), evaluated zero-shot across synthetic scales and domain-transferred to CiteSeer ($N=3327$).

| Scale (N) | Domain / Type | Algorithm | NCut ↓ | NMI ↑ | ARI ↑ | Time (s) |
|-----------|---------------|-----------|--------|-------|-------|----------|
| 100 | SBM (Scale N=100) | **spectral** | 0.8816 | 0.9491 | 0.9473 | 0.1 |
| 100 | SBM (Scale N=100) | **neurocut** | 0.9173 | 0.9696 | 0.9731 | 3.6 |
| 100 | SBM (Scale N=100) | **random** | 3.1116 | 0.0270 | -0.0084 | 0.1 |
| 300 | SBM (Scale N=300) | **spectral** | 0.7595 | 1.0000 | 1.0000 | 5.0 |
| 300 | SBM (Scale N=300) | **neurocut** | 0.7595 | 1.0000 | 1.0000 | 8.8 |
| 300 | SBM (Scale N=300) | **random** | 3.0160 | 0.0086 | -0.0018 | 4.1 |
| 1000 | SBM (Scale N=1000) | **spectral** | 0.7685 | 1.0000 | 1.0000 | 23.5 |
| 1000 | SBM (Scale N=1000) | **neurocut** | 0.7685 | 1.0000 | 1.0000 | 16.8 |
| 1000 | SBM (Scale N=1000) | **random** | 3.0017 | 0.0036 | 0.0004 | 13.0 |
| 3327 | CiteSeer (Cross-Domain) | **spectral** | 0.0408 | 0.2961 | 0.2463 | 40.6 |
| 3327 | CiteSeer (Cross-Domain) | **neurocut** | 0.2991 | 0.0861 | 0.0646 | 33.3 |
| 3327 | CiteSeer (Cross-Domain) | **random** | 2.9999 | 0.0017 | -0.0003 | 29.8 |


## 3. SS2V-D3QN Zero-Shot Scale Generalization (ER/BA Mixed)
SS2V edge contraction policy trained on $N=40$, evaluated zero-shot across ER/BA scales from $N=20$ to $N=200$.

| Scale (N) | Dataset | Algorithm | Mean Cost ↓ | Time (s) |
|-----------|---------|-----------|-------------|----------|
| 20 | ba_mcmp | **gaec** | 1.2609 | 0.0 |
| 20 | ba_mcmp | **ss2v_d3qn_hybrid** | 1.2609 | 4.7 |
| 20 | ba_mcmp | **ss2v_d3qn_mcts** | 1.3156 | 74.8 |
| 20 | ba_mcmp | **ss2v_d3qn_mpc** | 1.3165 | 31.8 |
| 20 | ba_mcmp | **ss2v_d3qn** | 2.7518 | 4.8 |
| 20 | er_mcmp | **ss2v_d3qn_mpc** | 3.3898 | 39.3 |
| 20 | er_mcmp | **gaec** | 3.4884 | 0.0 |
| 20 | er_mcmp | **ss2v_d3qn_hybrid** | 3.5042 | 6.5 |
| 20 | er_mcmp | **ss2v_d3qn_mcts** | 3.6389 | 57.2 |
| 20 | er_mcmp | **ss2v_d3qn** | 7.3612 | 7.0 |
| 40 | ba_mcmp | **gaec** | 2.5944 | 0.0 |
| 40 | ba_mcmp | **ss2v_d3qn_hybrid** | 2.6042 | 1.5 |
| 40 | ba_mcmp | **ss2v_d3qn_mpc** | 2.7266 | 3.5 |
| 40 | ba_mcmp | **ss2v_d3qn_mcts** | 2.9978 | 5.2 |
| 40 | ba_mcmp | **ss2v_d3qn** | 7.5156 | 2.1 |
| 40 | er_mcmp | **gaec** | 28.1101 | 0.0 |
| 40 | er_mcmp | **ss2v_d3qn_mpc** | 32.2084 | 86.0 |
| 40 | er_mcmp | **ss2v_d3qn_mcts** | 32.5345 | 115.5 |
| 40 | er_mcmp | **ss2v_d3qn_hybrid** | 43.7378 | 16.2 |
| 40 | er_mcmp | **ss2v_d3qn** | 51.7668 | 16.4 |
| 100 | ba_mcmp | **gaec** | 6.6741 | 0.0 |
| 100 | ba_mcmp | **ss2v_d3qn_mcts** | 9.7410 | 8.2 |
| 100 | ba_mcmp | **ss2v_d3qn_mpc** | 11.0747 | 4.6 |
| 100 | ba_mcmp | **ss2v_d3qn_hybrid** | 14.7220 | 1.0 |
| 100 | ba_mcmp | **ss2v_d3qn** | 21.1893 | 1.0 |
| 100 | er_mcmp | **gaec** | 252.3966 | 0.0 |
| 100 | er_mcmp | **ss2v_d3qn_mcts** | 323.1614 | 10.8 |
| 100 | er_mcmp | **ss2v_d3qn_mpc** | 323.7430 | 4.1 |
| 100 | er_mcmp | **ss2v_d3qn_hybrid** | 341.8964 | 1.3 |
| 100 | er_mcmp | **ss2v_d3qn** | 366.5039 | 1.4 |
| 200 | ba_mcmp | **gaec** | 11.4587 | 0.0 |
| 200 | ba_mcmp | **ss2v_d3qn_mcts** | 52.9422 | 413.2 |
| 200 | ba_mcmp | **ss2v_d3qn** | 53.0438 | 41.3 |
| 200 | ba_mcmp | **ss2v_d3qn_hybrid** | 54.2673 | 23.3 |
| 200 | ba_mcmp | **ss2v_d3qn_mpc** | 55.1567 | 250.3 |
| 200 | er_mcmp | **gaec** | 1139.6140 | 0.0 |
| 200 | er_mcmp | **ss2v_d3qn_mpc** | 1412.5259 | 28.1 |
| 200 | er_mcmp | **ss2v_d3qn_mcts** | 1412.5430 | 393.2 |
| 200 | er_mcmp | **ss2v_d3qn_hybrid** | 1423.9520 | 7.6 |
| 200 | er_mcmp | **ss2v_d3qn** | 1486.3572 | 13.1 |

