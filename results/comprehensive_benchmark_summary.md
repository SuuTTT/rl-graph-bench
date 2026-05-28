# Comprehensive Baseline Benchmark Table — rl-graph-bench

_Generated at: 2026-05-26 23:29:42 | Device: cuda_

This table provides an extensive overnight baseline evaluation covering all 6 RL algorithms and standard baselines across five task domains.

## 1. Graph Partitioning Results

| Dataset | Algorithm | NCut ↓ | NMI ↑ | ARI ↑ | Modularity ↑ | Time (s) |
|---------|-----------|--------|-------|-------|--------------|----------|
| blog_proxy | **leiden** | 0.4602 | 0.9878 | 0.9875 | 0.7051 | 0.1 |
| blog_proxy | **ac2cd** | 0.4602 | 0.9752 | 0.9746 | 0.7051 | 0.2 |
| blog_proxy | **spectral** | 0.4606 | 0.9085 | 0.8969 | 0.7034 | 0.1 |
| blog_proxy | **louvain** | 0.4824 | 0.9633 | 0.9619 | 0.7008 | 0.1 |
| blog_proxy | **neurocut** | 0.4849 | 0.9752 | 0.9746 | 0.7002 | 0.2 |
| blog_proxy | **wrt** | 0.5596 | 0.9752 | 0.9746 | 0.6854 | 0.3 |
| blog_proxy | **random** | 4.0366 | 0.0498 | -0.0035 | -0.0068 | 0.1 |
| citeseer_k4 | **spectral** | 0.0408 | 0.2961 | 0.2463 | 0.5211 | 20.9 |
| citeseer_k4 | **ac2cd** | 0.2988 | 0.0861 | 0.0646 | 0.6664 | 20.3 |
| citeseer_k4 | **neurocut** | 0.2991 | 0.0861 | 0.0646 | 0.6661 | 19.8 |
| citeseer_k4 | **wrt** | 0.3005 | 0.0858 | 0.0645 | 0.6660 | 19.8 |
| citeseer_k4 | **louvain** | 2.7296 | 0.3150 | 0.1454 | 0.8447 | 20.0 |
| citeseer_k4 | **leiden** | 2.9298 | 0.3252 | 0.1666 | 0.8498 | 21.0 |
| citeseer_k4 | **random** | 2.9877 | 0.0027 | 0.0003 | 0.0026 | 19.5 |
| cora_k4 | **spectral** | 0.2678 | 0.4576 | 0.3524 | 0.6089 | 20.9 |
| cora_k4 | **neurocut** | 0.4667 | 0.2358 | 0.1836 | 0.6250 | 20.5 |
| cora_k4 | **wrt** | 0.4668 | 0.2369 | 0.1840 | 0.6253 | 20.4 |
| cora_k4 | **ac2cd** | 0.4671 | 0.2368 | 0.1840 | 0.6252 | 20.6 |
| cora_k4 | **random** | 3.0032 | 0.0025 | -0.0004 | -0.0008 | 20.5 |
| cora_k4 | **louvain** | 3.5140 | 0.4510 | 0.2974 | 0.7713 | 20.1 |
| cora_k4 | **leiden** | 3.8792 | 0.4688 | 0.2917 | 0.7818 | 20.9 |
| email_proxy | **spectral** | 0.7340 | 0.7143 | 0.6045 | 0.6697 | 0.1 |
| email_proxy | **neurocut** | 0.8855 | 0.7680 | 0.6848 | 0.6926 | 0.2 |
| email_proxy | **ac2cd** | 0.8855 | 0.7680 | 0.6848 | 0.6926 | 0.2 |
| email_proxy | **wrt** | 1.0618 | 0.7469 | 0.6627 | 0.6583 | 0.3 |
| email_proxy | **louvain** | 1.3942 | 0.8203 | 0.7493 | 0.6987 | 0.1 |
| email_proxy | **leiden** | 1.4470 | 0.8442 | 0.7807 | 0.7042 | 0.1 |
| email_proxy | **random** | 5.0538 | 0.0670 | -0.0093 | -0.0085 | 0.1 |
| sbm_n300 | **leiden** | 1.1938 | 1.0000 | 1.0000 | 0.5613 | 1.4 |
| sbm_n300 | **louvain** | 1.1938 | 1.0000 | 1.0000 | 0.5613 | 0.9 |
| sbm_n300 | **spectral** | 1.1938 | 1.0000 | 1.0000 | 0.5613 | 1.6 |
| sbm_n300 | **neurocut** | 1.1938 | 1.0000 | 1.0000 | 0.5613 | 1.4 |
| sbm_n300 | **wrt** | 1.1938 | 1.0000 | 1.0000 | 0.5613 | 1.7 |
| sbm_n300 | **ac2cd** | 1.2040 | 0.9894 | 0.9916 | 0.5591 | 1.3 |
| sbm_n300 | **random** | 4.0032 | 0.0182 | 0.0015 | -0.0001 | 0.9 |


## 2. Multicut (MCMP) Results

| Dataset | Algorithm | Mean Cost ↓ | Time (s) |
|---------|-----------|-------------|----------|
| ba_n20 | **gaec** | 1.3060 | 0.0 |
| ba_n20 | **ss2v_d3qn** | 9.1436 | 0.8 |
| ba_n40 | **gaec** | 2.6146 | 0.0 |
| ba_n40 | **ss2v_d3qn** | 19.3751 | 1.7 |
| er_n20 | **gaec** | 3.6848 | 0.0 |
| er_n20 | **ss2v_d3qn** | 13.8660 | 0.8 |
| er_n40 | **gaec** | 27.2186 | 0.0 |
| er_n40 | **ss2v_d3qn** | 58.1417 | 1.7 |


## 3. SNAP Community Detection Results

| Dataset | Algorithm | F1 Score ↑ | NCut ↓ | Time (s) |
|---------|-----------|------------|--------|----------|
| amazon_test | **leiden** | 0.3701 | 0.9622 | 0.5 |
| amazon_test | **clare** | 0.2159 | 0.4688 | 1.0 |
| amazon_test | **slrl** | 0.2159 | 0.3684 | 0.6 |
| dblp_test | **clare** | 0.1667 | 0.8975 | 5.7 |
| dblp_test | **slrl** | 0.1667 | 0.7802 | 15.8 |
| dblp_test | **leiden** | 0.0883 | 1.9055 | 17.4 |


## 4. Dynamic Community Detection Results

| Dataset | Algorithm | Modularity Density ↑ | NMI ↑ | Time (s) |
|---------|-----------|----------------------|-------|----------|
| dynamic_sbm | **leiden** | 1.0542 | 1.0000 | 0.1 |
| dynamic_sbm | **ac2cd** | 1.0542 | 1.0000 | 0.2 |

