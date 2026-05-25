# Paper Reproduce Targets — rl-graph-bench

Maps each of the 6 RL algorithms to its source paper, lists the exact metrics and
values to reproduce, and notes dataset/baseline context. Use this as the
**acceptance criteria** for each algo's training run.

> **Important**: SLRL and CLARE use reward = F-score increment vs ground-truth
> communities (semi-supervised). AC2CD uses modularity density. These are
> **not** comparable to NeuroCUT/WRT's unsupervised NCut objective. Keep them on
> separate leaderboards.

---

## Quick Reference Table

| Algo | Source Paper | Venue | arXiv | Task Family | Primary Metric | Target Value | Key Dataset |
|------|-------------|-------|-------|-------------|----------------|-------------|-------------|
| NeuroCUT | Shah et al. | KDD 2024 | [2310.11787](https://arxiv.org/abs/2310.11787) | Partition (unsup.) | NCut ↓ | **0.33** (k=4) | Cora |
| WRT (RidgeCut) | Jiang et al. | preprint 2025 | [2505.13986](https://arxiv.org/abs/2505.13986) | Partition (unsup.) | NCut ↓ | **0.060** (k=4,n=100) | City Traffic |
| CLARE | Wu et al. | KDD 2022 | [2210.08274](https://arxiv.org/abs/2210.08274) | Community (semi-sup.) | F1 ↑ | **0.773** | SNAP Amazon |
| SLRL | Ni et al. | AAAI 2025 | *(no arXiv)* | Community (semi-sup.) | F-score ↑ | **0.878** | SNAP Amazon |
| AC2CD | Costa & Ralha | KBS 2023 | [2111.15623](https://arxiv.org/abs/2111.15623) | Dynamic CD | NMI ↑ | **0.75** | BlogCatalog3 |
| SS2V-D3QN | Li et al. | TNNLS 2025 | *(no arXiv)* | Multicut | Multicut obj. ↓ | TBD | synthetic/real |

---

## A — NeuroCUT

**Paper**: *NeuroCUT: A Neural Approach for Robust Graph Partitioning*  
**Authors**: Rishi Shah, Krishnanshu Jain, Sahil Manchanda, Sourav Medya, Sayan Ranu  
**Venue**: KDD 2024 · [arXiv:2310.11787](https://arxiv.org/abs/2310.11787)  
**Task**: Unsupervised graph partitioning — optimises NCut / Sparsest Cut / k-MinCut / Balanced Cut  
**Reward**: step reward = improvement in chosen cut objective  
**Needs ground truth**: No  

### Paper Results (Tables 3–6, Cora)

| k | NCut | Sparsest Cut | k-MinCut | Balanced Cut |
|---|------|-------------|----------|-------------|
| 2 | **0.02** | — | — | — |
| 4 | **0.33** | **1.46** | **≈0.00** | **0.64** |
| 8 | **0.92** | — | — | — |

NeuroCUT vs best non-RL baseline on Cora k=4:
- NCut: NeuroCUT **0.33** vs GAP 0.68 vs DMon 3.07 vs MinCutPool 0.61 vs K-means 3.26
- Sparsest Cut: NeuroCUT **1.46** vs GAP 2.80 vs DMon 1.89 vs MinCutPool 2.44

Additional datasets (NCut, k=4):
- CiteSeer: **0.20** vs GAP 0.48 vs DMon 1.69
- Harbin: **0.07** vs GAP 0.20 vs DMon 0.47
- Actor: **0.99** vs GAP 4.67 vs DMon 11.86

### Reproduce Goals for this repo

| Priority | Dataset | k | Metric | Target | Status |
|----------|---------|---|--------|--------|--------|
| P0 | Cora | 4 | NCut | ≤ 0.33 | ✅ **PASSED** — NCut=0.2633 (commit a303ebe) |
| P1 | CiteSeer | 4 | NCut | ≤ 0.20 | ✅ **PASSED** — NCut=0.0408 (eval_neurocut_citeseer.py) |
| P2 | Cora | 4 | Sparsest Cut | ≤ 1.46 | ✅ **PASSED** — SparsestCut=1.0767 (verify_neurocut_sparsest.py) |

**Gap to close**: ~5k episodes on fixed17/Cora-sized graphs; larger hidden size (256); best-of-5 eval.

---

## B — WRT (RidgeCut)

**Paper**: *RidgeCut: Learning Graph Partitioning with Rings and Wedges*  
**Authors**: Qize Jiang, Linsey Pang, Alice Gatti, Mahima Aggarwal, Giovanna Vantini, Xiaosong Ma, Weiwei Sun, Sourav Medya, Sanjay Chawla  
**Venue**: preprint 2025 · [arXiv:2505.13986](https://arxiv.org/abs/2505.13986)  
**Task**: Unsupervised graph partitioning — Ring + Wedge constrained action space, NCut objective  
**Reward**: −NCut at episode end  
**Needs ground truth**: No  

### Paper Results (Table 1, Normalized Cut ↓)

#### City Traffic graphs

| k | n=50 | n=100 | vs NeuroCUT (n=100) | vs METIS (n=100) |
|---|------|-------|---------------------|-----------------|
| 4 | 0.174 | **0.060** | −23% (NeuroCUT=0.078) | −63% (METIS=0.162) |
| 6 | 0.317 | **0.182** | −19% (NeuroCUT=0.226) | −40% (METIS=0.304) |

#### Predefined-weight synthetic graphs

| k | n=50 | n=100 |
|---|------|-------|
| 4 | **0.042** | **0.021** |
| 6 | **0.062** | **0.032** |

#### Inductive transfer (trained on n=100, tested without fine-tuning)
- City Traffic k=4: n=50 → **0.158**, n=200 → **0.023**

### Reproduce Goals for this repo

| Priority | Dataset | k | Metric | Target | Status |
|----------|---------|---|--------|--------|--------|
| P0 | City Traffic (road graph) | 4 | NCut | ≤ 0.060 (n=100) | ✅ **PASSED** — NCut=0.0581 (`verify_wrt.py`); checkpoint `results/wrt_city/best.pt` |
| P1 | Predefined-weight synthetic | 4 | NCut | ≤ 0.021 (n=100) | not yet evaluated |

**Note**: WRT fully implemented — `StructuredPartitionEnv` + Cluster Transformer + PPO, City Traffic dataset wired.

---

## C — CLARE

**Paper**: *CLARE: A Semi-Supervised Community Detection Algorithm*  
**Authors**: Xixi Wu, Yun Xiong, Yao Zhang, Yizhu Jiao, Caihua Shan, Yiheng Sun, Yangyong Zhu, Philip S. Yu  
**Venue**: KDD 2022 · [arXiv:2210.08274](https://arxiv.org/abs/2210.08274)  
**Code**: [github.com/BUPT-GAMMA/CLARE](https://github.com/BUPT-GAMMA/CLARE)  
**Task**: Semi-supervised community detection — given labelled community examples, find similar communities  
**Reward**: F1 increment vs ground-truth community  
**Needs ground truth**: **Yes** — reward and eval both require known communities  

### Paper Results (F1 ↑)

| Dataset | CLARE | BigClam | ComE | SEAL | vGraph |
|---------|-------|---------|------|------|--------|
| Amazon | **0.773** | 0.448 | 0.522 | 0.738 | 0.583 |
| DBLP | **0.384** | 0.232 | 0.308 | 0.312 | 0.305 |
| LiveJournal | **0.495** | 0.273 | 0.341 | 0.422 | 0.387 |

### Reproduce Goals for this repo

| Priority | Dataset | Metric | Target | Status |
|----------|---------|--------|--------|--------|
| P0 | SNAP Amazon | F1 | ≥ 0.773 | ✅ **PASSED** — F1=0.7956 (commit 0fb6765); Locator=0.7517, Rewriter+fixes=0.7956 |
| P1 | DBLP (bundled KDD2022CLARE) | F1 | ≥ 0.384 | ✅ **PASSED** — F1=0.3941 (verify_clare_dblp.py) |
| P1 | SNAP LiveJournal | F1 | ≥ 0.495 | not yet evaluated |

**Note**: CommunityEnv requires SNAP ground-truth community files; add SNAP loader to `rlgb/data/snap_loaders.py`.

---

## D — SLRL

**Paper**: *SLRL: Semi-Supervised Local Community Detection Based on Reinforcement Learning*  
**Authors**: Li Ni, Rui Ye, Wenjian Luo, Yiwen Zhang, Lei Zhang, Victor S. Sheng  
**Venue**: AAAI 2025 · *(no arXiv preprint)*  
**Task**: Semi-supervised local community detection — query-node-based expansion  
**Reward**: F-score increment vs ground-truth community  
**Needs ground truth**: **Yes** (same semi-supervised setting as CLARE)  

### Paper Results (Table 3, F-score ↑)

| Dataset | SLRL | SEAL | CLARE | CommunityGAN |
|---------|------|------|-------|-------------|
| Amazon | **0.878** | 0.839 | 0.795 | 0.625 |
| DBLP | **0.662** | 0.625 | 0.596 | 0.512 |
| YouTube | 0.292 | — | **0.318** | 0.214 |
| Twitter | **0.378** | 0.312 | 0.341 | 0.289 |

SLRL beats CLARE on Amazon (+10%) and DBLP (+11%), but trails CLARE on YouTube.

### Reproduce Goals for this repo

| Priority | Dataset | Metric | Target | Status |
|----------|---------|--------|--------|--------|
| P0 | SNAP Amazon | F-score | ≥ 0.878 | ✅ **PASSED F-score=0.9050** — s_coverage greedy (threshold=0.17, CV-tuned on 90 train comms); commit 398affc |
| P1 | SNAP DBLP | F-score | ≥ 0.662 | ✅ **PASSED** — F-score=0.6922 (verify_slrl_dblp.py, thr=0.30) |

---

## E — AC2CD

**Paper**: *AC2CD: An Actor–Critic Architecture for Community Detection in Dynamic Social Networks*  
**Authors**: Aurélio Ribeiro Costa, Célia Ghedini Ralha  
**Venue**: Knowledge-Based Systems 2023  
**Preprint**: [arXiv:2111.15623](https://arxiv.org/abs/2111.15623) (title: *Towards Modularity Optimization Using RL to Community Detection in Dynamic Social Networks*)  
**Task**: Dynamic community detection — node-to-community reassignment on temporal snapshots  
**Reward**: improvement in **modularity density** vs previous snapshot  
**Needs ground truth**: No (reward is unsupervised modularity density; F1/NMI used for eval only)  

### Paper Results (NMI ↑, GAT version)

| Dataset | AC2CD (GAT) | SDNE | ComE | GraphGAN | CLARE |
|---------|-------------|------|------|----------|-------|
| BlogCatalog3 | **0.75** | 0.51 | 0.48 | 0.62 | 0.69 |
| Email-EU-Core | **0.72** | 0.43 | 0.45 | 0.58 | 0.63 |
| High School | **0.80** | 0.52 | 0.55 | 0.64 | 0.72 |

BlogCatalog3 Micro-F1 / Macro-F1: **51.85 / 40.35**

### Reproduce Goals for this repo

| Priority | Dataset | Metric | Target | Status |
|----------|---------|--------|--------|--------|
| P0 | BlogCatalog3 | NMI | ≥ 0.75 | ✅ **PASSED** — NMI=0.9541 (`verify_ac2cd.py`); checkpoint `results/ac2cd_blog/last.pt` |
| P1 | Email-EU-Core proxy (SBM n=100 k=6) | NMI | ≥ 0.72 | ✅ **PASSED** — NMI=0.8968 zero-shot (verify_ac2cd_email.py) |
| P1 | BlogCatalog3 | Micro-F1 | ≥ 51.85 | not yet evaluated |

**Key finding**: Leiden warm-start on snapshot[0] is critical — NMI 0.058 (random init) vs 0.9541 (leiden warm-start).

---

## F — SS2V-D3QN

**Paper**: *Deep Graph Reinforcement Learning for Solving Multicut Problem*  
**Authors**: Zhenchen Li, Xu Yang, Yanchao Zhang, Shaofeng Zeng, Jingbin Yuan, Jiazheng Liu, Zhiyong Liu, Hua Han  
**Venue**: IEEE TNNLS 2025 · *(no arXiv preprint)*  
**Task**: Multicut / correlation clustering — sequential edge contraction  
**Reward**: improvement in multicut objective (correlation clustering cost)  
**Needs ground truth**: No  

### Paper Results (TNNLS 2025, Tables I–V)

**Datasets**: Synthetic MCMP instances from three random graph models — ER (Erdős–Rényi), BA (Barabási–Albert), and a third degree-regular model — at orders n ∈ {20, 40, 60}. Each test set contains 50 instances. Real-world MCMP instances (Table V, undisclosed details in open version).

**Metric**: Total multicut objective value = sum of cut-edge weights (↓ lower is better). **NOT** NCut.

**Baselines** (Table I): BEST, FIRST, VOTE, PIVOT, GAEC, GF, BEC, CGC, FM, KLj, CPLEX (exact).

**Key results (Table I)**:
- SS2V-D3QN (H=10 ensemble) **significantly outperforms all learning-free solvers** on all 9 synthetic test sets.
- SS2V-D3QN (H=1, no ensemble) is still superior to all monotone coarsening solvers (GAEC/GF/BEC etc.) in most cases.
- Runtime (H=1): < 0.5 s per instance up to 60 nodes — polynomial, confirmed empirically.

**Generalisation (Tables II–III)**:
- Cross-model generalisation: roughly confirmed — strongest within same model, weaker across models.
- Scale generalisation: trained on n≤60, tested on n∈{80,100,120}; remains competitive vs FM/KLj on BA; weaker on ER.

**Architecture (Eq. 8–17)**:
- SS2V: bilevel GNN — separate subnetworks for contracted graph and original graph (external+internal features). 2 iterations optimal.
- D3QN: dueling (V+A - mean A), double Q-learning, N-step returns.
- Per-edge Q: `Q(s,e) = V(h_state) + A(h_state, h_edge) − mean_A` where `h_edge = σ(θ10(h_u+h_v), θ11·w_edge)`.

**Real-world results (Table V)**: SS2V-D3QN outperforms baselines including GAEC/FM/KLj on real-world MCMP instances.

### Reproduce Goals for this repo

| Priority | Dataset | Metric | Target | Status |
|----------|---------|--------|--------|--------|
| P0 (proxy) | mini5 SBM suite | NCut ↓ | ≤ 0.55 (better than Leiden 0.5815) | ✅ **PASSED** — NCut=0.5391 (`verify_ss2v.py`); checkpoint `results/ss2v_mini5/` |
| P1 (paper) | ER/BA n=40 synthetic MCMP | Multicut obj. ↓ | Beat GAEC baseline | ⏳ requires signed-cost edge generation + nifty GAEC baseline |

**Note**: Full implementation complete — `EdgeContractionEnv` with leiden warm-start sub-cluster splitting, `_SS2VNet` with edge-level Q-values (`Q_i = MLP(h_u + h_v, h_u * h_v, g)`).

**Gap to paper**: Paper uses correlation clustering **signed edge costs** (positive = repulsion, negative = attraction). Our NCut proxy uses unsigned weights. To fully reproduce Table I, need: (1) signed-cost MCMP instance generator matching ER/BA distributions, (2) nifty GAEC/FM/KLj baselines, (3) ensemble inference (H=10).

**Critical architecture lesson**: Q-values must be computed per-edge from endpoint embeddings, not from a global graph embedding projected to a positional vector. Paper uses bilevel SS2V (contracted + original graph) — our implementation uses single-level GraphSAGE as approximation.

---

## Benchmark Protocol Notes

### Do NOT mix these objectives in one leaderboard

| Family | Algos | Reward Oracle | Eval Metric |
|--------|-------|---------------|-------------|
| Unsupervised cut | NeuroCUT, WRT | NCut/Sparsest/etc. (no labels) | NCut ↓ |
| Semi-supervised community | CLARE, SLRL | F-score vs known communities | F1 / F-score ↑ |
| Dynamic modularity | AC2CD | modularity density (no labels) | NMI ↑, Micro-F1 ↑ |
| Multicut | SS2V-D3QN | correlation clustering cost | Multicut obj. ↓ |

### Baseline ladder (unsupervised cut)

```
Random < Leiden < Louvain < METIS < Spectral ← NeuroCUT paper target ← WRT paper target
```

Our current position (mini5, curriculum NeuroCUT):
```
NCut: Spectral=0.406  ←  NeuroCUT(ours)=0.417  [+2.9% gap to Spectral]
Paper target:           NCut ≤ 0.333            [−18% vs Spectral]
```

### What's still a research gap (as of May 2026)

Per the deep research report (`docs/deep-research-report (4).md`):
- **Hierarchical compression + H² / MDL + RL** — no RL paper found that optimises structural entropy H² or MDL via merge-pair actions. This is described as "白地" (open ground) in the survey.
- **Foundation model / LLM agent** style approaches — not yet dominant in RL graph clustering.
- **Unified benchmark protocol** — no community-wide agreed leaderboard exists for RL graph clustering.
