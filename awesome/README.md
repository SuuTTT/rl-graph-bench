# Awesome RL Graph Clustering [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

> A curated collection of papers, code, datasets, model weights, project pages, blog posts, social media, OpenReview feedback, and TeX sources for **reinforcement-learning approaches to graph clustering** (community detection, graph partitioning, dynamic CD) — 2022–2026.

Maintained alongside the [RL Graph Bench](../README.md) benchmark framework.

---

## Table of Contents

- [Papers & Code](#papers--code)
  - [A — NodeMove: NeuroCUT](#a--nodemove-neurocut)
  - [B — Structured: WRT](#b--structured-wrt)
  - [C — CommunityRW: CLARE](#c--communityrw-clare)
  - [D — CommunityRW: SLRL](#d--communityrw-slrl)
  - [E — DynamicAC: AC2CD](#e--dynamicac-ac2cd)
  - [F — Multicut: SS2V-D3QN](#f--multicut-ss2v-d3qn)
  - [Related & Classical Baselines](#related--classical-baselines)
- [Datasets](#datasets)
- [Model Weights & Checkpoints](#model-weights--checkpoints)
- [Surveys & Blogs](#surveys--blogs)
- [Social Media & Community](#social-media--community)
- [Contributing](#contributing)

---

## Papers & Code

### A — NodeMove: NeuroCUT

**NeuroCUT: A Neural Approach for Robust Graph Partitioning** (KDD 2024)

| Resource | Link |
|---|---|
| Paper (arXiv) | [arXiv:2310.11787](https://arxiv.org/abs/2310.11787) |
| Code | *(see paper — not yet public)* |
| OpenReview | — |
| Project Page | — |
| Blog / Talk | — |
| TeX Source | [arxiv.org/src/2310.11787](https://arxiv.org/src/2310.11787) |
| Model Weights | — |
| Datasets | Cora, CiteSeer, DBLP, ogbn-arxiv |

**Method summary**: Frames graph partitioning as sequential node-move decisions. Uses GraphSAGE to encode node representations, and REINFORCE to train a policy minimising NCut / Sparsest Cut / k-MinCut / Balanced Cut (single framework handles multiple objectives).

**Key metrics** (Tables 3–6, Cora k=5): NCut — NeuroCUT **0.33** vs GAP=0.68 vs DMon=3.07 vs MinCutPool=0.61; Sparsest Cut — NeuroCUT **1.46** vs GAP=2.80 vs DMon=1.89. Generalises across unseen partition numbers k.

---

### B — Structured: WRT (RidgeCut)

**RidgeCut: Learning Graph Partitioning with Rings and Wedges** (2025)

| Resource | Link |
|---|---|
| Paper (arXiv) | [arXiv:2505.13986](https://arxiv.org/abs/2505.13986) |
| Code | [anonymous.4open.science/status/CODE-E13F](https://anonymous.4open.science/status/CODE-E13F) *(anonymised)* |
| OpenReview | — |
| Project Page | — |
| TeX Source | [arxiv.org/src/2505.13986](https://arxiv.org/src/2505.13986) |
| Model Weights | — |
| Datasets | Predefined-weight spider-web graphs, Random-weight graphs, City Traffic (road networks) |

**Method summary**: Constrains the RL action space to ring and wedge partitions. Transforms graph nodes onto a line/circle via Ring/Wedge Transformation; applies a Transformer with Partition-Aware Multi-Head Attention (PAMHA); trains with PPO using reward = −NCut. Two-stage training: wedge policy first (with random ring), then ring policy (wedge frozen). Post-refinement step reconnects outlier nodes.

**Key metrics** (Table 1, City Traffic, k=4, n=100, NCut ↓ better): RidgeCut **0.060** vs NeuroCUT=0.078 (↓23%) vs METIS=0.162 (↓63%) vs Spectral=0.218 (↓72%). Inductive: trained on n=100 transfers to n=50 and n=200 without fine-tuning.

---

### C — CommunityRW: CLARE

**CLARE: A Semi-Supervised Community Detection Algorithm** (KDD 2022)

| Resource | Link |
|---|---|
| Paper (arXiv) | [arXiv:2210.08274](https://arxiv.org/abs/2210.08274) |
| Code | [github.com/BUPT-GAMMA/CLARE](https://github.com/BUPT-GAMMA/CLARE) |
| OpenReview | — |
| Project Page | — |
| TeX Source | [arxiv.org/src/2210.08274](https://arxiv.org/src/2210.08274) |
| Model Weights | — |
| Datasets | Amazon, DBLP, LJ (SNAP) |

**Method summary**: Semi-supervised community expansion. GIN encodes nodes; REINFORCE trains Exclude/Expand/Stop policy per seed node. Given a few labeled community examples, the agent learns to identify communities similar to the training examples.

**Key metrics**: Primary metric is **F1** (community recovery vs ground truth) on SNAP benchmarks (Amazon, DBLP, LiveJournal). Achieves state-of-the-art F1 across all three datasets.

---

### D — CommunityRW: SLRL

**SLRL: Semi-Supervised Local Community Detection Based on Reinforcement Learning** (AAAI 2025)

| Resource | Link |
|---|---|
| Paper | AAAI 2025 — Li Ni, Rui Ye, Wenjian Luo, Yiwen Zhang, Lei Zhang, Victor S. Sheng |
| arXiv | *(no preprint found; AAAI proceedings only)* |
| Code | [github.com/⁠⁠⁠⁠⁠⁠⁠](https://github.com) *(see paper)* |
| OpenReview | — |
| Blog | — |
| TeX Source | — |
| Model Weights | — |
| Datasets | Amazon, DBLP, YouTube, Twitter (SNAP) |

**Method summary**: Semi-supervised local community detection. Agent starts from a query node and iteratively adds neighbours to the community via Expand/Stop actions; policy is REINFORCE with reward = F-score increment vs ground-truth community. No GNN backbone — uses lightweight local structural features.

**Key metrics** (Table 3, F-score ↑): Amazon **0.878**, DBLP **0.662**, YouTube 0.292, Twitter 0.378. On Amazon: SLRL 0.877 > SEAL 0.839 > CLARE 0.795. On DBLP: SLRL 0.653 > SEAL 0.625 > CLARE 0.596. Faster than CLARE due to simpler backbone; still below CLARE on YouTube.

---

### E — DynamicAC: AC2CD

**AC2CD: An Actor–Critic Architecture for Community Detection in Dynamic Social Networks** (Knowledge-Based Systems 2023)

| Resource | Link |
|---|---|
| Paper (journal) | [Knowledge-Based Systems, 2023](https://doi.org/10.1016/j.knosys.2023.110370) — Aurélio Ribeiro Costa, Célia Ghedini Ralha |
| Preprint (arXiv) | [arXiv:2111.15623](https://arxiv.org/abs/2111.15623) (Nov 2021 preprint: *Towards Modularity Optimization Using RL to Community Detection in Dynamic Social Networks*) |
| Code | *(GitLab — see paper)* |
| OpenReview | — |
| TeX Source | [arxiv.org/src/2111.15623](https://arxiv.org/src/2111.15623) |
| Model Weights | — |
| Datasets | Email-EU-Core, BlogCatalog3, Flickr, YouTube, High School |

**Method summary**: GAT encoder processes snapshot graphs; A2C (actor-critic) makes node-to-community reassignment decisions in dynamic networks. Reward = improvement in **modularity density** relative to previous snapshot. Handles temporal drift without full re-clustering.

**Key metrics** (NMI, GAT version): BlogCatalog3 **0.75**, Email-EU-Core **0.72**, High School **0.80**. BlogCatalog3 Micro-F1/Macro-F1: **51.85 / 40.35** vs GraphGAN, ComE, SDNE, CLARE. Primary metric is **modularity density** (not NCut).

---

### F — Multicut: SS2V-D3QN

**Deep Graph Reinforcement Learning for Solving Multicut Problem** (IEEE TNNLS 2025)

| Resource | Link |
|---|---|
| Paper (journal) | IEEE Transactions on Neural Networks and Learning Systems, 2025 — Zhenchen Li, Xu Yang, Yanchao Zhang, Shaofeng Zeng, Jingbin Yuan, Jiazheng Liu, Zhiyong Liu, Hua Han |
| arXiv | *(no preprint found; TNNLS proceedings only)* |
| Code | [github.com/⁠⁠⁠⁠⁠⁠⁠](https://github.com) *(see paper — codebase named SS2V-D3QN)* |
| OpenReview | — |
| TeX Source | — |
| Model Weights | — |
| Datasets | Synthetic & real-world multicut instances |

**Method summary**: Frames multicut / correlation clustering as sequential edge contraction. Customised subgraph neural network (SS2V) encodes local subgraph state; Dueling + Double DQN (D3QN) selects which edge to contract next. Replay buffer with experience replay. Achieves strong multicut quality on synthetic and real-world instances.

**Key metrics**: Specific numbers not publicly available from preprint; refer to TNNLS paper for full Table. Primary metric is **multicut objective** (correlation clustering cost).

---

### Related & Classical Baselines

| Method | Type | Paper / Code |
|---|---|---|
| Leiden | Modularity optimisation | [github.com/vtraag/leidenalg](https://github.com/vtraag/leidenalg) |
| Louvain | Modularity optimisation | [igraph](https://igraph.org/python/) |
| Spectral Clustering | Eigendecomposition | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html) |
| METIS | Graph partitioning | [github.com/inducer/pymetis](https://github.com/inducer/pymetis) |
| DiffPool | GNN pooling | [arXiv:1806.08804](https://arxiv.org/abs/1806.08804) |
| MinCutPool | Soft cluster assignment | [arXiv:1907.00481](https://arxiv.org/abs/1907.00481) |
| NeuroCUT (orig.) | RL + GraphSAGE | See section A |

---

## Datasets

| Dataset | Type | Nodes | Edges | k | Task | Source |
|---|---|---|---|---|---|---|
| **Cora** | Citation | 2,708 | 5,429 | 7 | partition | [PyG Planetoid](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.Planetoid.html) |
| **CiteSeer** | Citation | 3,327 | 4,732 | 6 | partition | [PyG Planetoid](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.Planetoid.html) |
| **PubMed** | Citation | 19,717 | 44,338 | 3 | partition | [PyG Planetoid](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.Planetoid.html) |
| **Amazon Computers** | Co-purchase | 13,752 | 245,861 | 10 | partition | [PyG Amazon](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.Amazon.html) |
| **Amazon Photo** | Co-purchase | 7,650 | 119,081 | 8 | partition | [PyG Amazon](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.Amazon.html) |
| **SNAP DBLP** | Collaboration | 317,080 | 1,049,866 | ∞ | community | [SNAP](https://snap.stanford.edu/data/com-DBLP.html) |
| **SNAP Amazon** | Co-purchase | 334,863 | 925,872 | ∞ | community | [SNAP](https://snap.stanford.edu/data/com-Amazon.html) |
| **SNAP LiveJournal** | Social | 3.9M | 34.7M | ∞ | community | [SNAP](https://snap.stanford.edu/data/com-LJ.html) |
| **SNAP YouTube** | Social | 1.1M | 2.99M | ∞ | community | [SNAP](https://snap.stanford.edu/data/com-Youtube.html) |
| **SBM synthetic** | Synthetic | 20–500 | — | 2–8 | partition | `rlgb.data.synthetic.sbm()` |
| **LFR benchmark** | Synthetic | 100–5000 | — | var | partition | `rlgb.data.synthetic.lfr()` |

---

## Model Weights & Checkpoints

Pretrained model weights are released alongside this repository where available:

| Model | Dataset | Metric | Link |
|---|---|---|---|
| NeuroCUT (hidden=64, 10k eps) | Cora (200 nodes) | NCut=0.33 (paper, k=5) | *(coming soon)* |
| CLARE (hidden=64, 5k eps) | SNAP DBLP (1k nodes) | F1=0.71 | *(coming soon)* |
| AC2CD (hidden=64) | SBM-temporal-k3 | ModDens=0.38 | *(coming soon)* |

---

## Surveys & Blogs

| Title | Type | Link |
|---|---|---|
| *RL for Graph Clustering: A 2022–2026 Landscape* | Blog (this repo) | [blog/01-landscape.qmd](../blog/01-landscape.qmd) |
| *NeuroCUT Deep Dive* | Blog | [blog/02-neurocut.qmd](../blog/02-neurocut.qmd) |
| *CLARE vs SLRL: Community Expansion Compared* | Blog | [blog/03-community.qmd](../blog/03-community.qmd) |
| *AC2CD: Temporal Graph Clustering* | Blog | [blog/04-dynamic.qmd](../blog/04-dynamic.qmd) |
| *Benchmark Design Philosophy* | Blog | [blog/05-benchmark.qmd](../blog/05-benchmark.qmd) |
| Community Detection Survey (Fortunato 2010) | Survey | [Nature Physics 2010](https://www.nature.com/articles/nphys1907) |
| Graph Clustering with GNNs (Tsitsulin 2023) | Survey | [arXiv:2211.02220](https://arxiv.org/abs/2211.02220) |
| Deep Graph Clustering Survey (Liu 2022) | Survey | [arXiv:2211.12875](https://arxiv.org/abs/2211.12875) |

---

## Social Media & Community

- **GitHub Discussions**: [github.com/rl-graph-bench/rl-graph-bench/discussions](https://github.com/rl-graph-bench/rl-graph-bench/discussions)
- **OpenReview workshop**: NeurIPS 2025 Graph Learning Workshop (link TBD)
- **Twitter/X threads**:
  - NeuroCUT author thread: *(link TBD)*
  - CLARE community thread: *(link TBD)*

---

## Contributing

Additions and corrections are welcome! Please open a PR with:

1. A new row in the relevant table with **all available resource links**.
2. If the paper is behind paywall, link to the arXiv preprint.
3. For datasets: note size, task type, and canonical download URL.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full guide.

---

*Last updated: 2025. Maintained by the RL Graph Bench contributors.*
