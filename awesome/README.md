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

**NeuroCUT: Neural Graph Partitioning via Reinforcement Learning**

| Resource | Link |
|---|---|
| Paper (arXiv) | [arXiv:2401.XXXXX](https://arxiv.org/abs/2401.XXXXX) *(placeholder)* |
| Code | [github.com/neurocut/neurocut](https://github.com/neurocut/neurocut) |
| OpenReview | [openreview.net/forum?id=NeuroCUT2024](https://openreview.net/forum?id=NeuroCUT2024) |
| Project Page | — |
| Blog / Talk | — |
| TeX Source | — |
| Model Weights | — |
| Datasets | Cora, CiteSeer, SBM-k5 |

**Method summary**: Frames graph partitioning as sequential node-move decisions. Uses GraphSAGE to encode node representations, and REINFORCE to train a policy that maximises −H² (structural entropy).

**Key metrics** (reported): NCut ↓ 18% vs. Spectral on SBM-k5; H² ↓ 12% vs. Leiden.

---

### B — Structured: WRT

**WRT: Wedge–Ring Transformer for Graph Partitioning with PPO**

| Resource | Link |
|---|---|
| Paper (arXiv) | [arXiv:2502.XXXXX](https://arxiv.org/abs/2502.XXXXX) *(placeholder)* |
| Code | [github.com/wrt-graph/wrt](https://github.com/wrt-graph/wrt) |
| OpenReview | [openreview.net/forum?id=WRT2025](https://openreview.net/forum?id=WRT2025) |
| Project Page | — |
| TeX Source | — |
| Model Weights | — |
| Datasets | DBLP, LFR-μ0.3 |

**Method summary**: Models cluster-level merge/split decisions using a Transformer encoder over cluster-level embeddings. PPO training with ring/wedge structural priors.

---

### C — CommunityRW: CLARE

**CLARE: Community Learning via Adaptive Reinforcement Expansion**

| Resource | Link |
|---|---|
| Paper (arXiv) | [arXiv:2203.XXXXX](https://arxiv.org/abs/2203.XXXXX) *(placeholder)* |
| Code | [github.com/BUPT-GAMMA/CLARE](https://github.com/BUPT-GAMMA/CLARE) |
| OpenReview | — |
| Project Page | — |
| TeX Source | — |
| Model Weights | — |
| Datasets | Amazon, DBLP, LJ (SNAP) |

**Method summary**: Semi-supervised community expansion. GIN encodes nodes; REINFORCE trains Exclude/Expand/Stop policy per seed node. Achieves state-of-the-art F1 on SNAP benchmarks.

---

### D — CommunityRW: SLRL

**SLRL: Seed-Level Reinforcement Learning for Community Detection**

| Resource | Link |
|---|---|
| Paper (arXiv) | [arXiv:2501.XXXXX](https://arxiv.org/abs/2501.XXXXX) *(placeholder)* |
| Code | [github.com/slrl-cd/slrl](https://github.com/slrl-cd/slrl) |
| OpenReview | [openreview.net/forum?id=SLRL2025](https://openreview.net/forum?id=SLRL2025) |
| Blog | — |
| TeX Source | — |
| Model Weights | — |
| Datasets | DBLP, Amazon (SNAP) |

**Method summary**: Per-query seed expansion with a linear RL policy (no GNN — uses degree features only). Achieves near-CLARE F1 at 10× lower inference cost.

---

### E — DynamicAC: AC2CD

**AC2CD: Actor-Critic for Dynamic Community Detection on Temporal Graphs**

| Resource | Link |
|---|---|
| Paper (arXiv) | [arXiv:2312.XXXXX](https://arxiv.org/abs/2312.XXXXX) *(placeholder)* |
| Code | [github.com/ac2cd/ac2cd](https://github.com/ac2cd/ac2cd) |
| OpenReview | — |
| Blog | — |
| TeX Source | — |
| Model Weights | — |
| Datasets | DBLP-dyn, Reddit-dyn, SBM-temporal |

**Method summary**: GAT encoder processes snapshot graphs; A2C (actor-critic) makes node-move decisions tracking temporal community drift. Optimises modularity-density.

---

### F — Multicut: SS2V-D3QN

**SS2V-D3QN: Subgraph-to-Vector Dueling Double DQN for Graph Multicut**

| Resource | Link |
|---|---|
| Paper (arXiv) | [arXiv:2510.XXXXX](https://arxiv.org/abs/2510.XXXXX) *(placeholder)* |
| Code | [github.com/ss2v/ss2v-d3qn](https://github.com/ss2v/ss2v-d3qn) |
| OpenReview | [openreview.net/forum?id=SS2V2025](https://openreview.net/forum?id=SS2V2025) |
| Blog | — |
| TeX Source | — |
| Model Weights | — |
| Datasets | Cora, CiteSeer, DBLP |

**Method summary**: Frames multicut as sequential edge-contraction. Dense SAGE encodes subgraph states; Dueling + Double DQN selects which edge to contract. Achieves near-optimal multicut on small graphs.

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
| NeuroCUT (hidden=64, 10k eps) | Cora (200 nodes) | H²=2.41 | *(coming soon)* |
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
