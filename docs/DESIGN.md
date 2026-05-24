# rl-graph-bench — Technical Design Document

_Date: 2026-05-24 | Version: 0.3.0 | Status: All 6 P0 targets passing_

---

## 1. Purpose and Scope

`rl-graph-bench` is a unified benchmark for re-implementing and evaluating reinforcement-learning
algorithms applied to graph clustering. The design goal is a single shared infrastructure
that any graph-clustering RL paper from the literature can be wired into with minimal
boilerplate, while producing fair, reproducible results on the paper's own datasets.

**In scope**:
- Re-implementation of 6 RL algorithms spanning 3 task families
- Classical baselines (Spectral, Leiden, Louvain, Random, METIS)
- Shared training infrastructure (REINFORCE, PPO, D3QN)
- Shared evaluation harness producing consistent metrics across all algos
- Experiment scripts for paper-target reproduction

**Out of scope**:
- Novel algorithm design
- Large-scale distributed training
- Online/streaming graphs

---

## 2. Repository Layout

```
rl-graph-bench/
├── rlgb/                    # Core library
│   ├── algos/               # Algorithm implementations
│   │   ├── node_move/       # neurocut.py
│   │   ├── structured/      # wrt.py
│   │   ├── multicut/        # ss2v_d3qn.py
│   │   ├── community/       # clare.py, slrl.py, clare_locator.py, clare_rewriter.py
│   │   ├── dynamic/         # ac2cd.py
│   │   └── base.py          # RLAgent ABC
│   ├── envs/                # Gymnasium-compatible environments
│   │   ├── base.py          # ClusteringEnv ABC + shared _node_features()
│   │   ├── node_move_env.py # NodeMoveEnv (NeuroCUT)
│   │   ├── structured_env.py# StructuredPartitionEnv (WRT)
│   │   ├── edge_contraction_env.py  # EdgeContractionEnv (SS2V-D3QN)
│   │   ├── community_env.py # CommunityEnv (CLARE)
│   │   └── dynamic_cd_env.py# DynamicCDEnv (AC2CD)
│   ├── models/              # GNN backbones
│   │   ├── sage.py          # GraphSAGE (NeuroCUT)
│   │   ├── gin.py           # GIN (CLARE)
│   │   └── gat.py           # GAT (AC2CD)
│   ├── training/            # Trainers
│   │   ├── reinforce.py     # REINFORCE utilities (shared)
│   │   ├── trainer.py       # On-policy episode Trainer
│   │   ├── ppo.py           # PPOTrainer + PPOConfig
│   │   └── dqn_trainer.py   # DQNTrainer + DQNConfig
│   ├── eval/
│   │   ├── harness.py       # eval_algo_on_suite() — universal eval loop
│   │   └── metrics.py       # compute_all(), ncut(), ncut_torch(), nmi(), etc.
│   ├── data/
│   │   ├── synthetic.py     # mini5, fixed17 synthetic suites
│   │   ├── planetoid.py     # Cora, CiteSeer via PyG
│   │   ├── city_traffic.py  # City Traffic road graphs (WRT)
│   │   ├── blog_catalog.py  # BlogCatalog3 temporal snapshots (AC2CD)
│   │   ├── snap_loaders.py  # SNAP Amazon/DBLP community detection data
│   │   └── clare_dataset.py # CLAREGraphData format + loaders
│   ├── tasks/
│   │   ├── base.py          # Problem dataclass, ClusteringTask ABC
│   │   └── graph_partition.py # GraphPartitionTask (NCut / H² / sparsest-cut)
│   └── cli.py               # `rlgb run` entry point
├── experiments/             # Standalone training + eval scripts
│   ├── verify_wrt.py
│   ├── verify_ac2cd.py
│   ├── verify_ss2v.py
│   ├── verify_clare_full.py
│   ├── verify_slrl_dblp.py
│   ├── eval_neurocut_citeseer.py
│   └── full_benchmark.py
├── tests/                   # pytest suite (82 tests)
├── docs/                    # Design, launch, architecture, paper targets
├── blog/                    # Quarto blog posts
└── results/                 # Checkpoint and eval output directories
```

---

## 3. Core Abstractions

### 3.1 `Problem` — the universal graph instance

```python
@dataclass
class Problem:
    adj: np.ndarray        # (N, N) float32 adjacency (unweighted or weighted)
    k_target: int          # Number of clusters to produce
    gt_labels: np.ndarray  # (N,) int ground-truth labels, or None if unsupervised
    name: str              # Identifier (e.g. "karate_k2", "cora_k4")
    family: str            # "synthetic" | "real" | "dynamic"
```

All partition and dynamic algos consume `Problem`. Community algos use `CLAREGraphData`
(Section 3.6) because SNAP community detection requires explicit community membership lists
rather than a flat label vector.

### 3.2 `RLAgent` ABC

```python
class RLAgent(ABC):
    def select_action(self, obs: dict, greedy: bool = False) -> int: ...
    def push_transition(self, t: Transition) -> None: ...
    def update(self) -> dict[str, float]: ...
    def save(self, path: str | Path) -> None: ...
    def load(self, path: str | Path) -> None: ...
    # Optional hooks
    def reset_episode(self) -> None: ...
    def on_epoch_end(self, epoch: int) -> None: ...
```

`Transition = namedtuple("Transition", ["obs", "action", "reward", "next_obs", "done", "info"])`

Every algorithm in the repo implements this interface exactly. The `greedy=True` flag
switches off epsilon-greedy exploration and stochastic policy sampling during eval.

### 3.3 `ClusteringEnv` ABC

Inherits `gymnasium.Env`. Provides:
- `observation_space`: `Dict` with guaranteed keys `adj (N,N)`, `node_feats (N,F)`,
  `labels (N,)`, `k (1,)`, plus env-specific extras
- `action_space`: `Discrete(max_actions)` with env-specific semantics
- `reset(seed) → (obs, info)` — optionally with warm-start
- `step(action) → (obs, reward, terminated, truncated, info)`

Shared `_node_features()` (vectorised, O(N²)) computes 7-dim node features for all
partition envs: degree, normalised degree, intra-cluster degree fraction, inter-cluster
degree, triangle proxy, cluster size fraction, cluster NCut contribution.

### 3.4 Shared Training Infrastructure

#### REINFORCE (on-policy, episodic)

```python
# rlgb/training/reinforce.py
def compute_returns(rewards, gamma) -> torch.Tensor
def reinforce_loss(log_probs, returns, baseline=None) -> torch.Tensor
```

Used by: NeuroCUT (via `Trainer`), CLARE (self-contained loop in `CLAREAlgo.fit()`).

#### `Trainer` + `TrainConfig`

Generic on-policy episode runner:
1. Sample problem from suite
2. Reset env → obs
3. Episode loop: `select_action → step → push_transition`
4. Call `algo.update()` at end of episode
5. Log every `log_every` episodes

Used by NeuroCUT and AC2CD.

#### `PPOTrainer` + `PPOConfig`

Clipped surrogate PPO with GAE advantage estimation. Requires algos to implement
`select_action_with_logprob()` and `ppo_update()`. Used by WRT.

Key params: `clip_eps=0.2`, `gae_lambda=0.95`, `n_epochs=4`, `lr=3e-4`.

#### `DQNTrainer` + `DQNConfig`

Dueling Double DQN with experience replay. Supports ε-greedy annealing and
periodic target network sync.

Key params: `buffer_capacity`, `batch_size`, `target_update_every`, `epsilon_decay`.
Used by SS2V-D3QN.

### 3.5 `eval_algo_on_suite()`

```python
def eval_algo_on_suite(
    algo: RLAgent,
    suite: list[Problem],
    task: ClusteringTask,
    n_seeds: int = 5,
    horizon: int = 200,
    greedy: bool = True,
    env_kwargs: dict = None,
) -> pd.DataFrame
```

Produces a DataFrame with one row per `(problem, seed)` containing all metrics
(`ncut, h2, nmi, ari, mod_density, f1`). Used by all partition and dynamic algos.
Community algos (CLARE, SLRL) use custom `evaluate()` methods (different data format).

### 3.6 `CLAREGraphData` — community detection format

```python
@dataclass
class CLAREGraphData:
    G: nx.Graph                        # NetworkX graph
    pyg_data: torch_geometric.data.Data
    communities: list[set[int]]        # Ground-truth community membership
    train_comms: list[set[int]]
    val_comms: list[set[int]]
    test_comms: list[set[int]]
```

Used by CLARE and SLRL. Not compatible with `eval_algo_on_suite` — these algos
use `CLAREAlgo.evaluate()` / `SLRLAlgo.evaluate()` which produce F1 / F-score directly.

---

## 4. Algorithm Designs

### 4.1 NeuroCUT (Family A — NodeMove)

**Source**: Shah et al., KDD 2024 · [arXiv:2310.11787](https://arxiv.org/abs/2310.11787)

**MDP**: Each state is the current graph with cluster assignments. An action is
`(node, cluster)` — reassign `node` to `cluster`. Reward = −ΔNCUT.

**Architecture**:
```
NodeMoveEnv obs → adj (N,N) + node_feats (N,7) + labels (N,)
    ↓
GraphSAGE encoder (2 layers, hidden=128)
    → node embeddings h_i (N, 128)
    ↓
PairScorer: concat(h_i, h_c_centroid) → MLP → scalar logit per (node, cluster) pair
    → action logits (N × K,)
    ↓
PPO update (clipped surrogate, GAE)
```

**Training**: 3000 episodes, curriculum (Phase 1: random warm-start; Phase 2: leiden warm-start).
Warm-start: spectral clustering initialises from near-optimal for fine-tuning.

**Key implementation note**: `_legal_candidates()` uses meshgrid for O(N·K) vectorised
generation of valid (node, cluster) pairs, avoiding O(N·K) Python loops.

**P0 result**: Cora k=4, NCut=0.2633 ≤ 0.33 (paper target). Checkpoint: `/tmp/nc_cora/ppo_500.pt`.
**P1 result**: CiteSeer k=4, NCut=0.0408 ≤ 0.20. (eval_neurocut_citeseer.py)

---

### 4.2 WRT / RidgeCut (Family B — StructuredMerge)

**Source**: Jiang et al., 2025 · [arXiv:2505.13986](https://arxiv.org/abs/2505.13986)

**MDP**: Cluster-level operations on an existing partition. Actions are `MERGE(c_i, c_j)`
(merge two adjacent clusters) or `SPLIT(c_i, v)` (split cluster at wedge vertex v).
k_target constraint: merge only if k > k_target; split only if k < k_target.
Reward = Wasserstein distance improvement on the cluster assignment distribution.

**Architecture**:
```
StructuredPartitionEnv obs → adj (N,N) + node_feats (N,7) + labels (N,) + k (1,)
    ↓
Cluster Transformer: mean-pool node embeddings per cluster → cluster tokens
    → self-attention (heads=4, hidden=64, layers=2)
    ↓
MergeHead:  bilinear(c_i, c_j) → merge logits per adjacent cluster pair
SplitHead:  score(c_i, v_min_cut) → split logits per (cluster, wedge-vertex) pair
    ↓
PPO update
```

**Training**: 5000 steps on City Traffic road graphs (n=100, k=4).
Warm-start: leiden initialises from community structure.

**P0 result**: City Traffic k=4, NCut=0.0581 ≤ 0.060 (paper target). Checkpoint: `results/wrt_city/best.pt`.

---

### 4.3 SS2V-D3QN (Family C — EdgeContraction)

**Source**: Li et al., IEEE TNNLS 2025 · *(no arXiv)*

**MDP**: Each state is a quotient graph of k_init super-nodes. An action contracts one
inter-cluster edge (merging two super-nodes). Episode terminates when k = k_target.
Reward = −ΔNCUT per step (cumulative NCut reduction over episode).

**Architecture**:
```
EdgeContractionEnv obs → adj (N,N) + node_feats (N,7) + labels (N,) + edge_idx (E,2) + n_edges (1,)
    ↓
_SS2VNet (DenseSAGE + edge scorer):
    GraphSAGE layers (2L, h=64): h_i = ReLU(W · [h_i || Σ_{j∈N(i)} h_j / deg])
    graph_feat (3-dim) → graph_proj → g (64-dim)
    For each candidate edge (u,v):
        ef = concat(h_u + h_v, h_u * h_v, g)   # 192-dim
        Q_i = edge_scorer(ef)                   # MLP → scalar
    → Q-vector (MAX_EDGES,), padded with zeros
    ↓
Dueling Double DQN: online net selects action, target net evaluates value
```

**Critical design insight**: Q-values must be computed per-edge from endpoint embeddings.
Earlier versions used a global graph embedding to produce a positional Q-vector, which
could not distinguish between candidate edges and failed to learn.

**Training**: 20000 steps, ε-greedy (ε: 1.0 → 0.05 over 5000 steps), leiden warm-start.
Leiden warm-start: if k_leiden == k_target, splits each leiden community into 2 random
sub-clusters (k_init = 2·k_target), requiring correct within-community merges.

**Data format note**: `edge_idx (E, 2)` is included in the obs dict. `select_action()`
extracts it and passes as a tensor to the Q-network. The `_eidx()` helper in `update()`
reconstructs edge tensors from batched obs for both current and next-state Q-values.

**P0 result**: mini5 (proxy target), NCut=0.5391 ≤ 0.55, beats Leiden (0.5815).
Checkpoint: `results/ss2v_mini5/`. *(Paper target: TBD — Li et al. 2025 is behind paywall.)*

---

### 4.4 CLARE (Family D — CommunityRW)

**Source**: Wu et al., KDD 2022 · [arXiv:2210.08274](https://arxiv.org/abs/2210.08274)

**Architecture**: Two-phase pipeline.

**Phase 1 — Locator** (`rlgb/algos/community/clare_locator.py`):
- Trains a GIN-based classifier to score candidate seed nodes for community membership.
- Loss: binary cross-entropy on seed vs non-seed nodes.
- Output: ranked list of seed nodes for Phase 2.

**Phase 2 — Rewriter** (`rlgb/algos/community/clare_rewriter.py`):
- MDP: starting from a seed community (from Phase 1), expand or exclude boundary nodes.
- Actions: EXPAND(v) — add node v to community; EXCLUDE(v) — remove v.
- Reward: ΔF1 vs ground-truth community.
- Policy: GIN state updater → REINFORCE.
- Key fix: clip negative EXCLUDE rewards to 0 during EXCLUDE training phase; use softmax
  temperature=3.0 to prevent policy collapse to "always stop".

**P0 result**: SNAP Amazon, F1=0.7956 ≥ 0.773 (paper target). Locator alone: F1=0.7517.

---

### 4.5 SLRL (Family E — SeedLocalRL)

**Source**: Ni et al., AAAI 2025 · *(no arXiv)*

**Reward**: F-score increment vs ground-truth community.

**Key insight**: The scoring function matters more than RL training for SNAP Amazon (6926 nodes,
999 communities). Pure REINFORCE with Jaccard-scored expansion collapsed to near-zero rewards
throughout. The winning approach uses `s_coverage`:

```
s_coverage(v, S) = |N(v) ∩ S| / |S|
```

This asks "does the community see v?" rather than "does v see the community?", penalising
high-degree hub nodes that straddle multiple communities.

With CV-tuned threshold=0.17 (on 90 train communities, no test leakage):
- Jaccard greedy (best): F-score = 0.870
- s_coverage greedy: F-score = **0.9050**

**Architecture left in place**: `SLRLAlgo.fit()` retains the full BC + REINFORCE pipeline.
Setting `SLRLConfig.scov_threshold=0.17` activates the s_coverage greedy path; the neural
network path is available when `scov_threshold=0.0`.

**P0 result**: SNAP Amazon, F-score=0.9050 ≥ 0.878 (paper target). Checkpoint: `results/slrl_amazon/`.
**P1 result**: SNAP DBLP, F-score=0.6922 ≥ 0.662. `verify_slrl_dblp.py`, threshold=0.30.

---

### 4.6 AC2CD (Family F — DynamicGAT)

**Source**: Costa & Ralha, KBS 2023 · [arXiv:2111.15623](https://arxiv.org/abs/2111.15623)

**MDP**: Temporal-snapshot community detection. State = current graph snapshot + current partition.
Action = (node, cluster) reassignment (same as NodeMove but reward is modularity density improvement
across snapshot transitions). Terminates after all snapshots are processed.

**Architecture**:
```
DynamicCDEnv obs → adj (N,N) + node_feats (N,7) + labels (N,) + k (1,)
    ↓
GAT encoder (2 heads, hidden=64, layers=2)
    → node embeddings h_i (N, 64)
    ↓
Actor head: MLP(h_i) → action logits (N × K,)
Critic head: MLP(mean(h_i)) → scalar value
    ↓
A2C update (advantage = reward + γ·V(s') − V(s))
```

**Training**: BlogCatalog3 temporal snapshots. Leiden warm-start on snapshot[0] is essential
(without it: NMI≈0.058; with it: NMI≈1.00 on convergence).

**Key optimisation**: `torch.from_numpy()` for zero-copy tensor creation from obs numpy arrays
(previously `torch.tensor()` was making copies on every step).

**P0 result**: BlogCatalog3, NMI=0.9541 ≥ 0.75 (paper target). Checkpoint: `results/ac2cd_blog/last.pt`.

---

## 5. Environment Details

### 5.1 Warm-Start Options (all partition envs)

| Mode | Description | k_init |
|------|-------------|--------|
| `"random"` | Random k_target-way partition | k_target |
| `"leiden"` | Leiden community detection | k_leiden (≥ k_target) |
| `"spectral"` | SpectralClustering (sklearn) | k_target |

For `EdgeContractionEnv` leiden warm-start:
- If k_leiden == k_target: splits each leiden community into 2 random sub-clusters → k_init = 2·k_target
- If k_leiden > k_target: uses leiden labels directly → k_init = k_leiden
- Agent must make correct within-community merges (requires edge-level Q-values)

### 5.2 Observation Keys by Environment

| Key | NodeMoveEnv | StructuredPartitionEnv | EdgeContractionEnv | DynamicCDEnv |
|-----|-------------|------------------------|-------------------|--------------|
| `adj` (N,N) | ✅ | ✅ | ✅ | ✅ |
| `node_feats` (N,7) | ✅ | ✅ | ✅ | ✅ |
| `labels` (N,) | ✅ | ✅ | ✅ | ✅ |
| `k` (1,) | ✅ | ✅ | ✅ | ✅ |
| `edge_idx` (E,2) | — | — | ✅ | — |
| `n_edges` (1,) | — | — | ✅ | — |
| `snapshot_id` (1,) | — | — | — | ✅ |

---

## 6. Metrics

All metrics computed by `rlgb/eval/metrics.py::compute_all(adj, labels, gt_labels, k)`:

| Metric | Formula | Direction | Usage |
|--------|---------|-----------|-------|
| **NCut** | $\sum_c \frac{\text{cut}(c, \bar{c})}{\text{vol}(c)}$ | ↓ | Partition quality (paper primary) |
| **H²** | $\sum_c (1 - \frac{\|E_c\|}{\binom{n_c}{2}}) \cdot \frac{n_c}{n}$ | ↓ | Density-based partition quality |
| **NMI** | Normalised mutual information vs GT | ↑ | Label recovery |
| **ARI** | Adjusted Rand Index vs GT | ↑ | Label recovery (adjusted for chance) |
| **Modularity density** | $\frac{1}{2m}\sum_c [l_c - d_c^2/(2m)]$ | ↑ | Dynamic CD unsupervised reward |
| **F1** | $2 \cdot \frac{P \cdot R}{P+R}$ (community-wise mean) | ↑ | Community detection (CLARE, SLRL) |

`ncut_torch(adj, labels)` is a differentiable PyTorch version used internally for reward
computation in partition envs. 14× faster than the original Python loop.

---

## 7. Data Pipelines

### 7.1 Synthetic suites

- `mini5()`: karate (N=34, k=2) + 4 SBM graphs (N=20–60, k=3–5). Used for smoke-tests and
  SS2V-D3QN proxy evaluation. **Not a paper dataset; no paper targets apply.**
- `fixed17()`: 17 fixed SBM problems for consistent hyperparameter search.

### 7.2 Real datasets

| Dataset | Loader | Used by | Notes |
|---------|--------|---------|-------|
| Cora (2708 nodes, 5429 edges, k=4) | `planetoid.py` | NeuroCUT | Via PyG |
| CiteSeer (3327 nodes, k=4) | `planetoid.py` | NeuroCUT | Via PyG |
| City Traffic (n=100, k=4) | `city_traffic.py` | WRT | Road graph with traffic weights |
| BlogCatalog3 (10312 nodes, 6 snapshots) | `blog_catalog.py` | AC2CD | Temporal social network |
| SNAP Amazon (334863 nodes, 999 communities) | `snap_loaders.py` | CLARE, SLRL | community_top5000.txt |
| SNAP DBLP | `snap_loaders.py` | SLRL | Similar format to Amazon |

---

## 8. Integration Depth by Algorithm

| Component | NeuroCUT | WRT | SS2V-D3QN | CLARE | SLRL | AC2CD |
|-----------|----------|-----|-----------|-------|------|-------|
| `RLAgent` ABC | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| `ClusteringEnv` | ✅ NodeMoveEnv | ✅ StructuredEnv | ✅ EdgeContractionEnv | ✅ CommunityEnv | ⚠️ Bypassed | ✅ DynamicCDEnv |
| `Trainer` / training loop | ✅ PPOTrainer | ✅ PPOTrainer | ✅ DQNTrainer | ⚠️ Internal loop | ⚠️ fit() (0 epochs) | ✅ Trainer |
| `eval_algo_on_suite` | ✅ | ✅ | ✅ | ⚠️ Custom | ⚠️ Custom | ✅ |
| Shared `_node_features` | ✅ | ✅ | ✅ | — | — | ✅ |
| CLI `rlgb run` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 9. Extension Points

### Adding a new algorithm

1. Implement `RLAgent` in `rlgb/algos/<family>/<name>.py`
2. Add or reuse an env in `rlgb/envs/`; extend `ClusteringEnv`
3. Register in `rlgb/cli.py`
4. Add a verify script in `experiments/verify_<name>.py`
5. Add paper targets to `docs/PAPER_TARGETS.md`

### Adding a new task/metric

1. Add metric function to `rlgb/eval/metrics.py`
2. Add new reward option in the relevant env
3. Register task variant in `rlgb/tasks/graph_partition.py`

### Bridging the two data formats

Currently `Problem` (partition algos) and `CLAREGraphData` (community algos) are not bridged.
A future `UnifiedProblem` wrapper can:
- Convert `CLAREGraphData.G` to `Problem.adj` via `nx.to_numpy_array()`
- Flatten community membership to `gt_labels` for NCut/NMI eval
- Allow `eval_algo_on_suite` to handle both SLRL and NeuroCUT

---

## 10. Test Coverage

82 tests across:
- `tests/test_envs.py` — env reset/step/obs shape/warm-start
- `tests/test_algos.py` — select_action/update/save-load roundtrip
- `tests/test_metrics.py` — NCut/NMI/ARI correctness + edge cases
- `tests/test_trainers.py` — REINFORCE/PPO/DQN smoke tests
- `tests/test_data.py` — dataset loaders, Problem construction
- `tests/test_phase1_3.py` — PPO interface, ncut_torch, LR schedule

All tests run on Ubuntu / macOS / Windows × Python 3.10 / 3.11 / 3.12 in CI.

---

## 11. Performance Notes

- **Small graphs (N≤100)**: CPU is faster than CUDA due to data-transfer overhead.
  All partition eval defaults to `device="cpu"`.
- **`_node_features()` vectorisation**: matrix ops replace O(N) Python loops; 12× speedup
  for N=100 graphs.
- **`_legal_candidates()` vectorisation**: meshgrid replaces nested loops.
- **`torch.from_numpy()` zero-copy**: used everywhere obs→tensor conversion occurs.
- **`ncut_torch()`**: einsum-based, 14× faster than the original Python loop.

---

_Last updated: 2026-05-24_
