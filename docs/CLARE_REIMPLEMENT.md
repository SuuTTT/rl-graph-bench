# CLARE Re-implementation Plan

**Context**: Our rlgb `CLAREAlgo` wrapper achieves F1=0.3714 vs the paper target of
F1≥0.773 on SNAP Amazon.  Running the original authors' code (`FDUDSDE/KDD2022CLARE`)
on the same dataset reproduces F1=0.7895 in ~6 minutes.  This document explains
exactly why we fail, what the paper actually does, and how to re-implement CLARE
natively inside rlgb.

---

## 1. Why the rlgb wrapper fails

### 1.1 Wrong problem formulation

The rlgb `CLAREAlgo` is wired to `CommunityEnv`, which treats community detection as
a **single-agent, online node-rewriting problem** over one global graph state.  The
actual CLARE paper solves a different problem:

> *Given a set of labelled exemplar communities (training set), find and refine all
> similar communities in the graph.*

This is **semi-supervised retrieval + local rewriting**, not online RL over a global
state.  The mismatch means the reward signal the rlgb wrapper receives is structurally
different from what the paper trains on.

### 1.2 Missing the Locator phase entirely

CLARE is a two-phase pipeline:

```
Phase 1 — Community Locator
  Input : full graph + 90 training communities
  Output: 1000 candidate communities (one per test query)

Phase 2 — Community Rewriter
  Input : one candidate community + its true community (training time)
  Output: refined community after ≤ max_step EXPAND/EXCLUDE actions
```

The rlgb wrapper skips Phase 1 completely.  Without the Locator, the Rewriter starts
from a random or arbitrary community boundary that has no relationship to any ground-
truth community, so the REINFORCE reward (delta-F1 against true_com) is nearly always
zero.  This is exactly the flat reward curve observed throughout training.

### 1.3 Reward signal is wrong

| | Original CLARE | rlgb wrapper |
|---|---|---|
| Reward per step | `F1(pred_com, true_com) × 10` — computed against the *known* true community that the Locator matched | `ΔF1` over full community assignment |
| Signal density | Dense (every EXCLUDE/EXPAND changes F1 by a measurable amount because the agent is close to the true boundary) | Sparse (most steps on a random 2k-node subgraph don't change macro-F1) |
| Scale | ×10 — rewards in range [0, 10] | ≈ 0.001 — machine-epsilon noise |

### 1.4 Wrong graph scope

| | Original CLARE | rlgb wrapper |
|---|---|---|
| Nodes | 6,926 (Amazon-1.90 filtered) | 2,000 (BFS subgraph) |
| Communities | 999 (train 90, val 10, test 900) | Top-5 largest (154–328 members) |
| Avg community size | 8–9 nodes | 154–328 nodes |
| Graph is dense enough? | Yes — every community's ego-net is connected | No — BFS subgraph cuts most inter-community edges |

The community sizes in the rlgb subgraph (avg ~278) are 30× larger than in the
original dataset (avg ~8.5).  Even if the agent did learn, the feature representation
and GIN ego-net assumption break at this scale.

### 1.5 Input representation mismatch

The original Rewriter builds a **local ego-net** around the candidate community plus
its outer boundary.  Node features are `[degree, min_neighbor_deg, max_neighbor_deg,
mean_neighbor_deg, std_neighbor_deg]` (5 dims) **plus a position flag** (1 if node is
currently inside the predicted community, 0 otherwise), giving 65-dim node features
fed to a GIN.  The GIN runs on the ego-net edge set only, not the full graph.

The rlgb wrapper feeds the full graph adjacency and a 64-dim feature to a different
GIN model, sharing no representational structure with the original.

---

## 2. Key design choices that enable CLARE's performance

### 2.1 Order embedding for community retrieval

The Locator trains a GCN to embed **ego-nets** using an order-embedding loss:

$$\mathcal{L} = \sum_{(G_a, G_b) \in \mathcal{P}} \max(0,\ \|[\mathbf{E}(G_a) - \mathbf{E}(G_b)]_+\|^2 - \text{margin}) + \ldots$$

The intuition: if community $A$ is a sub-community of $B$, then $E(A)$ should be
coordinate-wise ≤ $E(B)$ in embedding space.  This lets the model learn structural
containment relationships and retrieve test communities that are structurally similar
to training exemplars.

**Why it matters**: It provides the Rewriter with a seed community that is already
≈70–75% F1 (before any rewriting — see AvgF1=0.7323 after Locator only).  The Rewriter
only needs to push from 0.73 → 0.79, a small and tractable refinement.

### 2.2 Per-community REINFORCE with known true_com

The Rewriter agent trains on individual communities where the **true community is
known** (from the training split).  This means:
- Reward is always dense (every action changes F1 against a fixed target)
- The agent observes position flags (inside/outside prediction) — a direct hint about
  boundary quality
- EXCLUDE and EXPAND have **separate policy networks and optimizers**, preventing
  mode collapse where the agent only learns one action type

### 2.3 Virtual stop nodes

Instead of a fixed horizon, the original adds two virtual nodes (VIRTUAL_EXCLUDE,
VIRTUAL_EXPAND) to the candidate set.  Selecting a virtual node ends the EXCLUDE or
EXPAND phase.  This lets the agent learn *when to stop* rather than always taking all
`max_step` actions.

### 2.4 GIN on ego-net, not global graph

The GIN runs only on the subgraph induced by `pred_com ∪ outer_boundary`.  This
keeps computation O(community size) rather than O(graph size), enables batching over
many communities, and ensures the representation focuses on local boundary structure.
The position flag (inside vs outside bit) is prepended to node features, giving the
GIN explicit boundary information.

### 2.5 Semi-supervised training, transductive test

Training uses only 90 of 999 communities.  The Locator is trained to embed these 90
exemplars; the Rewriter is trained to refine them.  At test time, the Locator retrieves
900 new candidate communities by embedding similarity (it has never seen their true
communities), and the Rewriter refines them zero-shot.  This is why the algorithm
generalises: the Rewriter learns a *generic boundary-cleaning policy*, not a graph-
specific one.

---

## 3. Re-implementation plan for rlgb

### 3.1 New files to create

```
rlgb/
  algos/community/
    clare_native.py          ← CLAREAlgoNative: wraps both phases
  models/
    order_embedding.py       ← OrderEmbeddingLoss + EgoNetEncoder
    rewriter_net.py          ← ExcludeNet, ExpandNet, GINUpdater (ego-net)
  tasks/
    community_semisup.py     ← SemiSupCommunityTask (train/val/test split)
  data/
    clare_dataset.py         ← load_clare_dataset() → CLAREGraphData
```

### 3.2 `CLAREGraphData` — data layer

```python
@dataclass
class CLAREGraphData:
    nx_graph: nx.Graph
    communities: list[list[int]]   # all ground-truth communities
    node_feats: np.ndarray         # shape (N, 5) degree-based
    train_ids: list[int]           # indices into communities
    val_ids: list[int]
    test_ids: list[int]
```

`load_clare_dataset(name, train=90, val=10)` should:
1. Read `{name}-1.90.ungraph.txt` and `{name}-1.90.cmty.txt` from
   `~/.rlgb_data/CLARE/{name}/` (download if absent via `requests`).
2. Build degree features using the same 5-dim formula as the original.
3. Return a `CLAREGraphData` with randomised train/val/test split.

The Amazon files are already bundled in `/tmp/KDD2022CLARE/dataset/amazon/` and can
be copied to the data cache for bootstrapping.

### 3.3 `OrderEmbeddingLoss` — Locator training

Implement `rlgb/models/order_embedding.py`:
- `EgoNetEncoder(GNNEncoder)`: 2-layer GCN (input_dim=5, hidden=64, output=64) +
  `global_add_pool` — identical to `Locator/gnn.py:GNNEncoder`.
- `order_embedding_loss(emb_pos, emb_neg, margin=0.6)`: max-margin loss over
  positive (subgraph) and negative (random) pairs.
- `generate_ego_net(graph, community, n_layers=2)` → PyG `Data`: ego-net subgraph
  from utils/helper_funcs.py — copy verbatim and adapt to rlgb path conventions.

### 3.4 `CommunityLocator` — full Locator pipeline

```python
class CommunityLocator:
    def __init__(self, encoder: EgoNetEncoder): ...

    def fit(self, data: CLAREGraphData, epochs=30, lr=1e-3, batch_size=256):
        """Train order embedding on train communities."""

    def predict(self, data: CLAREGraphData, num_pred=1000,
                comm_max_size=12) -> list[list[int]]:
        """
        1. Embed all nodes.
        2. Generate candidate ego-nets (BFS from each node up to comm_max_size).
        3. Match candidates to training community embeddings via L2 distance.
        4. Return top-num_pred candidates, one per test community.
        """
```

The matching logic in `Locator/matching.py` is self-contained and can be ported
nearly verbatim — it's ~120 lines and has no external dependencies beyond PyTorch and
numpy.

### 3.5 `RewriterNet` + `RewritingAgent` — Rewriter

```python
# rlgb/models/rewriter_net.py
class RewriterNet(nn.Module):
    """Separate GIN updater + two MLP scorers for EXCLUDE and EXPAND."""
    gcn:        GINConv(nn.Sequential(Linear(65, 64), ReLU, Linear(64, 64)))
    exclude_net: MLP(65, 32, 1)
    expand_net:  MLP(65, 32, 1)
```

Key implementation details to preserve:
- **Position flag**: prepend 1-bit (inside=1 / outside=0) to 64-dim node embedding
  before feeding MLP scorers.
- **Virtual stop nodes**: append one VIRTUAL_EXCLUDE and one VIRTUAL_EXPAND node
  (zero embedding) to the candidate list.  Action = virtual → break the loop.
- **Separate optimisers**: `exclude_opt` and `expand_opt` update their respective
  MLPs independently per step.  Do NOT share a single REINFORCE update across both.
- **Reward scaling**: multiply F1 by 10 before REINFORCE loss.

### 3.6 `SemiSupCommunityTask` — task layer

```python
class SemiSupCommunityTask:
    """Semi-supervised community detection task matching CLARE paper protocol."""

    def train_episode(self, agent: CLAREAlgoNative, data: CLAREGraphData):
        """
        For each training community:
          1. Locator.predict() on train split to get seed community.
          2. Build Community object (pred_com, true_com, ego-net).
          3. Run Rewriter for max_step EXCLUDE steps, then max_step EXPAND steps.
          4. Compute REINFORCE loss and step optimisers.
        """

    def evaluate(self, agent, data, split="test") -> dict:
        """Return AvgF1, Jaccard, NMI, detect_percent matching paper metrics."""
```

### 3.7 `CLAREAlgoNative` — top-level wrapper

```python
class CLAREAlgoNative(RLAgent):
    """Native CLARE: Locator (order embedding) + Rewriter (REINFORCE).

    Call .fit(data) to train both phases end-to-end.
    Call .predict(data) to get test community predictions.
    """
    compatible_tasks = ["community_semisup"]
```

This inherits `RLAgent` for harness compatibility but overrides the standard
`select_action` / `update` loop with the two-phase training logic.

### 3.8 Eval harness integration

Add to `rlgb/eval/harness.py`:

```python
def eval_clare_native(algo: CLAREAlgoNative, data: CLAREGraphData,
                      n_seeds=3) -> pd.DataFrame:
    results = []
    for seed in range(n_seeds):
        algo.fit(data, seed=seed)
        metrics = algo.evaluate(data, split="test")
        results.append({"algo": "CLARE-native", "seed": seed, **metrics})
    return pd.DataFrame(results)
```

Paper target check: `df["f1"].mean() >= 0.773`.

---

## 4. Effort estimate and priority

| Component | LoC (est.) | Dependency | Priority |
|-----------|-----------|------------|----------|
| `clare_dataset.py` | ~80 | None | High — needed for everything |
| `order_embedding.py` (EgoNetEncoder + loss) | ~120 | PyG | High |
| `Locator/matching.py` port | ~130 | EgoNetEncoder | High |
| `rewriter_net.py` | ~80 | None | High |
| `Community` data object | ~100 | rewriter_net | High |
| `RewritingAgent` | ~60 | rewriter_net | High |
| `SemiSupCommunityTask` | ~120 | all above | Medium |
| `CLAREAlgoNative` | ~80 | all above | Medium |
| Harness integration + tests | ~60 | all above | Low |
| **Total** | **~830** | | |

Most of the Locator and Rewriter code can be ported near-verbatim from
`/tmp/KDD2022CLARE/`.  The main rlgb-specific work is the data layer
(`CLAREGraphData`) and harness wiring (`SemiSupCommunityTask`).

**Suggested order**: data layer → Locator (train + predict) → verify F1≥0.73 after
Locator alone → Rewriter → verify F1≥0.773 end-to-end.  Each stage has a clear
numerical checkpoint so regressions are caught early.

---

## 5. Notes on SLRL

SLRL (AAAI 2025) has no public code.  The paper describes a similar two-phase approach
(community seed selection + RL expansion), but without a reference implementation it
is impossible to determine whether the rlgb failure is a wrapper bug, a
hyperparameter gap, or a genuine over-report.  **Flagged as cannot-reproduce** until
authors release code or respond to a replication request.
