"""Community Locator — Phase 1 of native CLARE re-implementation.

Ported from FDUDSDE/KDD2022CLARE (Locator/ + utils/) with rlgb-style API.

Architecture: GCN order-embedding encoder trained with a max-margin loss on
ego-net vs sub-ego-net pairs.  At inference, each node's k-hop ego-net is
embedded and matched to training-community embeddings by L2 distance; the
nearest-neighbour ego-nets become candidate communities for the Rewriter.

Paper: CLARE (Wu et al., KDD 2022).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import k_hop_subgraph, subgraph

from rlgb.data.clare_dataset import CLAREGraphData


# ---------------------------------------------------------------------------
# GNN Encoder (Locator/gnn.py)
# ---------------------------------------------------------------------------

class GNNEncoder(nn.Module):
    """2-layer GCN with global-add-pool, returns (node_emb, graph_emb)."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64,
                 output_dim: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.act = nn.LeakyReLU()
        if n_layers < 2:
            raise ValueError("n_layers must be >= 2")
        layers = [GCNConv(input_dim, hidden_dim)]
        for _ in range(n_layers - 2):
            layers.append(GCNConv(hidden_dim, hidden_dim))
        layers.append(GCNConv(hidden_dim, output_dim))
        self.conv_layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        return_node: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for conv in self.conv_layers[:-1]:
            x = conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, training=self.training)
        node_emb = self.conv_layers[-1](x, edge_index)
        graph_emb = global_add_pool(node_emb, batch)
        return node_emb, graph_emb


# ---------------------------------------------------------------------------
# Ego-net helpers (utils/helper_funcs.py)
# ---------------------------------------------------------------------------

def _drop_nodes(graph_data: Data, aug_ratio: float = 0.15) -> Data:
    """Contrastive augmentation: randomly drop ~aug_ratio of nodes."""
    n = graph_data.x.size(0)
    drop_n = int(n * aug_ratio)
    perm = np.random.permutation(n)
    keep = np.sort(perm[drop_n:])
    idx_map = {old: new for new, old in enumerate(keep)}

    ei = graph_data.edge_index.cpu().numpy()
    new_ei = [
        [idx_map[ei[0, i]], idx_map[ei[1, i]]]
        for i in range(ei.shape[1])
        if ei[0, i] in idx_map and ei[1, i] in idx_map
    ]
    try:
        new_edge_index = torch.tensor(new_ei, dtype=torch.long).t().contiguous()
        return Data(x=graph_data.x[keep], edge_index=new_edge_index)
    except Exception:
        return graph_data


def _prepare_locator_batch(
    node_list: list[int],
    data: Data,
    max_size: int = 20,
    num_hop: int = 2,
) -> tuple[Batch, Batch]:
    """Build (ego-net batch, corrupted-subgraph batch) for order-embedding training."""
    batch, corrupt_batch = [], []
    num_nodes = data.x.size(0)

    for node in node_list:
        node_set, _, _, _ = k_hop_subgraph(
            node_idx=node, num_hops=num_hop,
            edge_index=data.edge_index, num_nodes=num_nodes,
        )
        if len(node_set) > max_size:
            perm = torch.randperm(node_set.shape[0])[:max_size]
            node_set = torch.unique(torch.cat([torch.tensor([node]), node_set[perm]]))

        nodes = node_set.tolist()
        # Ensure seed node is first
        if nodes[0] != node:
            idx = nodes.index(node)
            nodes[0], nodes[idx] = nodes[idx], nodes[0]
        assert nodes[0] == node

        ei, _ = subgraph(nodes, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
        g = Data(x=data.x[nodes], edge_index=ei)
        batch.append(g)
        corrupt_batch.append(_drop_nodes(g))

    return Batch.from_data_list(batch), Batch.from_data_list(corrupt_batch)


def _generate_ego_net_neighbors(
    nx_graph: nx.Graph,
    start_node: int,
    k: int = 1,
    max_size: int = 15,
) -> list[int]:
    """BFS k-hop neighbours of start_node (capped at max_size)."""
    queue, visited = [start_node], [start_node]
    for _ in range(k):
        nxt = []
        for u in queue:
            for v in nx_graph.neighbors(u):
                if v not in visited:
                    visited.append(v)
                    nxt.append(v)
                if len(visited) >= max_size:
                    return sorted(visited)
        queue = nxt
    return sorted(visited)


# ---------------------------------------------------------------------------
# Locator config + class
# ---------------------------------------------------------------------------

@dataclass
class LocatorConfig:
    hidden_dim:   int   = 64
    output_dim:   int   = 64
    n_layers:     int   = 2
    margin:       float = 0.6
    lr:           float = 1e-3
    weight_decay: float = 1e-5
    epochs:       int   = 30
    batch_size:   int   = 256
    subg_max_size: int  = 20
    num_hop:      int   = 2
    num_pred:     int   = 1000
    comm_max_size: int  = 12
    log_every:    int   = 5
    device:       str   = "cpu"


class CommunityLocator:
    """Order-embedding community locator (Phase 1 of native CLARE).

    Usage::

        locator = CommunityLocator(LocatorConfig())
        locator.fit(data)            # train GCN encoder
        pred_comms = locator.predict(data)  # list[list[int]], len=num_pred
    """

    def __init__(self, config: LocatorConfig | None = None) -> None:
        self.cfg = config or LocatorConfig()
        self.device = torch.device(self.cfg.device)

    def _make_encoder(self, input_dim: int) -> GNNEncoder:
        enc = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.output_dim,
            n_layers=self.cfg.n_layers,
        ).to(self.device)
        return enc

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, data: CLAREGraphData) -> None:
        """Train order-embedding GCN on training+validation community members."""
        input_dim = data.pyg_data.x.size(1)
        self.encoder = self._make_encoder(input_dim)
        opt = optim.Adam(self.encoder.parameters(),
                         lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        seed_nodes = list({n for com in data.train_communities + data.val_communities
                           for n in com})

        # keep PyG data on CPU for batch construction (k_hop_subgraph must be CPU)
        pyg_cpu = data.pyg_data  # always CPU from loader
        self.encoder.train()

        for epoch in range(1, self.cfg.epochs + 1):
            opt.zero_grad()
            t0 = time.time()

            sample = random.sample(seed_nodes, min(self.cfg.batch_size, len(seed_nodes)))
            batch, corrupt = _prepare_locator_batch(
                sample, pyg_cpu, max_size=self.cfg.subg_max_size, num_hop=self.cfg.num_hop
            )
            batch = batch.to(self.device)
            corrupt = corrupt.to(self.device)

            _, emb       = self.encoder(batch.x,   batch.edge_index,   batch.batch)
            _, emb_corr  = self.encoder(corrupt.x, corrupt.edge_index, corrupt.batch)

            # positive: (corrupt_subgraph, full) → corrupt should be ≤ full
            # negative: (random_shuffled, full) → no containment relationship
            shuf = torch.randperm(emb.size(0))
            emb_neg = emb[shuf]

            emb_as = torch.cat([emb_corr, emb_neg], dim=0)
            emb_bs = torch.cat([emb,      emb],     dim=0)
            labels = torch.tensor(
                [1] * emb.size(0) + [0] * emb.size(0), device=self.device
            )

            e = torch.sum(torch.clamp(emb_as - emb_bs, min=0.0) ** 2, dim=1)
            loss_pos = e[labels == 1]
            loss_neg = torch.clamp(self.cfg.margin - e[labels == 0], min=0.0)
            loss = (loss_pos.mean() + loss_neg.mean()) / 2

            loss.backward()
            opt.step()

            if epoch % self.cfg.log_every == 0 or epoch == self.cfg.epochs:
                print(f"  [Locator] epoch {epoch:04d}/{self.cfg.epochs}"
                      f"  loss={loss.item():.4f}  t={time.time()-t0:.2f}s")

        self.encoder.eval()
        print("  [Locator] training done.\n")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _embed_communities(self, data: CLAREGraphData,
                           comms: list[list[int]]) -> np.ndarray:
        pyg = data.pyg_data  # CPU
        batch_list = []
        for com in comms:
            ei, _ = subgraph(com, pyg.edge_index, relabel_nodes=True,
                             num_nodes=pyg.x.size(0))
            batch_list.append(Data(x=pyg.x[com], edge_index=ei))
        b = Batch.from_data_list(batch_list).to(self.device)
        _, emb = self.encoder(b.x, b.edge_index, b.batch)
        return emb.cpu().numpy()

    @torch.no_grad()
    def _embed_all_nodes(self, data: CLAREGraphData) -> np.ndarray:
        pyg = data.pyg_data  # CPU
        n = pyg.x.size(0)
        chunk = 4096
        out = np.zeros((n, self.cfg.output_dim), dtype=np.float32)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            nodes = list(range(start, end))
            b, _ = _prepare_locator_batch(
                nodes, pyg, max_size=self.cfg.subg_max_size, num_hop=self.cfg.num_hop
            )
            b = b.to(self.device)
            _, emb = self.encoder(b.x, b.edge_index, b.batch)
            out[start:end] = emb.cpu().numpy()
            print(f"  [Locator] embedded nodes {start}–{end}")
        return out

    @torch.no_grad()
    def get_node_embeddings(self, data: CLAREGraphData) -> np.ndarray:
        """Return per-node GCN embeddings (N, output_dim) using the full graph.

        These are the 64-dim base features consumed by the Rewriter.
        Matches CommMatching.generate_all_node_emb() in the original code.
        """
        pyg = data.pyg_data.to(self.device)
        node_emb, _ = self.encoder(pyg.x, pyg.edge_index,
                                   torch.zeros(pyg.x.size(0), dtype=torch.long,
                                               device=self.device))
        return node_emb.cpu().numpy()

    def predict(self, data: CLAREGraphData) -> list[list[int]]:
        """Retrieve `num_pred` candidate communities by nearest-neighbour matching."""
        query_emb  = self._embed_communities(
            data, data.train_communities + data.val_communities
        )
        node_emb   = self._embed_all_nodes(data)
        seed_nodes = {n for com in data.train_communities + data.val_communities
                      for n in com}

        num_pred   = self.cfg.num_pred
        per_query  = max(1, num_pred // query_emb.shape[0])

        pred_comms: list[list[int]] = []
        used_seeds: set[int] = set()

        for q_emb in query_emb:
            dist     = np.sqrt(np.sum((q_emb - node_emb) ** 2, axis=1))
            ranked   = np.argsort(dist).tolist()
            added    = 0

            for node in ranked:
                if added >= per_query or len(pred_comms) >= num_pred:
                    break
                if node in seed_nodes or node in used_seeds:
                    continue
                nbrs = _generate_ego_net_neighbors(
                    data.nx_graph, node,
                    k=self.cfg.num_hop,
                    max_size=self.cfg.comm_max_size,
                )
                if nbrs not in pred_comms:
                    pred_comms.append(nbrs)
                    used_seeds.add(node)
                    added += 1

        lengths = [len(c) for c in pred_comms]
        print(f"  [Locator] predicted {len(pred_comms)} communities,"
              f" avg_len={np.mean(lengths):.2f}")
        return pred_comms
