"""CLARE Community Rewriter — Phase 2 of native CLARE re-implementation.

Architecture: two MLP scorers (exclude_net, expand_net) over 64-dim GCN node
embeddings + 1-dim position flag = 65-dim.  GINConv updates the state embedding
after each step.  Separate REINFORCE with discounted returns for EXCLUDE and EXPAND
actions.  Virtual stop nodes are appended to each community so the policy can halt.

Paper: CLARE (Wu et al., KDD 2022).
Ported from FDUDSDE/KDD2022CLARE/Rewriter/{data_obj,rewriter_core,rewriting}.py.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.nn import GINConv

from rlgb.data.clare_dataset import CLAREGraphData

# ---------------------------------------------------------------------------
# Symbols (Rewriter/symbol.py)
# ---------------------------------------------------------------------------
EXPAND  = "expand"
EXCLUDE = "exclude"
VSTOP_EXCLUDE = -1   # virtual stop node for EXCLUDE phase
VSTOP_EXPAND  = -2   # virtual stop node for EXPAND phase


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, in_dim: int = 65, hidden: int = 32, out_dim: int = 1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        return self.fc3(h)


class _GIN(nn.Module):
    """Single GINConv for community state update."""
    def __init__(self, in_dim: int = 65, out_dim: int = 64) -> None:
        super().__init__()
        self.conv = GINConv(
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                          nn.Linear(out_dim, out_dim))
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index)


# ---------------------------------------------------------------------------
# Community object (Rewriter/data_obj.py)
# ---------------------------------------------------------------------------

class Community:
    """Mutable community state for the Rewriter episode."""

    def __init__(self, feat_mat: np.ndarray, pred_com: list[int],
                 true_com: list[int] | None,
                 nodes: list[int], nx_subgraph: nx.Graph,
                 mapping: dict[int, int], expand: bool = True) -> None:
        self.nodes      = nodes[:]       # local node IDs
        self.feat_mat   = feat_mat.copy()
        self.pred_com   = pred_com[:]
        self.true_com   = true_com
        self.graph      = nx_subgraph
        self.mapping    = mapping.copy()
        self.expand     = expand

        # Append virtual stop nodes
        self.nodes.append(VSTOP_EXCLUDE)
        self.pred_com.append(VSTOP_EXCLUDE)
        self.mapping[len(self.nodes) - 1] = VSTOP_EXCLUDE

        self.nodes.append(VSTOP_EXPAND)
        self.mapping[len(self.nodes) - 1] = VSTOP_EXPAND

        # Zero embeddings for virtual nodes (64-dim base)
        self.feat_mat = np.vstack([self.feat_mat, np.zeros((2, self.feat_mat.shape[1]))])

        # Prepend position flag (1-dim): 1 if node is in pred_com
        pos = np.zeros((self.feat_mat.shape[0], 1))
        for idx, node in self.mapping.items():
            if node in self.pred_com:
                pos[idx, 0] = 1.0
        self.feat_mat = np.hstack([pos, self.feat_mat])   # (N, 65)

    def _position_flag(self) -> np.ndarray:
        pos = np.zeros((self.feat_mat.shape[0], 1))
        for idx, node in self.mapping.items():
            if node in self.pred_com:
                pos[idx, 0] = 1.0
        return pos

    def compute_cost(self, choice: str = "f1") -> float:
        assert self.true_com is not None
        pred_set = set(self.pred_com) - {VSTOP_EXCLUDE, VSTOP_EXPAND}
        true_set = set(self.true_com)
        tp = len(pred_set & true_set)
        if tp == 0 or len(pred_set) == 0:
            return 0.0
        prec = tp / len(pred_set)
        rec  = tp / len(true_set)
        f    = 2 * prec * rec / (prec + rec + 1e-9)
        j    = tp / (len(pred_set) + len(true_set) - tp)
        if choice == "f1":
            return f * 10.0
        elif choice == "jaccard":
            return j * 10.0
        elif choice == "hybrid":
            return (f + j) * 10.0
        raise ValueError(choice)

    def apply_exclude(self, node: int, cost_choice: str) -> float:
        pre = self.compute_cost(cost_choice)
        if node in self.pred_com:
            self.pred_com.remove(node)
        return self.compute_cost(cost_choice) - pre

    def apply_expand(self, node: int, cost_choice: str) -> float:
        pre = self.compute_cost(cost_choice)
        if node not in self.pred_com:
            self.pred_com.append(node)
        return self.compute_cost(cost_choice) - pre

    def step(self, gin: _GIN) -> np.ndarray:
        """Run GIN on pred_com subgraph edges → update feat_mat."""
        revert = {node: idx for idx, node in self.mapping.items()}
        edges  = list(self.graph.subgraph(self.pred_com).edges())

        if edges:
            ei = torch.zeros((2, len(edges)), dtype=torch.long)
            for i, (u, v) in enumerate(edges):
                ei[0, i] = revert.get(u, 0)
                ei[1, i] = revert.get(v, 0)
        else:
            ei = torch.zeros((2, 0), dtype=torch.long)

        x_new = gin(torch.FloatTensor(self.feat_mat), ei)
        pos   = self._position_flag()
        return np.hstack([pos, x_new.detach().numpy()])


# ---------------------------------------------------------------------------
# Rewriter config
# ---------------------------------------------------------------------------

@dataclass
class RewriterConfig:
    agent_lr:        float = 1e-3
    gamma:           float = 0.99
    n_episode:       int   = 10     # episodes per epoch
    n_epoch:         int   = 1000
    max_step:        int   = 10     # max steps per training episode
    max_rewrite_step: int  = 4      # max steps at inference
    cost_choice:     str   = "f1"
    n_layers:        int   = 2      # for ego-net generation in DataProcessor
    comm_max_size:   int   = 12
    log_every:       int   = 100


# ---------------------------------------------------------------------------
# DataProcessor (Rewriter/data_obj.py) — generates training Community objects
# ---------------------------------------------------------------------------

def _outer_boundary(nx_graph: nx.Graph, com_nodes: list[int],
                    max_size: int = 20) -> list[int]:
    outer = set()
    for n in com_nodes:
        outer.update(nx_graph.neighbors(n))
    outer -= set(com_nodes)
    result = sorted(outer)
    return result[:max_size]


def _ego_net_neighbors(nx_graph: nx.Graph, start: int, k: int = 2,
                       max_size: int = 12) -> list[int]:
    q, visited = [start], [start]
    for _ in range(k):
        nxt = []
        for u in q:
            for v in nx_graph.neighbors(u):
                if v not in visited:
                    visited.append(v)
                    nxt.append(v)
                if len(visited) >= max_size:
                    return sorted(visited)
        q = nxt
    return sorted(visited)


def _make_community(nx_graph: nx.Graph, feat_mat: np.ndarray,
                    true_com: list[int], root_node: int,
                    cfg: RewriterConfig) -> Community:
    ego = _ego_net_neighbors(nx_graph, root_node, k=cfg.n_layers,
                             max_size=cfg.comm_max_size)
    outer = _outer_boundary(nx_graph, ego, max_size=10)
    expand = len(outer) > 0
    all_nodes = sorted(ego + outer)
    mapping = {idx: node for idx, node in enumerate(all_nodes)}
    return Community(
        feat_mat=feat_mat[all_nodes, :],
        pred_com=ego,
        true_com=true_com,
        nodes=all_nodes,
        nx_subgraph=nx_graph.subgraph(all_nodes).copy(),
        mapping=mapping,
        expand=expand,
    )


def _sample_root_by_degree(nx_graph: nx.Graph, true_com: list[int]) -> int:
    sub = nx_graph.subgraph(true_com)
    nodes  = sorted(sub.nodes())
    degs   = np.array([sub.degree(n) for n in nodes], dtype=np.float32)
    degs   = degs / (degs.sum() + 1e-9)
    return int(np.random.choice(nodes, p=degs))


# ---------------------------------------------------------------------------
# Agent (Rewriter/rewriter_core.py)
# ---------------------------------------------------------------------------

class _Agent:
    def __init__(self, cfg: RewriterConfig) -> None:
        self.cfg         = cfg
        self.exclude_net = _MLP(65, 32, 1)
        self.expand_net  = _MLP(65, 32, 1)
        self.gin         = _GIN(65, 64)
        # Small init so softmax is nearly uniform at start (avoids cold-start
        # where the virtual-stop node always scores highest with default init)
        for net in [self.exclude_net, self.expand_net]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -0.01, 0.01)
                    nn.init.zeros_(m.bias)
        self.excl_opt    = optim.Adam(self.exclude_net.parameters(), lr=cfg.agent_lr)
        self.expd_opt    = optim.Adam(self.expand_net.parameters(),  lr=cfg.agent_lr)

    def _score(self, feat: np.ndarray, kind: str) -> torch.Tensor:
        x = torch.FloatTensor(feat)
        if kind == EXCLUDE:
            return self.exclude_net(x).squeeze(-1)
        return self.expand_net(x).squeeze(-1)

    def choose_action(self, com: Community, kind: str) -> dict | None:
        if kind == EXCLUDE:
            idx_list = sorted(
                idx for idx, node in com.mapping.items() if node in com.pred_com
            )
            if len(idx_list) <= 1:
                return None
        else:
            expanders = set(com.nodes) - set(com.pred_com)
            if not expanders or not com.expand:
                return None
            idx_list = sorted(
                idx for idx, node in com.mapping.items() if node in expanders
            )
            if len(idx_list) <= 1:
                return None

        tmp = {i: com.mapping[v] for i, v in enumerate(idx_list)}
        feat = com.feat_mat[idx_list, :]
        scores = self._score(feat, kind)
        # Higher temperature for EXCLUDE during training prevents premature
        # convergence to "always stop" caused by mostly-negative rewards.
        temp = 3.0 if (kind == EXCLUDE and self.exclude_net.training) else 1.0
        probs  = F.softmax(scores / temp, dim=0)
        dist   = Categorical(probs)
        action = dist.sample()
        lp     = dist.log_prob(action)
        node   = tmp[action.item()]

        stop = VSTOP_EXCLUDE if kind == EXCLUDE else VSTOP_EXPAND
        if node == stop:
            return None
        return {"log_prob": lp, "node": node}

    def learn(self, log_probs: list, rewards_norm: np.ndarray, kind: str) -> None:
        loss = -(torch.stack(log_probs) * torch.from_numpy(rewards_norm)).sum()
        opt  = self.excl_opt if kind == EXCLUDE else self.expd_opt
        opt.zero_grad()
        loss.backward()
        opt.step()


# ---------------------------------------------------------------------------
# CommunityRewriter
# ---------------------------------------------------------------------------

class CommunityRewriter:
    """REINFORCE-based community rewriter (Phase 2 of CLARE).

    Usage::

        rewriter = CommunityRewriter(RewriterConfig())
        rewriter.fit(data, feat_mat, pred_comms_from_locator)
        refined   = rewriter.predict(pred_comms_from_locator, data, feat_mat)
    """

    def __init__(self, config: RewriterConfig | None = None) -> None:
        self.cfg   = config or RewriterConfig()
        self.agent = _Agent(self.cfg)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, data: CLAREGraphData, feat_mat: np.ndarray,
            pred_comms: list[list[int]]) -> None:
        """Train on training communities (randomly sampled ego-net episodes)."""
        g   = data.nx_graph
        cfg = self.cfg

        self.agent.exclude_net.train()
        self.agent.expand_net.train()
        t0  = time.time()

        for epoch in range(1, cfg.n_epoch + 1):
            excl_lps,  excl_rew  = [], []
            expd_lps,  expd_rew  = [], []

            for _ in range(cfg.n_episode):
                true_com  = random.choice(data.train_communities)
                root      = _sample_root_by_degree(g, true_com)
                obj       = _make_community(g, feat_mat, true_com, root, cfg)

                ep_excl_r, ep_expd_r = [], []
                step = 0
                do_excl, do_expd = True, True

                while True:
                    if do_excl:
                        a = self.agent.choose_action(obj, EXCLUDE)
                        if a is not None:
                            excl_lps.append(a["log_prob"])
                            r = obj.apply_exclude(a["node"], cfg.cost_choice)
                            # Clip to >= 0: only reward successful excludes.
                            # Prevents negative gradients from collapsing the
                            # policy to "always select VSTOP_EXCLUDE".
                            r = max(0.0, r)
                            ep_excl_r.append(r)
                        else:
                            do_excl = False

                    if do_expd:
                        a = self.agent.choose_action(obj, EXPAND)
                        if a is not None:
                            expd_lps.append(a["log_prob"])
                            r = obj.apply_expand(a["node"], cfg.cost_choice)
                            ep_expd_r.append(r)
                        else:
                            do_expd = False

                    obj.feat_mat = obj.step(self.agent.gin)

                    if (not do_excl and not do_expd) or step >= cfg.max_step:
                        # Discounted returns
                        if ep_excl_r:
                            n = len(ep_excl_r)
                            r = np.array([
                                np.sum(ep_excl_r[i:] * (cfg.gamma ** np.arange(n - i)))
                                for i in range(n)
                            ])
                            excl_rew.append(r)
                        if ep_expd_r:
                            n = len(ep_expd_r)
                            r = np.array([
                                np.sum(ep_expd_r[i:] * (cfg.gamma ** np.arange(n - i)))
                                for i in range(n)
                            ])
                            expd_rew.append(r)
                        break
                    step += 1

            # Normalised REINFORCE update
            if excl_lps and excl_rew:
                rr = np.concatenate(excl_rew)
                rr = (rr - rr.mean()) / (rr.std() + 1e-9)
                self.agent.learn(excl_lps, rr, EXCLUDE)
            if expd_lps and expd_rew:
                rr = np.concatenate(expd_rew)
                rr = (rr - rr.mean()) / (rr.std() + 1e-9)
                self.agent.learn(expd_lps, rr, EXPAND)

            if epoch % cfg.log_every == 0 or epoch == cfg.n_epoch:
                elapsed = time.time() - t0
                print(f"  [Rewriter] epoch {epoch:04d}/{cfg.n_epoch}"
                      f"  excl_steps={len(excl_lps):3d}"
                      f"  expd_steps={len(expd_lps):3d}"
                      f"  t={elapsed:.0f}s")

        self.agent.exclude_net.eval()
        self.agent.expand_net.eval()
        print("  [Rewriter] training done.\n")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, pred_comms: list[list[int]], data: CLAREGraphData,
                feat_mat: np.ndarray) -> list[list[int]]:
        """Refine Locator candidates using the trained policy.

        Mirrors CommRewriting.rewrite_community(valid=False).
        """
        g   = data.nx_graph
        cfg = self.cfg
        refined = []

        self.agent.exclude_net.eval()
        self.agent.expand_net.eval()

        for pred in pred_comms:
            pred = sorted(pred)
            outer = _outer_boundary(g, pred, max_size=20)
            nodes = sorted(pred + outer)
            expand = len(outer) > 0
            mapping = {idx: node for idx, node in enumerate(nodes)}

            obj = Community(
                feat_mat=feat_mat[nodes, :],
                pred_com=pred,
                true_com=None,        # unknown at inference
                nodes=nodes,
                nx_subgraph=g.subgraph(nodes).copy(),
                mapping=mapping,
                expand=expand,
            )

            step = 0
            do_excl, do_expd = True, True
            while True:
                step += 1
                if do_excl:
                    a = self.agent.choose_action(obj, EXCLUDE)
                    if a is not None:
                        node = a["node"]
                        if node in obj.pred_com:
                            obj.pred_com.remove(node)
                    else:
                        do_excl = False
                if do_expd:
                    a = self.agent.choose_action(obj, EXPAND)
                    if a is not None:
                        node = a["node"]
                        if node not in obj.pred_com:
                            obj.pred_com.append(node)
                    else:
                        do_expd = False
                if (not do_excl and not do_expd) or step >= cfg.max_rewrite_step:
                    break
                obj.feat_mat = obj.step(self.agent.gin)

            # Remove virtual stop nodes and keep non-empty communities
            pc = [n for n in obj.pred_com if n not in (VSTOP_EXCLUDE, VSTOP_EXPAND)]
            if pc:
                refined.append(pc)

        print(f"  [Rewriter] refined {len(refined)} communities,"
              f" avg_len={np.mean([len(c) for c in refined]):.2f}")
        return refined
