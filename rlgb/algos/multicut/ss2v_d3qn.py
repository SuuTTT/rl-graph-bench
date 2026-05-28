"""SS2V-D3QN algorithm stub — Family E: Multicut + D3QN sequential edge contraction.

Paper: SS2V-D3QN 2025 — Subgraph-to-Vector with Dueling Double DQN.
Action: sequentially contract edges (merge two endpoints into one supernode).
The agent keeps contracting until exactly k supernodes remain.

This implementation uses a replay buffer and ε-greedy exploration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlgb.algos.base import RLAgent, ReplayBuffer, Transition


@dataclass
class SS2VConfig:
    hidden: int = 64
    n_layers: int = 2
    lr: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 2000
    buffer_capacity: int = 5000
    batch_size: int = 32
    target_update_every: int = 50
    grad_clip: float = 1.0
    device: str = "cpu"
    hybrid: bool = False
    hybrid_mode: str = "top_k"  # "top_k" or "blend"
    hybrid_top_k: int = 5
    hybrid_alpha: float = 0.5
    mpc_planning: bool = False
    mpc_horizon: int = 3
    mpc_top_k: int = 5
    actor_prior: bool = False
    prior_coef: float = 0.1
    rank_fusion: bool = False
    depth_decay: bool = False
    mcts_planning: bool = False
    mcts_simulations: int = 30
    mcts_cpuct: float = 1.5


class _DuelingHead(nn.Module):
    def __init__(self, in_dim: int, n_actions: int) -> None:
        super().__init__()
        self.value    = nn.Sequential(nn.Linear(in_dim, in_dim // 2), nn.ReLU(), nn.Linear(in_dim // 2, 1))
        self.advantage = nn.Sequential(nn.Linear(in_dim, in_dim // 2), nn.ReLU(), nn.Linear(in_dim // 2, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        V = self.value(x)
        A = self.advantage(x)
        return V + (A - A.mean(dim=-1, keepdim=True))


class _SS2VNet(nn.Module):
    """SAGE encoder + edge-level Q-values.

    Q-value for each candidate edge is computed from the embeddings of the
    two endpoint nodes, so the network can differentiate edges by their local
    structure rather than position.  This is critical for identifying which
    pairs of leiden sub-clusters belong to the same community.
    """

    GRAPH_FEAT_DIM = 3

    def __init__(self, feat_dim: int, hidden: int, n_layers: int, max_edges: int) -> None:
        super().__init__()
        self._max_edges = max_edges
        dims = [feat_dim] + [hidden] * n_layers
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(dims[i] * 2, dims[i + 1]))
        self.sage_layers = nn.ModuleList(layers)
        self.graph_proj = nn.Linear(self.GRAPH_FEAT_DIM, hidden)
        # Edge scorer: (h_u+h_v || h_u*h_v || g || w_uv || cluster_sum) → Q  (+2 for edge features)
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden * 3 + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        # Prior scorer (Policy Prior Head for distillation)
        self.prior_scorer = nn.Sequential(
            nn.Linear(hidden * 3 + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        feats: torch.Tensor,
        adj: torch.Tensor,
        graph_feat: torch.Tensor,
        edge_idx: torch.Tensor,            # (n_cands, 2) — node indices for each candidate
        edge_w: torch.Tensor | None = None,  # (n_cands,) signed edge weights
        return_prior: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = feats
        deg = adj.sum(1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / deg
        for layer in self.sage_layers:
            agg = adj_norm @ h
            h = F.relu(layer(torch.cat([h, agg], dim=1)))

        g = self.graph_proj(graph_feat)  # (hidden,)

        n_cands = edge_idx.shape[0]
        if n_cands == 0:
            if return_prior:
                return torch.zeros(self._max_edges, device=feats.device), torch.zeros(self._max_edges, device=feats.device)
            return torch.zeros(self._max_edges, device=feats.device)

        u_emb = h[edge_idx[:, 0]]                              # (n_cands, hidden)
        v_emb = h[edge_idx[:, 1]]                              # (n_cands, hidden)
        g_exp = g.unsqueeze(0).expand(n_cands, -1)             # (n_cands, hidden)
        # Signed edge weight + cluster-level sum: explicit signals about merge quality
        if edge_w is not None:
            ew = edge_w[:n_cands]  # (n_cands, 2): [w_uv, cluster_sum]
            if ew.dim() == 1:
                ew = ew.unsqueeze(1)  # backward compat: (n_cands, 1)
                ew = torch.cat([ew, torch.zeros_like(ew)], dim=1)  # pad to (n_cands, 2)
        else:
            ew = torch.zeros(n_cands, 2, device=feats.device)  # fallback zeros
        ef    = torch.cat([u_emb + v_emb, u_emb * v_emb, g_exp, ew], dim=1)  # (n_cands, hidden*3+2)
        q     = self.edge_scorer(ef).squeeze(1)                # (n_cands,)

        # Pad to fixed max_edges size (unused slots will be masked by caller)
        padded_q = torch.zeros(self._max_edges, device=feats.device)
        n = min(n_cands, self._max_edges)
        padded_q[:n] = q[:n]

        if return_prior:
            prior = self.prior_scorer(ef).squeeze(1)
            padded_prior = torch.zeros(self._max_edges, device=feats.device)
            padded_prior[:n] = prior[:n]
            return padded_q, padded_prior

        return padded_q


class _MCTSNode:
    """A node in the AlphaZero-style MCTS search tree for sequential edge contraction."""
    def __init__(self, obs: dict, parent: _MCTSNode | None = None, parent_action: int | None = None, reward: float = 0.0) -> None:
        self.obs = obs
        self.parent = parent
        self.parent_action = parent_action
        self.reward = reward

        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0

        self.children: dict[int, _MCTSNode] = {}
        self.priors: dict[int, float] = {}
        self.valid_actions = list(range(len(obs["edge_idx"]))) if "edge_idx" in obs else []

    def select_puct(self, cpuct: float) -> int:
        best_score = -float("inf")
        best_act = -1

        total_visits = sum(child.visit_count for child in self.children.values())
        sqrt_total_visits = np.sqrt(max(1, total_visits))

        for act in self.valid_actions:
            prior = self.priors.get(act, 0.0)

            if act in self.children:
                child = self.children[act]
                q_val = child.mean_value
                n_val = child.visit_count
            else:
                q_val = 0.0
                n_val = 0

            u_score = q_val + cpuct * prior * (sqrt_total_visits / (1.0 + n_val))

            if u_score > best_score:
                best_score = u_score
                best_act = act

        return best_act


class SS2VAlgo(RLAgent):
    """SS2V-D3QN: sequential edge-contraction with Dueling Double DQN.

    Note: This stub uses node-move action semantics (merge two clusters
    = contract all edges between them) to remain compatible with existing envs.
    Full subgraph-vector encoding is a TODO once the env supports it natively.
    """

    name = "ss2v_d3qn"
    compatible_tasks = ["partition"]

    MAX_EDGES = 100  # fixed Q-head output size; padded / masked

    def __init__(self, config: SS2VConfig | None = None) -> None:
        self._cfg = config or SS2VConfig()
        self._device = torch.device(self._cfg.device)
        feat_dim = 7  # node_feats dim
        self._online = _SS2VNet(feat_dim, self._cfg.hidden, self._cfg.n_layers, self.MAX_EDGES).to(self._device)
        self._target = _SS2VNet(feat_dim, self._cfg.hidden, self._cfg.n_layers, self.MAX_EDGES).to(self._device)
        self._target.load_state_dict(self._online.state_dict())
        self._optimizer = torch.optim.Adam(self._online.parameters(), lr=self._cfg.lr)
        self._replay = ReplayBuffer(capacity=self._cfg.buffer_capacity)
        self._rng = np.random.default_rng(42)
        self._step = 0
        self._epsilon = self._cfg.epsilon_start

    @property
    def _eps(self) -> float:
        cfg = self._cfg
        decay = (cfg.epsilon_start - cfg.epsilon_end) * max(0, 1 - self._step / cfg.epsilon_decay)
        return cfg.epsilon_end + decay

    def _make_edge_w(self, obs: dict, eidx_np: np.ndarray, n_cands: int) -> torch.Tensor:
        """Build (n_cands, 2) edge feature tensor [w_uv, cluster_sum]."""
        adj_s = obs.get("adj_signed")
        if adj_s is not None and len(eidx_np) > 0:
            ew = adj_s[eidx_np[:n_cands, 0], eidx_np[:n_cands, 1]].astype(np.float32)
        else:
            ew = np.zeros(n_cands, dtype=np.float32)

        cs = obs.get("cluster_sums")
        if cs is not None and len(cs) > 0:
            cs_vals = np.asarray(cs[:n_cands], dtype=np.float32)
        else:
            cs_vals = np.zeros(n_cands, dtype=np.float32)

        combined = np.stack([ew, cs_vals], axis=1)  # (n_cands, 2)
        return torch.from_numpy(combined).to(self._device)

    def _simulate_node_feats(self, adj: np.ndarray, labels: np.ndarray) -> np.ndarray:
        n = adj.shape[0]
        deg = adj.sum(axis=1)                            # (N,)
        m2 = deg.sum()
        k = max(int(labels.max()) + 1, 1)

        # Cluster size
        sizes = np.bincount(labels, minlength=k).astype(np.float32)
        cluster_size_ratio = sizes[labels] / n      # (N,)

        # Intra-cluster degree — vectorized (no Python loop over nodes)
        same_cluster = (labels[:, None] == labels[None, :])  # (N, N) bool
        intra = (adj * same_cluster).sum(axis=1).astype(np.float32)    # (N,)
        intra_ratio = intra / (deg + 1e-9)               # (N,)

        # Triangle proxy
        adj2 = adj @ adj
        tri = (adj * adj2).sum(axis=1).astype(np.float32)  # (N,)
        vol = (deg * (deg - 1))
        cc = np.where(vol > 0, tri / (vol + 1e-9), 0.0)

        # Normalise
        deg_n  = deg  / (deg.max()  + 1e-9)
        cc_n   = cc   / (cc.max()   + 1e-9)
        intra_n = intra_ratio
        csr_n  = cluster_size_ratio

        # Pad/clip to 7 dims
        feats = np.stack([
            deg_n, cc_n, intra_n, csr_n,
            deg / (m2 + 1e-9),                           # global degree ratio
            (deg_n) ** 2,                                # degree-squared proxy
            np.ones(n, dtype=np.float32) * k / n,        # k/n global signal
        ], axis=1).astype(np.float32)
        return feats

    def _simulate_contraction(self, obs: dict, action_idx: int) -> tuple[dict, float]:
        labels = obs["labels"].copy()
        adj = obs["adj"]
        cost_adj = obs.get("adj_signed", adj)
        edge_idx = obs["edge_idx"]
        
        if len(edge_idx) == 0 or action_idx >= len(edge_idx):
            return obs, 0.0
            
        u, v = edge_idx[action_idx]
        c_u, c_v = int(labels[u]), int(labels[v])
        if c_u != c_v:
            # Merge smaller cluster into larger (stable merge)
            labels[labels == c_v] = c_u
            # Canonicalize labels
            _, inv = np.unique(labels, return_inverse=True)
            next_labels = inv.astype(np.int32)
        else:
            next_labels = labels
            
        # Reward is the cluster sum for this edge
        r = 0.0
        if "cluster_sums" in obs and action_idx < len(obs["cluster_sums"]):
            r = float(obs["cluster_sums"][action_idx])
        elif c_u != c_v:
            r = float(cost_adj[np.ix_(labels == c_u, labels == c_v)].sum())
            
        # Get next edges and sums
        next_edges, next_sums = self._simulate_get_next_edges_and_sums(adj, cost_adj, next_labels)
        
        next_obs = {
            "adj": adj,
            "labels": next_labels,
            "edge_idx": next_edges,
            "cluster_sums": next_sums,
            "n_edges": np.array([len(next_edges)], dtype=np.int32),
            "k": np.array([obs["k"][0]], dtype=np.int32),
        }
        if "adj_signed" in obs:
            next_obs["adj_signed"] = cost_adj
            
        return next_obs, r

    def _simulate_get_next_edges_and_sums(self, adj: np.ndarray, cost_adj: np.ndarray, labels: np.ndarray):
        rows, cols = np.where(adj > 0)
        mask = (labels[rows] != labels[cols]) & (rows < cols)
        edge_idx = np.stack([rows[mask], cols[mask]], axis=1) # (N_edges, 2)
        
        if len(edge_idx) == 0:
            return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32)
            
        weights = cost_adj[edge_idx[:, 0], edge_idx[:, 1]]
        pos_mask = weights > 0
        filtered_edges = edge_idx[pos_mask]
        
        if len(filtered_edges) == 0:
            return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32)
            
        k = int(labels.max()) + 1
        N = len(labels)
        S = np.zeros((N, k), dtype=np.float32)
        S[np.arange(N), labels] = 1.0
        
        C_clust = S.T @ (cost_adj @ S)
        
        lu = labels[filtered_edges[:, 0]]
        lv = labels[filtered_edges[:, 1]]
        cs = C_clust[lu, lv]
        
        return filtered_edges, cs

    def _percentile_rank(self, arr: np.ndarray) -> np.ndarray:
        if len(arr) <= 1:
            return np.zeros_like(arr)
        ranks = np.argsort(np.argsort(arr))
        return ranks.astype(np.float32) / (len(arr) - 1)

    def _get_q_values(self, obs: dict, return_prior: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        adj_np    = obs["adj"]
        labels_np = obs["labels"]
        N = len(labels_np)
        k = int(labels_np.max()) + 1

        if "node_feats" in obs:
            node_feats = obs["node_feats"]
        else:
            node_feats = self._simulate_node_feats(adj_np, labels_np)

        adj_t   = torch.from_numpy(np.ascontiguousarray(adj_np, dtype=np.float32)).to(self._device)
        feats_t = torch.from_numpy(np.ascontiguousarray(node_feats, dtype=np.float32)).to(self._device)
        g_feat  = torch.tensor([N, k, float(adj_np.sum()) / max(N, 1)],
                               dtype=torch.float32, device=self._device)

        if "edge_idx" in obs:
            edge_idx = obs["edge_idx"]
        else:
            rows, cols = np.where(adj_np > 0)
            inter_mask = (labels_np[rows] != labels_np[cols]) & (rows < cols)
            edge_idx = np.stack([rows[inter_mask], cols[inter_mask]], axis=1)

        n_edges = len(edge_idx)
        n_cands = min(n_edges, self.MAX_EDGES)

        if n_cands == 0:
            zeros = torch.zeros(self.MAX_EDGES, device=self._device)
            if return_prior:
                return zeros, zeros
            return zeros

        eidx_np = edge_idx[:n_cands]
        eidx_t = torch.from_numpy(np.ascontiguousarray(eidx_np, dtype=np.int64)).to(self._device)
        edge_w_t = self._make_edge_w(obs, eidx_np, n_cands)

        with torch.no_grad():
            if return_prior:
                q, prior = self._online(feats_t, adj_t, g_feat, eidx_t, edge_w_t, return_prior=True)
                return q, prior
            else:
                q = self._online(feats_t, adj_t, g_feat, eidx_t, edge_w_t)
                return q

    def _hybrid_select_action_from_q(self, q: torch.Tensor, obs: dict, eidx_np: np.ndarray, n_cands: int) -> int:
        if len(q) == 0 or n_cands == 0:
            return 0
            
        labels_np = obs["labels"]
        if getattr(self._cfg, "hybrid", False) and "adj_signed" in obs:
            cost_adj = obs["adj_signed"]
            if "cluster_sums" in obs and len(obs["cluster_sums"]) >= n_cands:
                cluster_sums = obs["cluster_sums"][:n_cands]
            else:
                cluster_sums = np.zeros(n_cands, dtype=np.float32)
                seen = {}
                for i in range(n_cands):
                    u, v = int(eidx_np[i, 0]), int(eidx_np[i, 1])
                    cu, cv = int(labels_np[u]), int(labels_np[v])
                    key = (min(cu, cv), max(cu, cv))
                    if key not in seen:
                        seen[key] = float(cost_adj[np.ix_(labels_np == cu, labels_np == cv)].sum())
                    cluster_sums[i] = seen[key]

            mode = getattr(self._cfg, "hybrid_mode", "top_k")
            if mode == "top_k":
                q_sorted_indices = torch.argsort(q[:n_cands], descending=True).cpu().numpy()
                top_k = getattr(self._cfg, "hybrid_top_k", 5)
                top_indices = q_sorted_indices[:min(top_k, n_cands)]
                idx = int(top_indices[np.argmax(cluster_sums[top_indices])])
            elif mode == "blend":
                alpha = getattr(self._cfg, "hybrid_alpha", 0.5)
                if getattr(self._cfg, "rank_fusion", False):
                    q_rank = self._percentile_rank(q[:n_cands].cpu().numpy())
                    w_rank = self._percentile_rank(cluster_sums)
                    score = alpha * q_rank + (1.0 - alpha) * w_rank
                else:
                    q_np = q[:n_cands].cpu().numpy()
                    q_std = q_np.std()
                    q_norm = (q_np - q_np.mean()) / (q_std + 1e-8) if q_std > 1e-6 else np.zeros_like(q_np)

                    w_std = cluster_sums.std()
                    w_norm = (cluster_sums - cluster_sums.mean()) / (w_std + 1e-8) if w_std > 1e-6 else np.zeros_like(cluster_sums)
                    score = alpha * q_norm + (1.0 - alpha) * w_norm
                idx = int(score.argmax())
            else:
                idx = int(q[:n_cands].argmax().item())
        else:
            idx = int(q[:n_cands].argmax().item())
        return idx

    def _mpc_rollout_search(self, obs: dict, eidx_np: np.ndarray, n_cands: int) -> int:
        if getattr(self._cfg, "actor_prior", False):
            q, prior_logits = self._get_q_values(obs, return_prior=True)
            prior_probs = torch.softmax(prior_logits[:n_cands], dim=-1).cpu().numpy()
            q_sorted_indices = np.argsort(-prior_probs)
        else:
            q = self._get_q_values(obs)
            q_sorted_indices = torch.argsort(q[:n_cands], descending=True).cpu().numpy()
            
        top_k = min(getattr(self._cfg, "mpc_top_k", 5), n_cands)
        candidates = q_sorted_indices[:top_k]
        
        if len(candidates) == 0:
            return 0
        if len(candidates) == 1:
            return int(candidates[0])
            
        gamma = self._cfg.gamma
        horizon = getattr(self._cfg, "mpc_horizon", 3)
        
        path_rewards = []
        for c in candidates:
            next_obs, r = self._simulate_contraction(obs, c)
            path_reward = r
            
            curr_gamma = gamma
            for step in range(1, horizon):
                n_edges_next = next_obs["n_edges"][0] if "n_edges" in next_obs else 0
                if n_edges_next == 0:
                    break
                if "cluster_sums" in next_obs and len(next_obs["cluster_sums"]) > 0:
                    if next_obs["cluster_sums"].max() <= 0.0:
                        break
                    rollout_act = int(next_obs["cluster_sums"].argmax())
                else:
                    break
                    
                next_obs, r_step = self._simulate_contraction(next_obs, rollout_act)
                path_reward += curr_gamma * r_step
                curr_gamma *= gamma
                
            n_edges_terminal = next_obs["n_edges"][0] if "n_edges" in next_obs else 0
            if n_edges_terminal > 0:
                terminal_q = self._get_q_values(next_obs)
                if len(terminal_q) > 0:
                    terminal_val = float(terminal_q[:n_edges_terminal].max().item())
                    # Apply depth-dependent decay to the terminal value estimate
                    decay_factor = 0.5 if getattr(self._cfg, "depth_decay", False) else 1.0
                    path_reward += curr_gamma * decay_factor * terminal_val
                    
            path_rewards.append(path_reward)
            
        best_candidate_idx = np.argmax(path_rewards)
        return int(candidates[best_candidate_idx])

    def _mcts_search(self, obs: dict, simulations: int) -> int:
        adj_np = obs["adj"]
        labels_np = obs["labels"]
        
        # Calculate n_edges
        if "n_edges" in obs:
            n_edges = int(obs["n_edges"][0])
        else:
            rows, cols = np.where(adj_np > 0)
            inter_mask = (labels_np[rows] != labels_np[cols]) & (rows < cols)
            n_edges = int(inter_mask.sum())
            
        n_cands = min(n_edges, self.MAX_EDGES)
        
        if n_cands == 0:
            return 0
        if n_cands == 1:
            return 0
            
        root = _MCTSNode(obs)
        cpuct = getattr(self._cfg, "mcts_cpuct", 1.5)
        
        # 1. Initialize root priors from GNN Q-values
        q_gnn = self._get_q_values(obs)[:n_cands]
        priors = torch.softmax(q_gnn, dim=0).cpu().numpy()
        for i in range(n_cands):
            root.priors[i] = float(priors[i])
            
        for sim in range(simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse down tree using PUCT selection guided by GNN priors
            while len(node.valid_actions) > 0:
                act = node.select_puct(cpuct)
                if act == -1:
                    break
                if act not in node.children:
                    # Expand this action!
                    next_obs, r = self._simulate_contraction(node.obs, act)
                    child = _MCTSNode(next_obs, parent=node, parent_action=act, reward=r)
                    node.children[act] = child
                    node = child
                    search_path.append(node)
                    break
                else:
                    # Traverse to existing child
                    node = node.children[act]
                    search_path.append(node)
                    
            # Evaluation
            next_adj_np = node.obs["adj"]
            next_labels_np = node.obs["labels"]
            if "n_edges" in node.obs:
                n_edges_next = int(node.obs["n_edges"][0])
            else:
                next_rows, next_cols = np.where(next_adj_np > 0)
                next_inter_mask = (next_labels_np[next_rows] != next_labels_np[next_cols]) & (next_rows < next_cols)
                n_edges_next = int(next_inter_mask.sum())
                
            n_cands_next = min(n_edges_next, self.MAX_EDGES)
            
            if n_cands_next == 0:
                v = 0.0
            else:
                q_vals_next = self._get_q_values(node.obs)[:n_cands_next]
                priors_next = torch.softmax(q_vals_next, dim=0).cpu().numpy()
                for i in range(n_cands_next):
                    node.priors[i] = float(priors_next[i])
                v = float(q_vals_next.max().item())
                
            # Backup
            g = v
            gamma = self._cfg.gamma
            for idx in range(len(search_path) - 2, -1, -1):
                child_node = search_path[idx + 1]
                g = child_node.reward + gamma * g
                
                child_node.visit_count += 1
                child_node.total_value += g
                child_node.mean_value = child_node.total_value / child_node.visit_count
                
        # Select action based on maximum visit count at root
        best_act = -1
        best_visits = -1
        for act, child in root.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_act = act
                
        if best_act == -1:
            return 0
        return best_act

    def select_action(self, obs: dict, greedy: bool = False) -> int:
        """Select an inter-cluster edge to contract."""
        adj_np    = obs["adj"]
        labels_np = obs["labels"]

        if "n_edges" in obs:
            n_edges = int(obs["n_edges"][0])
        else:
            rows, cols = np.where(adj_np > 0)
            inter_mask = (labels_np[rows] != labels_np[cols]) & (rows < cols)
            n_edges = int(inter_mask.sum())

        if n_edges == 0:
            return 0

        n_cands = min(n_edges, self.MAX_EDGES)

        if not greedy and random.random() < self._eps:
            idx = random.randrange(n_cands)
        else:
            if getattr(self._cfg, "mcts_planning", False):
                idx = self._mcts_search(obs, getattr(self._cfg, "mcts_simulations", 30))
            elif getattr(self._cfg, "mpc_planning", False):
                eidx_np = obs["edge_idx"][:n_cands] if "edge_idx" in obs else np.zeros((0, 2), dtype=np.int32)
                idx = self._mpc_rollout_search(obs, eidx_np, n_cands)
            else:
                q = self._get_q_values(obs)
                eidx_np = obs["edge_idx"][:n_cands] if "edge_idx" in obs else np.zeros((0, 2), dtype=np.int32)
                idx = self._hybrid_select_action_from_q(q, obs, eidx_np, n_cands)

        self._step += 1
        return idx

    def push_transition(self, t: Transition) -> None:
        self._replay.push(t)

    def update(self) -> dict[str, float]:
        if len(self._replay) < self._cfg.batch_size:
            return {}
        batch = self._replay.sample(self._cfg.batch_size, self._rng)
        total_loss = 0.0
        for t in batch:
            # Use numpy for cheap CPU metadata to avoid GPU→CPU syncs
            labels_np = t.obs["labels"]
            k         = int(labels_np.max()) + 1
            N         = len(labels_np)
            adj_sum   = float(np.sum(t.obs["adj"]))

            adj_t    = torch.from_numpy(np.ascontiguousarray(t.obs["adj"], dtype=np.float32)).to(self._device)
            feats_t  = torch.from_numpy(np.ascontiguousarray(t.obs["node_feats"], dtype=np.float32)).to(self._device)
            g_feat   = torch.tensor([N, k, adj_sum / max(N, 1)],
                                    dtype=torch.float32, device=self._device)
            def _eidx(ob: dict, n_valid: int) -> torch.Tensor:
                if "edge_idx" in ob and ob["edge_idx"].shape[0] > 0:
                    return torch.from_numpy(
                        np.ascontiguousarray(ob["edge_idx"][:n_valid], dtype=np.int64)
                    ).to(self._device)
                return torch.zeros((0, 2), dtype=torch.int64, device=self._device)

            n_valid_cur = int(t.obs.get("n_edges", [self.MAX_EDGES])[0])
            n_valid_cur = max(1, min(n_valid_cur, self.MAX_EDGES))
            eidx_cur    = _eidx(t.obs, n_valid_cur)
            eidx_cur_np = t.obs["edge_idx"][:n_valid_cur] if "edge_idx" in t.obs and t.obs["edge_idx"].shape[0] > 0 else np.empty((0, 2), dtype=np.int32)
            ew_cur      = self._make_edge_w(t.obs, eidx_cur_np, n_valid_cur)
            
            if getattr(self._cfg, "actor_prior", False):
                q_vals, prior_logits = self._online(feats_t, adj_t, g_feat, eidx_cur, ew_cur, return_prior=True)
                prior_logits = prior_logits[:n_valid_cur]
            else:
                q_vals = self._online(feats_t, adj_t, g_feat, eidx_cur, ew_cur)
                prior_logits = None
                
            act_idx = min(int(t.action), n_valid_cur - 1)
            q_pred  = q_vals[act_idx]
            
            with torch.no_grad():
                lab_n_np = t.next_obs["labels"]
                k_n      = int(lab_n_np.max()) + 1
                adj_n_sum = float(np.sum(t.next_obs["adj"]))
                adj_n  = torch.from_numpy(np.ascontiguousarray(t.next_obs["adj"], dtype=np.float32)).to(self._device)
                feat_n = torch.from_numpy(np.ascontiguousarray(t.next_obs["node_feats"], dtype=np.float32)).to(self._device)
                g2     = torch.tensor([N, k_n, adj_n_sum / max(N, 1)],
                                      dtype=torch.float32, device=self._device)
                n_valid_next = int(t.next_obs.get("n_edges", [self.MAX_EDGES])[0])
                n_valid_next = max(1, min(n_valid_next, self.MAX_EDGES))
                eidx_next    = _eidx(t.next_obs, n_valid_next)
                eidx_next_np = t.next_obs["edge_idx"][:n_valid_next] if "edge_idx" in t.next_obs and t.next_obs["edge_idx"].shape[0] > 0 else np.empty((0, 2), dtype=np.int32)
                ew_next      = self._make_edge_w(t.next_obs, eidx_next_np, n_valid_next)
                
                # Co-Adapted action selection: bootstrap from online hybrid/MPC action
                q_online_next = self._online(feat_n, adj_n, g2, eidx_next, ew_next)[:n_valid_next]
                if getattr(self._cfg, "mcts_planning", False):
                    best_next_act = self._mcts_search(t.next_obs, getattr(self._cfg, "mcts_simulations", 30))
                elif getattr(self._cfg, "mpc_planning", False):
                    best_next_act = self._mpc_rollout_search(t.next_obs, eidx_next_np, n_valid_next)
                elif getattr(self._cfg, "hybrid", False):
                    best_next_act = self._hybrid_select_action_from_q(q_online_next, t.next_obs, eidx_next_np, n_valid_next)
                else:
                    best_next_act = int(q_online_next.argmax().item())
                    
                q_next = self._target(feat_n, adj_n, g2, eidx_next, ew_next)[best_next_act]
                q_tgt  = t.reward + self._cfg.gamma * q_next * (1 - float(t.done))
                
            dqn_loss = F.smooth_l1_loss(q_pred, q_tgt)
            if prior_logits is not None:
                target_act = torch.tensor([act_idx], dtype=torch.long, device=self._device)
                prior_loss = F.cross_entropy(prior_logits.unsqueeze(0), target_act)
                dqn_loss += getattr(self._cfg, "prior_coef", 0.1) * prior_loss
                
            total_loss += dqn_loss

        loss = total_loss / self._cfg.batch_size
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._online.parameters(), self._cfg.grad_clip)
        self._optimizer.step()

        if self._step % self._cfg.target_update_every == 0:
            self._target.load_state_dict(self._online.state_dict())

        return {"loss": float(loss.item()), "epsilon": self._eps}

    def bc_update(self, obs: dict, expert_action: int) -> float:
        """Behavioral cloning update: train Q to rank expert_action highest.

        Cross-entropy loss: Q[expert_action] maximised relative to all candidate
        edges.  This teaches the agent GAEC's greedy "pick max-weight positive
        edge" policy before expensive RL exploration begins.
        """
        if "edge_idx" not in obs or obs["edge_idx"].shape[0] == 0:
            return 0.0
        n_cands = int(obs.get("n_edges", [0])[0])
        if n_cands <= 1:
            return 0.0
        n_cands = min(n_cands, self.MAX_EDGES)

        labels_np = obs["labels"]
        k = int(labels_np.max()) + 1
        N = len(labels_np)
        adj_sum = float(np.sum(obs["adj"]))

        adj_t   = torch.from_numpy(np.ascontiguousarray(obs["adj"], dtype=np.float32)).to(self._device)
        feats_t = torch.from_numpy(np.ascontiguousarray(obs["node_feats"], dtype=np.float32)).to(self._device)
        g_feat  = torch.tensor([N, k, adj_sum / max(N, 1)],
                                dtype=torch.float32, device=self._device)

        eidx_np = obs["edge_idx"][:n_cands]
        eidx_t  = torch.from_numpy(np.ascontiguousarray(eidx_np, dtype=np.int64)).to(self._device)
        ew_t    = self._make_edge_w(obs, eidx_np, n_cands)

        act    = min(int(expert_action), n_cands - 1)
        target = torch.tensor([act], dtype=torch.long, device=self._device)

        if getattr(self._cfg, "actor_prior", False):
            q_vals, prior_logits = self._online(feats_t, adj_t, g_feat, eidx_t, ew_t, return_prior=True)
            q_vals = q_vals[:n_cands]
            prior_logits = prior_logits[:n_cands]
            dqn_loss = F.cross_entropy(q_vals.unsqueeze(0), target)
            prior_loss = F.cross_entropy(prior_logits.unsqueeze(0), target)
            loss = dqn_loss + getattr(self._cfg, "prior_coef", 0.1) * prior_loss
        else:
            q_vals = self._online(feats_t, adj_t, g_feat, eidx_t, ew_t)[:n_cands]  # (n_cands,)
            loss   = F.cross_entropy(q_vals.unsqueeze(0), target)

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._online.parameters(), self._cfg.grad_clip)
        self._optimizer.step()
        self._step += 1

        return float(loss.item())

    def reset_episode(self) -> None:
        pass

    def save(self, path: str | Path) -> None:
        torch.save({"model_state_dict": self._online.state_dict(),
                    "algo": "ss2v_d3qn", "version": "0.1.0"}, str(path))

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(str(path), map_location=self._device, weights_only=False)
        self._online.load_state_dict(ckpt["model_state_dict"], strict=False)
        self._target.load_state_dict(self._online.state_dict())
