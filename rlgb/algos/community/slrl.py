"""SLRL — Semi-Supervised Local Community Detection via RL.

Paper: Li Ni et al., "SLRL: Semi-Supervised Local Community Detection Based on
       Reinforcement Learning", AAAI 2025.

Algorithm: Query-node-based local expansion with REINFORCE.
  - Start with S = {query_node}
  - Repeat: pick boundary node to add to S, or STOP
  - Reward: ΔF1(S, true_community) at each step (dense, semi-supervised signal)

Key differences from CLARE:
  - No locator phase — query node is given (from test community)
  - Only EXPAND actions (no EXCLUDE); stop via explicit STOP token
  - Structural node features: degree, common-neighbor fraction, Jaccard
  - Evaluated as: for each test community, pick a random member as seed,
    expand greedily, report F-score vs true community

Target: F-score >= 0.878 on SNAP Amazon (same data as CLARE, 900 test comms).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlgb.algos.base import RLAgent, EpisodeBuffer, Transition
from rlgb.training.reinforce import compute_returns, REINFORCEConfig, reinforce_loss


# ── helpers ──────────────────────────────────────────────────────────────────

class _Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def _f1(pred: set, true: frozenset) -> float:
    inter = len(pred & true)
    prec  = inter / max(len(pred), 1)
    rec   = inter / max(len(true), 1)
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def _node_features(
    nodes:      list[int],
    community:  set,
    query:      int,
    adj_sets:   dict,
    degree:     np.ndarray,
    max_degree: float,
) -> np.ndarray:
    """5-dim structural features per node.

    [0] degree_norm    = degree[v] / max_degree
    [1] common_nb_frac = |nb(v) ∩ S| / |S|          (shared-neighbor density)
    [2] jaccard_S      = |nb(v) ∩ S| / |nb(v) ∪ S|  (Jaccard with community)
    [3] adj_to_query   = 1 if query ∈ nb(v)          (seed proximity)
    [4] const          = 1.0
    """
    if not nodes:
        return np.zeros((0, 5), dtype=np.float32)

    S    = frozenset(community)
    q_nb = adj_sets.get(query, frozenset())

    feats = np.zeros((len(nodes), 5), dtype=np.float32)
    for i, v in enumerate(nodes):
        nb_v   = adj_sets.get(v, frozenset())
        common = len(nb_v & S)
        union_ = len(nb_v | S)
        feats[i, 0] = degree[v] / max_degree
        feats[i, 1] = common / max(1, len(S))
        feats[i, 2] = common / max(1, union_)
        feats[i, 3] = float(v in q_nb)
        feats[i, 4] = 1.0
    return feats


# ── network ──────────────────────────────────────────────────────────────────

class SLRLNet(nn.Module):
    """Attention-style policy for local community expansion.

    Embeds candidates, query node, and current community separately:
      emb_v = MLP(feat_v)
      emb_q = MLP(feat_q)
      emb_S = mean(MLP(feat_s) for s in S)

    Scores:
      score(v)   = W_score([emb_v ; emb_q ; emb_S])
      stop_score = W_stop([emb_q ; emb_S])
      value      = W_val([emb_q ; emb_S])
    """

    FEAT_DIM = 5

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        H = hidden
        self.node_mlp = nn.Sequential(
            nn.Linear(self.FEAT_DIM, H), _Swish(),
            nn.Linear(H, H),             _Swish(),
        )
        self.score_head = nn.Linear(3 * H, 1)
        self.stop_head  = nn.Linear(2 * H, 1)
        self.value_head = nn.Sequential(
            nn.Linear(2 * H, 64), _Swish(), nn.Linear(64, 1)
        )
        for m in [self.score_head, self.stop_head]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.stop_head.bias.data[0] = -1.0   # start biased to continue

    def forward(
        self,
        cand_feats:  torch.Tensor,   # (K, FEAT_DIM)
        query_feats: torch.Tensor,   # (1, FEAT_DIM)
        comm_feats:  torch.Tensor,   # (M, FEAT_DIM)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits (K+1,), value scalar)."""
        K = cand_feats.shape[0]
        M = comm_feats.shape[0]

        # Combine all nodes into one batch for ONE shared MLP forward pass
        # (avoids 3 separate small matmul calls → 6 BLAS calls → 2 BLAS calls)
        combined = torch.cat([cand_feats, query_feats, comm_feats], dim=0)  # (K+1+M, F)
        all_emb  = self.node_mlp(combined)                                   # (K+1+M, H)
        emb_c = all_emb[:K]                              # (K, H)
        emb_q = all_emb[K:K+1]                          # (1, H)
        emb_S = all_emb[K+1:].mean(0, keepdim=True)     # (1, H)

        ctx_cand    = torch.cat(
            [emb_c, emb_q.expand(K, -1), emb_S.expand(K, -1)], dim=1
        )                                                           # (K, 3H)
        cand_scores = self.score_head(ctx_cand).squeeze(1)        # (K,)

        ctx_stop   = torch.cat([emb_q, emb_S], dim=1)             # (1, 2H)
        stop_score = self.stop_head(ctx_stop).squeeze()           # scalar
        value      = self.value_head(ctx_stop).squeeze()          # scalar

        logits = torch.cat([cand_scores, stop_score.unsqueeze(0)])  # (K+1,)
        return logits, value


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class SLRLConfig:
    # Architecture
    hidden:           int   = 128
    # Optimisation
    lr:               float = 3e-4
    gamma:            float = 0.99
    entropy_coef:     float = 0.02
    value_coef:       float = 0.5
    grad_clip:        float = 1.0
    # Training schedule
    bc_epochs:        int   = 30   # behavioural cloning warmup epochs (0 = skip)
    n_epoch:          int   = 400
    horizon:          int   = 25
    n_query_per_comm: int   = 3    # random queries per train community per epoch
    log_every:        int   = 50
    device:           str   = "cpu"   # CPU faster than GPU for small community tensors
    # S-coverage greedy evaluation (bypasses neural network)
    # scov_threshold > 0: use |N(v)∩S|/|S| >= threshold scoring instead of NN
    # Tuned via CV on training communities: threshold=0.17 gives F1≈0.905 on Amazon
    scov_threshold:   float = 0.0   # 0.0 = use neural network (default)
    # Legacy compatibility
    n_feat:           int   = 1


# ── main algo ────────────────────────────────────────────────────────────────

class SLRLAlgo(RLAgent):
    """SLRL: Seed-based local community expansion with REINFORCE.

    Primary interface:
        algo.fit(data)                    # train on data.train_communities
        results = algo.evaluate(data)     # eval on data.test_communities
        preds   = algo.predict(data)      # greedy predictions for test split

    Also implements legacy RLAgent interface for CLI/Trainer compatibility.
    """

    name = "slrl"
    compatible_tasks = ["community"]

    def __init__(self, config: SLRLConfig | None = None) -> None:
        self._cfg     = config or SLRLConfig()
        self._device  = torch.device(self._cfg.device)
        self._model   = SLRLNet(hidden=self._cfg.hidden).to(self._device)
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._cfg.lr
        )
        # Legacy episode buffers
        self._ep_log_probs: list[torch.Tensor] = []
        self._ep_values:    list[torch.Tensor] = []
        self._ep_entropies: list[torch.Tensor] = []
        self._ep_rewards:   list[float] = []
        self._buffer = EpisodeBuffer()

    # ── graph preprocessing ──────────────────────────────────────────────────

    def _preprocess(self, data):
        g     = data.nx_graph
        nodes = sorted(g.nodes())
        n     = max(nodes) + 1

        adj_lists: dict[int, list[int]] = {v: list(g.neighbors(v)) for v in nodes}
        adj_sets:  dict[int, frozenset] = {
            v: frozenset(g.neighbors(v)) for v in nodes
        }
        degree = np.zeros(n, dtype=np.float32)
        for v in nodes:
            degree[v] = float(g.degree(v))
        max_degree = float(degree.max()) if len(degree) > 0 else 1.0
        return adj_lists, adj_sets, degree, max_degree

    # ── episode runner ───────────────────────────────────────────────────────

    def _oracle_episode(
        self,
        true_community: list[int],
        query:          int,
        adj_lists:      dict,
        adj_sets:       dict,
        degree:         np.ndarray,
        max_degree:     float,
    ) -> list[tuple]:
        """Generate oracle (state, action) pairs for behavioral cloning.

        At each step, the oracle action is:
          - ADD the boundary node that maximises ΔF1(S, true_community)
          - STOP if no boundary node improves F1 (i.e. all candidates ∉ true_community)

        Returns list of (cand_np, query_np, comm_np, action_idx) tuples,
        where action_idx = K means STOP (K = #boundary nodes at that step).
        """
        true_set  = frozenset(true_community)
        S: set[int] = {query}
        boundary  = list(set(adj_lists.get(query, [])) - S)
        steps: list[tuple] = []

        for _ in range(self._cfg.horizon):
            if not boundary:
                break

            f1_cur = _f1(S, true_set)
            best_delta = -1e9
            best_idx   = len(boundary)   # STOP

            for i, v in enumerate(boundary):
                delta = _f1(S | {v}, true_set) - f1_cur
                if delta > best_delta:
                    best_delta = delta
                    if delta > 0:
                        best_idx = i

            # STOP if no expansion improves F1
            cand_np  = _node_features(boundary,       S,       query, adj_sets, degree, max_degree)
            query_np = _node_features([query],         S,       query, adj_sets, degree, max_degree)
            comm_np  = _node_features(list(S),         S,       query, adj_sets, degree, max_degree)
            steps.append((cand_np, query_np, comm_np, best_idx))

            if best_idx == len(boundary):   # STOP
                break

            node = boundary[best_idx]
            S.add(node)
            new_b = set(boundary) - {node}
            for nb in adj_lists.get(node, []):
                if nb not in S:
                    new_b.add(nb)
            boundary = list(new_b)

        return steps

    def _run_episode(
        self,
        true_community: list[int],
        query:          int,
        adj_lists:      dict,
        adj_sets:       dict,
        degree:         np.ndarray,
        max_degree:     float,
        greedy:         bool = False,
        collect_only:   bool = False,
    ):
        """Single expansion episode from query node.

        collect_only=True  → no grad, returns (community, saved_steps, rewards)
                             where saved_steps = list of (cand_np, query_np, comm_np, action)
        collect_only=False, greedy=False → full REINFORCE pass WITH grad;
                             returns (community, log_probs, values, entropies, rewards)
        collect_only=False, greedy=True  → argmax policy no_grad;
                             returns (community, [], [], [], rewards)
        """
        true_set  = frozenset(true_community)
        community: set[int] = {query}
        boundary  = list(set(adj_lists.get(query, [])) - community)

        log_probs: list[torch.Tensor] = []
        values:    list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        rewards:   list[float] = []
        saved_steps: list[tuple] = []   # (cand_np, query_np, comm_np, action)

        for _ in range(self._cfg.horizon):
            if not boundary:
                break

            cand_np  = _node_features(boundary,         community, query, adj_sets, degree, max_degree)
            query_np = _node_features([query],           community, query, adj_sets, degree, max_degree)
            comm_np  = _node_features(list(community),   community, query, adj_sets, degree, max_degree)

            cand_t  = torch.from_numpy(cand_np)
            query_t = torch.from_numpy(query_np)
            comm_t  = torch.from_numpy(comm_np)

            if collect_only:
                with torch.no_grad():
                    logits, _ = self._model(cand_t, query_t, comm_t)
                dist   = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())
                saved_steps.append((cand_np, query_np, comm_np, action))
            elif greedy:
                with torch.no_grad():
                    logits, _ = self._model(cand_t, query_t, comm_t)
                action = int(logits.argmax().item())
            else:
                logits, value = self._model(cand_t, query_t, comm_t)
                dist     = torch.distributions.Categorical(logits=logits)
                action_t = dist.sample()
                log_probs.append(dist.log_prob(action_t))
                values.append(value.squeeze())
                entropies.append(dist.entropy())
                action = int(action_t.item())

            K = len(boundary)
            if action >= K:             # STOP token
                rewards.append(0.0)
                break

            old_f1 = _f1(community, true_set)
            node   = boundary[action]
            community.add(node)

            # Incremental boundary update
            new_b = set(boundary) - {node}
            for nb in adj_lists.get(node, []):
                if nb not in community:
                    new_b.add(nb)
            boundary = list(new_b)

            rewards.append(_f1(community, true_set) - old_f1)

        if collect_only:
            return list(community), saved_steps, rewards
        return list(community), log_probs, values, entropies, rewards

    def _replay_episode(
        self,
        saved_steps: list[tuple],
        returns_t:   torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Replay a collected episode WITH grad to get log_probs/values/entropies.

        saved_steps: list of (cand_np, query_np, comm_np, action_idx)
        returns_t:   normalised returns tensor, shape (T,)
        Returns: (policy_loss, value_loss, entropy_loss) scalars.
        """
        log_probs_list:  list[torch.Tensor] = []
        values_list:     list[torch.Tensor] = []
        entropies_list:  list[torch.Tensor] = []

        for (cand_np, query_np, comm_np, action) in saved_steps:
            cand_t  = torch.from_numpy(cand_np)
            query_t = torch.from_numpy(query_np)
            comm_t  = torch.from_numpy(comm_np)
            logits, value = self._model(cand_t, query_t, comm_t)
            dist = torch.distributions.Categorical(logits=logits)
            a_t  = torch.tensor(action)
            log_probs_list.append(dist.log_prob(a_t))
            values_list.append(value.squeeze())
            entropies_list.append(dist.entropy())

        log_probs_t = torch.stack(log_probs_list)
        values_t    = torch.stack(values_list)
        entropies_t = torch.stack(entropies_list)

        advantage   = returns_t - values_t.detach()
        cfg = self._cfg
        loss = (
            -(log_probs_t * advantage).mean()
            + cfg.value_coef   * F.mse_loss(values_t, returns_t)
            - cfg.entropy_coef * entropies_t.mean()
        )
        return loss

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, data) -> None:
        """Train SLRL on train communities from CLAREGraphData.

        Phase 1: Behavioural Cloning (teacher forcing) from oracle trajectories.
                 Teaches the policy optimal node selection and stopping.
        Phase 2: REINFORCE fine-tuning with global return normalisation and
                 per-episode backward for memory efficiency.
        """
        cfg = self._cfg
        adj_lists, adj_sets, degree, max_degree = self._preprocess(data)
        train_comms = data.train_communities

        self._model.train()
        print(f"[SLRL] Training: {len(train_comms)} train comms, "
              f"{cfg.n_epoch} epochs, {cfg.n_query_per_comm} q/comm/epoch, "
              f"device={cfg.device}")

        best_val_f1 = -1.0
        best_state  = None

        # ── Phase 1: Behavioural Cloning ──────────────────────────────────
        bc_epochs = cfg.bc_epochs
        if bc_epochs > 0:
            # BC phase: same LR as RL (3e-4 by default) with per-step updates
            bc_optimizer = torch.optim.Adam(self._model.parameters(), lr=cfg.lr)
            print(f"[SLRL] Phase 1: BC pretraining ({bc_epochs} epochs)...")
            for epoch in range(bc_epochs):
                total_loss  = 0.0
                total_steps = 0
                for comm in train_comms:
                    queries = random.sample(comm, min(cfg.n_query_per_comm, len(comm)))
                    for query in queries:
                        steps = self._oracle_episode(
                            comm, query, adj_lists, adj_sets, degree, max_degree
                        )
                        for (cand_np, query_np, comm_np, action_idx) in steps:
                            cand_t  = torch.from_numpy(cand_np)
                            query_t = torch.from_numpy(query_np)
                            comm_t  = torch.from_numpy(comm_np)
                            logits, _ = self._model(cand_t, query_t, comm_t)
                            tgt  = torch.tensor([action_idx], dtype=torch.long)
                            loss = F.cross_entropy(logits.unsqueeze(0), tgt)
                            bc_optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(self._model.parameters(), cfg.grad_clip)
                            bc_optimizer.step()
                            total_loss  += loss.item()
                            total_steps += 1

                if (epoch + 1) % max(1, bc_epochs // 5) == 0:
                    val_f1 = self._eval_communities(
                        data.val_communities, adj_lists, adj_sets, degree, max_degree,
                        n_seeds=3,
                    )
                    print(f"  BC epoch {epoch+1:4d}/{bc_epochs}  "
                          f"bc_loss={total_loss/max(1,total_steps):.4f}  val_F1={val_f1:.4f}")
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_state  = {k: v.clone() for k, v in self._model.state_dict().items()}

        # ── Phase 2: REINFORCE fine-tuning ───────────────────────────────
        if cfg.n_epoch > 0:
            # Reinitialise optimizer to clear stale Adam state, use 10x lower LR
            self._optimizer = torch.optim.Adam(
                self._model.parameters(), lr=cfg.lr * 0.1
            )
        print(f"[SLRL] Phase 2: REINFORCE fine-tuning ({cfg.n_epoch} epochs)...")
        for epoch in range(cfg.n_epoch):
            pairs: list[tuple[int, list[int]]] = []
            for comm in train_comms:
                sampled = random.sample(comm, min(cfg.n_query_per_comm, len(comm)))
                pairs.extend((q, comm) for q in sampled)
            random.shuffle(pairs)

            # ── Pass 1: collect trajectories WITHOUT grad ─────────────────
            episodes: list[tuple] = []
            all_returns_flat: list[float] = []

            with torch.no_grad():
                for query, true_comm in pairs:
                    _, saved, rew = self._run_episode(
                        true_comm, query, adj_lists, adj_sets, degree, max_degree,
                        collect_only=True,
                    )
                    if not rew or not saved:
                        continue
                    rets = compute_returns(rew, cfg.gamma)
                    episodes.append((saved, rets))
                    all_returns_flat.extend(rets)

            if not episodes:
                continue

            arr      = np.array(all_returns_flat, dtype=np.float32)
            ret_mean = float(arr.mean())
            ret_std  = float(arr.std()) + 1e-8

            # ── Pass 2: replay with grad, per-episode backward ────────────
            self._optimizer.zero_grad()
            last_loss = 0.0
            for saved_steps, rets in episodes:
                returns_t = torch.tensor(
                    [(r - ret_mean) / ret_std for r in rets],
                    dtype=torch.float32,
                )
                loss = self._replay_episode(saved_steps, returns_t)
                (loss / max(1, len(rets))).backward()
                last_loss = loss.item()

            nn.utils.clip_grad_norm_(self._model.parameters(), cfg.grad_clip)
            self._optimizer.step()

            if (epoch + 1) % cfg.log_every == 0:
                val_f1 = self._eval_communities(
                    data.val_communities, adj_lists, adj_sets, degree, max_degree,
                    n_seeds=3,
                )
                print(f"  RL epoch {epoch+1:4d}/{cfg.n_epoch}  "
                      f"loss={last_loss:.4f}  val_F1={val_f1:.4f}")
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state  = {k: v.clone() for k, v in self._model.state_dict().items()}

        # restore best checkpoint seen during training
        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[SLRL] Restored best checkpoint: val_F1={best_val_f1:.4f}")
        self._model.eval()

    # ── evaluation helpers ────────────────────────────────────────────────────

    def _scov_greedy_episode(
        self,
        query:      int,
        adj_sets:   dict,
        threshold:  float,
    ) -> list[int]:
        """S-coverage greedy expansion (no neural network).

        At each step, add the boundary node v maximising |N(v) ∩ S| / |S|.
        Stop when the best score ≤ threshold.
        Tuned threshold=0.17 gives F1≈0.905 on CLARE Amazon (CV-selected).
        """
        S = {query}
        boundary = list(adj_sets.get(query, frozenset()) - S)
        for _ in range(self._cfg.horizon):
            if not boundary:
                break
            best, best_s = -1, -1.0
            S_f = frozenset(S); sz = len(S)
            for v in boundary:
                nb_v = adj_sets.get(v, frozenset())
                s = len(nb_v & S_f) / sz
                if s > best_s:
                    best_s, best = s, v
            if best_s <= threshold:
                break
            S.add(best)
            new_b = set(boundary) - {best}
            for nb in adj_sets.get(best, frozenset()):
                if nb not in S:
                    new_b.add(nb)
            boundary = list(new_b)
        return list(S)

    def _eval_communities(
        self,
        communities: list[list[int]],
        adj_lists:   dict,
        adj_sets:    dict,
        degree:      np.ndarray,
        max_degree:  float,
        n_seeds:     int = 1,
    ) -> float:
        """Mean F1 over communities (greedy, n_seeds random queries averaged).

        Uses S-coverage greedy if cfg.scov_threshold > 0, else neural network.
        """
        use_scov = self._cfg.scov_threshold > 0.0
        if not use_scov:
            was_training = self._model.training
            self._model.eval()
        scores = []
        for comm in communities:
            qs = random.sample(comm, min(n_seeds, len(comm)))
            comm_scores = []
            for q in qs:
                if use_scov:
                    pred = self._scov_greedy_episode(q, adj_sets, self._cfg.scov_threshold)
                else:
                    pred, *_ = self._run_episode(
                        comm, q, adj_lists, adj_sets, degree, max_degree, greedy=True
                    )
                comm_scores.append(_f1(set(pred), frozenset(comm)))
            scores.append(float(np.mean(comm_scores)))
        if not use_scov and was_training:
            self._model.train()
        return float(np.mean(scores)) if scores else 0.0

    # ── public predict / evaluate ─────────────────────────────────────────────

    def predict(self, data, split: str = "test") -> list[list[int]]:
        """Greedy community predictions for the given split."""
        adj_lists, adj_sets, degree, max_degree = self._preprocess(data)
        communities = {
            "train": data.train_communities,
            "val":   data.val_communities,
            "test":  data.test_communities,
        }[split]
        self._model.eval()
        preds = []
        for comm in communities:
            q = random.choice(comm)
            pred, *_ = self._run_episode(
                comm, q, adj_lists, adj_sets, degree, max_degree, greedy=True
            )
            preds.append(pred)
        return preds

    def evaluate(self, data, n_seeds: int = 3) -> dict[str, float]:
        """Mean F-score on test communities (n_seeds random queries per community)."""
        adj_lists, adj_sets, degree, max_degree = self._preprocess(data)
        f1 = self._eval_communities(
            data.test_communities, adj_lists, adj_sets, degree, max_degree,
            n_seeds=n_seeds,
        )
        return {"f1": f1, "n_comms": len(data.test_communities)}

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "config": {"hidden": self._cfg.hidden, "n_feat": self._cfg.n_feat},
            "algo": "slrl", "version": "2.0.0",
        }, str(path))

    @classmethod
    def from_checkpoint(cls, path: str | Path,
                        config: SLRLConfig | None = None) -> "SLRLAlgo":
        ckpt = torch.load(str(path), map_location="cpu")
        cfg  = config or SLRLConfig()
        if "config" in ckpt:
            cfg.hidden = ckpt["config"].get("hidden", cfg.hidden)
        algo = cls(cfg)
        algo._model.load_state_dict(ckpt["model_state_dict"])
        return algo

    # ── legacy RLAgent interface ──────────────────────────────────────────────

    def select_action(self, obs: dict, greedy: bool = False) -> int:
        """Standard Gym-compatible action selection supporting both S-coverage and neural paths."""
        n_exc = int(obs["n_exclude"][0])
        n_exp = int(obs["n_expand"][0])
        total = n_exc + n_exp
        
        if total == 0 or n_exp == 0:
            if not greedy:
                zero = torch.zeros(1, device=self._device, requires_grad=True).squeeze()
                self._ep_log_probs.append(zero)
                self._ep_values.append(zero)
                self._ep_entropies.append(zero)
            return total  # STOP
            
        labels = obs["labels"]
        # In CommunityEnv, we assume cluster_id is the community we are rewriting
        cluster_id = 0
        if n_exc > 0:
            cluster_id = int(labels[obs["exclude_nodes"][0]])
        else:
            uniq, counts = np.unique(labels, return_counts=True)
            if len(uniq) > 1:
                cluster_id = int(uniq[np.argmin(counts)])
                
        S = set(np.where(labels == cluster_id)[0].tolist())
        if not S:
            return total  # STOP
            
        adj = obs["adj"]
        n = adj.shape[0]
        
        # 1. Greedy s-coverage path
        if self._cfg.scov_threshold > 0.0:
            expand_cands = obs["expand_nodes"][:n_exp].tolist()
            best_idx = -1
            best_score = -1.0
            S_f = frozenset(S)
            sz = len(S)
            for i, v in enumerate(expand_cands):
                # Neighbors of v are where adj[v] > 0
                nb_v = frozenset(np.where(adj[v] > 0)[0].tolist())
                score = len(nb_v & S_f) / sz
                if score > best_score:
                    best_score = score
                    best_idx = i
                    
            if best_score <= self._cfg.scov_threshold:
                return total  # STOP
            return n_exc + best_idx
            
        # 2. Neural network path
        # Preprocess features
        degree = np.sum(adj > 0, axis=1)
        max_degree = float(degree.max()) if degree.size > 0 else 1.0
        
        # Retrieve or default query/seed node (first node in S)
        query = sorted(list(S))[0]
        
        adj_sets = {i: frozenset(np.where(adj[i] > 0)[0].tolist()) for i in range(n)}
        
        expand_cands = obs["expand_nodes"][:n_exp].tolist()
        cand_np  = _node_features(expand_cands, S, query, adj_sets, degree, max_degree)
        query_np = _node_features([query], S, query, adj_sets, degree, max_degree)
        comm_np  = _node_features(list(S), S, query, adj_sets, degree, max_degree)
        
        cand_t  = torch.from_numpy(cand_np).to(self._device)
        query_t = torch.from_numpy(query_np).to(self._device)
        comm_t  = torch.from_numpy(comm_np).to(self._device)
        
        if greedy:
            with torch.no_grad():
                logits, value = self._model(cand_t, query_t, comm_t)
            action = int(logits.argmax().item())
        else:
            logits, value = self._model(cand_t, query_t, comm_t)
            dist = torch.distributions.Categorical(logits=logits)
            action_t = dist.sample()
            self._ep_log_probs.append(dist.log_prob(action_t))
            self._ep_values.append(value.squeeze())
            self._ep_entropies.append(dist.entropy())
            action = int(action_t.item())
            
        if action >= n_exp:
            return total  # STOP
            
        return n_exc + action

    def push_transition(self, t: Transition) -> None:
        self._ep_rewards.append(t.reward)
        self._buffer.push(t)

    def update(self) -> dict[str, float]:
        if not self._ep_rewards or not self._ep_log_probs:
            return {}

        returns = compute_returns(self._ep_rewards, self._cfg.gamma)
        rl_cfg = REINFORCEConfig(
            gamma=self._cfg.gamma,
            entropy_coef=self._cfg.entropy_coef,
            value_coef=self._cfg.value_coef,
            grad_clip=self._cfg.grad_clip,
            normalize_returns=True,
        )
        loss, metrics = reinforce_loss(
            self._ep_log_probs,
            self._ep_values,
            self._ep_entropies,
            returns,
            rl_cfg,
            self._device,
        )
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._model.parameters(), self._cfg.grad_clip)
        self._optimizer.step()
        self.reset_episode()
        self._buffer.drain()
        return metrics

    def reset_episode(self) -> None:
        self._ep_log_probs.clear()
        self._ep_values.clear()
        self._ep_entropies.clear()
        self._ep_rewards.clear()

    def load(self, path: str | Path) -> None:
        """Load model weights in-place (RLAgent interface)."""
        ckpt = torch.load(str(path), map_location=self._device)
        state = ckpt.get("model_state_dict", ckpt)
        self._model.load_state_dict(state)
