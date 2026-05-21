"""Pytest test suite for rl-graph-bench Phase 1-3 components.

Run with: pytest tests/ -v
"""
import json
import os
import tempfile

import numpy as np
import pytest
import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def _two_cliques():
    """6-node graph with two triangles connected by one bridge edge."""
    adj = np.zeros((6, 6), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            if i != j:
                adj[i, j] = adj[3+i, 3+j] = 1.0
    adj[2, 3] = adj[3, 2] = 1.0
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    return adj, labels


# ── metrics ──────────────────────────────────────────────────────────────────

class TestMetrics:
    def setup_method(self):
        from rlgb.eval.metrics import ncut, h2, modularity, nmi, ari, compute_all
        self.ncut = ncut
        self.h2 = h2
        self.mod = modularity
        self.nmi = nmi
        self.ari = ari
        self.compute_all = compute_all
        self.adj, self.labels = _two_cliques()

    def test_ncut_gt(self):
        """NCut with ground-truth partition should be < 1."""
        val = self.ncut(self.adj, self.labels)
        assert 0 < val < 1.0, f"NCut={val}"

    def test_ncut_worse_single_cluster(self):
        """All nodes in one cluster → NCut=0 (no cut)."""
        val = self.ncut(self.adj, np.zeros(6, dtype=np.int32))
        assert val == 0.0

    def test_ncut_torch_matches_numpy(self):
        """ncut_torch must return value within 1e-3 of numpy ncut."""
        import torch
        from rlgb.eval.metrics import ncut_torch
        adj_t = torch.tensor(self.adj, dtype=torch.float32)
        lab_t = torch.tensor(self.labels, dtype=torch.long)
        torch_val = ncut_torch(adj_t, lab_t).item()
        numpy_val = self.ncut(self.adj, self.labels)
        assert abs(torch_val - numpy_val) < 1e-3, (
            f"torch={torch_val:.4f} numpy={numpy_val:.4f}"
        )

    def test_ncut_torch_single_cluster(self):
        """ncut_torch: all-one-cluster → NCut≈0."""
        import torch
        from rlgb.eval.metrics import ncut_torch
        adj_t = torch.tensor(self.adj, dtype=torch.float32)
        lab_t = torch.zeros(6, dtype=torch.long)
        assert ncut_torch(adj_t, lab_t).item() == pytest.approx(0.0, abs=1e-6)

    def test_h2_positive(self):
        val = self.h2(self.adj, self.labels)
        assert val > 0

    def test_nmi_perfect(self):
        assert abs(self.nmi(self.labels, self.labels) - 1.0) < 1e-6

    def test_ari_perfect(self):
        assert abs(self.ari(self.labels, self.labels) - 1.0) < 1e-6

    def test_compute_all_keys(self):
        out = self.compute_all(self.adj, self.labels, gt_labels=self.labels)
        for key in ("h2", "ncut", "nmi", "ari", "modularity", "conductance"):
            assert key in out, f"Missing key: {key}"


# ── synthetic data ────────────────────────────────────────────────────────────

class TestSyntheticData:
    def test_sbm_shape(self):
        from rlgb.data.synthetic import sbm
        adj, labels, k = sbm(n=60, k=3, p_in=0.7, p_out=0.05)
        assert adj.shape == (60, 60)
        assert len(np.unique(labels)) == 3
        assert k == 3

    def test_lfr_shape(self):
        from rlgb.data.synthetic import lfr
        adj, labels, k = lfr(n=100, mu=0.3, min_community=20, seed=0)
        assert adj.shape == (100, 100)
        assert k >= 2

    def test_mini5_count(self):
        from rlgb.data.synthetic import mini5
        suite = mini5()
        assert len(suite) == 5

    def test_fixed17_count(self):
        from rlgb.data.synthetic import fixed17
        suite = fixed17()
        assert len(suite) == 17

    def test_problem_fields(self):
        from rlgb.data.synthetic import mini5
        p = mini5()[0]
        assert p.adj.shape == (p.n, p.n)
        assert len(p.gt_labels) == p.n
        assert p.k_target >= 2


# ── graph partition task ──────────────────────────────────────────────────────

class TestGraphPartitionTask:
    def setup_method(self):
        from rlgb.tasks.graph_partition import GraphPartitionTask
        self.task = GraphPartitionTask(objective="h2")

    def test_build_suite_mini(self):
        suite = self.task.build_suite("mini")
        assert len(suite) == 5

    def test_reward_positive_on_improvement(self):
        from rlgb.data.synthetic import mini5
        p = mini5()[0]
        # random partition vs. gt → reward direction is unconstrained but finite
        rng = np.random.default_rng(0)
        labels_a = rng.integers(0, p.k_target, size=p.n).astype(np.int32)
        labels_b = rng.integers(0, p.k_target, size=p.n).astype(np.int32)
        r = self.task.reward(p.adj, labels_a, labels_b, p)
        assert isinstance(r, float)

    def test_evaluate_keys(self):
        from rlgb.data.synthetic import mini5
        p = mini5()[0]
        out = self.task.evaluate(p.adj, p.gt_labels, p)
        assert "h2" in out
        assert "nmi" in out
        assert "primary_value" in out

    def test_invalid_objective_raises(self):
        from rlgb.tasks.graph_partition import GraphPartitionTask
        with pytest.raises(ValueError):
            GraphPartitionTask(objective="bad_obj")


# ── NodeMove environment ──────────────────────────────────────────────────────

class TestNodeMoveEnv:
    def _make_env(self, horizon=5):
        from rlgb.envs.node_move_env import NodeMoveEnv
        from rlgb.tasks.graph_partition import GraphPartitionTask
        from rlgb.data.synthetic import mini5
        task = GraphPartitionTask(objective="h2")
        p = mini5()[1]
        return NodeMoveEnv(task=task, problem=p, horizon=horizon, warm_start="random")

    def test_reset_obs_keys(self):
        env = self._make_env()
        obs, _ = env.reset()
        for key in ("adj", "node_feats", "labels", "candidates", "n_candidates"):
            assert key in obs
        env.close()

    def test_step_returns_valid(self):
        env = self._make_env(horizon=3)
        obs, _ = env.reset(seed=42)
        legal = env.legal_action_indices()
        assert len(legal) > 0
        obs2, reward, term, trunc, info = env.step(legal[0])
        assert isinstance(reward, float)
        assert "adj" in obs2
        env.close()

    def test_episode_terminates(self):
        env = self._make_env(horizon=3)
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 10:
            legal = env.legal_action_indices()
            action = legal[0] if legal else 0
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            steps += 1
        assert done, "Episode should terminate within horizon"
        env.close()


# ── NeuroCUT model ────────────────────────────────────────────────────────────

class TestNeuroCUTPolicy:
    def setup_method(self):
        from rlgb.models.sage import NeuroCUTPolicy, SAGEConfig
        self.model = NeuroCUTPolicy(SAGEConfig(node_feat_dim=7, hidden=16, n_layers=1))

    def _inputs(self, N=8, K=3):
        adj = torch.rand(N, N)
        adj = (adj + adj.T) / 2
        adj.fill_diagonal_(0)
        feats = torch.rand(N, 7)
        labels = torch.randint(0, K, (N,))
        cands = torch.tensor(
            [[n, c] for n in range(N) for c in range(K) if labels[n] != c]
        )
        return adj, feats, labels, cands

    def test_forward_shapes(self):
        adj, feats, labels, cands = self._inputs()
        logits, value = self.model(adj, feats, labels, cands)
        assert logits.shape == (cands.shape[0],)
        assert value.shape == (1, 1)

    def test_greedy_in_range(self):
        adj, feats, labels, cands = self._inputs()
        idx = self.model.select_greedy(adj, feats, labels, cands)
        assert 0 <= idx < cands.shape[0]

    def test_empty_candidates(self):
        """With no legal moves, model should return without error."""
        adj, feats, labels, _ = self._inputs()
        cands = torch.zeros((0, 2), dtype=torch.long)
        logits, value = self.model(adj, feats, labels, cands)
        assert logits.shape[0] == 1  # dummy logit

    def test_param_count(self):
        n = sum(p.numel() for p in self.model.parameters())
        assert n > 0


# ── NeuroCUT algo ─────────────────────────────────────────────────────────────

class TestNeuroCUTAlgo:
    def _make_algo(self):
        from rlgb.algos.node_move.neurocut import NeuroCUTAlgo, NeuroCUTConfig
        return NeuroCUTAlgo(NeuroCUTConfig(hidden=16, n_layers=1))

    def _make_env(self):
        from rlgb.envs.node_move_env import NodeMoveEnv
        from rlgb.tasks.graph_partition import GraphPartitionTask
        from rlgb.data.synthetic import mini5
        task = GraphPartitionTask()
        return NodeMoveEnv(task=task, problem=mini5()[1], horizon=3, warm_start="random")

    def test_select_action_returns_int(self):
        from rlgb.algos.base import Transition
        algo = self._make_algo()
        env = self._make_env()
        obs, _ = env.reset()
        action = algo.select_action(obs)
        assert isinstance(action, int)
        env.close()

    def test_update_returns_metrics(self):
        from rlgb.algos.base import Transition
        algo = self._make_algo()
        env = self._make_env()
        obs, _ = env.reset()
        for _ in range(3):
            action = algo.select_action(obs, greedy=False)
            obs2, reward, term, trunc, info = env.step(action)
            algo.push_transition(Transition(obs, action, reward, obs2, term or trunc, info))
            obs = obs2
            if term or trunc:
                break
        metrics = algo.update()
        assert "loss" in metrics
        env.close()

    def test_save_load_roundtrip(self):
        algo = self._make_algo()
        from rlgb.algos.node_move.neurocut import NeuroCUTAlgo
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            algo.save(path)
            algo2 = NeuroCUTAlgo.from_checkpoint(path)
            assert algo2.name == "neurocut"

    def test_ppo_interface_smoke(self):
        """NeuroCUTAlgo exposes select_action_with_logprob + ppo_update for PPOTrainer."""
        import torch
        from rlgb.training.ppo import PPOTrainer, PPOConfig
        from rlgb.tasks.graph_partition import GraphPartitionTask
        from rlgb.data.synthetic import mini5
        import random

        algo = self._make_algo()
        assert hasattr(algo, "select_action_with_logprob")
        assert hasattr(algo, "ppo_update")

        task = GraphPartitionTask(objective="ncut")
        suite = mini5()[:2]
        rng = random.Random(1)
        env_fn = lambda: task.build_env(rng.choice(suite), horizon=3)

        trainer = PPOTrainer(
            algo=algo, env_fn=env_fn,
            config=PPOConfig(n_episodes=8, horizon=3, log_every=8,
                             save_every=0, out_dir="/tmp/test_ppo_ci",
                             n_episodes_per_update=4),
        )
        assert trainer._ppo_mode, "PPOTrainer should detect PPO interface on NeuroCUTAlgo"
        trainer.train()   # must not raise

    def test_ppo_cosine_lr_smoke(self):
        """PPOConfig lr_schedule='cosine' completes without error."""
        from rlgb.training.ppo import PPOTrainer, PPOConfig
        from rlgb.tasks.graph_partition import GraphPartitionTask
        from rlgb.data.synthetic import mini5
        import random

        algo = self._make_algo()
        task = GraphPartitionTask(objective="ncut")
        suite = mini5()[:2]
        rng = random.Random(3)
        env_fn = lambda: task.build_env(rng.choice(suite), horizon=3)

        PPOTrainer(
            algo=algo, env_fn=env_fn,
            config=PPOConfig(n_episodes=8, horizon=3, log_every=8,
                             save_every=0, out_dir="/tmp/test_ppo_cosine",
                             n_episodes_per_update=4,
                             lr_schedule="cosine", lr_min_ratio=0.1),
        ).train()  # must not raise


# ── Trainer end-to-end ────────────────────────────────────────────────────────

class TestTrainer:
    def test_train_runs_and_logs(self):
        import random
        from rlgb.algos.node_move.neurocut import NeuroCUTAlgo, NeuroCUTConfig
        from rlgb.envs.node_move_env import NodeMoveEnv
        from rlgb.tasks.graph_partition import GraphPartitionTask
        from rlgb.training.trainer import Trainer, TrainConfig
        from rlgb.data.synthetic import mini5

        task = GraphPartitionTask()
        problems = mini5()[1:]

        def env_fn():
            return NodeMoveEnv(task=task, problem=random.choice(problems),
                               horizon=3, warm_start="random")

        algo = NeuroCUTAlgo(NeuroCUTConfig(hidden=16, n_layers=1))
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainConfig(
                n_episodes=12, horizon=3, log_every=6,
                save_every=12, out_dir=tmp, verbose=False,
            )
            trainer = Trainer(algo, env_fn, cfg)
            trainer.train()

            assert os.path.exists(os.path.join(tmp, "last.pt"))
            lines = open(os.path.join(tmp, "train_log.jsonl")).readlines()
            assert len(lines) == 2
            row = json.loads(lines[-1])
            assert "mean_return" in row
            assert "loss" in row
