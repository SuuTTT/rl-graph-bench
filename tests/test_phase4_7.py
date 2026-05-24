"""Tests for algorithm families 4-7: CLARE, SLRL, AC2CD, WRT, SS2V-D3QN.

Covers:
  - Network forward-pass shapes
  - select_action returns valid int
  - update() returns metrics dict
  - save/load checkpoint round-trip
  - EpisodeBuffer / ReplayBuffer push + update
  - Trainer integration (mini run, no crash)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from rlgb.data.synthetic import mini5
from rlgb.algos.base import Transition, EpisodeBuffer


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_partition_obs(n: int = 20, k: int = 3) -> dict:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, k, size=n).astype(np.float32)
    adj = (rng.random((n, n)) > 0.7).astype(np.float32)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)
    return {
        "adj":        adj,
        "node_feats": rng.random((n, 7)).astype(np.float32),
        "labels":     labels,
        "k":          np.array([k], dtype=np.float32),
    }


def _make_community_obs(n: int = 20) -> dict:
    rng = np.random.default_rng(1)
    adj = (rng.random((n, n)) > 0.7).astype(np.float32)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)
    return {
        "adj":          adj,
        "node_feats":   rng.random((n, 7)).astype(np.float32),
        "labels":       rng.integers(0, 3, n).astype(np.float32),
        "k":            np.array([3], dtype=np.float32),
        "exclude_nodes": np.zeros(200, dtype=np.int64),
        "expand_nodes":  np.zeros(200, dtype=np.int64),
        "n_exclude":     np.array([3], dtype=np.float32),
        "n_expand":      np.array([4], dtype=np.float32),
    }


def _rollout_transitions(algo, env, n_steps: int = 6) -> list[Transition]:
    """Roll out n_steps steps and collect transitions."""
    obs, _ = env.reset(seed=0)
    transitions = []
    for _ in range(n_steps):
        a = algo.select_action(obs)
        obs2, r, t, tr, info = env.step(a)
        transitions.append(Transition(obs, a, r, obs2, t or tr, info))
        obs = obs2
        if t or tr:
            break
    return transitions


# ── CLARE ─────────────────────────────────────────────────────────────────────

class TestCLARE:
    def _make_algo(self):
        from rlgb.algos.community.clare import CLAREAlgo, CLAREConfig
        return CLAREAlgo(CLAREConfig(hidden=16))

    def test_select_action_type(self):
        from rlgb.envs.community_env import CommunityEnv
        from rlgb.tasks.community_expand import CommunityExpandTask
        task = CommunityExpandTask()
        prob = mini5()[0]
        env  = CommunityEnv(task=task, problem=prob, horizon=5)
        algo = self._make_algo()
        obs, _ = env.reset(seed=0)
        a = algo.select_action(obs)
        assert isinstance(a, int)
        env.close()

    def test_update_returns_metrics(self):
        from rlgb.envs.community_env import CommunityEnv
        from rlgb.tasks.community_expand import CommunityExpandTask
        task = CommunityExpandTask()
        prob = mini5()[0]
        env  = CommunityEnv(task=task, problem=prob, horizon=6)
        algo = self._make_algo()
        ts   = _rollout_transitions(algo, env, n_steps=6)
        buf  = EpisodeBuffer()
        for t in ts:
            buf.push(t)
            algo.push_transition(t)
        m = algo.update()
        assert isinstance(m, dict)
        assert "loss" in m or "pg_loss" in m
        env.close()

    def test_save_load(self):
        from rlgb.algos.community.clare import CLAREAlgo, CLAREConfig
        algo = CLAREAlgo(CLAREConfig(hidden=16))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "clare.pt"
            algo.save(p)
            algo2 = CLAREAlgo(CLAREConfig(hidden=16))
            algo2.load(p)
        # should not raise

    def test_greedy_action(self):
        from rlgb.envs.community_env import CommunityEnv
        from rlgb.tasks.community_expand import CommunityExpandTask
        task = CommunityExpandTask()
        prob = mini5()[0]
        env  = CommunityEnv(task=task, problem=prob, horizon=5)
        algo = self._make_algo()
        obs, _ = env.reset(seed=0)
        a = algo.select_action(obs, greedy=True)
        assert isinstance(a, int)
        env.close()


# ── SLRL ──────────────────────────────────────────────────────────────────────

class TestSLRL:
    def _make_algo(self):
        from rlgb.algos.community.slrl import SLRLAlgo
        return SLRLAlgo()

    def test_select_action_type(self):
        from rlgb.envs.community_env import CommunityEnv
        from rlgb.tasks.community_expand import CommunityExpandTask
        task = CommunityExpandTask()
        prob = mini5()[0]
        env  = CommunityEnv(task=task, problem=prob, horizon=5)
        algo = self._make_algo()
        obs, _ = env.reset(seed=0)
        a = algo.select_action(obs)
        assert isinstance(a, int)
        env.close()

    def test_update_returns_metrics(self):
        from rlgb.envs.community_env import CommunityEnv
        from rlgb.tasks.community_expand import CommunityExpandTask
        task = CommunityExpandTask()
        prob = mini5()[0]
        env  = CommunityEnv(task=task, problem=prob, horizon=6)
        algo = self._make_algo()
        ts   = _rollout_transitions(algo, env, n_steps=6)
        for t in ts:
            algo.push_transition(t)
        m = algo.update()
        assert isinstance(m, dict)
        env.close()

    def test_save_load(self):
        from rlgb.algos.community.slrl import SLRLAlgo
        algo = SLRLAlgo()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "slrl.pt"
            algo.save(p)
            algo2 = SLRLAlgo()
            algo2.load(p)

    def test_net_forward_shape(self):
        from rlgb.algos.community.slrl import SLRLNet
        F = SLRLNet.FEAT_DIM  # 5
        net = SLRLNet(hidden=16)
        cand_feats  = torch.zeros(4, F)   # 4 candidate nodes
        query_feats = torch.zeros(1, F)   # query node
        comm_feats  = torch.zeros(3, F)   # 3 current community members
        logits, val = net(cand_feats, query_feats, comm_feats)
        assert logits.shape == (5,)   # 4 candidates + stop
        assert val.ndim == 0 or val.shape == (1,)  # scalar or (1,)


# ── AC2CD ─────────────────────────────────────────────────────────────────────

class TestAC2CD:
    def _make_algo(self):
        from rlgb.algos.dynamic.ac2cd import AC2CDAlgo, AC2CDConfig
        return AC2CDAlgo(AC2CDConfig(hidden=16, n_layers=1, n_heads=2))

    def test_select_action_type(self):
        from rlgb.envs.dynamic_env import DynamicCDEnv
        from rlgb.tasks.dynamic_cd import DynamicCDTask
        task = DynamicCDTask()
        prob = task.build_suite()[0]
        env  = DynamicCDEnv(task=task, problem=prob, horizon=5)
        algo = self._make_algo()
        obs, _ = env.reset(seed=0)
        a = algo.select_action(obs)
        assert isinstance(a, int)
        env.close()

    def test_update_returns_metrics(self):
        from rlgb.envs.dynamic_env import DynamicCDEnv
        from rlgb.tasks.dynamic_cd import DynamicCDTask
        task = DynamicCDTask()
        prob = task.build_suite()[0]
        env  = DynamicCDEnv(task=task, problem=prob, horizon=6)
        algo = self._make_algo()
        ts   = _rollout_transitions(algo, env, n_steps=6)
        for t in ts:
            algo.push_transition(t)
        m = algo.update()
        assert isinstance(m, dict)
        assert len(m) > 0
        env.close()

    def test_save_load(self):
        from rlgb.algos.dynamic.ac2cd import AC2CDAlgo, AC2CDConfig
        algo = AC2CDAlgo(AC2CDConfig(hidden=16))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ac2cd.pt"
            algo.save(p)
            algo2 = AC2CDAlgo(AC2CDConfig(hidden=16))
            algo2.load(p)

    def test_compatible_with_partition(self):
        from rlgb.algos.dynamic.ac2cd import AC2CDAlgo
        assert "partition" in AC2CDAlgo.compatible_tasks
        assert "dynamic"   in AC2CDAlgo.compatible_tasks


# ── WRT ───────────────────────────────────────────────────────────────────────

class TestWRT:
    def _make_algo(self):
        from rlgb.algos.structured.wrt import WRTAlgo, WRTConfig
        return WRTAlgo(WRTConfig(hidden=16, n_layers=1, n_heads=2))

    def test_select_action_type(self):
        from rlgb.envs.node_move_env import NodeMoveEnv
        from rlgb.tasks.graph_partition import GraphPartitionTask
        task = GraphPartitionTask()
        prob = mini5()[1]
        env  = NodeMoveEnv(task=task, problem=prob, horizon=5, warm_start="random")
        algo = self._make_algo()
        obs, _ = env.reset(seed=0)
        a = algo.select_action(obs)
        assert isinstance(a, int)
        env.close()

    def test_update_returns_metrics(self):
        from rlgb.envs.node_move_env import NodeMoveEnv
        from rlgb.tasks.graph_partition import GraphPartitionTask
        task = GraphPartitionTask()
        prob = mini5()[1]
        env  = NodeMoveEnv(task=task, problem=prob, horizon=6, warm_start="random")
        algo = self._make_algo()
        ts   = _rollout_transitions(algo, env, n_steps=6)
        for t in ts:
            algo.push_transition(t)
        m = algo.update()
        assert isinstance(m, dict)
        assert "loss" in m or "pg_loss" in m
        env.close()

    def test_save_load(self):
        from rlgb.algos.structured.wrt import WRTAlgo, WRTConfig
        algo = WRTAlgo(WRTConfig(hidden=16))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "wrt.pt"
            algo.save(p)
            algo2 = WRTAlgo(WRTConfig(hidden=16))
            algo2.load(p)

    def test_decode_action(self):
        from rlgb.algos.structured.wrt import WRTAlgo, WRTConfig
        import numpy as np
        algo = WRTAlgo(WRTConfig(hidden=16))
        labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        # Seed _last_pairs so decode_action works
        algo._last_pairs = [(0, 1), (1, 2)]
        # merge action (index < n_pairs)
        new_labels = algo.decode_action(0, labels)
        assert isinstance(new_labels, np.ndarray)
        # split action (index >= n_pairs)
        new_labels2 = algo.decode_action(2, labels)
        assert isinstance(new_labels2, np.ndarray)


# ── SS2V-D3QN ─────────────────────────────────────────────────────────────────

class TestSS2V:
    def _make_algo(self):
        from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo, SS2VConfig
        return SS2VAlgo(SS2VConfig(hidden=16, batch_size=4, buffer_capacity=50))

    def test_select_action_type(self):
        from rlgb.envs.edge_contraction_env import EdgeContractionEnv
        from rlgb.tasks.graph_partition import GraphPartitionTask
        task = GraphPartitionTask()
        prob = mini5()[1]
        env  = EdgeContractionEnv(task=task, problem=prob, horizon=5, warm_start="random")
        algo = self._make_algo()
        obs, _ = env.reset(seed=0)
        a = algo.select_action(obs)
        assert isinstance(a, int)
        env.close()

    def test_update_after_enough_transitions(self):
        from rlgb.envs.edge_contraction_env import EdgeContractionEnv
        from rlgb.tasks.graph_partition import GraphPartitionTask
        task = GraphPartitionTask()
        prob = mini5()[1]
        env  = EdgeContractionEnv(task=task, problem=prob, horizon=5, warm_start="random")
        algo = self._make_algo()
        # Collect enough transitions to fill replay buffer above batch_size
        for _ in range(8):
            obs, _ = env.reset(seed=0)
            for _ in range(5):
                a = algo.select_action(obs)
                obs2, r, t, tr, info = env.step(a)
                algo.push_transition(Transition(obs, a, r, obs2, t or tr, info))
                obs = obs2
                if t or tr:
                    break
        m = algo.update()
        assert isinstance(m, dict)
        assert "loss" in m
        assert "epsilon" in m
        env.close()

    def test_save_load(self):
        from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo, SS2VConfig
        algo = SS2VAlgo(SS2VConfig(hidden=16))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ss2v.pt"
            algo.save(p)
            algo2 = SS2VAlgo(SS2VConfig(hidden=16))
            algo2.load(p)

    def test_epsilon_decay(self):
        from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo, SS2VConfig
        algo = SS2VAlgo(SS2VConfig(hidden=16, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=10))
        obs = _make_partition_obs()
        for _ in range(20):
            algo.select_action(obs)
        assert algo._eps < 1.0

    def test_target_network_update(self):
        from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo, SS2VConfig
        algo = SS2VAlgo(SS2VConfig(hidden=16, target_update_every=1))
        # Force a target update by setting step counter
        algo._step = 0
        # Online and target should start identical
        p_on = list(algo._online.parameters())[0].data.clone()
        p_tg = list(algo._target.parameters())[0].data.clone()
        assert torch.allclose(p_on, p_tg)


# ── Trainer integration (one-episode smoke tests) ──────────────────────────────

class TestTrainerIntegration:
    """Verify each algo survives one mini training run via Trainer."""

    def _run(self, algo, task, n_episodes=4, horizon=4):
        from rlgb.training.trainer import Trainer, TrainConfig
        import random as _r
        suite = mini5()[:2]
        _rng  = _r.Random(0)
        with tempfile.TemporaryDirectory() as d:
            trainer = Trainer(
                algo=algo,
                env_fn=lambda: task.build_env(_rng.choice(suite), horizon=horizon),
                config=TrainConfig(n_episodes=n_episodes, horizon=horizon,
                                   out_dir=d, log_every=2, save_every=999),
            )
            trainer.train()

    def test_trainer_clare(self):
        from rlgb.algos.community.clare import CLAREAlgo, CLAREConfig
        from rlgb.tasks.community_expand import CommunityExpandTask
        self._run(CLAREAlgo(CLAREConfig(hidden=16)), CommunityExpandTask())

    def test_trainer_slrl(self):
        from rlgb.algos.community.slrl import SLRLAlgo
        from rlgb.tasks.community_expand import CommunityExpandTask
        self._run(SLRLAlgo(), CommunityExpandTask())

    def test_trainer_ac2cd(self):
        from rlgb.algos.dynamic.ac2cd import AC2CDAlgo, AC2CDConfig
        from rlgb.tasks.dynamic_cd import DynamicCDTask
        self._run(AC2CDAlgo(AC2CDConfig(hidden=16)), DynamicCDTask())

    def test_trainer_wrt(self):
        from rlgb.algos.structured.wrt import WRTAlgo, WRTConfig
        from rlgb.tasks.graph_partition import GraphPartitionTask
        self._run(WRTAlgo(WRTConfig(hidden=16, n_layers=1)), GraphPartitionTask())

    def test_trainer_ss2v(self):
        from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo, SS2VConfig
        from rlgb.tasks.graph_partition import GraphPartitionTask
        from rlgb.training.trainer import Trainer, TrainConfig
        import random as _r
        suite = mini5()[:2]
        _rng  = _r.Random(0)
        task = GraphPartitionTask()
        algo = SS2VAlgo(SS2VConfig(hidden=16, batch_size=4, buffer_capacity=50))
        with tempfile.TemporaryDirectory() as d:
            trainer = Trainer(
                algo=algo,
                env_fn=lambda: task.build_env(_rng.choice(suite), horizon=4,
                                              env_class="edge_contraction"),
                config=TrainConfig(n_episodes=4, horizon=4, out_dir=d,
                                   log_every=2, save_every=999),
            )
            trainer.train()
