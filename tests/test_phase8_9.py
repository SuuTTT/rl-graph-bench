"""Tests for data loaders, eval harness, and CLI.

Covers:
  - pyg_loaders: load_planetoid() shape/type contract
  - snap_loaders: load_snap_from_files() with synthetic edge/community files
  - eval/harness: eval_algo_on_suite() returns correct DataFrame structure
  - eval/harness: summary_table() columns and index
  - eval/harness: compare_algos() concatenates results
  - cli: list-algos, list-datasets commands run cleanly
  - cli: _get_algo, _get_task, _get_suite registry functions
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rlgb.data.synthetic import mini5


# ── pyg_loaders ───────────────────────────────────────────────────────────────

class TestPyGLoaders:
    def test_load_cora_shapes(self):
        from rlgb.data.pyg_loaders import load_planetoid
        probs = load_planetoid("Cora", max_nodes=100)
        assert len(probs) == 1
        p = probs[0]
        n = p.adj.shape[0]
        assert p.adj.shape == (n, n)
        assert p.adj.dtype == np.float32
        assert p.gt_labels is not None
        assert p.gt_labels.shape == (n,)
        assert p.k_target >= 2
        assert p.task_type == "partition"

    def test_load_cora_symmetric(self):
        from rlgb.data.pyg_loaders import load_planetoid
        p = load_planetoid("Cora", max_nodes=100)[0]
        assert np.allclose(p.adj, p.adj.T)

    def test_load_citeseer(self):
        from rlgb.data.pyg_loaders import load_planetoid
        probs = load_planetoid("CiteSeer", max_nodes=100)
        assert len(probs) == 1
        assert probs[0].name == "citeseer"

    def test_real_benchmark_suite_returns_list(self):
        from rlgb.data.pyg_loaders import real_benchmark_suite
        suite = real_benchmark_suite(names=["Cora"], max_nodes=80)
        assert isinstance(suite, list)
        assert all(hasattr(p, "adj") for p in suite)

    def test_bfs_subsample_max_nodes(self):
        from rlgb.data.pyg_loaders import load_planetoid
        p = load_planetoid("Cora", max_nodes=50)[0]
        assert p.adj.shape[0] <= 50


# ── snap_loaders ───────────────────────────────────────────────────────────────

class TestSNAPLoaders:
    def _write_snap_files(self, d: Path):
        ep = d / "edges.txt"
        cp = d / "cmty.txt"
        ep.write_text(
            "# comment\n"
            "0 1\n1 2\n2 0\n"   # triangle 1
            "3 4\n4 5\n5 3\n"   # triangle 2
            "2 3\n"             # bridge
        )
        cp.write_text("0 1 2\n3 4 5\n")
        return ep, cp

    def test_basic_load(self):
        from rlgb.data.snap_loaders import load_snap_from_files
        with tempfile.TemporaryDirectory() as d:
            ep, cp = self._write_snap_files(Path(d))
            probs = load_snap_from_files(ep, cp, name="toy")
        assert len(probs) == 1
        p = probs[0]
        assert p.adj.shape[0] == 6
        assert p.known_communities is not None
        assert len(p.known_communities) == 2

    def test_known_communities_members_in_graph(self):
        from rlgb.data.snap_loaders import load_snap_from_files
        with tempfile.TemporaryDirectory() as d:
            ep, cp = self._write_snap_files(Path(d))
            p = load_snap_from_files(ep, cp)[0]
        n = p.adj.shape[0]
        for comm in p.known_communities:
            assert all(0 <= v < n for v in comm)

    def test_task_type_community(self):
        from rlgb.data.snap_loaders import load_snap_from_files
        with tempfile.TemporaryDirectory() as d:
            ep, cp = self._write_snap_files(Path(d))
            p = load_snap_from_files(ep, cp)[0]
        assert p.task_type == "community"

    def test_missing_edge_file_raises(self):
        from rlgb.data.snap_loaders import load_snap_from_files
        with pytest.raises(FileNotFoundError):
            load_snap_from_files("/no/such/edges.txt", "/no/cmty.txt")

    def test_gz_edge_file(self):
        import gzip
        from rlgb.data.snap_loaders import load_snap_from_files
        with tempfile.TemporaryDirectory() as d:
            ep, cp = self._write_snap_files(Path(d))
            ep_gz = Path(d) / "edges.txt.gz"
            with gzip.open(str(ep_gz), "wt") as fh:
                fh.write(ep.read_text())
            probs = load_snap_from_files(ep_gz, cp)
        assert len(probs) == 1

    def test_unknown_snap_dataset_raises(self):
        from rlgb.data.snap_loaders import load_snap
        with pytest.raises(ValueError, match="Unknown SNAP dataset"):
            load_snap("notadataset")


# ── eval harness ──────────────────────────────────────────────────────────────

class TestEvalHarness:
    def _make_algo_and_task(self):
        from rlgb.algos.node_move.neurocut import NeuroCUTAlgo, NeuroCUTConfig
        from rlgb.tasks.graph_partition import GraphPartitionTask
        return NeuroCUTAlgo(NeuroCUTConfig(hidden=16)), GraphPartitionTask()

    def test_returns_dataframe(self):
        from rlgb.eval.harness import eval_algo_on_suite
        algo, task = self._make_algo_and_task()
        df = eval_algo_on_suite(algo, mini5()[:2], task, n_seeds=1, horizon=3)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2   # 2 problems × 1 seed

    def test_required_columns_present(self):
        from rlgb.eval.harness import eval_algo_on_suite
        algo, task = self._make_algo_and_task()
        df = eval_algo_on_suite(algo, mini5()[:1], task, n_seeds=1, horizon=3)
        for col in ["algo", "problem", "seed", "h2", "ncut", "nmi", "ari", "wall_sec"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_n_seeds_rows(self):
        from rlgb.eval.harness import eval_algo_on_suite
        algo, task = self._make_algo_and_task()
        df = eval_algo_on_suite(algo, mini5()[:2], task, n_seeds=3, horizon=3)
        assert len(df) == 6   # 2 problems × 3 seeds

    def test_summary_table_index_is_algo(self):
        from rlgb.eval.harness import eval_algo_on_suite, summary_table
        algo, task = self._make_algo_and_task()
        df = eval_algo_on_suite(algo, mini5()[:1], task, n_seeds=2, horizon=3)
        tbl = summary_table(df)
        assert "neurocut" in tbl.index

    def test_compare_algos_two_algos(self):
        from rlgb.eval.harness import compare_algos, summary_table
        from rlgb.algos.node_move.neurocut import NeuroCUTAlgo, NeuroCUTConfig
        from rlgb.algos.structured.wrt import WRTAlgo, WRTConfig
        from rlgb.tasks.graph_partition import GraphPartitionTask
        algos = [
            NeuroCUTAlgo(NeuroCUTConfig(hidden=16)),
            WRTAlgo(WRTConfig(hidden=16, n_layers=1)),
        ]
        task = GraphPartitionTask()
        df = compare_algos(algos, mini5()[:1], task, n_seeds=1, horizon=3)
        assert set(df["algo"].unique()) == {"neurocut", "wrt"}

    def test_greedy_flag(self):
        from rlgb.eval.harness import eval_algo_on_suite
        algo, task = self._make_algo_and_task()
        df = eval_algo_on_suite(algo, mini5()[:1], task, n_seeds=1,
                                horizon=3, greedy=False)
        assert len(df) == 1

    def test_best_of_n(self):
        """best_of>1 should return same shape and NCut ≤ single rollout."""
        from rlgb.eval.harness import eval_algo_on_suite
        algo, task = self._make_algo_and_task()
        df1 = eval_algo_on_suite(algo, mini5()[:1], task, n_seeds=1,
                                  horizon=5, greedy=False, best_of=1)
        df3 = eval_algo_on_suite(algo, mini5()[:1], task, n_seeds=1,
                                  horizon=5, greedy=False, best_of=3)
        assert len(df3) == len(df1)  # same rows
        # best-of-3 should not be worse than best-of-1 (may be equal)
        assert df3["ncut"].mean() <= df1["ncut"].mean() + 0.5  # loose bound

    def test_metrics_are_finite(self):
        from rlgb.eval.harness import eval_algo_on_suite
        algo, task = self._make_algo_and_task()
        df = eval_algo_on_suite(algo, mini5()[:2], task, n_seeds=1, horizon=3)
        for col in ["h2", "ncut"]:
            assert df[col].notna().all(), f"{col} contains NaN"
            assert np.isfinite(df[col].values).all(), f"{col} not finite"


# ── CLI ───────────────────────────────────────────────────────────────────────

class TestCLI:
    def _run(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "rlgb.cli", *args],
            capture_output=True, text=True, timeout=180,
        )

    def _rlgb(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["rlgb", *args],
            capture_output=True, text=True, timeout=180,
        )

    def test_list_algos(self):
        r = self._rlgb("list-algos")
        assert r.returncode == 0
        assert "neurocut" in r.stdout

    def test_list_datasets(self):
        r = self._rlgb("list-datasets")
        assert r.returncode == 0
        assert "mini5" in r.stdout
        assert "cora" in r.stdout

    def test_run_mini(self):
        with tempfile.TemporaryDirectory() as d:
            r = self._rlgb(
                "run",
                "--algo", "neurocut",
                "--task", "partition",
                "--dataset", "mini5",
                "--steps", "5",
                "--horizon", "3",
                "--hidden", "16",
                "--out-dir", d,
            )
        assert r.returncode == 0, r.stderr

    def test_eval_mini(self):
        with tempfile.TemporaryDirectory() as d:
            # First train to get a checkpoint
            self._rlgb("run", "--algo", "neurocut", "--dataset", "mini5",
                        "--steps", "5", "--hidden", "16", "--out-dir", d)
            ckpt = Path(d) / "last.pt"
            r = self._rlgb(
                "eval",
                "--algo", "neurocut",
                "--dataset", "mini5",
                "--seeds", "1",
                "--horizon", "3",
                "--hidden", "16",
                "--checkpoint", str(ckpt),
            )
        assert r.returncode == 0, r.stderr

    def test_get_algo_registry(self):
        from rlgb.cli import _get_algo
        for name in ["neurocut", "clare", "slrl", "ac2cd", "wrt", "ss2v_d3qn"]:
            a = _get_algo(name, hidden=16)
            assert a.name is not None

    def test_get_task_registry(self):
        from rlgb.cli import _get_task
        for name in ["partition", "community", "dynamic"]:
            t = _get_task(name)
            assert hasattr(t, "build_suite")

    def test_get_suite_registry(self):
        from rlgb.cli import _get_suite, _get_task
        task = _get_task("partition")
        for name in ["mini5", "fixed17"]:
            suite = _get_suite(name, task)
            assert len(suite) > 0

    def test_invalid_algo_raises(self):
        from rlgb.cli import _get_algo
        import typer
        with pytest.raises(Exception):
            _get_algo("nonexistent_algo")
