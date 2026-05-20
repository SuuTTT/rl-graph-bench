"""RL Graph Bench — Streamlit Dashboard.

Launch with:
    streamlit run dashboard/app.py -- --results_dir results/
or via CLI:
    rlgb serve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Parse CLI args passed after "--" ─────────────────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--results_dir", default="results", type=Path)
try:
    _cli_args, _ = _parser.parse_known_args(
        [a for a in sys.argv[1:] if not a.startswith("--server")]
    )
except SystemExit:
    class _Defaults:
        results_dir = Path("results")
    _cli_args = _Defaults()  # type: ignore[assignment]

RESULTS_DIR = Path(_cli_args.results_dir)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RL Graph Bench",
    page_icon="🕸",
    layout="wide",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _load_jsonl(path: Path) -> pd.DataFrame:
    """Load a JSONL training log into a DataFrame."""
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(records)


@st.cache_data(ttl=30)
def _load_all_csvs(results_dir: Path) -> pd.DataFrame:
    """Concatenate all eval_*.csv files in results_dir."""
    csvs = sorted(results_dir.glob("eval_*.csv"))
    if not csvs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)


def _discover_training_logs(results_dir: Path) -> dict[str, Path]:
    """Return {run_name: jsonl_path} for all training logs."""
    logs = {}
    for p in sorted(results_dir.glob("*.jsonl")):
        logs[p.stem] = p
    for p in sorted(results_dir.glob("**/*.jsonl")):
        if p.parent != results_dir:
            logs[str(p.relative_to(results_dir))] = p
    return logs


def _quick_demo_df() -> pd.DataFrame:
    """Generate synthetic demo data so the UI renders even with no saved results."""
    rng = np.random.default_rng(0)
    rows = []
    for algo in ["neurocut", "wrt", "ss2v_d3qn", "ac2cd", "clare", "slrl"]:
        for prob in ["sbm_k3_n30", "sbm_k5_n50", "lfr_mu04"]:
            for seed in range(3):
                rows.append({
                    "algo": algo, "problem": prob, "seed": seed,
                    "h2": float(rng.uniform(1.5, 5.0)),
                    "ncut": float(rng.uniform(0.3, 0.9)),
                    "nmi": float(rng.uniform(0.3, 0.95)),
                    "ari": float(rng.uniform(0.2, 0.9)),
                    "wall_sec": float(rng.uniform(0.01, 0.2)),
                    "k_target": 3, "n_nodes": 30,
                })
    return pd.DataFrame(rows)


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🕸 RL Graph Bench")
st.sidebar.markdown("---")
tab_choice = st.sidebar.radio(
    "View",
    ["Training Curves", "Comparison Table", "Graph Visualiser", "Quick Train"],
    index=0,
)

# ── Tab 1: Training Curves ────────────────────────────────────────────────────

if tab_choice == "Training Curves":
    st.header("Training Curves")
    logs = _discover_training_logs(RESULTS_DIR)
    if not logs:
        st.info(f"No training logs (*.jsonl) found in `{RESULTS_DIR}/`. Run `rlgb run …` first.")
    else:
        selected = st.multiselect("Select runs", list(logs.keys()), default=list(logs.keys())[:3])
        if selected:
            metric = st.selectbox("Metric", ["reward", "loss", "pg_loss", "v_loss", "entropy"])
            smooth = st.slider("Smoothing (rolling avg)", 1, 50, 5)
            fig = go.Figure()
            for run in selected:
                try:
                    df = _load_jsonl(logs[run])
                    if metric in df.columns and "episode" in df.columns:
                        y = df[metric].rolling(smooth, min_periods=1).mean()
                        fig.add_trace(go.Scatter(x=df["episode"], y=y, name=run, mode="lines"))
                except Exception as e:
                    st.warning(f"Could not load {run}: {e}")
            fig.update_layout(xaxis_title="Episode", yaxis_title=metric,
                              template="plotly_white", height=450)
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Comparison Table ───────────────────────────────────────────────────

elif tab_choice == "Comparison Table":
    st.header("Algo Comparison")
    eval_df = _load_all_csvs(RESULTS_DIR)
    if eval_df.empty:
        st.info("No eval CSV files found. Showing synthetic demo data.")
        eval_df = _quick_demo_df()

    all_algos = sorted(eval_df["algo"].unique()) if "algo" in eval_df.columns else []
    all_probs = sorted(eval_df["problem"].unique()) if "problem" in eval_df.columns else []
    sel_algos = st.multiselect("Algorithms", all_algos, default=all_algos)
    sel_probs = st.multiselect("Problems", all_probs, default=all_probs[:5])

    if sel_algos and sel_probs:
        mask = eval_df["algo"].isin(sel_algos) & eval_df["problem"].isin(sel_probs)
        sub = eval_df[mask]
        numeric_cols = [c for c in ["h2", "ncut", "nmi", "ari", "wall_sec"] if c in sub.columns]
        metric = st.selectbox("Primary metric", numeric_cols, index=0)

        # Mean±std table
        agg = sub.groupby(["algo", "problem"])[metric].agg(["mean", "std"]).reset_index()
        agg["mean±std"] = agg.apply(lambda r: f"{r['mean']:.4f}±{r['std']:.4f}", axis=1)
        pivot = agg.pivot(index="algo", columns="problem", values="mean±std").fillna("—")
        st.dataframe(pivot, use_container_width=True)

        # Bar chart
        bar_df = sub.groupby("algo")[metric].mean().reset_index()
        fig_bar = px.bar(bar_df, x="algo", y=metric, color="algo",
                         title=f"Mean {metric} by algorithm",
                         template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Download
        csv_bytes = sub.to_csv(index=False).encode()
        st.download_button("Download CSV", csv_bytes, "eval_results.csv", "text/csv")

# ── Tab 3: Graph Visualiser ───────────────────────────────────────────────────

elif tab_choice == "Graph Visualiser":
    st.header("Graph Visualiser")
    st.markdown("Visualise a synthetic benchmark graph with community labels.")

    col1, col2, col3 = st.columns(3)
    n = col1.slider("Nodes", 20, 200, 50)
    k = col2.slider("Clusters k", 2, 8, 3)
    mu = col3.slider("LFR μ", 0.1, 0.8, 0.3, step=0.05)

    try:
        from rlgb.data.synthetic import lfr
        prob = lfr(n=n, mu=mu, k=k)
    except Exception:
        from rlgb.data.synthetic import sbm
        prob = sbm(n=n, k=k)

    adj = prob.adj
    labels = prob.gt_labels if prob.gt_labels is not None else np.zeros(n, dtype=int)

    # Simple spring layout via networkx
    try:
        import networkx as nx
        G = nx.from_numpy_array(adj)
        pos = nx.spring_layout(G, seed=42)
        xs = [pos[i][0] for i in range(len(pos))]
        ys = [pos[i][1] for i in range(len(pos))]

        edge_x, edge_y = [], []
        for u, v in G.edges():
            edge_x += [pos[u][0], pos[v][0], None]
            edge_y += [pos[u][1], pos[v][1], None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                  line=dict(width=0.5, color="#ccc"), hoverinfo="none"))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=8, color=labels.tolist(), colorscale="Viridis",
                        showscale=True, colorbar=dict(title="cluster")),
            text=[f"node {i} | cluster {labels[i]}" for i in range(len(labels))],
            hoverinfo="text",
        ))
        fig.update_layout(showlegend=False, template="plotly_white",
                          xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
                          height=480, title=f"{prob.name} (n={n}, k={k})")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("networkx required for graph visualisation (`pip install networkx`).")

# ── Tab 4: Quick Train ────────────────────────────────────────────────────────

elif tab_choice == "Quick Train":
    st.header("Quick Train")
    st.markdown("Run a short training session and view results inline.")

    col_a, col_b, col_c = st.columns(3)
    algo_name = col_a.selectbox("Algo", ["neurocut", "wrt", "ss2v_d3qn", "ac2cd", "clare", "slrl"])
    task_name = col_b.selectbox("Task", ["partition", "community", "dynamic"])
    episodes  = col_c.slider("Episodes", 5, 200, 20)

    if st.button("Run Training"):
        from rlgb.cli import _get_algo, _get_task, _get_suite
        from rlgb.training.trainer import Trainer, TrainConfig
        import tempfile

        task_obj = _get_task(task_name)
        algo_obj = _get_algo(algo_name, hidden=32)
        suite    = _get_suite("mini5", task_obj)

        def env_fn():
            import random as _r
            return task_obj.build_env(_r.choice(suite), horizon=5)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainConfig(n_episodes=episodes, horizon=5, out_dir=tmp, log_every=1)
            trainer = Trainer(algo=algo_obj, env_fn=env_fn, config=cfg)
            with st.spinner(f"Training {algo_name} for {episodes} episodes …"):
                trainer.train()

            log_file = next(Path(tmp).glob("*.jsonl"), None)
            if log_file:
                df = _load_jsonl(log_file)
                if "reward" in df.columns:
                    fig = px.line(df, x="episode", y="reward",
                                  title=f"{algo_name} training reward", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df.tail(10), use_container_width=True)
            else:
                st.info("Training completed. No log file found.")
        st.success("Done!")
