#!/usr/bin/env python3
"""Track E1: Comprehensive Academic Ablation and Empirical Validation Engine.

Performs three key studies:
  1. MPC Horizon (H) & Candidate Branching (K) Sensitivity analysis.
  2. Co-Adapted Target Updates vs. Standard Double DQN updates ablation.
  3. Extreme scale zero-shot generalization (N=300 & N=500).

Outputs LaTeX publication-ready tables and a unified markdown report.
"""
import sys
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rlgb.algos.multicut.ss2v_d3qn import SS2VAlgo, SS2VConfig
from rlgb.baselines.multicut import GAECBaseline
from rlgb.data.mcmp_instances import er_mcmp, ba_mcmp, mcmp_train_suite
from rlgb.tasks.multicut import MulticutTask, multicut_cost_fast
from rlgb.envs.edge_contraction_env import EdgeContractionEnv
from rlgb.training.trainer import Trainer, TrainConfig
from experiments.train_hybrid_mpc import MCMPEnvWrapper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = Path("results/checkpoints")
OUT_DIR = Path("results")
DOC_DIR = Path("docs/research")

CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOC_DIR.mkdir(parents=True, exist_ok=True)


def load_model(ckpt_path: Path, hidden: int, device: str, **kwargs) -> SS2VAlgo:
    cfg = SS2VConfig(hidden=hidden, n_layers=2, device=device, **kwargs)
    algo = SS2VAlgo(cfg)
    algo.load(ckpt_path)
    return algo


def format_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Format a pandas DataFrame as a LaTeX booktabs table."""
    try:
        latex_str = df.to_latex(index=False, column_format="l" + "c" * (len(df.columns) - 1),
                                caption=caption, label=label, position="t",
                                float_format="%.4f")
        return latex_str
    except Exception:
        # Fallback manual formatting in case older pandas is used
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("  \\centering")
        lines.append(f"  \\caption{{{caption}}}")
        lines.append(f"  \\label{{tab:{label}}}")
        cols = "l" + "c" * (len(df.columns) - 1)
        lines.append(f"  \\begin{{tabular}}{{{cols}}}")
        lines.append("    \\toprule")
        lines.append("    " + " & ".join(df.columns) + " \\\\")
        lines.append("    \\midrule")
        for _, row in df.iterrows():
            formatted_vals = []
            for val in row:
                if isinstance(val, float):
                    formatted_vals.append(f"{val:.4f}")
                else:
                    formatted_vals.append(str(val))
            lines.append("    " + " & ".join(formatted_vals) + " \\\\")
        lines.append("    \\bottomrule")
        lines.append("  \\end{tabular}")
        lines.append("\\end{table}")
        return "\n".join(lines)


def run_eval_rollouts(algo_obj, probs, ds_name, scale, algo_label) -> dict:
    costs = []
    t0 = time.perf_counter()
    for p in probs:
        cost_adj = p.meta["cost_matrix"]
        inner = EdgeContractionEnv(task=MulticutTask(), problem=p, horizon=scale * 2, warm_start="singleton")
        env = MCMPEnvWrapper(inner, cost_adj)
        obs, _ = env.reset()
        done = False
        while not done:
            a = algo_obj.select_action(obs, greedy=True)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
        cost = multicut_cost_fast(cost_adj, inner.labels)
        costs.append(cost)
        inner.close()
    elapsed = time.perf_counter() - t0
    return {
        "scale": scale,
        "algo": algo_label,
        "dataset": ds_name,
        "mean_cost": float(np.mean(costs)),
        "wall_sec": elapsed,
    }


def main():
    print("=" * 80)
    print("RL GRAPH BENCH — COMPREHENSIVE ACADEMIC ABLATION SUITE")
    print(f"Device: {DEVICE}  |  Out: {OUT_DIR}")
    print("=" * 80)

    # Load primary checkpoint paths
    ss2v_mpc_ckpt = CKPT_DIR / "ss2v_mpc_trained.pt"
    ss2v_mcmp_ckpt = CKPT_DIR / "ss2v_mcmp.pt"

    if not ss2v_mpc_ckpt.exists():
        print(f"[Error] Active MPC checkpoint {ss2v_mpc_ckpt} not found! Run training first.")
        return

    # Load active MPC model
    print("\nLoading trained Active MPC Model...")
    algo_mpc = load_model(
        ss2v_mpc_ckpt, hidden=32, device=DEVICE,
        hybrid=True, hybrid_mode="top_k", hybrid_top_k=5,
        mpc_planning=True, mpc_horizon=3, mpc_top_k=5
    )
    print("  Loaded Active MPC GNN successfully ✓")

    # ==========================================================================
    # STUDY 1: MPC Horizon (H) and Branching Factor (K) Sensitivity
    # ==========================================================================
    print("\n--- Study 1: MPC Horizon (H) & Branching Factor (K) Sensitivity Study ---")
    h_grid = [1, 2, 3]
    k_grid = [1, 3, 5]
    
    # Generate 5 ER + 5 BA N=40 problems for sensitivity study
    sens_probs_er = er_mcmp(40, n_instances=5, seed_offset=777)
    sens_probs_ba = ba_mcmp(40, n_instances=5, seed_offset=888)
    
    sens_results = []
    
    # Calculate GAEC baseline for reference
    gaec = GAECBaseline()
    er_gaec_cost = float(np.mean([multicut_cost_fast(p.meta["cost_matrix"], gaec.partition(p.meta["cost_matrix"])) for p in sens_probs_er]))
    ba_gaec_cost = float(np.mean([multicut_cost_fast(p.meta["cost_matrix"], gaec.partition(p.meta["cost_matrix"])) for p in sens_probs_ba]))
    print(f"  Reference GAEC Costs: ER={er_gaec_cost:.4f}, BA={ba_gaec_cost:.4f}")

    for h in h_grid:
        for k in k_grid:
            print(f"  Evaluating Config: Horizon H={h}, Branching K={k} ...")
            algo_mpc._cfg.mpc_horizon = h
            algo_mpc._cfg.mpc_top_k = k
            
            # Run ER
            res_er = run_eval_rollouts(algo_mpc, sens_probs_er, "er", 40, f"mpc_h{h}_k{k}")
            # Run BA
            res_ba = run_eval_rollouts(algo_mpc, sens_probs_ba, "ba", 40, f"mpc_h{h}_k{k}")
            
            sens_results.append({
                "H": h, "K": k,
                "ER Cost": res_er["mean_cost"], "ER Time (s)": res_er["wall_sec"],
                "BA Cost": res_ba["mean_cost"], "BA Time (s)": res_ba["wall_sec"]
            })
            print(f"    H={h} K={k} | ER Cost: {res_er['mean_cost']:.4f} ({res_er['wall_sec']:.1f}s) | BA Cost: {res_ba['mean_cost']:.4f} ({res_ba['wall_sec']:.1f}s)")

    df_sens = pd.DataFrame(sens_results)
    df_sens.to_csv(OUT_DIR / "academic_sensitivity_grid.csv", index=False)
    
    latex_sens = format_latex_table(
        df_sens,
        caption="MPC Look-Ahead Sensitivity Study. Evaluates structural multicut cost and wall-clock execution times across search horizons $H$ and branching factors $K$ on $N=40$ ER and BA synthetic scales.",
        label="sens_grid"
    )

    # ==========================================================================
    # STUDY 2: Co-Adapted Updates vs. Standard DQN updates Ablation
    # ==========================================================================
    print("\n--- Study 2: Co-Adapted Updates vs. Standard DQN target updates Ablation ---")
    
    # Train Standard Double DQN Model (no co-adaptation, standard DQN argmax targets)
    print("Training Standard Double DQN comparative model (100 episodes)...")
    train_suite = mcmp_train_suite()[:5]
    task = MulticutTask()
    
    cfg_std = SS2VConfig(
        hidden=32, n_layers=2, lr=1e-4, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=200,
        buffer_capacity=5000, batch_size=32, target_update_every=20,
        device=DEVICE, hybrid=False, mpc_planning=False
    )
    algo_std = SS2VAlgo(cfg_std)
    
    rng_train = random.Random(999)
    def env_fn():
        p = rng_train.choice(train_suite)
        env = EdgeContractionEnv(task=task, problem=p, horizon=40, warm_start="singleton")
        return MCMPEnvWrapper(env, p.meta["cost_matrix"])
        
    train_cfg = TrainConfig(
        n_episodes=100, horizon=40, n_episode_per_update=5,
        log_every=20, save_every=0, out_dir=str(CKPT_DIR),
        lr_schedule="none", verbose=False
    )
    
    t_train0 = time.perf_counter()
    trainer = Trainer(algo=algo_std, env_fn=env_fn, config=train_cfg)
    trainer.train()
    print(f"  Comparative standard training complete in {time.perf_counter()-t_train0:.1f}s.")
    
    # Save standard comparative checkpoint
    std_ckpt_path = CKPT_DIR / "ss2v_standard_trained.pt"
    algo_std.save(std_ckpt_path)
    print(f"  Saved comparative model checkpoint: {std_ckpt_path}")
    
    # Now evaluate both models under look-ahead (H=3, K=5)
    # This evaluates whether representation learning during training matches look-ahead search
    print("\nEvaluating Standard trained GNN under look-ahead search...")
    algo_std_eval = load_model(
        std_ckpt_path, hidden=32, device=DEVICE,
        hybrid=True, hybrid_mode="top_k", hybrid_top_k=5,
        mpc_planning=True, mpc_horizon=3, mpc_top_k=5
    )
    
    # Evaluate on our sensitivity test problems
    res_std_er = run_eval_rollouts(algo_std_eval, sens_probs_er, "er", 40, "standard_double_dqn")
    res_std_ba = run_eval_rollouts(algo_std_eval, sens_probs_ba, "ba", 40, "standard_double_dqn")
    
    # Evaluate Co-Adapted MPC model under identical parameters (H=3, K=5)
    algo_mpc._cfg.mpc_planning = True
    algo_mpc._cfg.mpc_horizon = 3
    algo_mpc._cfg.mpc_top_k = 5
    res_co_er = run_eval_rollouts(algo_mpc, sens_probs_er, "er", 40, "co_adapted_mpc")
    res_co_ba = run_eval_rollouts(algo_mpc, sens_probs_ba, "ba", 40, "co_adapted_mpc")
    
    ablation_rows = [
        {"Model": "GAEC Baseline", "ER Cost": er_gaec_cost, "BA Cost": ba_gaec_cost},
        {"Model": "Standard Double DQN GNN", "ER Cost": res_std_er["mean_cost"], "BA Cost": res_std_ba["mean_cost"]},
        {"Model": "Co-Adapted Active MPC GNN", "ER Cost": res_co_er["mean_cost"], "BA Cost": res_co_ba["mean_cost"]}
    ]
    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(OUT_DIR / "academic_coadaptation_ablation.csv", index=False)
    
    latex_ablation = format_latex_table(
        df_ablation,
        caption="Co-Adaptation Ablation Analysis. Demonstrates the performance impact of training GNN representations under co-adapted target updates versus standard double Q-learning target updates, evaluated under identical $H=3, K=5$ look-ahead search.",
        label="coadaptation_ablation"
    )
    print("\nAblation Results Summary:")
    for _, r in df_ablation.iterrows():
        print(f"  {r['Model']:<28} | ER Cost: {r['ER Cost']:.4f} | BA Cost: {r['BA Cost']:.4f}")

    # ==========================================================================
    # STUDY 3: Extreme Out-of-Distribution scale Generalization (N=300 & N=500)
    # ==========================================================================
    print("\n--- Study 3: Extreme scale Zero-Shot Generalization ---")
    extreme_scales = [300, 500]
    extreme_rows = []
    
    # Load 64 hidden baselines if available
    ss2v_pure = None
    ss2v_hybrid = None
    if ss2v_mcmp_ckpt.exists():
        ss2v_pure = load_model(ss2v_mcmp_ckpt, hidden=64, device=DEVICE, hybrid=False)
        ss2v_hybrid = load_model(ss2v_mcmp_ckpt, hidden=64, device=DEVICE, hybrid=True, hybrid_mode="top_k", hybrid_top_k=10)
        
    for n in extreme_scales:
        print(f"  Evaluating MCMP scale N={n} (3 instances each) ...")
        probs_er = er_mcmp(n, n_instances=3, seed_offset=5000 + n)
        probs_ba = ba_mcmp(n, n_instances=3, seed_offset=6000 + n)
        
        for ds_name, probs in [("er", probs_er), ("ba", probs_ba)]:
            # 1. GAEC
            costs_gaec = [multicut_cost_fast(p.meta["cost_matrix"], gaec.partition(p.meta["cost_matrix"])) for p in probs]
            extreme_rows.append({
                "Scale": n, "Dataset": ds_name, "Algorithm": "GAEC",
                "Cost": float(np.mean(costs_gaec)), "Time (s)": 0.0
            })
            print(f"    GAEC         | scale={n:<4} | ds={ds_name} | cost={extreme_rows[-1]['Cost']:.4f}")
            
            # 2. Pure (if available)
            if ss2v_pure is not None:
                res = run_eval_rollouts(ss2v_pure, probs, ds_name, n, "ss2v_d3qn")
                extreme_rows.append({
                    "Scale": n, "Dataset": ds_name, "Algorithm": "SS2V (Pure Paper)",
                    "Cost": res["mean_cost"], "Time (s)": res["wall_sec"]
                })
                print(f"    Pure GNN     | scale={n:<4} | ds={ds_name} | cost={extreme_rows[-1]['Cost']:.4f} ({res['wall_sec']:.1f}s)")
                
            # 3. Hybrid (if available)
            if ss2v_hybrid is not None:
                res = run_eval_rollouts(ss2v_hybrid, probs, ds_name, n, "ss2v_d3qn_hybrid")
                extreme_rows.append({
                    "Scale": n, "Dataset": ds_name, "Algorithm": "SS2V (One-Step Hybrid)",
                    "Cost": res["mean_cost"], "Time (s)": res["wall_sec"]
                })
                print(f"    One-step Hyb | scale={n:<4} | ds={ds_name} | cost={extreme_rows[-1]['Cost']:.4f} ({res['wall_sec']:.1f}s)")
                
            # 4. MPC (H=3, K=5)
            algo_mpc._cfg.mpc_planning = True
            algo_mpc._cfg.mpc_horizon = 3
            algo_mpc._cfg.mpc_top_k = 5
            res = run_eval_rollouts(algo_mpc, probs, ds_name, n, "ss2v_d3qn_mpc")
            extreme_rows.append({
                "Scale": n, "Dataset": ds_name, "Algorithm": "SS2V (Active MPC)",
                "Cost": res["mean_cost"], "Time (s)": res["wall_sec"]
            })
            print(f"    Active MPC   | scale={n:<4} | ds={ds_name} | cost={extreme_rows[-1]['Cost']:.4f} ({res['wall_sec']:.1f}s)")

    df_extreme = pd.DataFrame(extreme_rows)
    df_extreme.to_csv(OUT_DIR / "academic_extreme_generalization.csv", index=False)
    
    latex_extreme = format_latex_table(
        df_extreme,
        caption="Extreme Out-of-Distribution Scale Generalization. zero-shot evaluation performance and search-computation time scaling on massive synthetic scales $N=300$ and $N=500$.",
        label="extreme_scale"
    )

    # ==========================================================================
    # WRITE ACADEMIC REPORT
    # ==========================================================================
    print("\n--- Compiling Final Academic Reports ---")
    report = []
    report.append("# Academic Empirical Validation & Ablation Report — rl-graph-bench\n")
    report.append(f"_Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')} | Device: {DEVICE}_\n")
    report.append("This report documents the rigorous publication-ready academic ablation studies for the hybrid spatial-relaxation Graph Reinforcement Learning paradigm paired with Model Predictive Control (MPC) and a deterministic analytical Graph World Model.\n")
    
    report.append("## 1. Study 1: MPC Horizon ($H$) & Candidate Branching ($K$) Sensitivity Study")
    report.append("Evaluates the optimization-computation scaling trade-offs of the GNN-guided look-ahead planner across different parameters.\n")
    
    report.append("### LaTeX booktabs Code:")
    report.append("```latex")
    report.append(latex_sens)
    report.append("```\n")
    
    report.append("### Markdown Table:")
    report.append("| Horizon (H) | Branching (K) | ER Cost ↓ | ER Time (s) | BA Cost ↓ | BA Time (s) |")
    report.append("|-------------|---------------|-----------|-------------|-----------|-------------|")
    for _, r in df_sens.iterrows():
        report.append(f"| {int(r['H'])} | {int(r['K'])} | {r['ER Cost']:.4f} | {r['ER Time (s)']:.1f} | {r['BA Cost']:.4f} | {r['BA Time (s)']:.1f} |")
    report.append("\n")
    
    report.append("## 2. Study 2: Co-Adaptation DQN Target Update Ablation Study")
    report.append("Validates whether training representations under co-adapted updates aligns network Q-values with planned MPC look-ahead search, compared to standard Double DQN bootstrapping targets.\n")
    
    report.append("### LaTeX booktabs Code:")
    report.append("```latex")
    report.append(latex_ablation)
    report.append("```\n")
    
    report.append("### Markdown Table:")
    report.append("| Model | ER Cost ↓ | BA Cost ↓ |")
    report.append("|-------|-----------|-----------|")
    for _, r in df_ablation.iterrows():
        report.append(f"| {r['Model']} | {r['ER Cost']:.4f} | {r['BA Cost']:.4f} |")
    report.append("\n")
    
    report.append("## 3. Study 3: Extreme Out-of-Distribution Scale Generalization ($N=300, 500$)")
    report.append("Tests the limits of GNN generalization. Our look-ahead active GNN is evaluated zero-shot at scales up to $N=500$ (over 12x the training scale $N=40$).\n")
    
    report.append("### LaTeX booktabs Code:")
    report.append("```latex")
    report.append(latex_extreme)
    report.append("```\n")
    
    report.append("### Markdown Table:")
    report.append("| Scale (N) | Dataset | Algorithm | Mean Cost ↓ | Time (s) |")
    report.append("|-----------|---------|-----------|-------------|----------|")
    for _, r in df_extreme.iterrows():
        report.append(f"| {int(r['Scale'])} | {r['Dataset']} | **{r['Algorithm']}** | {r['Cost']:.4f} | {r['Time (s)']:.1f} |")
    report.append("\n")
    
    # Save the reports
    summary_md = OUT_DIR / "academic_ablation_summary.md"
    summary_md.write_text("\n".join(report))
    print(f"  Summary Markdown saved → {summary_md}")
    
    doc_md = DOC_DIR / "academic_ablation_report.md"
    doc_md.write_text("\n".join(report))
    print(f"  Research Report saved → {doc_md}")
    
    print("\n" + "=" * 80)
    print("ACADEMIC ABLATION SUITE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
