import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def metric_candidates():
    return [
        ("total_cost", "Cost", "cost_vs_sigma_by_nloc.png"),
        ("rmse_ee", "RMSE", "rmse_vs_sigma_by_nloc.png"),
        ("solve_time_mean_ms", "Solve Time (ms)", "solve_time_vs_sigma_by_nloc.png"),
        ("slack_inf_max", "Slack max ||sigma||_inf", "slack_max_vs_sigma_by_nloc.png"),
        ("Kopt_mean", "K_opt mean", "kopt_vs_sigma_by_nloc.png"),
        ("sigma_min_Mk", "sigma_min(Mk)", "sigma_min_mk_vs_sigma_by_nloc.png"),
    ]


def find_sigma_col(columns):
    for c in ["sigma_min_Mk_min", "sigma_min_Mk", "sigma_min_mk", "sigma_min", "min_sigma_Mk"]:
        if c in columns:
            return c
    return None


def plot_metric(df, metric_col, ylabel, outpath):
    if metric_col not in df.columns:
        return False
    sub = df[["N_loc", "sigma_bar", metric_col]].dropna()
    if sub.empty:
        return False
    nlocs = sorted(sub["N_loc"].unique())
    plt.figure(figsize=(10, 6))
    for nloc in nlocs:
        g = sub[sub["N_loc"] == nloc].sort_values("sigma_bar")
        plt.plot(
            g["sigma_bar"].to_numpy(dtype=float),
            g[metric_col].to_numpy(dtype=float),
            marker="o",
            linewidth=1.6,
            label=f"N_loc={int(nloc)}",
        )
    plt.xlabel("sigma_bar")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs sigma_bar (by N_loc)")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Plot sigma_bar sweep curves by N_loc.")
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join("logs", "reacher", "loca-PE-gate", "greed_search_dir", "greedy_trial_results.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("logs", "reacher", "loca-PE-gate", "greed_search_dir", "sweep_by_nloc"),
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv).replace([np.inf, -np.inf], np.nan)

    # Map optional sigma_min column if present.
    sigma_col = find_sigma_col(df.columns)
    produced = []
    skipped = []
    for col, label, fname in metric_candidates():
        use_col = col
        if col == "sigma_min_Mk" and sigma_col is not None:
            use_col = sigma_col
        ok = plot_metric(df, use_col, label, os.path.join(args.outdir, fname))
        if ok:
            produced.append((label, fname, use_col))
        else:
            skipped.append(label)

    # Also emit pivot tables for quick inspection.
    for col, label, _ in metric_candidates():
        use_col = col
        if col == "sigma_min_Mk" and sigma_col is not None:
            use_col = sigma_col
        if use_col not in df.columns:
            continue
        piv = (
            df[["N_loc", "sigma_bar", use_col]]
            .groupby(["N_loc", "sigma_bar"], as_index=False)
            .mean()
            .pivot(index="N_loc", columns="sigma_bar", values=use_col)
        )
        piv.to_csv(os.path.join(args.outdir, f"pivot_{use_col}.csv"))

    print(f"Saved directory: {args.outdir}")
    print("Produced plots:")
    for label, fname, use_col in produced:
        print(f"  - {fname} ({label}, col={use_col})")
    if skipped:
        print("Skipped (missing column):")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
