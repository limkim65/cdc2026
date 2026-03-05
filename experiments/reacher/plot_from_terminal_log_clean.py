import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Clean trend plots from parsed terminal CSV.")
    parser.add_argument(
        "--greedy_csv",
        type=str,
        default=os.path.join("logs", "reacher", "parsed_from_terminal", "parsed_greedy_rows.csv"),
    )
    parser.add_argument(
        "--rho_csv",
        type=str,
        default=os.path.join("logs", "reacher", "parsed_from_terminal", "parsed_adaptive_rho_rows.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("logs", "reacher", "parsed_from_terminal"),
    )
    parser.add_argument("--nloc_max", type=int, default=2000)
    parser.add_argument("--sigma_max", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    g = pd.read_csv(args.greedy_csv)
    # Remove heavy/pathological region that dominates scale and hides trends.
    g_clean = g[(g["N_loc"] <= args.nloc_max) & (g["sigma_bar"] <= args.sigma_max)].copy()
    g_clean.to_csv(os.path.join(args.outdir, "parsed_greedy_rows_clean.csv"), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    metrics = [("rmse", "RMSE"), ("t_mean_ms", "t_mean (ms)"), ("slack_max", "slack max"), ("Kopt_mean", "Kopt mean")]
    for ax, (col, ylab) in zip(axes, metrics):
        for nloc, df in g_clean.groupby("N_loc"):
            df = df.sort_values("sigma_bar")
            ax.plot(df["sigma_bar"], df[col], marker="o", linewidth=1.7, label=f"N={int(nloc)}")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("sigma_bar")
    axes[-2].set_xlabel("sigma_bar")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle(f"Clean Greedy Trends (N_loc <= {args.nloc_max}, sigma <= {args.sigma_max})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "greedy_trends_clean.png"), dpi=220)
    plt.close(fig)

    # Correlation by N_loc for quick read.
    rows = []
    for nloc, df in g_clean.groupby("N_loc"):
        if len(df) < 3:
            continue
        rows.append(
            {
                "N_loc": int(nloc),
                "corr_sigma_rmse": np.corrcoef(df["sigma_bar"], df["rmse"])[0, 1],
                "corr_sigma_tmean": np.corrcoef(df["sigma_bar"], df["t_mean_ms"])[0, 1],
                "corr_sigma_slack": np.corrcoef(df["sigma_bar"], df["slack_max"])[0, 1],
                "corr_sigma_kopt": np.corrcoef(df["sigma_bar"], df["Kopt_mean"])[0, 1],
                "success_rate": float(df["success"].mean()),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "greedy_clean_correlations_by_nloc.csv"), index=False)

    # Adaptive-rho (already cleaner) summary plot.
    r = pd.read_csv(args.rho_csv)
    rg = r.groupby("rho", as_index=False).agg(cost=("cost", "mean"), rmse=("rmse", "mean"), succ=("success", "mean"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(rg["rho"], rg["cost"], marker="o")
    axes[0].set_title("cost vs rho")
    axes[1].plot(rg["rho"], rg["rmse"], marker="o")
    axes[1].set_title("rmse vs rho")
    axes[2].plot(rg["rho"], rg["succ"], marker="o")
    axes[2].set_title("success vs rho")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlabel("rho")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "adaptive_rho_trends_clean.png"), dpi=220)
    plt.close(fig)

    print(f"Saved: {args.outdir}")


if __name__ == "__main__":
    main()

