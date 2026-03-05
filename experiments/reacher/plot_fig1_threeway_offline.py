import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Figure 1: sigma_min(Hu) vs tracking vs instability.")
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=os.path.join(
            "logs",
            "history",
            "reacher",
            "fixed_k_sweep_dir",
            "reacher_k_sweep_quick_fixedK",
            "summary.csv",
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(
            "logs",
            "history",
            "reacher",
            "fixed_k_sweep_dir",
            "reacher_k_sweep_quick_fixedK",
            "paper_figs",
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.summary_csv)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Aggregate across seeds per K.
    g = (
        df.groupby("K", as_index=False)
        .agg(
            sigma_min_Hu_med=("sigma_min_Hu", "median"),
            tracking_err_med=("ISE", "median"),  # tracking error proxy
            slack_max_med=("slack_norm_max", "median"),
            solve_time_med=("mean_solve_time_ms", "median"),
            runs=("seed", "count"),
        )
        .sort_values("K")
    )

    # Main 3-way figure.
    fig, ax1 = plt.subplots(figsize=(10.5, 6))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 65))

    l1 = ax1.plot(
        g["K"],
        g["sigma_min_Hu_med"],
        color="#1f77b4",
        marker="o",
        linewidth=2.2,
        label=r"$\sigma_{\min}(H_u)$ (median)",
    )
    l2 = ax2.plot(
        g["K"],
        g["tracking_err_med"],
        color="#d62728",
        marker="s",
        linewidth=2.0,
        label="Tracking error (ISE median)",
    )
    l3 = ax3.plot(
        g["K"],
        g["slack_max_med"],
        color="#2ca02c",
        marker="^",
        linewidth=2.0,
        label="Instability proxy (slack max median)",
    )

    ax1.set_xlabel("K (selection strength)")
    ax1.set_ylabel(r"$\sigma_{\min}(H_u)$", color="#1f77b4")
    ax2.set_ylabel("Tracking error (ISE)", color="#d62728")
    ax3.set_ylabel("Slack max", color="#2ca02c")
    ax1.tick_params(axis="y", colors="#1f77b4")
    ax2.tick_params(axis="y", colors="#d62728")
    ax3.tick_params(axis="y", colors="#2ca02c")
    ax1.grid(alpha=0.3)
    ax1.set_title("Figure 1: Conditioning-Performance-Instability (Offline Fixed-K)")

    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best", frameon=True)
    plt.tight_layout()
    out_png = os.path.join(args.outdir, "fig1_threeway_sigma_tracking_slack.png")
    plt.savefig(out_png, dpi=240)
    plt.close(fig)

    # Save aggregated table for paper numbers.
    out_csv = os.path.join(args.outdir, "fig1_threeway_aggregated_by_K.csv")
    g.to_csv(out_csv, index=False)

    # Correlation summary (for quick sanity check).
    corr = {
        "corr_K_sigmaMinHu": float(np.corrcoef(g["K"], g["sigma_min_Hu_med"])[0, 1]),
        "corr_sigmaMinHu_trackingErr": float(np.corrcoef(g["sigma_min_Hu_med"], g["tracking_err_med"])[0, 1]),
        "corr_sigmaMinHu_slackMax": float(np.corrcoef(g["sigma_min_Hu_med"], g["slack_max_med"])[0, 1]),
    }
    pd.DataFrame([corr]).to_csv(os.path.join(args.outdir, "fig1_threeway_correlations.csv"), index=False)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {os.path.join(args.outdir, 'fig1_threeway_correlations.csv')}")


if __name__ == "__main__":
    main()

