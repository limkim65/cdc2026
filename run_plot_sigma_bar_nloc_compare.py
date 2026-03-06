import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_plot_sigma_bar_iid_representative import METRICS, aggregate_representative, load_runs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare sigma_bar sweep representative metrics across N_loc settings."
    )
    parser.add_argument(
        "--logdirs",
        nargs="+",
        required=True,
        help="List of log directories for each N_loc condition.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Legend labels corresponding to --logdirs (e.g., N_loc=200 N_loc=500 N_loc=1000).",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for comparison plots and merged CSV.",
    )
    parser.add_argument(
        "--fig_name",
        default="fig_sigma_bar_compare_Nloc_9metrics.png",
    )
    parser.add_argument(
        "--csv_name",
        default="sigma_bar_compare_Nloc_representative.csv",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.logdirs) != len(args.labels):
        raise ValueError("--logdirs and --labels must have the same length.")

    os.makedirs(args.outdir, exist_ok=True)

    reps = []
    for logdir, label in zip(args.logdirs, args.labels):
        runs = load_runs(logdir)
        rep = aggregate_representative(runs)
        rep["group"] = label
        reps.append(rep)

    merged = pd.concat(reps, ignore_index=True)
    merged.to_csv(os.path.join(args.outdir, args.csv_name), index=False)

    fig, axs = plt.subplots(3, 3, figsize=(16, 12), dpi=180)
    axs = axs.flatten()

    for i, (metric, label) in enumerate(METRICS):
        ax = axs[i]
        for group_label, sub in merged.groupby("group"):
            x = sub["sigma_bar"].to_numpy(dtype=float)
            med = sub[f"{metric}_median"].to_numpy(dtype=float)
            p10 = sub[f"{metric}_p10"].to_numpy(dtype=float)
            p90 = sub[f"{metric}_p90"].to_numpy(dtype=float)
            ax.plot(x, med, "-o", linewidth=1.8, label=f"{group_label} median")
            ax.fill_between(x, p10, p90, alpha=0.15)

        ax.set_title(f"{label} vs sigma_bar")
        ax.set_xlabel("sigma_bar")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(args.outdir, args.fig_name), dpi=int(args.dpi))
    plt.close(fig)

    print(f"Saved CSV: {os.path.join(args.outdir, args.csv_name)}")
    print(f"Saved Figure: {os.path.join(args.outdir, args.fig_name)}")


if __name__ == "__main__":
    main()
