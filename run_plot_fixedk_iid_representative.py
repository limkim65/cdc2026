import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ("sigma_mk_min", "sigma_min(Mk)"),
    ("cost", "Total Cost"),
    ("solve_mean", "Mean Solve Time (ms)"),
    ("K_opt_mean", "K_opt Mean"),
    ("rmse", "RMSE"),
    ("ISE", "ISE"),
    ("IAE", "IAE"),
    ("slack_max", "Slack Max"),
    ("solver_opt_rate", "Solver Optimal Rate"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot fixed-k iid representative 9-metrics figure."
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.path.join("logs", "reacher", "fixed_k", "dataset_iid"),
        help="Directory containing run_fixedk_K*_seed*.npz files.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Output directory. Default: <logdir>/analysis_fullstats",
    )
    parser.add_argument(
        "--exclude_k",
        type=int,
        nargs="*",
        default=[10],
        help="K values to exclude from plotting.",
    )
    parser.add_argument(
        "--fig_name",
        type=str,
        default="fig_fixedk_iid_representative_9metrics_noK10.png",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="K_stats_representative_noK10.csv",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_runs(logdir):
    paths = sorted(glob.glob(os.path.join(logdir, "run_fixedk_K*_seed*.npz")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No run_fixedk_K*_seed*.npz found in: {logdir}")

    rows = []
    for path in paths:
        d = np.load(path, allow_pickle=True)

        if "K" in d.files:
            K = int(np.asarray(d["K"]).item())
        else:
            m = re.search(r"_K(\d+)_", os.path.basename(path))
            if m is None:
                continue
            K = int(m.group(1))

        m_seed = re.search(r"_seed(\d+)\.npz$", os.path.basename(path))
        seed = int(m_seed.group(1)) if m_seed is not None else -1

        rmse = float(np.asarray(d["rmse_ee"]).item()) if "rmse_ee" in d.files else np.nan
        cost = float(np.asarray(d["total_cost"]).item()) if "total_cost" in d.files else np.nan
        ise = float(np.asarray(d["ISE"]).item()) if "ISE" in d.files else np.nan
        iae = float(np.asarray(d["IAE"]).item()) if "IAE" in d.files else np.nan

        solve = np.asarray(d["solve_time_ms"], dtype=float).reshape(-1) if "solve_time_ms" in d.files else np.array([], dtype=float)
        sigma_mk = np.asarray(d["sigma_min_Mk"], dtype=float).reshape(-1) if "sigma_min_Mk" in d.files else np.array([], dtype=float)
        slack = np.asarray(d["slack_inf"], dtype=float).reshape(-1) if "slack_inf" in d.files else np.array([], dtype=float)
        kopt = np.asarray(d["K_opt"], dtype=float).reshape(-1) if "K_opt" in d.files else np.array([], dtype=float)
        status = np.asarray(d["solver_status"], dtype=object).reshape(-1) if "solver_status" in d.files else np.array([], dtype=object)
        status_l = np.array([str(x).lower() for x in status], dtype=object)
        opt_rate = float(np.mean([("optimal" in s) for s in status_l])) if status_l.size else np.nan

        rows.append(
            {
                "K": K,
                "seed": seed,
                "sigma_mk_min": float(np.nanmin(sigma_mk)) if sigma_mk.size else np.nan,
                "cost": cost,
                "solve_mean": float(np.nanmean(solve)) if solve.size else np.nan,
                "K_opt_mean": float(np.nanmean(kopt)) if kopt.size else np.nan,
                "rmse": rmse,
                "ISE": ise,
                "IAE": iae,
                "slack_max": float(np.nanmax(slack)) if slack.size else np.nan,
                "solver_opt_rate": opt_rate,
            }
        )

    return pd.DataFrame(rows).sort_values(["K", "seed"])


def aggregate_representative(df):
    rep_rows = []
    for K, sub in df.groupby("K"):
        row = {"K": int(K)}
        for metric, _ in METRICS:
            v = sub[metric].to_numpy(dtype=float)
            row[f"{metric}_median"] = float(np.nanmedian(v))
            row[f"{metric}_p10"] = float(np.nanpercentile(v, 10))
            row[f"{metric}_p90"] = float(np.nanpercentile(v, 90))
            row[f"{metric}_min"] = float(np.nanmin(v))
            row[f"{metric}_max"] = float(np.nanmax(v))
        rep_rows.append(row)
    return pd.DataFrame(rep_rows).sort_values("K")


def plot_representative(rep_df, out_path, dpi):
    x = rep_df["K"].to_numpy(dtype=float)
    fig, axs = plt.subplots(3, 3, figsize=(16, 12), dpi=180)
    axs = axs.flatten()

    for i, (metric, label) in enumerate(METRICS):
        med = rep_df[f"{metric}_median"].to_numpy(dtype=float)
        p10 = rep_df[f"{metric}_p10"].to_numpy(dtype=float)
        p90 = rep_df[f"{metric}_p90"].to_numpy(dtype=float)
        mn = rep_df[f"{metric}_min"].to_numpy(dtype=float)
        mx = rep_df[f"{metric}_max"].to_numpy(dtype=float)

        axs[i].plot(x, med, "-o", color="#2ca02c", label="median")
        axs[i].fill_between(x, p10, p90, color="#2ca02c", alpha=0.2, label="p10-p90")
        axs[i].plot(x, mn, "--", color="#7f7f7f", linewidth=1, label="min")
        axs[i].plot(x, mx, "--", color="#9467bd", linewidth=1, label="max")
        axs[i].set_title(f"{label} vs K (representative)")
        axs[i].set_xlabel("K")
        axs[i].set_ylabel(label)
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = args.outdir if args.outdir else os.path.join(args.logdir, "analysis_fullstats")
    os.makedirs(outdir, exist_ok=True)

    runs = load_runs(args.logdir)
    if args.exclude_k:
        runs = runs[~runs["K"].isin(args.exclude_k)].copy()
    if runs.empty:
        raise RuntimeError("No runs left after applying exclude_k.")

    rep = aggregate_representative(runs)

    csv_path = os.path.join(outdir, args.csv_name)
    fig_path = os.path.join(outdir, args.fig_name)
    rep.to_csv(csv_path, index=False)
    plot_representative(rep, fig_path, dpi=int(args.dpi))

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Figure: {fig_path}")


if __name__ == "__main__":
    main()
