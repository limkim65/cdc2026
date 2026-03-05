import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GREEDY_RE = re.compile(
    r"\[(?P<run_tag>N(?P<N_loc>\d+)_sigma[^\]]+)\]\s+"
    r"N_loc=\s*(?P<N_loc_2>\d+),\s+sigma_bar=(?P<sigma>[-+eE0-9\.]+)\s+\|\s+"
    r"succ=(?P<succ>\d+)\s+rmse=(?P<rmse>[-+eE0-9\.]+)\s+"
    r"t_mean=(?P<tmean>[-+eE0-9\.]+)ms\s+slack_max=(?P<slack>[-+eE0-9\.]+)\s+"
    r"Kopt_mean=(?P<kopt>[-+eE0-9\.]+)"
)

ADAPT_RHO_RE = re.compile(
    r"\[adaptive_rho\]\s+noise=(?P<noise>[-+eE0-9\.]+)\s+rho=(?P<rho>[-+eE0-9\.]+)\s+"
    r"seed=\s*(?P<seed>\d+)\s+cost=(?P<cost>[-+eE0-9\.]+)\s+rmse=(?P<rmse>[-+eE0-9\.]+)\s+succ=(?P<succ>\d+)"
)


def parse_log(log_text: str):
    greedy_rows = []
    rho_rows = []
    for line in log_text.splitlines():
        line = line.strip()
        if not line:
            continue
        mg = GREEDY_RE.search(line)
        if mg:
            d = mg.groupdict()
            greedy_rows.append(
                {
                    "run_tag": d["run_tag"],
                    "N_loc": int(d["N_loc"]),
                    "sigma_bar": float(d["sigma"]),
                    "success": int(d["succ"]),
                    "rmse": float(d["rmse"]),
                    "t_mean_ms": float(d["tmean"]),
                    "slack_max": float(d["slack"]),
                    "Kopt_mean": float(d["kopt"]),
                }
            )
            continue

        mr = ADAPT_RHO_RE.search(line)
        if mr:
            d = mr.groupdict()
            rho_rows.append(
                {
                    "noise": float(d["noise"]),
                    "rho": float(d["rho"]),
                    "seed": int(d["seed"]),
                    "cost": float(d["cost"]),
                    "rmse": float(d["rmse"]),
                    "success": int(d["succ"]),
                }
            )

    return pd.DataFrame(greedy_rows), pd.DataFrame(rho_rows)


def draw_greedy(df: pd.DataFrame, outdir: str):
    if df.empty:
        return
    df = df.sort_values(["N_loc", "sigma_bar"]).copy()
    df.to_csv(os.path.join(outdir, "parsed_greedy_rows.csv"), index=False)

    metrics = [
        ("rmse", "RMSE"),
        ("t_mean_ms", "Solve Time Mean (ms)"),
        ("slack_max", "Slack Max"),
        ("Kopt_mean", "Kopt Mean"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()
    for ax, (col, ylabel) in zip(axes, metrics):
        for nloc, g in df.groupby("N_loc"):
            g = g.sort_values("sigma_bar")
            ax.plot(g["sigma_bar"], g[col], marker="o", linewidth=1.7, label=f"N={int(nloc)}")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("sigma_bar")
    axes[-2].set_xlabel("sigma_bar")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Greedy Search Trends from Terminal Log")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "greedy_trends_by_sigma.png"), dpi=220)
    plt.close(fig)

    # Heatmaps for quick trend reading.
    for col in ["rmse", "t_mean_ms", "slack_max", "Kopt_mean", "success"]:
        piv = (
            df.groupby(["N_loc", "sigma_bar"], as_index=False)[col]
            .mean()
            .pivot(index="N_loc", columns="sigma_bar", values=col)
            .sort_index()
        )
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", origin="lower")
        ax.set_title(f"Heatmap: {col}")
        ax.set_xlabel("sigma_bar")
        ax.set_ylabel("N_loc")
        ax.set_xticks(np.arange(piv.shape[1]))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(piv.shape[0]))
        ax.set_yticklabels([str(int(r)) for r in piv.index])
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f"greedy_heatmap_{col}.png"), dpi=220)
        plt.close(fig)


def draw_adaptive_rho(df: pd.DataFrame, outdir: str):
    if df.empty:
        return
    df = df.sort_values(["rho", "seed"]).copy()
    df.to_csv(os.path.join(outdir, "parsed_adaptive_rho_rows.csv"), index=False)

    g = df.groupby("rho", as_index=False).agg(
        cost_mean=("cost", "mean"),
        cost_std=("cost", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        success_rate=("success", "mean"),
        n=("seed", "count"),
    )
    g.to_csv(os.path.join(outdir, "summary_adaptive_rho_from_log.csv"), index=False)

    x = g["rho"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].errorbar(x, g["cost_mean"], yerr=g["cost_std"], marker="o", capsize=3)
    axes[0].set_title("Cost vs rho")
    axes[0].set_xlabel("rho")
    axes[0].set_ylabel("cost")
    axes[0].grid(alpha=0.3)
    axes[1].errorbar(x, g["rmse_mean"], yerr=g["rmse_std"], marker="o", capsize=3, color="#d62728")
    axes[1].set_title("RMSE vs rho")
    axes[1].set_xlabel("rho")
    axes[1].set_ylabel("rmse")
    axes[1].grid(alpha=0.3)
    axes[2].plot(x, g["success_rate"], marker="o", color="#2ca02c")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Success Rate vs rho")
    axes[2].set_xlabel("rho")
    axes[2].set_ylabel("success rate")
    axes[2].grid(alpha=0.3)
    fig.suptitle("Adaptive-rho Trends from Terminal Log")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "adaptive_rho_trends.png"), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Parse terminal logs and draw quick trend plots.")
    parser.add_argument("--log_txt", type=str, required=True, help="Path to pasted terminal text log.")
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("logs", "reacher", "parsed_from_terminal"),
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.log_txt, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    greedy_df, rho_df = parse_log(text)
    draw_greedy(greedy_df, args.outdir)
    draw_adaptive_rho(rho_df, args.outdir)

    print(f"Saved directory: {args.outdir}")
    print(f"Parsed greedy rows: {len(greedy_df)}")
    print(f"Parsed adaptive_rho rows: {len(rho_df)}")


if __name__ == "__main__":
    main()

