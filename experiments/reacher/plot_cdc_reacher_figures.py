import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def read_csv_rows(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def to_float(v, default=np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def group_median(rows: List[dict], x_key: str, y_key: str) -> Dict[float, float]:
    bucket = {}
    for r in rows:
        x = to_float(r.get(x_key, np.nan))
        y = to_float(r.get(y_key, np.nan))
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        bucket.setdefault(x, []).append(y)
    return {k: float(np.median(v)) for k, v in bucket.items() if len(v) > 0}


def save_table(path: str, rows: List[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_valid(ax, x: np.ndarray, y: np.ndarray, **kwargs) -> bool:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        ax.text(0.5, 0.5, "no valid data", ha="center", va="center", transform=ax.transAxes)
        return False
    ax.plot(x[m], y[m], **kwargs)
    return True


def can_log(arr: np.ndarray) -> bool:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return bool(a.size > 0 and np.all(a > 0))


def main():
    parser = argparse.ArgumentParser(description="Plot CDC-style figures from reacher benchmark logs.")
    parser.add_argument("--indir", type=str, required=True, help="Directory containing summary csv files.")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for figures/tables.")
    args = parser.parse_args()

    outdir = args.outdir if args.outdir is not None else os.path.join(args.indir, "figures_cdc")
    os.makedirs(outdir, exist_ok=True)

    summary = read_csv_rows(os.path.join(args.indir, "summary.csv"))
    by_k = read_csv_rows(os.path.join(args.indir, "summary_by_K.csv"))
    by_rho = read_csv_rows(os.path.join(args.indir, "summary_by_rho.csv"))

    fixed_rows = [r for r in summary if str(r.get("mode", "")) == "fixed_k"]
    adapt_rows = [r for r in summary if str(r.get("mode", "")) == "adaptive_rho"]

    # Fallback: if summary_by_K is missing, compute medians from summary.csv
    if not by_k and fixed_rows:
        k_vals = sorted(set(int(to_float(r.get("K", np.nan))) for r in fixed_rows if np.isfinite(to_float(r.get("K", np.nan)))))
        for k in k_vals:
            s = [r for r in fixed_rows if int(to_float(r.get("K", np.nan))) == k]
            by_k.append(
                {
                    "K": k,
                    "median_sigma_min_Ag": np.median([to_float(r.get("sigma_min_Ag", np.nan)) for r in s]),
                    "median_localness": np.median([to_float(r.get("localness_mean", np.nan)) for r in s]),
                    "median_slack_max": np.median([to_float(r.get("slack_max", np.nan)) for r in s]),
                    "median_kkt_max": np.median([to_float(r.get("kkt_res_max", np.nan)) for r in s]),
                    "median_rmse_ee": np.median([to_float(r.get("rmse_ee", np.nan)) for r in s]),
                    "success_rate": np.mean([to_float(r.get("success", 0)) for r in s]),
                    "median_total_cost": np.median([to_float(r.get("total_cost", np.nan)) for r in s]),
                    "median_jerk_sum_sq": np.median([to_float(r.get("jerk_sum_sq", np.nan)) for r in s]),
                }
            )
    # Fallback: if summary_by_rho is missing, compute medians from summary.csv
    if not by_rho and adapt_rows:
        rho_vals = sorted(set(to_float(r.get("rho", np.nan)) for r in adapt_rows if np.isfinite(to_float(r.get("rho", np.nan)))))
        for rho in rho_vals:
            s = [r for r in adapt_rows if np.isclose(to_float(r.get("rho", np.nan)), rho)]
            by_rho.append(
                {
                    "rho": rho,
                    "median_k_opt_mean": np.median([to_float(r.get("k_opt_mean", np.nan)) for r in s]),
                    "median_sigma_min_Ag": np.median([to_float(r.get("sigma_min_Ag", np.nan)) for r in s]),
                    "median_localness": np.median([to_float(r.get("localness_mean", np.nan)) for r in s]),
                    "median_slack_max": np.median([to_float(r.get("slack_max", np.nan)) for r in s]),
                    "median_kkt_max": np.median([to_float(r.get("kkt_res_max", np.nan)) for r in s]),
                    "median_rmse_ee": np.median([to_float(r.get("rmse_ee", np.nan)) for r in s]),
                    "success_rate": np.mean([to_float(r.get("success", 0)) for r in s]),
                    "median_total_cost": np.median([to_float(r.get("total_cost", np.nan)) for r in s]),
                    "median_solve_ms": np.median([to_float(r.get("solve_ms_mean", np.nan)) for r in s]),
                }
            )

    # Figure 1: K vs sigma_min(A_g), K vs localness
    if by_k:
        k = np.array([to_float(r["K"]) for r in by_k], dtype=float)
        sig = np.array([to_float(r.get("median_sigma_min_Ag", np.nan)) for r in by_k], dtype=float)
        loc = np.array([to_float(r.get("median_localness", np.nan)) for r in by_k], dtype=float)
        order = np.argsort(k)
        k, sig, loc = k[order], sig[order], loc[order]

        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
        ok0 = plot_valid(axs[0], k, sig, marker="o", lw=1.5)
        axs[0].set_xlabel("K")
        axs[0].set_ylabel(r"$\sigma_{\min}(A_g)$")
        axs[0].set_title("Conditioning Phase Transition")
        axs[0].grid(True, alpha=0.25)
        if ok0 and can_log(sig):
            axs[0].set_yscale("log")

        ok1 = plot_valid(axs[1], k, loc, marker="o", lw=1.5, color="tab:orange")
        axs[1].set_xlabel("K")
        axs[1].set_ylabel("Loc(K)")
        axs[1].set_title("Localness")
        axs[1].grid(True, alpha=0.25)
        if ok1 and can_log(loc):
            axs[1].set_yscale("log")

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "figure1_k_vs_conditioning_localness.png"), bbox_inches="tight")
        plt.close(fig)

    # Figure 2: K vs slack, K vs KKT residual
    if by_k:
        k = np.array([to_float(r["K"]) for r in by_k], dtype=float)
        slack = np.array([to_float(r.get("median_slack_max", np.nan)) for r in by_k], dtype=float)
        kkt = np.array([to_float(r.get("median_kkt_max", np.nan)) for r in by_k], dtype=float)
        order = np.argsort(k)
        k, slack, kkt = k[order], slack[order], kkt[order]

        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
        ok0 = plot_valid(axs[0], k, slack, marker="o", lw=1.5, color="tab:red")
        axs[0].set_xlabel("K")
        axs[0].set_ylabel(r"$\|\sigma_y\|_\infty$ (max)")
        axs[0].set_title("Slack Explosion")
        axs[0].grid(True, alpha=0.25)
        if ok0 and can_log(slack):
            axs[0].set_yscale("log")

        ok1 = plot_valid(axs[1], k, kkt, marker="o", lw=1.5, color="tab:purple")
        axs[1].set_xlabel("K")
        axs[1].set_ylabel("KKT Residual (proxy)")
        axs[1].set_title("Solver Stability")
        axs[1].grid(True, alpha=0.25)
        if ok1 and can_log(kkt):
            axs[1].set_yscale("log")

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "figure2_k_vs_solver_stability.png"), bbox_inches="tight")
        plt.close(fig)

    # Figure 3: K vs RMSE, K vs success
    if by_k:
        k = np.array([to_float(r["K"]) for r in by_k], dtype=float)
        rmse = np.array([to_float(r.get("median_rmse_ee", np.nan)) for r in by_k], dtype=float)
        succ = np.array([to_float(r.get("success_rate", np.nan)) for r in by_k], dtype=float)
        order = np.argsort(k)
        k, rmse, succ = k[order], rmse[order], succ[order]

        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
        ok0 = plot_valid(axs[0], k, rmse, marker="o", lw=1.5, color="tab:green")
        axs[0].set_xlabel("K")
        axs[0].set_ylabel("End-effector RMSE")
        axs[0].set_title("Control Performance")
        axs[0].grid(True, alpha=0.25)
        if ok0 and can_log(rmse):
            axs[0].set_yscale("log")

        plot_valid(axs[1], k, succ, marker="o", lw=1.5, color="tab:blue")
        axs[1].set_xlabel("K")
        axs[1].set_ylabel("Success Rate")
        axs[1].set_ylim(-0.02, 1.02)
        axs[1].set_title("Task Success")
        axs[1].grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "figure3_k_vs_performance.png"), bbox_inches="tight")
        plt.close(fig)

    # Figure 4: adaptive rho trade-off
    if by_rho:
        rho = np.array([to_float(r["rho"]) for r in by_rho], dtype=float)
        kopt = np.array([to_float(r.get("median_k_opt_mean", np.nan)) for r in by_rho], dtype=float)
        cost = np.array([to_float(r.get("median_total_cost", np.nan)) for r in by_rho], dtype=float)
        solve = np.array([to_float(r.get("median_solve_ms", np.nan)) for r in by_rho], dtype=float)
        succ = np.array([to_float(r.get("success_rate", np.nan)) for r in by_rho], dtype=float)
        order = np.argsort(rho)
        rho, kopt, cost, solve, succ = rho[order], kopt[order], cost[order], solve[order], succ[order]

        fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), dpi=150)
        ok0 = plot_valid(axs[0], rho, kopt, marker="o", lw=1.5)
        axs[0].set_xlabel(r"$\rho$")
        axs[0].set_ylabel("Mean $K_{opt}$")
        axs[0].set_title("Adaptive-K Allocation")
        axs[0].grid(True, alpha=0.25)
        if ok0 and can_log(rho):
            axs[0].set_xscale("log")

        ok1 = plot_valid(axs[1], rho, cost, marker="o", lw=1.5, color="tab:red")
        axs[1].set_xlabel(r"$\rho$")
        axs[1].set_ylabel("Total Cost")
        axs[1].set_title("Cost Trade-off")
        axs[1].grid(True, alpha=0.25)
        if ok1 and can_log(rho):
            axs[1].set_xscale("log")
        if ok1 and can_log(cost):
            axs[1].set_yscale("log")

        ok2 = plot_valid(axs[2], rho, succ, marker="o", lw=1.5, color="tab:green")
        axs[2].set_xlabel(r"$\rho$")
        axs[2].set_ylabel("Success Rate")
        axs[2].set_ylim(-0.02, 1.02)
        axs[2].set_title("Robustness")
        axs[2].grid(True, alpha=0.25)
        if ok2 and can_log(rho):
            axs[2].set_xscale("log")

        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "figure4_adaptive_rho_tradeoff.png"), bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0), dpi=150)
        m = np.isfinite(solve) & np.isfinite(cost) & np.isfinite(rho)
        if np.any(m):
            sc = ax.scatter(solve[m], cost[m], c=rho[m], cmap="viridis", s=70)
        else:
            sc = ax.scatter([], [])
            ax.text(0.5, 0.5, "no valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Solve Time [ms]")
        ax.set_ylabel("Total Cost")
        ax.set_title("Adaptive-K Compute vs Performance")
        ax.grid(True, alpha=0.25)
        if np.any(m) and can_log(cost[m]):
            ax.set_yscale("log")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"$\rho$")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "figure4b_adaptive_cost_vs_solve.png"), bbox_inches="tight")
        plt.close(fig)

    # Table export
    table_rows = []
    if by_k:
        for r in by_k:
            table_rows.append(
                {
                    "method": "fixed_k",
                    "x": to_float(r.get("K", np.nan)),
                    "sigma_min_Ag": to_float(r.get("median_sigma_min_Ag", np.nan)),
                    "localness": to_float(r.get("median_localness", np.nan)),
                    "slack_max": to_float(r.get("median_slack_max", np.nan)),
                    "kkt_max": to_float(r.get("median_kkt_max", np.nan)),
                    "rmse_ee": to_float(r.get("median_rmse_ee", np.nan)),
                    "success_rate": to_float(r.get("success_rate", np.nan)),
                    "total_cost": to_float(r.get("median_total_cost", np.nan)),
                    "jerk_sum_sq": to_float(r.get("median_jerk_sum_sq", np.nan)),
                }
            )
    if by_rho:
        for r in by_rho:
            table_rows.append(
                {
                    "method": "adaptive_rho",
                    "x": to_float(r.get("rho", np.nan)),
                    "sigma_min_Ag": to_float(r.get("median_sigma_min_Ag", np.nan)),
                    "localness": to_float(r.get("median_localness", np.nan)),
                    "slack_max": to_float(r.get("median_slack_max", np.nan)),
                    "kkt_max": to_float(r.get("median_kkt_max", np.nan)),
                    "rmse_ee": to_float(r.get("median_rmse_ee", np.nan)),
                    "success_rate": to_float(r.get("success_rate", np.nan)),
                    "total_cost": to_float(r.get("median_total_cost", np.nan)),
                    "jerk_sum_sq": np.nan,
                }
            )
    save_table(os.path.join(outdir, "table_cdc_metrics.csv"), table_rows)

    print(f"Loaded: summary={len(summary)} by_k={len(by_k)} by_rho={len(by_rho)}")
    print(f"Saved CDC figures/tables under: {outdir}")


if __name__ == "__main__":
    main()
