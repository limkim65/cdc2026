import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pareto_mask_minimize(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.size
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
        if np.any(dominated):
            keep[i] = False
    return keep


def pivot_grid(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nlocs = np.sort(df["N_loc"].unique())
    sigmas = np.sort(df["sigma_bar"].unique())
    grid = np.full((nlocs.size, sigmas.size), np.nan, dtype=float)
    for i, nloc in enumerate(nlocs):
        for j, sigma in enumerate(sigmas):
            sub = df[(df["N_loc"] == nloc) & (df["sigma_bar"] == sigma)]
            if not sub.empty:
                grid[i, j] = float(sub[value_col].mean())
    return nlocs, sigmas, grid


def draw_heatmap(ax, grid, nlocs, sigmas, title, cbar_label):
    im = ax.imshow(grid, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("sigma_bar")
    ax.set_ylabel("N_loc")
    ax.set_xticks(np.arange(sigmas.size))
    ax.set_xticklabels([f"{v:g}" for v in sigmas], rotation=45, ha="right")
    ax.set_yticks(np.arange(nlocs.size))
    ax.set_yticklabels([str(int(v)) for v in nlocs])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)


def main():
    parser = argparse.ArgumentParser(description="Greedy search result plots for cost/RMSE.")
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join("logs", "reacher", "loca-PE-gate", "greed_search_dir", "greedy_trial_results.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("logs", "reacher", "loca-PE-gate", "greed_search_dir", "paper_cost_rmse"),
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["total_cost", "rmse_ee"]).copy()

    nlocs, sigmas, cost_grid = pivot_grid(df, "total_cost")
    _, _, rmse_grid = pivot_grid(df, "rmse_ee")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    draw_heatmap(axes[0], cost_grid, nlocs, sigmas, "Cost Heatmap (N_loc x sigma_bar)", "total_cost")
    draw_heatmap(axes[1], rmse_grid, nlocs, sigmas, "RMSE Heatmap (N_loc x sigma_bar)", "rmse_ee")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_heatmaps_cost_rmse.png"), dpi=220)
    plt.close(fig)

    # Pareto front in (cost, rmse), color by solve time.
    solve = df["solve_time_mean_ms"].to_numpy(dtype=float)
    cost = df["total_cost"].to_numpy(dtype=float)
    rmse = df["rmse_ee"].to_numpy(dtype=float)
    pmask = pareto_mask_minimize(cost, rmse)
    pidx = np.where(pmask)[0]
    porder = np.argsort(cost[pidx])
    pidx = pidx[porder]

    fig = plt.figure(figsize=(8, 6))
    sc = plt.scatter(cost, rmse, c=solve, cmap="viridis", s=55, alpha=0.9)
    plt.colorbar(sc, label="solve_time_mean_ms")
    plt.plot(cost[pidx], rmse[pidx], "r-", lw=2, label="Pareto front")
    plt.scatter(cost[pidx], rmse[pidx], c="red", s=70)
    for i in pidx:
        plt.annotate(
            f"N{int(df.iloc[i]['N_loc'])}/s{df.iloc[i]['sigma_bar']:.2f}",
            (cost[i], rmse[i]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    plt.xlabel("total_cost (lower is better)")
    plt.ylabel("rmse_ee (lower is better)")
    plt.title("Cost-RMSE Pareto Plot")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_pareto_cost_rmse.png"), dpi=220)
    plt.close(fig)

    # Combined normalized score (equal weights) for "good overall".
    c_n = (df["total_cost"] - df["total_cost"].min()) / (df["total_cost"].max() - df["total_cost"].min() + 1e-12)
    r_n = (df["rmse_ee"] - df["rmse_ee"].min()) / (df["rmse_ee"].max() - df["rmse_ee"].min() + 1e-12)
    df["score_cost_rmse"] = 0.5 * c_n + 0.5 * r_n
    nlocs, sigmas, score_grid = pivot_grid(df, "score_cost_rmse")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    draw_heatmap(ax, score_grid, nlocs, sigmas, "Balanced Score Heatmap (cost/rmse)", "normalized score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_score_heatmap_cost_rmse.png"), dpi=220)
    plt.close(fig)

    # At-a-glance summary figure: best condition + top-10 table-like view.
    best_idx = int(df["score_cost_rmse"].idxmin())
    best = df.loc[best_idx]
    top10 = df.sort_values(["score_cost_rmse", "total_cost", "rmse_ee"]).head(10).copy()

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], width_ratios=[1.2, 1.0])

    # Panel A: score heatmap + best point marker.
    ax1 = fig.add_subplot(gs[:, 0])
    draw_heatmap(ax1, score_grid, nlocs, sigmas, "At-a-Glance: Best Region (lower is better)", "score")
    i_best = int(np.where(nlocs == best["N_loc"])[0][0])
    j_best = int(np.where(sigmas == best["sigma_bar"])[0][0])
    ax1.scatter([j_best], [i_best], s=220, facecolors="none", edgecolors="red", linewidths=2.5)
    ax1.text(
        j_best + 0.2,
        i_best + 0.2,
        f"BEST\nN_loc={int(best['N_loc'])}\nsigma={best['sigma_bar']:.2f}",
        color="red",
        fontsize=10,
        weight="bold",
    )

    # Panel B: top-10 score bars.
    ax2 = fig.add_subplot(gs[0, 1])
    labels = [f"N{int(r.N_loc)}/s{r.sigma_bar:.2f}" for _, r in top10.iterrows()]
    ax2.barh(np.arange(len(top10))[::-1], top10["score_cost_rmse"].to_numpy(), color="#4C78A8")
    ax2.set_yticks(np.arange(len(top10))[::-1])
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("score_cost_rmse")
    ax2.set_title("Top-10 Conditions (lower better)")
    ax2.grid(axis="x", alpha=0.25)

    # Panel C: compact metrics table as text.
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    header = "rank | condition      | cost      | rmse      | solve_ms"
    rows = [header, "-" * len(header)]
    for rank, (_, r) in enumerate(top10.iterrows(), start=1):
        rows.append(
            f"{rank:>4d} | N{int(r.N_loc):<5d}/s{r.sigma_bar:>4.2f} | {r.total_cost:>9.4g} | {r.rmse_ee:>9.4g} | {r.solve_time_mean_ms:>8.3f}"
        )
    ax3.text(0.01, 0.98, "\n".join(rows), va="top", family="monospace", fontsize=10)
    ax3.set_title("Top-10 Metrics")

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_at_a_glance_best_conditions.png"), dpi=220)
    plt.close(fig)

    # Save concise tables.
    cols = [
        "trial_id",
        "N_loc",
        "sigma_bar",
        "total_cost",
        "rmse_ee",
        "solve_time_mean_ms",
        "Kopt_mean",
        "score_cost_rmse",
    ]
    df.sort_values(["total_cost", "rmse_ee"], ascending=[True, True])[cols].head(20).to_csv(
        os.path.join(args.outdir, "top20_by_cost_then_rmse.csv"), index=False
    )
    df.sort_values(["rmse_ee", "total_cost"], ascending=[True, True])[cols].head(20).to_csv(
        os.path.join(args.outdir, "top20_by_rmse_then_cost.csv"), index=False
    )
    df[df.index.isin(pidx)][cols].sort_values(["total_cost", "rmse_ee"]).to_csv(
        os.path.join(args.outdir, "pareto_points_cost_rmse.csv"), index=False
    )
    top10.to_csv(os.path.join(args.outdir, "top10_at_a_glance.csv"), index=False)

    print(f"Saved plots/tables to: {args.outdir}")


if __name__ == "__main__":
    main()
