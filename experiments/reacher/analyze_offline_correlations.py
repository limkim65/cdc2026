import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def align_kopt_to_steps(k_raw: np.ndarray, n_steps: int) -> np.ndarray:
    k = np.asarray(k_raw, dtype=float).reshape(-1)
    if k.size == n_steps:
        return k
    if k.size == 2 * n_steps:
        # In current controller, K_opt can be appended twice per step.
        # Use selected-size entry (second of each pair).
        return k.reshape(n_steps, 2)[:, 1]
    if k.size > n_steps:
        return k[:n_steps]
    # Pad with NaN if shorter.
    out = np.full(n_steps, np.nan, dtype=float)
    out[: k.size] = k
    return out


def heatmap_from_df(corr_df: pd.DataFrame, title: str, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    mat = corr_df.to_numpy(dtype=float)
    im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(corr_df.shape[1]))
    ax.set_xticklabels(corr_df.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(corr_df.shape[0]))
    ax.set_yticklabels(corr_df.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("correlation")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Correlation analysis for single offline npz run.")
    parser.add_argument(
        "--npz",
        type=str,
        default=os.path.join("logs", "reacher", "offline_ksweep_dir", "run_fixed_K080_rho0.3_seed000.npz"),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("logs", "reacher", "offline_ksweep_dir", "paper_figs"),
    )
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = np.load(args.npz, allow_pickle=True)
    n_steps = np.asarray(d["sigma_min_Hu"], dtype=float).reshape(-1).size
    kopt_step = align_kopt_to_steps(np.asarray(d["K_opt_history"]), n_steps)
    df = pd.DataFrame(
        {
            "K_opt": kopt_step,
            "sigma_min_Hu": np.asarray(d["sigma_min_Hu"], dtype=float).reshape(-1),
            "tracking_error": np.asarray(d["tracking_error"], dtype=float).reshape(-1),
            "slack_inf": np.asarray(d["slack_inf"], dtype=float).reshape(-1),
            "solve_time_ms": np.asarray(d["solve_time_ms"], dtype=float).reshape(-1),
        }
    )
    df["step"] = np.arange(len(df), dtype=int)

    pearson = df.drop(columns=["step"]).corr(method="pearson")
    spearman = df.drop(columns=["step"]).corr(method="spearman")
    pearson.to_csv(os.path.join(args.outdir, "corr_pearson.csv"))
    spearman.to_csv(os.path.join(args.outdir, "corr_spearman.csv"))

    heatmap_from_df(pearson, "Pearson Correlation", os.path.join(args.outdir, "corr_pearson_heatmap.png"))
    heatmap_from_df(spearman, "Spearman Correlation", os.path.join(args.outdir, "corr_spearman_heatmap.png"))

    # Pair scatters (colored by step)
    pairs = [
        ("K_opt", "sigma_min_Hu"),
        ("K_opt", "tracking_error"),
        ("K_opt", "slack_inf"),
        ("K_opt", "solve_time_ms"),
        ("sigma_min_Hu", "tracking_error"),
        ("sigma_min_Hu", "slack_inf"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for ax, (x, y) in zip(axes, pairs):
        sc = ax.scatter(df[x], df[y], c=df["step"], s=18, cmap="viridis", alpha=0.85)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.25)
    for ax in axes[len(pairs) :]:
        ax.axis("off")
    cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("step")
    fig.suptitle("Pairwise Relationships")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "pairwise_scatter_grid.png"), dpi=220)
    plt.close(fig)

    # Optional: low-sigma regime correlations (bottom 25% sigma_min_Hu)
    q25 = np.quantile(df["sigma_min_Hu"], 0.25)
    low = df[df["sigma_min_Hu"] <= q25].drop(columns=["step"])
    if len(low) >= 10:
        low_corr = low.corr(method="pearson")
        low_corr.to_csv(os.path.join(args.outdir, "corr_pearson_low_sigma25.csv"))
        heatmap_from_df(
            low_corr,
            "Pearson Correlation (Low sigma_min_Hu 25%)",
            os.path.join(args.outdir, "corr_pearson_low_sigma25_heatmap.png"),
        )

    print(f"Saved analysis to: {args.outdir}")


if __name__ == "__main__":
    main()
