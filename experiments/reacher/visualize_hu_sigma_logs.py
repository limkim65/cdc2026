import argparse
import csv
import glob
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=160)
    plt.close()


def _plot_summary(summary_csv: str, outdir: str) -> List[Tuple[str, str]]:
    outputs = []
    if not os.path.exists(summary_csv):
        return outputs

    df = pd.read_csv(summary_csv)
    needed = [
        "rho",
        "min_sigma_min_Hu",
        "mean_solve_time_ms",
        "k_opt_median",
        "slack_fail_rate",
        "total_cost",
    ]
    for col in needed:
        if col not in df.columns:
            raise RuntimeError(f"Missing column in {summary_csv}: {col}")

    # 1) Hu sigma vs solve time
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(
        df["min_sigma_min_Hu"],
        df["mean_solve_time_ms"],
        c=df["rho"],
        cmap="viridis",
        s=36,
        alpha=0.85,
    )
    plt.xscale("log")
    plt.xlabel("min sigma_min(H_u) per run")
    plt.ylabel("mean solve time [ms]")
    plt.title("H_u conditioning vs computation")
    plt.grid(alpha=0.25)
    cb = plt.colorbar(sc)
    cb.set_label("rho")
    p = os.path.join(outdir, "A_hu_sigma_vs_solve_scatter.png")
    _savefig(p)
    outputs.append((summary_csv, p))

    # 2) Hu sigma by rho (median + quantile band)
    g = (
        df.groupby("rho")["min_sigma_min_Hu"]
        .agg(
            q10=lambda s: np.quantile(s, 0.1),
            med="median",
            q90=lambda s: np.quantile(s, 0.9),
        )
        .reset_index()
        .sort_values("rho")
    )
    plt.figure(figsize=(7, 5))
    x = g["rho"].to_numpy(dtype=float)
    med = g["med"].to_numpy(dtype=float)
    q10 = g["q10"].to_numpy(dtype=float)
    q90 = g["q90"].to_numpy(dtype=float)
    plt.plot(x, med, marker="o", lw=1.8, label="median")
    plt.fill_between(x, q10, q90, alpha=0.2, label="q10-q90")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("rho")
    plt.ylabel("min sigma_min(H_u)")
    plt.title("H_u minimum singular value across rho")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    p = os.path.join(outdir, "B_hu_sigma_by_rho_band.png")
    _savefig(p)
    outputs.append((summary_csv, p))

    # 3) Hu sigma vs Kopt
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(
        df["k_opt_median"],
        df["min_sigma_min_Hu"],
        c=df["rho"],
        cmap="plasma",
        s=36,
        alpha=0.85,
    )
    plt.yscale("log")
    plt.xlabel("k_opt_median")
    plt.ylabel("min sigma_min(H_u)")
    plt.title("Selected K vs H_u conditioning")
    plt.grid(alpha=0.25)
    cb = plt.colorbar(sc)
    cb.set_label("rho")
    p = os.path.join(outdir, "C_hu_sigma_vs_kopt_scatter.png")
    _savefig(p)
    outputs.append((summary_csv, p))

    # 4) Hu sigma vs slack fail
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(
        df["min_sigma_min_Hu"],
        df["slack_fail_rate"],
        c=df["rho"],
        cmap="cividis",
        s=36,
        alpha=0.85,
    )
    plt.xscale("log")
    plt.xlabel("min sigma_min(H_u)")
    plt.ylabel("slack fail rate")
    plt.title("Conditioning vs feasibility")
    plt.grid(alpha=0.25)
    cb = plt.colorbar(sc)
    cb.set_label("rho")
    p = os.path.join(outdir, "D_hu_sigma_vs_slack_fail_scatter.png")
    _savefig(p)
    outputs.append((summary_csv, p))

    return outputs


def _plot_summary_by_rho(by_rho_csv: str, outdir: str) -> List[Tuple[str, str]]:
    outputs = []
    if not os.path.exists(by_rho_csv):
        return outputs
    df = pd.read_csv(by_rho_csv).sort_values("rho")

    cols = [
        "rho",
        "median_sigma_inf_max",
        "median_sigma_inf_mean",
        "median_k_opt",
        "median_mean_solve_time_ms",
    ]
    for col in cols:
        if col not in df.columns:
            raise RuntimeError(f"Missing column in {by_rho_csv}: {col}")

    fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axs[0].plot(df["rho"], df["median_sigma_inf_max"], marker="o", label="sigma_inf_max")
    axs[0].plot(df["rho"], df["median_sigma_inf_mean"], marker="s", label="sigma_inf_mean")
    axs[0].set_xscale("log")
    axs[0].set_ylabel("median sigma_inf")
    axs[0].set_title("Sigma-inf summaries by rho")
    axs[0].grid(alpha=0.25)
    axs[0].legend(loc="best")

    axs[1].plot(df["rho"], df["median_k_opt"], marker="o", label="median_k_opt")
    axs[1].plot(df["rho"], df["median_mean_solve_time_ms"], marker="s", label="median solve ms")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("rho")
    axs[1].set_ylabel("value")
    axs[1].grid(alpha=0.25)
    axs[1].legend(loc="best")

    p = os.path.join(outdir, "E_sigma_inf_and_compute_by_rho.png")
    _savefig(p)
    outputs.append((by_rho_csv, p))
    return outputs


def _parse_diag_txt(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Diagnostic") or line.startswith("K="):
                continue
            if line.startswith("step"):
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 5:
                continue
            try:
                rows.append(
                    {
                        "step": int(parts[0]),
                        "sigma_k40": float(parts[1]),
                        "status_k40": parts[2],
                        "sigma_k50": float(parts[3]),
                        "status_k50": parts[4],
                    }
                )
            except Exception:
                continue
    return pd.DataFrame(rows)


def _plot_diag_txt(diag_txt: str, outdir: str) -> List[Tuple[str, str]]:
    outputs = []
    df = _parse_diag_txt(diag_txt)
    if df.empty:
        return outputs

    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axs[0].plot(df["step"], df["sigma_k40"], label="K=40", lw=1.6)
    axs[0].plot(df["step"], df["sigma_k50"], label="K=50", lw=1.6)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("sigma_min_u")
    axs[0].set_title(os.path.basename(diag_txt))
    axs[0].grid(alpha=0.25)
    axs[0].legend(loc="best")

    fail40 = df[~df["status_k40"].str.contains("optimal", case=False, na=False)]
    fail50 = df[~df["status_k50"].str.contains("optimal", case=False, na=False)]
    axs[1].plot(df["step"], np.zeros(len(df)), alpha=0)
    if not fail40.empty:
        axs[1].scatter(fail40["step"], np.ones(len(fail40)) * 1.0, s=16, label="K40 non-optimal")
    if not fail50.empty:
        axs[1].scatter(fail50["step"], np.ones(len(fail50)) * 0.0, s=16, label="K50 non-optimal")
    axs[1].set_yticks([0.0, 1.0])
    axs[1].set_yticklabels(["K50", "K40"])
    axs[1].set_xlabel("step")
    axs[1].set_ylabel("solver status")
    axs[1].grid(alpha=0.25)
    axs[1].legend(loc="best")

    stem = os.path.splitext(os.path.basename(diag_txt))[0]
    p = os.path.join(outdir, f"F_diag_{stem}.png")
    _savefig(p)
    outputs.append((diag_txt, p))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize all found H_u sigma-min related logs.")
    parser.add_argument("--history_root", type=str, default="logs/history")
    parser.add_argument(
        "--outdir",
        type=str,
        default="logs/history/h_u_sigma_visualization_bundle",
    )
    args = parser.parse_args()

    _ensure_dir(args.outdir)

    summary_csv = os.path.join(args.history_root, "adaptive_k", "summary.csv")
    by_rho_csv = os.path.join(
        args.history_root,
        "adaptive_k",
        "summary_by_rho_Kmin40_Kmax200_Kstep1_gmin1e-06.csv",
    )
    diag_glob = os.path.join(args.history_root, "results", "diag_k40_vs_k50*.txt")
    diag_files = sorted(glob.glob(diag_glob))

    produced: List[Tuple[str, str]] = []
    produced.extend(_plot_summary(summary_csv, args.outdir))
    produced.extend(_plot_summary_by_rho(by_rho_csv, args.outdir))
    for d in diag_files:
        produced.extend(_plot_diag_txt(d, args.outdir))

    manifest = os.path.join(args.outdir, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "output_plot"])
        for src, out in produced:
            w.writerow([src, out])

    print(f"Saved {len(produced)} plots")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
