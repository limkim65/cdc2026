import argparse
import csv
import glob
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def get_scalar(d, key, default=np.nan):
    if key not in d.files:
        return default
    a = np.asarray(d[key])
    if a.ndim == 0:
        try:
            return float(a)
        except Exception:
            return default
    if a.size == 1:
        try:
            return float(a.reshape(-1)[0])
        except Exception:
            return default
    return default


def compute_sse_and_reach(err: np.ndarray, eps: float, hold: int):
    e = np.asarray(err, dtype=float).reshape(-1)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return np.nan, np.nan, 0.0
    w = int(max(1, hold))
    sse = float(np.mean(e[-w:]))

    if e.size < w:
        reached = int(np.all(e <= eps))
        rt = float(e.size if not reached else 0)
        return sse, rt, float(reached)

    ok = (e <= float(eps)).astype(int)
    run = np.convolve(ok, np.ones(w, dtype=int), mode="valid")
    idx = np.where(run >= w)[0]
    if idx.size > 0:
        return sse, float(idx[0]), 1.0
    # Not reached: censored at horizon
    return sse, float(e.size), 0.0


def summarize(vals):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(v)), float(np.quantile(v, 0.1)), float(np.quantile(v, 0.9))


def load_rows(indir: str, reach_eps: float, reach_hold: int) -> List[Dict]:
    rows = []
    for p in glob.glob(os.path.join(indir, "*.npz")):
        d = np.load(p, allow_pickle=True)
        mode = str(np.asarray(d["mode"]).reshape(-1)[0]) if "mode" in d.files else ""
        noise = get_scalar(d, "measurement_noise_std", np.nan)
        if not np.isfinite(noise):
            m = re.search(r"noise([0-9.]+)", os.path.basename(p))
            if m:
                noise = float(m.group(1))
        r = {
            "file": os.path.basename(p),
            "mode": mode,
            "noise": noise,
            "K": get_scalar(d, "K", np.nan),
            "rho": get_scalar(d, "rho", np.nan),
            "cost": get_scalar(d, "total_cost", np.nan),
            "rmse": get_scalar(d, "rmse_ee", np.nan),
            "success": get_scalar(d, "success", np.nan),
            "slack": get_scalar(d, "slack_max", np.nan),
            "sigma": get_scalar(d, "sigma_min_Ag_episode_min", np.nan),
            "solve_ms": get_scalar(d, "solve_ms_mean", np.nan),
        }
        if not np.isfinite(r["sigma"]) and "sigma_min_Ag" in d.files:
            arr = np.asarray(d["sigma_min_Ag"], dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                r["sigma"] = float(np.min(arr))
        if not np.isfinite(r["solve_ms"]) and "solve_time_ms" in d.files:
            arr = np.asarray(d["solve_time_ms"], dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                r["solve_ms"] = float(np.mean(arr))
        if "tracking_error" in d.files:
            sse, reach_time, reach_hit = compute_sse_and_reach(
                np.asarray(d["tracking_error"], dtype=float),
                eps=float(reach_eps),
                hold=int(reach_hold),
            )
            r["sse"] = sse
            r["reach_time"] = reach_time
            r["reach_hit"] = reach_hit
        else:
            r["sse"] = np.nan
            r["reach_time"] = np.nan
            r["reach_hit"] = np.nan
        rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Plot Reacher noise experiment results.")
    parser.add_argument("--indir", type=str, required=True)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--reach_eps", type=float, default=0.01)
    parser.add_argument("--reach_hold", type=int, default=20)
    args = parser.parse_args()

    outdir = args.outdir or os.path.join("logs", "reacher", "figure", "noise_plots")
    os.makedirs(outdir, exist_ok=True)
    rows = load_rows(args.indir, args.reach_eps, args.reach_hold)
    if not rows:
        print(f"No npz files found under: {args.indir}")
        return

    csv_path = os.path.join(outdir, "noise_results_tidy.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    metrics = [
        ("cost", "Total Cost", False),
        ("rmse", "EE RMSE", False),
        ("sse", "Steady-state Error", True),
        ("success", "Success Rate", False),
        ("slack", "Slack Max", True),
        ("sigma", "Sigma min(A_g)", True),
        ("solve_ms", "Solve Time [ms]", False),
    ]

    modes = sorted(set(r["mode"] for r in rows))
    noises = sorted(set(r["noise"] for r in rows if np.isfinite(r["noise"])))

    # 1) Aggregated by mode vs noise
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), dpi=220)
    axs = axs.reshape(-1)
    for ax, (m, title, ylog) in zip(axs, metrics):
        for md in modes:
            ys = []
            for nz in noises:
                vals = np.array(
                    [r[m] for r in rows if r["mode"] == md and np.isfinite(r["noise"]) and abs(r["noise"] - nz) < 1e-12],
                    dtype=float,
                )
                vals = vals[np.isfinite(vals)]
                ys.append(float(np.median(vals)) if vals.size else np.nan)
            x = np.array(noises, dtype=float)
            y = np.array(ys, dtype=float)
            mk = np.isfinite(x) & np.isfinite(y)
            if np.any(mk):
                ax.plot(x[mk], y[mk], "-o", label=md)
        ax.set_title(title)
        ax.set_xlabel("measurement_noise_std")
        ax.grid(True, alpha=0.25)
        if m == "success":
            ax.set_ylim(-0.02, 1.02)
        if ylog:
            vals = []
            for line in ax.get_lines():
                vals.extend(line.get_ydata())
            vals = np.array(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size and np.all(vals > 0):
                ax.set_yscale("log")
        ax.legend(frameon=False)
    fig.suptitle("Noise Effect (Aggregated by Mode)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = os.path.join(outdir, "noise_vs_metrics_by_mode.png")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)

    # 2) Fixed-K: metric vs K by noise
    fixed = [r for r in rows if r["mode"] == "fixed_k" and np.isfinite(r["K"])]
    if fixed:
        fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), dpi=220)
        axs = axs.reshape(-1)
        ks = sorted(set(int(round(r["K"])) for r in fixed))
        for ax, (m, title, ylog) in zip(axs, metrics):
            for nz in noises:
                ys = []
                for k in ks:
                    vals = np.array(
                        [
                            r[m]
                            for r in fixed
                            if int(round(r["K"])) == k and np.isfinite(r["noise"]) and abs(r["noise"] - nz) < 1e-12
                        ],
                        dtype=float,
                    )
                    vals = vals[np.isfinite(vals)]
                    ys.append(float(np.median(vals)) if vals.size else np.nan)
                x = np.array(ks, dtype=float)
                y = np.array(ys, dtype=float)
                mk = np.isfinite(x) & np.isfinite(y)
                if np.any(mk):
                    ax.plot(x[mk], y[mk], "-o", label=f"noise={nz:g}")
            ax.set_title(title)
            ax.set_xlabel("K (fixed)")
            ax.grid(True, alpha=0.25)
            if m == "success":
                ax.set_ylim(-0.02, 1.02)
            if ylog:
                vals = []
                for line in ax.get_lines():
                    vals.extend(line.get_ydata())
                vals = np.array(vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size and np.all(vals > 0):
                    ax.set_yscale("log")
            ax.legend(frameon=False)
        fig.suptitle("Fixed-K: Metrics vs K under Measurement Noise", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        p2 = os.path.join(outdir, "fixedK_metrics_vs_K_by_noise.png")
        fig.savefig(p2, bbox_inches="tight")
        plt.close(fig)

        # Reach-time plot (fixed)
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8), dpi=220)
        for nz in noises:
            med, lo, hi = [], [], []
            for k in ks:
                vals = np.array(
                    [
                        r["reach_time"]
                        for r in fixed
                        if int(round(r["K"])) == k and np.isfinite(r["noise"]) and abs(r["noise"] - nz) < 1e-12
                    ],
                    dtype=float,
                )
                m, l, h = summarize(vals)
                med.append(m)
                lo.append(l)
                hi.append(h)
            x = np.array(ks, dtype=float)
            y = np.array(med, dtype=float)
            l = np.array(lo, dtype=float)
            h = np.array(hi, dtype=float)
            mk = np.isfinite(x) & np.isfinite(y)
            if np.any(mk):
                ax.plot(x[mk], y[mk], "-o", label=f"reach time noise={nz:g}")
                if np.any(np.isfinite(l[mk])) and np.any(np.isfinite(h[mk])):
                    ax.fill_between(x[mk], l[mk], h[mk], alpha=0.18)
        ax.set_title("Fixed-K: Reach Time vs K by Noise")
        ax.set_xlabel("K (fixed)")
        ax.set_ylabel("Reach time [step]")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        p2b = os.path.join(outdir, "fixedK_reach_time_vs_K_by_noise.png")
        fig.savefig(p2b, bbox_inches="tight")
        plt.close(fig)
    else:
        p2 = "(no fixed_k data)"
        p2b = "(no fixed_k data)"

    # 3) Adaptive-rho: metric vs rho by noise
    adap = [r for r in rows if r["mode"] == "adaptive_rho" and np.isfinite(r["rho"])]
    if adap:
        fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), dpi=220)
        axs = axs.reshape(-1)
        rhos = sorted(set(float(r["rho"]) for r in adap))
        for ax, (m, title, ylog) in zip(axs, metrics):
            for nz in noises:
                ys = []
                for rho in rhos:
                    vals = np.array(
                        [
                            r[m]
                            for r in adap
                            if abs(r["rho"] - rho) < 1e-12 and np.isfinite(r["noise"]) and abs(r["noise"] - nz) < 1e-12
                        ],
                        dtype=float,
                    )
                    vals = vals[np.isfinite(vals)]
                    ys.append(float(np.median(vals)) if vals.size else np.nan)
                x = np.array(rhos, dtype=float)
                y = np.array(ys, dtype=float)
                mk = np.isfinite(x) & np.isfinite(y)
                if np.any(mk):
                    ax.plot(x[mk], y[mk], "-o", label=f"noise={nz:g}")
            ax.set_title(title)
            ax.set_xlabel("rho (adaptive)")
            ax.grid(True, alpha=0.25)
            if m == "success":
                ax.set_ylim(-0.02, 1.02)
            if np.all(np.array(rhos) > 0):
                ax.set_xscale("log")
            if ylog:
                vals = []
                for line in ax.get_lines():
                    vals.extend(line.get_ydata())
                vals = np.array(vals, dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size and np.all(vals > 0):
                    ax.set_yscale("log")
            ax.legend(frameon=False)
        fig.suptitle("Adaptive-rho: Metrics vs rho under Measurement Noise", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        p3 = os.path.join(outdir, "adaptive_metrics_vs_rho_by_noise.png")
        fig.savefig(p3, bbox_inches="tight")
        plt.close(fig)

        # Reach-time plot (adaptive)
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8), dpi=220)
        for nz in noises:
            med, lo, hi = [], [], []
            for rho in rhos:
                vals = np.array(
                    [
                        r["reach_time"]
                        for r in adap
                        if abs(r["rho"] - rho) < 1e-12 and np.isfinite(r["noise"]) and abs(r["noise"] - nz) < 1e-12
                    ],
                    dtype=float,
                )
                m, l, h = summarize(vals)
                med.append(m)
                lo.append(l)
                hi.append(h)
            x = np.array(rhos, dtype=float)
            y = np.array(med, dtype=float)
            l = np.array(lo, dtype=float)
            h = np.array(hi, dtype=float)
            mk = np.isfinite(x) & np.isfinite(y)
            if np.any(mk):
                ax.plot(x[mk], y[mk], "-o", label=f"reach time noise={nz:g}")
                if np.any(np.isfinite(l[mk])) and np.any(np.isfinite(h[mk])):
                    ax.fill_between(x[mk], l[mk], h[mk], alpha=0.18)
        if np.all(np.array(rhos) > 0):
            ax.set_xscale("log")
        ax.set_title("Adaptive-rho: Reach Time vs rho by Noise")
        ax.set_xlabel("rho (adaptive)")
        ax.set_ylabel("Reach time [step]")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        p3b = os.path.join(outdir, "adaptive_reach_time_vs_rho_by_noise.png")
        fig.savefig(p3b, bbox_inches="tight")
        plt.close(fig)
    else:
        p3 = "(no adaptive_rho data)"
        p3b = "(no adaptive_rho data)"

    print(f"Saved: {csv_path}")
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    print(f"Saved: {p2b}")
    print(f"Saved: {p3}")
    print(f"Saved: {p3b}")
    print("counts mode/noise:")
    for md in modes:
        for nz in noises:
            n = sum(1 for r in rows if r["mode"] == md and np.isfinite(r["noise"]) and abs(r["noise"] - nz) < 1e-12)
            print(f"  {md} noise={nz:g} n={n}")


if __name__ == "__main__":
    main()
