import argparse
import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from custom.setup_deepc import setup_DeePC  # noqa: E402
from seed_sweep_sigma005 import (  # noqa: E402
    build_controller,
    check_runtime_dependencies,
    ensure_env_registered,
    load_or_build_offline_dataset,
    run_episode,
    save_csv,
)
from wandb_utils import finish_wandb, log_rows_table, log_summary_dict, save_artifacts, setup_wandb  # noqa: E402


def parse_alpha_list(s: str):
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    if not vals:
        raise ValueError("alpha-list is empty")
    return vals


def summarize_alpha(rows, alpha):
    alpha_rows = [r for r in rows if np.isclose(r["alpha"], alpha)]
    success = np.array([r["success"] for r in alpha_rows], dtype=float)
    ttu = np.array([r["time_to_upright"] for r in alpha_rows], dtype=float)
    solve = np.array([r["avg_solve_time"] for r in alpha_rows], dtype=float)
    mean_k = np.array([r["mean_K_used"] for r in alpha_rows], dtype=float)
    max_k = np.array([r["max_K_used"] for r in alpha_rows], dtype=float)
    cert = np.array([r["certificate_rate"] for r in alpha_rows], dtype=float)
    succ_mask = success > 0.5
    ttu_succ = ttu[succ_mask]
    return {
        "success_rate": float(np.mean(success)) if success.size > 0 else np.nan,
        "mean_ttu_success": float(np.mean(ttu_succ)) if ttu_succ.size > 0 else np.nan,
        "mean_avg_solve_time": float(np.nanmean(solve)) if solve.size > 0 else np.nan,
        "mean_of_mean_K": float(np.nanmean(mean_k)) if mean_k.size > 0 else np.nan,
        "mean_of_max_K": float(np.nanmean(max_k)) if max_k.size > 0 else np.nan,
        "mean_certificate_rate": float(np.nanmean(cert)) if cert.size > 0 else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="Alpha sensitivity sweep for unified selection at sigma_y=0.05.")
    parser.add_argument("--sigma-y", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--alpha-list", type=str, default="0.005,0.01,0.02")
    parser.add_argument("--local-quantile", type=float, default=0.05)
    parser.add_argument("--k-min", type=int, default=40)
    parser.add_argument("--k-max", type=int, default=200)
    parser.add_argument("--k-step", type=int, default=5)
    parser.add_argument("--local-min-cands", type=int, default=100)
    parser.add_argument("--local-max-cands", type=int, default=1000)
    parser.add_argument("--t-sim", type=int, default=300)
    parser.add_argument("--ref-switch-step", type=int, default=100)
    parser.add_argument("--outdir", type=str, default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument(
        "--offline-cache",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "cache", "offline_dataset_cache.pkl"),
    )
    parser.add_argument("--rebuild-offline", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cdc2026")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="alpha_sensitivity_sigma005")
    parser.add_argument("--wandb-run-name", type=str, default="")
    args = parser.parse_args()

    alpha_vals = parse_alpha_list(args.alpha_list)
    seeds = [int(args.seed0 + i) for i in range(int(args.seeds))]

    check_runtime_dependencies()
    ensure_env_registered()
    os.makedirs(args.outdir, exist_ok=True)

    trajectory_data = load_or_build_offline_dataset(
        cache_path=args.offline_cache,
        rebuild_offline=bool(args.rebuild_offline),
    )
    controller_args = setup_DeePC(trajectory_data)
    wb_run = setup_wandb(args, "alpha_sensitivity_sigma005")

    rows = []
    print(
        "[config] "
        f"sigma_y={args.sigma_y} alphas={alpha_vals} seeds={len(seeds)} "
        f"k_min={args.k_min} k_max={args.k_max} k_step={args.k_step} "
        f"local_quantile={args.local_quantile}"
    )

    for alpha in alpha_vals:
        for seed in seeds:
            ctl_args = SimpleNamespace(
                k=int(args.k_min),
                adaptive_k=True,
                k_min=int(args.k_min),
                k_max=int(args.k_max),
                k_step=int(args.k_step),
                alpha=float(alpha),
                local_gate=True,
                local_quantile=float(args.local_quantile),
                local_min_cands=int(args.local_min_cands),
                local_max_cands=int(args.local_max_cands),
            )
            controller = build_controller(controller_args, ctl_args)
            ep = run_episode(
                controller=controller,
                sigma_y=float(args.sigma_y),
                seed=int(seed),
                t_sim=int(args.t_sim),
                ref_switch_step=int(args.ref_switch_step),
            )
            row = {
                "alpha": float(alpha),
                "seed": int(seed),
                "success": int(ep["success"]),
                "time_to_upright": int(ep["time_to_upright"]),
                "avg_solve_time": float(ep["avg_solve_time"]),
                "mean_K_used": float(ep["mean_K_used"]),
                "max_K_used": float(ep["max_K_used"]),
                "certificate_rate": float(ep["certificate_rate"]),
            }
            rows.append(row)
            print(
                f"[alpha={alpha:.4f}][seed={seed:02d}] success={row['success']} "
                f"ttu={row['time_to_upright']} solve={row['avg_solve_time']:.4e}s "
                f"meanK={row['mean_K_used']:.2f}"
            )

    out_csv = os.path.join(args.outdir, "alpha_sensitivity_sigma005.csv")
    cols = [
        "alpha",
        "seed",
        "success",
        "time_to_upright",
        "avg_solve_time",
        "mean_K_used",
        "max_K_used",
        "certificate_rate",
    ]
    save_csv(rows, out_csv, fieldnames=cols)

    summaries = {a: summarize_alpha(rows, a) for a in alpha_vals}
    print("\n[alpha-summary]")
    header = (
        f"{'alpha':>7} | {'success_rate':>12} | {'mean_ttu_success':>16} | "
        f"{'mean_avg_solve':>14} | {'mean_K':>10} | {'mean_maxK':>10} | {'cert_rate':>10}"
    )
    print(header)
    print("-" * len(header))
    for a in alpha_vals:
        s = summaries[a]
        print(
            f"{a:7.4f} | {s['success_rate']:12.3f} | {s['mean_ttu_success']:16.2f} | "
            f"{s['mean_avg_solve_time']:14.4e} | {s['mean_of_mean_K']:10.2f} | "
            f"{s['mean_of_max_K']:10.2f} | {s['mean_certificate_rate']:10.3f}"
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    y = [summaries[a]["success_rate"] for a in alpha_vals]
    ax.plot(alpha_vals, y, marker="o", linewidth=2.0, color="tab:green", label="unified_a(alpha)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("success_rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Alpha Sensitivity at sigma_y=0.05")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    out_png = os.path.join(args.outdir, "alpha_sensitivity_success.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n[outputs]")
    print(f"CSV: {out_csv}")
    print(f"Plot: {out_png}")

    log_rows_table(wb_run, "episodes", rows)
    summary_rows = []
    for a in alpha_vals:
        s = summaries[a]
        summary_rows.append({"alpha": float(a), **{k: float(v) for k, v in s.items()}})
    log_rows_table(wb_run, "summary", summary_rows)
    log_summary_dict(wb_run, {"num_rows": int(len(rows)), "num_alphas": int(len(alpha_vals))})
    save_artifacts(wb_run, [out_csv, out_png])
    finish_wandb(wb_run)


if __name__ == "__main__":
    main()
