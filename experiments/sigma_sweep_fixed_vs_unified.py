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
    summarize_by_method,
)
from wandb_utils import finish_wandb, log_rows_table, log_summary_dict, save_artifacts, setup_wandb  # noqa: E402


def parse_sigma_list(s: str):
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("sigma-list is empty")
    return vals


def print_sigma_summary(rows, sigma_vals, methods):
    print("\n[sigma-summary]")
    header = (
        f"{'sigma':>7} | {'method':>12} | {'success_rate':>12} | {'reached_rate':>12} | "
        f"{'fall_down_rate':>14} | {'mean_ttu_success':>16} | {'mean_avg_solve':>14} | "
        f"{'mean_of_mean_k':>14} | {'mean_of_max_k':>13}"
    )
    print(header)
    print("-" * len(header))
    for sigma in sigma_vals:
        sigma_rows = [r for r in rows if np.isclose(r["sigma_y"], sigma)]
        for method in methods:
            s = summarize_by_method(sigma_rows, method)
            print(
                f"{sigma:7.3f} | {method:12s} | {s['success_rate']:12.3f} | {s['success_reached_rate']:12.3f} | "
                f"{s['fall_down_rate']:14.3f} | {s['mean_ttu_success']:16.2f} | {s['mean_avg_solve_time']:14.4e} | "
                f"{s['mean_of_mean_k']:14.2f} | {s['mean_of_max_k']:13.2f}"
            )


def build_summary_series(rows, sigma_vals, methods, key):
    out = {m: [] for m in methods}
    for sigma in sigma_vals:
        sigma_rows = [r for r in rows if np.isclose(r["sigma_y"], sigma)]
        for method in methods:
            s = summarize_by_method(sigma_rows, method)
            out[method].append(float(s[key]))
    return out


def style_line_ax(ax, xvals, ylabel, title):
    ax.set_xlabel("sigma_y")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(xvals)
    ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.7)
    ax.legend(frameon=True)


def main():
    parser = argparse.ArgumentParser(
        description="Sigma sweep for fixedK40 vs unified_a001 (seed sweep per sigma)."
    )
    parser.add_argument("--sigma-list", type=str, default="0.02,0.05,0.08,0.10,0.12")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--t-sim", type=int, default=300)
    parser.add_argument("--ref-switch-step", type=int, default=100)
    parser.add_argument("--outdir", type=str, default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--traj-dir", type=str, default="")
    parser.add_argument(
        "--save-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-run trajectory/history npz files.",
    )

    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--local-quantile", type=float, default=0.05)
    parser.add_argument("--k-min", type=int, default=40)
    parser.add_argument("--k-max", type=int, default=200)
    parser.add_argument("--k-step", type=int, default=5)
    parser.add_argument("--local-min-cands", type=int, default=100)
    parser.add_argument("--local-max-cands", type=int, default=1000)

    parser.add_argument(
        "--offline-cache",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "cache", "offline_dataset_cache.pkl"),
    )
    parser.add_argument("--rebuild-offline", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cdc2026")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="sigma_sweep_fixed_vs_unified")
    parser.add_argument("--wandb-run-name", type=str, default="")
    args = parser.parse_args()

    sigma_vals = parse_sigma_list(args.sigma_list)
    seed_list = [int(args.seed0 + i) for i in range(int(args.seeds))]
    methods = ["fixedK40", "unified_a001"]

    check_runtime_dependencies()
    ensure_env_registered()
    os.makedirs(args.outdir, exist_ok=True)
    traj_dir = args.traj_dir if args.traj_dir else os.path.join(args.outdir, "sigma_sweep_traj")
    if args.save_trajectories:
        os.makedirs(traj_dir, exist_ok=True)

    trajectory_data = load_or_build_offline_dataset(
        cache_path=args.offline_cache,
        rebuild_offline=bool(args.rebuild_offline),
    )
    controller_args = setup_DeePC(trajectory_data)
    wb_run = setup_wandb(args, "sigma_sweep_fixed_vs_unified")

    rows = []
    print(
        "[config] "
        f"sigma_list={sigma_vals} seeds={len(seed_list)} seed0={args.seed0} "
        f"alpha={args.alpha} local_quantile={args.local_quantile} "
        f"local_min_cands={args.local_min_cands} local_max_cands={args.local_max_cands} "
        f"k_min={args.k_min} k_max={args.k_max} k_step={args.k_step}"
    )

    for sigma in sigma_vals:
        for seed in seed_list:
            method_cfgs = [
                {
                    "name": "fixedK40",
                    "k": 40,
                    "adaptive_k": False,
                    "local_gate": False,
                },
                {
                    "name": "unified_a001",
                    "k": int(args.k_min),
                    "adaptive_k": True,
                    "local_gate": True,
                },
            ]
            for mcfg in method_cfgs:
                ctl_args = SimpleNamespace(
                    k=int(mcfg["k"]),
                    adaptive_k=bool(mcfg["adaptive_k"]),
                    k_min=int(args.k_min),
                    k_max=int(args.k_max),
                    k_step=int(args.k_step),
                    alpha=float(args.alpha),
                    local_gate=bool(mcfg["local_gate"]),
                    local_quantile=float(args.local_quantile),
                    local_min_cands=int(args.local_min_cands),
                    local_max_cands=int(args.local_max_cands),
                )
                controller = build_controller(controller_args, ctl_args)
                ep = run_episode(
                    controller=controller,
                    sigma_y=float(sigma),
                    seed=int(seed),
                    t_sim=int(args.t_sim),
                    ref_switch_step=int(args.ref_switch_step),
                )
                clamp_reason = ""
                if mcfg["name"] == "unified_a001" and np.isfinite(ep["max_K_used"]):
                    if int(round(ep["max_K_used"])) >= int(args.k_max):
                        clamp_reason = "kmax_reached"
                    else:
                        clamp_reason = "none"

                rows.append(
                    {
                        "sigma_y": float(sigma),
                        "seed": int(seed),
                        "method": mcfg["name"],
                        "success": int(ep["success"]),
                        "success_reached": int(ep["success_reached"]),
                        "fall_down": int(ep["fall_down"]),
                        "time_to_upright": int(ep["time_to_upright"]),
                        "avg_solve_time": float(ep["avg_solve_time"]),
                        "mean_K_used": float(40.0 if mcfg["name"] == "fixedK40" else ep["mean_K_used"]),
                        "max_K_used": float(40.0 if mcfg["name"] == "fixedK40" else ep["max_K_used"]),
                        "certificate_rate": float(np.nan if mcfg["name"] == "fixedK40" else ep["certificate_rate"]),
                        "fail_step": int(ep["fail_step"]),
                        "mean_tau": float(np.nan if mcfg["name"] == "fixedK40" else ep["mean_tau"]),
                        "mean_d90": float(np.nan if mcfg["name"] == "fixedK40" else ep["mean_d90"]),
                        "mean_C_size": float(np.nan if mcfg["name"] == "fixedK40" else ep["mean_C_size"]),
                        "local_min_cands": int(args.local_min_cands),
                        "local_max_cands": int(args.local_max_cands),
                        "clamp_reason": clamp_reason,
                    }
                )
                if args.save_trajectories:
                    traj_path = os.path.join(
                        traj_dir,
                        f"traj_sigma{float(sigma):.3f}_seed{int(seed):03d}_{mcfg['name']}.npz",
                    )
                    np.savez(
                        traj_path,
                        sigma_y=np.array(float(sigma), dtype=float),
                        seed=np.array(int(seed), dtype=int),
                        method=np.array(mcfg["name"], dtype=object),
                        success=np.array(int(ep["success"]), dtype=int),
                        time_to_upright=np.array(int(ep["time_to_upright"]), dtype=int),
                        fail_step=np.array(int(ep["fail_step"]), dtype=int),
                        x=np.asarray(ep["x"], dtype=float),
                        theta=np.asarray(ep["theta"], dtype=float),
                        cos_theta=np.asarray(ep["cos_theta"], dtype=float),
                        x_ref=np.asarray(ep["x_ref"], dtype=float),
                        solve_times=np.asarray(ep["solve_times"], dtype=float),
                        K_used=np.asarray(ep["k_used_hist"], dtype=float),
                        met_gamma=np.asarray(ep["met_gamma_hist"], dtype=bool),
                        local_tau=np.asarray(ep["local_tau_hist"], dtype=float),
                        d_90=np.asarray(ep["d_90_hist"], dtype=float),
                        C_size=np.asarray(ep["c_size_hist"], dtype=float),
                    )
                print(
                    f"[sigma={sigma:.3f}][seed={seed:02d}][{mcfg['name']}] "
                    f"success={int(ep['success'])} reached={int(ep['success_reached'])} "
                    f"fall={int(ep['fall_down'])} ttu={int(ep['time_to_upright'])} "
                    f"fail={int(ep['fail_step'])} solve={float(ep['avg_solve_time']):.4e}s"
                )

    out_csv = os.path.join(args.outdir, "sigma_sweep.csv")
    csv_cols = [
        "sigma_y",
        "seed",
        "method",
        "success",
        "success_reached",
        "fall_down",
        "time_to_upright",
        "fail_step",
        "avg_solve_time",
        "mean_K_used",
        "max_K_used",
        "certificate_rate",
        "mean_tau",
        "mean_d90",
        "mean_C_size",
        "local_min_cands",
        "local_max_cands",
        "clamp_reason",
    ]
    save_csv(rows, out_csv, fieldnames=csv_cols)

    succ_series = build_summary_series(rows, sigma_vals, methods, "success_rate")
    reached_series = build_summary_series(rows, sigma_vals, methods, "success_reached_rate")
    fall_series = build_summary_series(rows, sigma_vals, methods, "fall_down_rate")
    time_series = build_summary_series(rows, sigma_vals, methods, "mean_avg_solve_time")

    # Plot 1: success rate vs sigma
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(
        sigma_vals,
        succ_series["fixedK40"],
        marker="o",
        markersize=6,
        linewidth=2.0,
        label="fixedK40",
        color="tab:blue",
    )
    ax1.plot(
        sigma_vals,
        succ_series["unified_a001"],
        marker="o",
        markersize=6,
        linewidth=2.0,
        label="unified_a001 (alpha=0.01)",
        color="tab:green",
    )
    ax1.set_ylim(0.0, 1.0)
    style_line_ax(ax1, sigma_vals, "success rate", "Success Rate vs Measurement Noise")
    for x, y in zip(sigma_vals, succ_series["fixedK40"]):
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    for x, y in zip(sigma_vals, succ_series["unified_a001"]):
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)
    fig1.tight_layout()
    out_success = os.path.join(args.outdir, "sigma_sweep_success.png")
    fig1.savefig(out_success, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: success_reached_rate vs sigma
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(
        sigma_vals,
        reached_series["fixedK40"],
        marker="o",
        markersize=6,
        linewidth=2.0,
        label="fixedK40",
        color="tab:blue",
    )
    ax2.plot(
        sigma_vals,
        reached_series["unified_a001"],
        marker="o",
        markersize=6,
        linewidth=2.0,
        label="unified_a001 (alpha=0.01)",
        color="tab:green",
    )
    ax2.set_ylim(0.0, 1.0)
    style_line_ax(ax2, sigma_vals, "success reached rate", "Upright-Reached Rate vs Measurement Noise")
    fig2.tight_layout()
    out_reached = os.path.join(args.outdir, "sigma_sweep_reached.png")
    fig2.savefig(out_reached, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # Plot 3: fall_down_rate vs sigma (conditioned on reached)
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(
        sigma_vals,
        fall_series["fixedK40"],
        marker="o",
        markersize=6,
        linewidth=2.0,
        label="fixedK40",
        color="tab:blue",
    )
    ax3.plot(
        sigma_vals,
        fall_series["unified_a001"],
        marker="o",
        markersize=6,
        linewidth=2.0,
        label="unified_a001 (alpha=0.01)",
        color="tab:green",
    )
    ax3.set_ylim(0.0, 1.0)
    style_line_ax(ax3, sigma_vals, "fall-down rate | reached", "Fall-Down Rate vs Measurement Noise")
    fig3.tight_layout()
    out_fall = os.path.join(args.outdir, "sigma_sweep_falldown.png")
    fig3.savefig(out_fall, dpi=200, bbox_inches="tight")
    plt.close(fig3)

    # Plot 4: solve time vs sigma
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.plot(
        sigma_vals,
        time_series["fixedK40"],
        marker="o",
        markersize=6,
        linewidth=2.0,
        label="fixedK40",
        color="tab:blue",
    )
    ax4.plot(
        sigma_vals,
        time_series["unified_a001"],
        marker="o",
        markersize=6,
        linewidth=2.0,
        label="unified_a001 (alpha=0.01)",
        color="tab:green",
    )
    style_line_ax(ax4, sigma_vals, "mean avg solve time [s]", "Solve Time vs Measurement Noise")
    fig4.tight_layout()
    out_time = os.path.join(args.outdir, "sigma_sweep_time.png")
    fig4.savefig(out_time, dpi=200, bbox_inches="tight")
    plt.close(fig4)

    print_sigma_summary(rows, sigma_vals, methods)
    print("\n[outputs]")
    print(f"CSV: {out_csv}")
    print(f"Plot(success): {out_success}")
    print(f"Plot(reached): {out_reached}")
    print(f"Plot(falldown): {out_fall}")
    print(f"Plot(time): {out_time}")
    if args.save_trajectories:
        print(f"Trajectories: {traj_dir}")

    log_rows_table(wb_run, "episodes", rows)
    summary_rows = []
    for sigma in sigma_vals:
        sigma_rows = [r for r in rows if np.isclose(r["sigma_y"], sigma)]
        for method in methods:
            s = summarize_by_method(sigma_rows, method)
            summary_rows.append({"sigma_y": float(sigma), "method": str(method), **{k: float(v) for k, v in s.items()}})
    log_rows_table(wb_run, "summary", summary_rows)
    log_summary_dict(wb_run, {"num_rows": int(len(rows)), "num_sigmas": int(len(sigma_vals))})
    save_artifacts(wb_run, [out_csv, out_success, out_reached, out_fall, out_time])
    finish_wandb(wb_run)


if __name__ == "__main__":
    main()
