import argparse
import csv
import os
import pickle
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CUSTOM_DIR = os.path.join(REPO_ROOT, "custom")
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
if CUSTOM_DIR not in sys.path:
    sys.path.append(CUSTOM_DIR)

from custom.offline_hankel_collection import (  # noqa: E402
    collect_closedloop_controller_data,
    create_trajectory_dataset,
)
from custom.setup_deepc import setup_DeePC  # noqa: E402
from realtime_verify_run import (  # noqa: E402
    build_controller,
    check_runtime_dependencies,
    ensure_env_registered,
    featurize_obs,
)
from wandb_utils import finish_wandb, log_rows_table, log_summary_dict, save_artifacts, setup_wandb  # noqa: E402


N_UP = 20
N_FAIL = 20


@dataclass
class MethodCfg:
    name: str
    k: int
    adaptive_k: bool
    local_gate: bool
    alpha: float
    local_quantile: float
    k_min: int
    k_max: int
    k_step: int
    local_min_cands: int
    local_max_cands: int


def evaluate_upright_and_fail(cos_theta: np.ndarray, n_up: int, n_fail: int):
    success_reached = False
    up_count = 0
    down_count = 0
    upright_step = None
    fail_step = None

    for k, c in enumerate(cos_theta):
        c = float(c)
        if not success_reached:
            if c > 0.95:
                up_count += 1
                if up_count >= n_up:
                    success_reached = True
                    upright_step = int(k)
            else:
                up_count = 0
            continue

        if c < 0.5:
            down_count += 1
            if down_count >= n_fail:
                fail_step = int(k - n_fail + 1)
                break
        else:
            down_count = 0

    final_success = bool(success_reached and fail_step is None)
    return final_success, success_reached, upright_step, fail_step


def load_or_build_offline_dataset(cache_path: str, rebuild_offline: bool):
    if (not rebuild_offline) and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    _, reset_time, u_off, y_off = collect_closedloop_controller_data(animation=False)
    trajectory_data = create_trajectory_dataset(reset_time, u_off, y_off, "swingup_data")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(trajectory_data, f)
    return trajectory_data


def run_episode(controller, sigma_y: float, seed: int, t_sim: int, ref_switch_step: int):
    env = gym.make("InvertedPendulum-v4-swingup")
    obs_raw, _ = env.reset(seed=seed)
    rng_noise = np.random.default_rng(seed + 1000)

    traj_true = []
    x_ref_hist = []
    cos_theta = []
    solve_times = []
    for step in range(t_sim):
        obs_true = featurize_obs(np.asarray(obs_raw, dtype=float))
        obs_meas = obs_true + rng_noise.normal(0.0, sigma_y, size=obs_true.shape)

        x_ref = 0.0 if step <= ref_switch_step else 0.5
        ref = np.array([x_ref, 0.0, 1.0, 0.0, 0.0], dtype=float)

        t0 = time.perf_counter()
        action = controller.compute_action(obs_meas, ref)
        solve_times.append(time.perf_counter() - t0)

        obs_raw, _, terminated, truncated, _ = env.step(action)
        traj_true.append(obs_true)
        x_ref_hist.append(float(x_ref))
        cos_theta.append(float(np.cos(obs_raw[1])))
        if terminated or truncated:
            break

    env.close()
    traj_true = np.asarray(traj_true, dtype=float)
    x_ref_hist = np.asarray(x_ref_hist, dtype=float)
    theta = np.arctan2(traj_true[:, 1], traj_true[:, 2]) if traj_true.size > 0 else np.array([])
    cos_theta = np.asarray(cos_theta, dtype=float)
    steps_done = int(cos_theta.size)

    final_success, success_reached, upright_step, fail_step = evaluate_upright_and_fail(
        cos_theta=cos_theta,
        n_up=N_UP,
        n_fail=N_FAIL,
    )

    k_used_hist = np.asarray(controller.get_k_used_history(), dtype=float)
    if k_used_hist.size == 0:
        k_used_hist = np.full(steps_done, np.nan, dtype=float)
    met_gamma_hist = np.asarray(controller.get_met_gamma_history(), dtype=bool)
    local_tau_hist = np.asarray(controller.get_local_tau_history(), dtype=float)
    d_90_hist = np.asarray(controller.get_dist_p90_history(), dtype=float)
    c_size_hist = np.asarray(controller.get_local_pool_size_history(), dtype=float)

    fall_down = bool(success_reached and (fail_step is not None))

    return {
        "traj_true": traj_true,
        "x": traj_true[:, 0] if traj_true.size > 0 else np.array([]),
        "theta": theta,
        "cos_theta": cos_theta,
        "x_ref": x_ref_hist,
        "k_used_hist": k_used_hist,
        "met_gamma_hist": met_gamma_hist,
        "local_tau_hist": local_tau_hist,
        "d_90_hist": d_90_hist,
        "c_size_hist": c_size_hist,
        "solve_times": np.asarray(solve_times, dtype=float),
        "success": int(final_success),
        "success_reached": int(success_reached),
        "fall_down": int(fall_down),
        "time_to_upright": int(upright_step) if upright_step is not None else int(t_sim),
        "fail_step": int(fail_step) if fail_step is not None else -1,
        "avg_solve_time": float(np.mean(solve_times)) if solve_times else np.nan,
        "mean_K_used": float(np.nanmean(k_used_hist)) if k_used_hist.size > 0 else np.nan,
        "max_K_used": float(np.nanmax(k_used_hist)) if k_used_hist.size > 0 else np.nan,
        "certificate_rate": float(np.mean(met_gamma_hist.astype(float))) if met_gamma_hist.size > 0 else np.nan,
        "mean_tau": float(np.nanmean(local_tau_hist)) if local_tau_hist.size > 0 else np.nan,
        "mean_d90": float(np.nanmean(d_90_hist)) if d_90_hist.size > 0 else np.nan,
        "mean_C_size": float(np.nanmean(c_size_hist)) if c_size_hist.size > 0 else np.nan,
    }


def summarize_by_method(rows: List[Dict[str, float]], method: str):
    method_rows = [r for r in rows if r["method"] == method]
    success = np.array([r["success"] for r in method_rows], dtype=float)
    reached = np.array([r.get("success_reached", np.nan) for r in method_rows], dtype=float)
    fall = np.array([r.get("fall_down", np.nan) for r in method_rows], dtype=float)
    ttu = np.array([r["time_to_upright"] for r in method_rows], dtype=float)
    avg_solve = np.array([r["avg_solve_time"] for r in method_rows], dtype=float)
    mean_k = np.array([r["mean_K_used"] for r in method_rows], dtype=float)
    max_k = np.array([r["max_K_used"] for r in method_rows], dtype=float)
    cert = np.array([r["certificate_rate"] for r in method_rows], dtype=float)
    success_mask = success > 0.5
    ttu_succ = ttu[success_mask]
    reached_mask = reached > 0.5
    fall_reached = fall[reached_mask]
    return {
        "success_rate": float(np.mean(success)) if success.size > 0 else np.nan,
        "success_reached_rate": float(np.nanmean(reached)) if reached.size > 0 else np.nan,
        "fall_down_rate": float(np.nanmean(fall_reached)) if fall_reached.size > 0 else np.nan,
        "mean_ttu_success": float(np.mean(ttu_succ)) if ttu_succ.size > 0 else np.nan,
        "mean_avg_solve_time": float(np.nanmean(avg_solve)) if avg_solve.size > 0 else np.nan,
        "mean_of_mean_k": float(np.nanmean(mean_k)) if mean_k.size > 0 else np.nan,
        "mean_of_max_k": float(np.nanmean(max_k)) if max_k.size > 0 else np.nan,
        "mean_certificate_rate": float(np.nanmean(cert)) if cert.size > 0 else np.nan,
    }


def save_csv(rows: List[Dict[str, float]], out_csv: str, fieldnames: Optional[List[str]] = None):
    if fieldnames is None:
        fieldnames = [
            "seed",
            "method",
            "success",
            "time_to_upright",
            "avg_solve_time",
            "mean_K_used",
            "max_K_used",
            "certificate_rate",
            "fail_step",
            "mean_tau",
            "mean_d90",
            "mean_C_size",
        ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def make_plots(rows: List[Dict[str, float]], outdir: str):
    methods = ["fixedK40", "unified_a001"]
    summaries = {m: summarize_by_method(rows, m) for m in methods}

    # Plot 1: success rate bar
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    x = np.arange(len(methods))
    vals = [summaries[m]["success_rate"] for m in methods]
    ax1.bar(x, vals, color=["tab:blue", "tab:green"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("success rate")
    ax1.grid(True, alpha=0.3, axis="y")
    fig1.tight_layout()
    out1 = os.path.join(outdir, "seed_sweep_success_rate.png")
    fig1.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: solve-time vs success trade-off (avg solve bar + success line)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    solve_vals = [summaries[m]["mean_avg_solve_time"] for m in methods]
    bars = ax2.bar(x, solve_vals, color=["tab:blue", "tab:green"], alpha=0.85, label="avg solve time/step")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel("avg solve time per step [s]")
    ax2.grid(True, alpha=0.3, axis="y")

    ax2b = ax2.twinx()
    succ_vals = [summaries[m]["success_rate"] for m in methods]
    ax2b.plot(x, succ_vals, color="tab:red", marker="o", linewidth=1.8, label="success rate")
    ax2b.set_ylim(0.0, 1.0)
    ax2b.set_ylabel("success rate")

    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    for bar in bars:
        h = bar.get_height()
        if np.isfinite(h):
            ax2.text(bar.get_x() + bar.get_width() * 0.5, h, f"{h:.3e}", ha="center", va="bottom", fontsize=8)
    fig2.tight_layout()
    out2 = os.path.join(outdir, "seed_sweep_time_tradeoff.png")
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    return out1, out2


def main():
    parser = argparse.ArgumentParser(description="Seed sweep at sigma_y=0.05: fixed-K40 vs unified.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of sequential seeds.")
    parser.add_argument("--seed0", type=int, default=0, help="Starting seed.")
    parser.add_argument("--sigma-y", type=float, default=0.05)
    parser.add_argument("--t-sim", type=int, default=300)
    parser.add_argument("--ref-switch-step", type=int, default=100)
    parser.add_argument("--outdir", type=str, default=os.path.join(REPO_ROOT, "results"))

    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--local-quantile", type=float, default=0.05)
    parser.add_argument("--local-min-cands", type=int, default=100)
    parser.add_argument("--local-max-cands", type=int, default=1000)
    parser.add_argument("--k-min", type=int, default=40)
    parser.add_argument("--k-max", type=int, default=200)
    parser.add_argument("--k-step", type=int, default=5)

    parser.add_argument(
        "--offline-cache",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "cache", "offline_dataset_cache.pkl"),
    )
    parser.add_argument("--rebuild-offline", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cdc2026")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="seed_sweep_sigma005")
    parser.add_argument("--wandb-run-name", type=str, default="")
    args = parser.parse_args()

    check_runtime_dependencies()
    ensure_env_registered()
    os.makedirs(args.outdir, exist_ok=True)

    trajectory_data = load_or_build_offline_dataset(
        cache_path=args.offline_cache,
        rebuild_offline=bool(args.rebuild_offline),
    )
    controller_args = setup_DeePC(trajectory_data)
    wb_run = setup_wandb(args, "seed_sweep_sigma005")

    methods = [
        MethodCfg(
            name="fixedK40",
            k=40,
            adaptive_k=False,
            local_gate=False,
            alpha=float(args.alpha),
            local_quantile=float(args.local_quantile),
            k_min=int(args.k_min),
            k_max=int(args.k_max),
            k_step=int(args.k_step),
            local_min_cands=int(args.local_min_cands),
            local_max_cands=int(args.local_max_cands),
        ),
        MethodCfg(
            name="unified_a001",
            k=int(args.k_min),
            adaptive_k=True,
            local_gate=True,
            alpha=float(args.alpha),
            local_quantile=float(args.local_quantile),
            k_min=int(args.k_min),
            k_max=int(args.k_max),
            k_step=int(args.k_step),
            local_min_cands=int(args.local_min_cands),
            local_max_cands=int(args.local_max_cands),
        ),
    ]

    seed_list = [int(args.seed0 + i) for i in range(int(args.seeds))]
    rows: List[Dict[str, float]] = []
    print(
        "[config] "
        f"seeds={len(seed_list)} seed0={args.seed0} sigma_y={args.sigma_y} "
        f"alpha={args.alpha} local_quantile={args.local_quantile} "
        f"local_min_cands={args.local_min_cands} local_max_cands={args.local_max_cands} "
        f"k_min={args.k_min} k_max={args.k_max} k_step={args.k_step}"
    )

    for seed in seed_list:
        for cfg in methods:
            ctl_args = SimpleNamespace(
                k=cfg.k,
                adaptive_k=cfg.adaptive_k,
                k_min=cfg.k_min,
                k_max=cfg.k_max,
                k_step=cfg.k_step,
                alpha=cfg.alpha,
                local_gate=cfg.local_gate,
                local_quantile=cfg.local_quantile,
                local_min_cands=cfg.local_min_cands,
                local_max_cands=cfg.local_max_cands,
            )
            controller = build_controller(controller_args, ctl_args)
            ep = run_episode(
                controller=controller,
                sigma_y=float(args.sigma_y),
                seed=seed,
                t_sim=int(args.t_sim),
                ref_switch_step=int(args.ref_switch_step),
            )

            row = {
                "seed": int(seed),
                "method": cfg.name,
                "success": int(ep["success"]),
                "time_to_upright": int(ep["time_to_upright"]),
                "avg_solve_time": float(ep["avg_solve_time"]),
                "mean_K_used": float(40.0 if cfg.name == "fixedK40" else ep["mean_K_used"]),
                "max_K_used": float(40.0 if cfg.name == "fixedK40" else ep["max_K_used"]),
                "certificate_rate": float(np.nan if cfg.name == "fixedK40" else ep["certificate_rate"]),
                "fail_step": int(ep["fail_step"]),
                "mean_tau": float(np.nan if cfg.name == "fixedK40" else ep["mean_tau"]),
                "mean_d90": float(np.nan if cfg.name == "fixedK40" else ep["mean_d90"]),
                "mean_C_size": float(np.nan if cfg.name == "fixedK40" else ep["mean_C_size"]),
            }
            rows.append(row)
            print(
                f"[seed={seed:02d}][{cfg.name}] success={row['success']} "
                f"ttu={row['time_to_upright']} fail_step={row['fail_step']} "
                f"solve={row['avg_solve_time']:.4e}s meanK={row['mean_K_used']:.2f}"
            )

    out_csv = os.path.join(args.outdir, "seed_sweep_sigma005.csv")
    save_csv(rows, out_csv)
    out_success_png, out_tradeoff_png = make_plots(rows, args.outdir)

    print("\n[summary]")
    print(
        f"local_min_cands={args.local_min_cands}, "
        f"local_max_cands={args.local_max_cands}"
    )
    summary_rows = []
    for cfg in methods:
        s = summarize_by_method(rows, cfg.name)
        summary_rows.append({"method": str(cfg.name), **{k: float(v) for k, v in s.items()}})
        print(
            f"{cfg.name:12s} success_rate={s['success_rate']:.3f} "
            f"mean_ttu_success={s['mean_ttu_success']:.2f} "
            f"avg_solve_time={s['mean_avg_solve_time']:.4e}s "
            f"meanK={s['mean_of_mean_k']:.2f} "
            f"meanMaxK={s['mean_of_max_k']:.2f} "
            f"cert_rate={s['mean_certificate_rate']:.3f}"
        )

    print("\n[outputs]")
    print(f"CSV: {out_csv}")
    print(f"Plot(success rate): {out_success_png}")
    print(f"Plot(time tradeoff): {out_tradeoff_png}")

    log_rows_table(wb_run, "episodes", rows)
    log_rows_table(wb_run, "summary", summary_rows)
    log_summary_dict(wb_run, {"num_rows": int(len(rows)), "num_seeds": int(len(seed_list))})
    save_artifacts(wb_run, [out_csv, out_success_png, out_tradeoff_png])
    finish_wandb(wb_run)


if __name__ == "__main__":
    main()
