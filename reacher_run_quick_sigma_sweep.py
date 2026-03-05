import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from experiments.reacher.run_cdc_reacher_benchmark import (  # noqa: E402
    ReacherSetpointScheduler,
    compute_success,
    compute_tracking_error,
    fix_dataset,
    get_cost_dict,
)
from select_deepc.data_selectors import AdaptiveLkSelector  # noqa: E402
from select_deepc.deepc_controller import SelectDeePC  # noqa: E402
from select_deepc.deepc_dataclasses import (  # noqa: E402
    DeePCConstraints,
    DeePCCost,
    DeePCControllerArgs,
    DeePCDims,
)
from select_deepc.deepc_utils import (  # noqa: E402
    DeePCCostAccumulator,
    IntegralAbsoluteError,
    IntegralSquareError,
    PerformanceAccumulatorWrapper,
    get_reacher_simulator,
    load_data_from_folder,
    run_reacher_simulator,
)


def safe_stat(values, fn, default=np.nan):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return float(default)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(fn(arr))


def make_env(outdir: str, max_steps: int, run_tag: str):
    video_dir = os.path.join(outdir, "video", run_tag)
    os.makedirs(video_dir, exist_ok=True)
    return get_reacher_simulator(
        num_steps=int(max_steps),
        video_folder=video_dir,
        video_title=f"reacher_{run_tag}",
    )


def make_env_no_video(max_steps: int):
    from gymnasium.envs.registration import register, registry

    env_id = "Reacher-v4-custom"
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="gymnasium.envs.mujoco.reacher_v4:ReacherEnv",
            max_episode_steps=int(max_steps),
            reward_threshold=-3.75,
        )
    return gym.make(env_id, render_mode="rgb_array")


def format_run_tag(n_loc: int, sigma_bar: float, seed: int, mode: str) -> str:
    sigma_txt = f"{float(sigma_bar):.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{mode}_N{int(n_loc)}_sigma{sigma_txt}_seed{int(seed):03d}"


def build_controller_args(data):
    m = 2
    p = 8
    n = 75
    t_past = 2
    t_fut = 15
    t_hankel = (m + 1) * (t_past + t_fut) + n - 1

    q = np.diag([0, 0, 0, 0, 40000, 40000, 10, 10])
    r = np.diag([10, 10])
    controller_costs = DeePCCost(q, r, 5000000.0, 10.0, 0.0)

    a_u = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b_u = np.array([0.5, 0.5, 0.5, 0.5])
    controller_constraints = DeePCConstraints(a_u, b_u, None, None)

    return DeePCControllerArgs(
        data,
        DeePCDims(2, 15, 8, 2),
        t_hankel,
        controller_costs,
        controller_constraints,
        [0, 0],
        False,
    )


def main():
    parser = argparse.ArgumentParser(description="Quick sigma_bar sweep at fixed N_loc for 3 seeds.")
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("logs", "reacher", "quick_sigma_sweep_nloc500"),
    )
    parser.add_argument("--dataset", choices=["iid", "random_walk", "both"], default="iid")
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--N_loc", type=int, default=500)
    parser.add_argument(
        "--K",
        type=int,
        default=180,
        help="Base Hankel column count (num_hankel_cols). Fixed-K mode uses this exact value.",
    )
    parser.add_argument(
        "--adaptive_k",
        action="store_true",
        help="Enable adaptive-K mode. If omitted, runs fixed-K mode.",
    )
    parser.add_argument(
        "--sigma_bar_values",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8],
    )
    parser.add_argument("--d_max", type=float, default=10.0)
    parser.add_argument("--Kmin", type=int, default=40)
    parser.add_argument("--Kmax", type=int, default=10000)
    parser.add_argument("--num_extra_targets", type=int, default=0)
    parser.add_argument("--success_eps", type=float, default=0.01)
    parser.add_argument("--success_window", type=int, default=20)
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Enable video recording (default: off).",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    iid = load_data_from_folder(os.path.join(REPO_ROOT, "data", "reacher", "dataset_iid"), "iid")
    rw = load_data_from_folder(os.path.join(REPO_ROOT, "data", "reacher", "dataset_random_walk"), "random_walk")
    for ds in [iid, rw]:
        fix_dataset(ds)
    if args.dataset == "iid":
        data = iid
    elif args.dataset == "random_walk":
        data = rw
    else:
        data = iid + rw

    controller_args = build_controller_args(data)
    mode_tag = "adaptiveK" if bool(args.adaptive_k) else "fixedK"
    rows = []
    total_runs = len(args.seeds) * len(args.sigma_bar_values)
    run_idx = 0

    for seed in args.seeds:
        for sigma_bar in args.sigma_bar_values:
            run_idx += 1
            run_tag = format_run_tag(args.N_loc, sigma_bar, seed, mode_tag)
            if bool(args.save_video):
                env = make_env(args.outdir, args.max_steps, run_tag)
            else:
                env = make_env_no_video(args.max_steps)
            try:
                deepc = SelectDeePC(
                    controller_args,
                    selector_callback=AdaptiveLkSelector(order=2),
                    num_hankel_cols=int(args.K),
                    sigma_bar=float(sigma_bar),
                    N_loc=int(args.N_loc),
                    d_max=float(args.d_max),
                    n_iter=1,
                    adaptive_k=bool(args.adaptive_k),
                    K_min=int(args.Kmin),
                    K_max=int(args.Kmax),
                )
                scheduler = ReacherSetpointScheduler(
                    env, controller_args.deepc_dims, num_extra_targets=int(args.num_extra_targets)
                )
                x_traj, _, _, _, cost_obj, _ = run_reacher_simulator(
                    env,
                    deepc,
                    scheduler,
                    seed=int(seed),
                    deepc_cost_accumulator=PerformanceAccumulatorWrapper(
                        DeePCCostAccumulator(controller_args.controller_costs),
                        IntegralSquareError(mask=np.array([0, 0, 0, 0, 1, 1, 0, 0])),
                        IntegralAbsoluteError(mask=np.array([0, 0, 0, 0, 1, 1, 0, 0])),
                    ),
                )

                target_xy = np.asarray(
                    scheduler._targets.targets[scheduler._target_idx], dtype=float
                ).reshape(1, 2)
                target_traj = np.repeat(target_xy, repeats=max(1, len(x_traj)), axis=0)
                err = compute_tracking_error(np.asarray(x_traj, dtype=float), target_traj)

                solve_hist = np.asarray(getattr(deepc, "_solve_time_ms_history", []), dtype=float)
                slack_hist = np.asarray(getattr(deepc, "_slack_norm_inf_history", []), dtype=float)
                kopt_hist = np.asarray(getattr(deepc, "_K_opt_history", []), dtype=float)
                sigma_hu_hist = np.asarray(getattr(deepc, "_min_selected_sigma_history", []), dtype=float)
                sigma_mk_hist = np.asarray(getattr(deepc, "_sigma_min_Mk_history", []), dtype=float)

                row = {
                    "run_tag": run_tag,
                    "mode": mode_tag,
                    "adaptive_k": int(bool(args.adaptive_k)),
                    "seed": int(seed),
                    "N_loc": int(args.N_loc),
                    "K": int(args.K),
                    "sigma_bar": float(sigma_bar),
                    "success": int(
                        compute_success(err, eps=float(args.success_eps), window=int(args.success_window))
                    ),
                    "rmse_ee": float(np.sqrt(safe_stat(err * err, np.mean))),
                    "total_cost": float(get_cost_dict(cost_obj).get("cost", np.nan)),
                    "solve_time_mean_ms": safe_stat(solve_hist, np.mean),
                    "solve_time_p95_ms": safe_stat(solve_hist, lambda x: np.percentile(x, 95)),
                    "slack_inf_max": safe_stat(slack_hist, np.max),
                    "slack_inf_mean": safe_stat(slack_hist, np.mean),
                    "sigma_min_Hu_min": safe_stat(sigma_hu_hist, np.min),
                    "sigma_min_Hu_mean": safe_stat(sigma_hu_hist, np.mean),
                    "sigma_min_Mk_min": safe_stat(sigma_mk_hist, np.min),
                    "sigma_min_Mk_mean": safe_stat(sigma_mk_hist, np.mean),
                    "sigma_min_Mk_repr": safe_stat(sigma_mk_hist, np.median),
                    "local_gate_fail": int(getattr(deepc, "_local_gate_fail", 0)),
                    "cond_gate_fail": int(getattr(deepc, "_cond_gate_fail", 0)),
                    "Kopt_mean": safe_stat(kopt_hist, np.mean),
                    "Kopt_p95": safe_stat(kopt_hist, lambda x: np.percentile(x, 95)),
                    "error": "",
                }
            except Exception as e:
                row = {
                    "run_tag": run_tag,
                    "mode": mode_tag,
                    "adaptive_k": int(bool(args.adaptive_k)),
                    "seed": int(seed),
                    "N_loc": int(args.N_loc),
                    "K": int(args.K),
                    "sigma_bar": float(sigma_bar),
                    "success": 0,
                    "rmse_ee": np.nan,
                    "total_cost": np.nan,
                    "solve_time_mean_ms": np.nan,
                    "solve_time_p95_ms": np.nan,
                    "slack_inf_max": np.nan,
                    "slack_inf_mean": np.nan,
                    "sigma_min_Hu_min": np.nan,
                    "sigma_min_Hu_mean": np.nan,
                    "sigma_min_Mk_min": np.nan,
                    "sigma_min_Mk_mean": np.nan,
                    "sigma_min_Mk_repr": np.nan,
                    "local_gate_fail": np.nan,
                    "cond_gate_fail": np.nan,
                    "Kopt_mean": np.nan,
                    "Kopt_p95": np.nan,
                    "error": str(e),
                }
                print(f"[{run_idx}/{total_runs}] {run_tag} failed: {e}")
            finally:
                env.close()

            rows.append(row)
            print(
                f"[{run_idx}/{total_runs}] {run_tag} "
                f"cost={row['total_cost']:.4g} rmse={row['rmse_ee']:.4g} "
                f"solve={row['solve_time_mean_ms']:.2f}ms"
            )

    csv_path = os.path.join(args.outdir, "quick_sigma_sweep_runs.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    num_cols = [
        "success",
        "rmse_ee",
        "total_cost",
        "solve_time_mean_ms",
        "slack_inf_max",
        "Kopt_mean",
        "sigma_min_Hu_min",
        "sigma_min_Mk_repr",
    ]
    agg = df.groupby("sigma_bar", as_index=False)[num_cols].agg(["mean", "std"])
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.to_flat_index()]
    summary_path = os.path.join(args.outdir, "quick_sigma_sweep_summary_by_sigma.csv")
    agg.to_csv(summary_path, index=False)

    # quick plot (mean +- std over seeds)
    plot_metrics = [
        ("total_cost", "cost"),
        ("rmse_ee", "rmse"),
        ("solve_time_mean_ms", "solve_time_ms"),
        ("slack_inf_max", "slack_inf_max"),
        ("Kopt_mean", "Kopt_mean"),
        ("sigma_min_Mk_repr", "sigma_min_Mk_repr"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    x = agg["sigma_bar"].to_numpy(dtype=float)
    for ax, (col, ylab) in zip(axes, plot_metrics):
        m = agg[f"{col}_mean"].to_numpy(dtype=float)
        s = agg[f"{col}_std"].to_numpy(dtype=float)
        ax.plot(x, m, marker="o", linewidth=2)
        ax.fill_between(x, m - s, m + s, alpha=0.2)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("sigma_bar")
    axes[-2].set_xlabel("sigma_bar")
    fig.suptitle(
        f"Quick Sigma Sweep ({mode_tag}) @ N_loc={args.N_loc}, K={args.K} (seeds={args.seeds})",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "quick_sigma_sweep_summary_plots.png"), dpi=220)
    plt.close(fig)

    print(f"Saved runs: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {os.path.join(args.outdir, 'quick_sigma_sweep_summary_plots.png')}")


if __name__ == "__main__":
    main()
