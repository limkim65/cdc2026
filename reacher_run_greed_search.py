import argparse
import os
import sys

import numpy as np


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Reuse existing, validated setup helpers.
from experiments.reacher.run_cdc_reacher_benchmark import (  # noqa: E402
    ReacherSetpointScheduler,
    compute_tracking_error,
    compute_success,
    fix_dataset,
    get_cost_dict,
    setup_deepc_reacher,
)
from select_deepc.data_selectors import LkSelector  # noqa: E402
from select_deepc.data_selectors import AdaptiveLkSelector  # noqa: E402
from select_deepc.deepc_controller import SelectDeePC  # noqa: E402
from select_deepc.deepc_utils import (  # noqa: E402
    DeePCCostAccumulator,
    IntegralAbsoluteError,
    IntegralSquareError,
    PerformanceAccumulatorWrapper,
    get_reacher_simulator,
    load_data_from_folder,
    run_reacher_simulator,
)
from select_deepc.deepc_dataclasses import (  # noqa: E402
    DeePCConstraints,
    DeePCCost,
    DeePCControllerArgs,
    DeePCDims,
    TrajectoryDataSet,
)

def safe_stat(values, fn, default=np.nan):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return float(default)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(fn(arr))

def _format_run_tag(n_loc: int, sigma_bar: float, seed: int) -> str:
    sigma_txt = f"{float(sigma_bar):.4f}".rstrip("0").rstrip(".")
    sigma_txt = sigma_txt.replace(".", "p")
    return f"N{int(n_loc)}_sigma{sigma_txt}_seed{int(seed):03d}"


def make_env(outdir, max_steps, run_tag):
    outdir = outdir or os.path.join("logs", "reacher", "loca-PE-gate", "greed_search_dir")
    os.makedirs(outdir, exist_ok=True)

    video_dir = os.path.join(outdir, "video", run_tag)
    os.makedirs(video_dir, exist_ok=True)

    return get_reacher_simulator(
        num_steps=int(max_steps),
        video_folder=video_dir,   # None 금지
        video_title=f"reacher_{run_tag}",
    )


def main():
    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser(description="Greedy search for (N_loc, sigma_bar) on Reacher.")
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("logs", "reacher", "loca-PE-gate", "greed_search_dir"),
    )
    parser.add_argument("--dataset", choices=["iid", "random_walk", "both"], default="iid")
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--num_extra_targets", type=int, default=0)
    parser.add_argument("--success_eps", type=float, default=0.01)
    parser.add_argument("--success_window", type=int, default=20)
    # argparse



    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    seed_list = args.seeds if args.seeds is not None else [args.seed]

    # ---------------------------
    # Load datasets
    # ---------------------------
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

    # ---------------------------
    # DeePC config (same as your current manual config)
    # ---------------------------
    p = 8
    m = 2
    n = 75
    t_past = 2
    t_fut = 15
    t_hankel = (m + 1) * (t_past + t_fut) + n - 1

    q = np.diag([0, 0, 0, 0, 40000, 40000, 10, 10])
    r = np.diag([10, 10])
    controller_costs = DeePCCost(q, r, 5000000.0, 10.0, 0.0)

    enable_measurement_constraint = False
    a_u = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b_u = np.array([0.5, 0.5, 0.5, 0.5])
    a_y = None
    b_y = None
    if enable_measurement_constraint:
        a_y = np.array([[0, 0, 0, 0, 0, 1, 0, 0]])
        b_y = np.array([[0.1]])
    controller_constraints = DeePCConstraints(a_u, b_u, a_y, b_y)

    controller_args = DeePCControllerArgs(
        data,
        DeePCDims(2, 15, 8, 2),
        t_hankel,
        controller_costs,
        controller_constraints,
        [0, 0],
        False,
    )

    # ---------------------------
    # Search space (keep modest for 1-seed runs)
    # ---------------------------
    N_loc_values = [50, 100, 200, 500, 1000, 2000]
    sigma_bar_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    d_max = 10.0
    Kmin = 40
    Kmax = 10000

    # ---------------------------
    # Result logging (buffer in memory, save once at end)
    # ---------------------------
    csv_filename = os.path.join(args.outdir, "greedy_trial_results.csv")
    fieldnames = [
        "run_tag",
        "N_loc",
        "sigma_bar",
        "success",
        "rmse_ee",
        "total_cost",
        "solve_time_mean_ms",
        "solve_time_p95_ms",
        "slack_inf_max",
        "slack_inf_mean",
        "sigma_min_Hu_min",
        "sigma_min_Hu_mean",
        "sigma_min_Mk_min",
        "sigma_min_Mk_mean",
        "local_gate_fail",
        "cond_gate_fail",
        "Kopt_mean",
        "Kopt_p95",
    ]
    trial_results = []
    trial_id = 0

    # ---------------------------
    # Main loop
    # ---------------------------
    for seed in seed_list:
        for N_loc_trial in N_loc_values:
            for sigma_bar_trial in sigma_bar_values:
                trial_id += 1
                run_tag = _format_run_tag(N_loc_trial, sigma_bar_trial, seed)
                env = make_env(args.outdir, args.max_steps, run_tag)

                try:
                    deepc_trial = SelectDeePC(
                        controller_args,
                        selector_callback=AdaptiveLkSelector(order=2),
                        sigma_bar=float(sigma_bar_trial),
                        N_loc=int(N_loc_trial),
                        d_max=float(d_max),
                        n_iter=1,
                        K_min=int(Kmin),
                        K_max=int(Kmax),
                    )

                    scheduler_trial = ReacherSetpointScheduler(
                        env, controller_args.deepc_dims, num_extra_targets=int(args.num_extra_targets)
                    )

                    x_traj_trial, u_traj_trial, _, _, cost_obj_trial, _ = run_reacher_simulator(
                        env,
                        deepc_trial,
                        scheduler_trial,
                        seed=int(seed),
                        deepc_cost_accumulator=PerformanceAccumulatorWrapper(
                            DeePCCostAccumulator(controller_args.controller_costs),
                            IntegralSquareError(mask=np.array([0, 0, 0, 0, 1, 1, 0, 0])),
                            IntegralAbsoluteError(mask=np.array([0, 0, 0, 0, 1, 1, 0, 0])),
                        ),
                    )

                    # Metrics
                    target_xy = np.asarray(
                        scheduler_trial._targets.targets[scheduler_trial._target_idx], dtype=float
                    ).reshape(1, 2)
                    target_traj = np.repeat(target_xy, repeats=max(1, len(x_traj_trial)), axis=0)
                    err = compute_tracking_error(np.asarray(x_traj_trial, dtype=float), target_traj)
                    success = compute_success(err, eps=float(args.success_eps), window=int(args.success_window))
                    cost = float(get_cost_dict(cost_obj_trial).get("cost", np.nan))

                    solve_hist = np.asarray(getattr(deepc_trial, "_solve_time_ms_history", []), dtype=float)
                    slack_hist = np.asarray(getattr(deepc_trial, "_slack_norm_inf_history", []), dtype=float)
                    Kopt_hist = np.asarray(getattr(deepc_trial, "_K_opt_history", []), dtype=float)
                    sigma_hu_hist = np.asarray(getattr(deepc_trial, "_min_selected_sigma_history", []), dtype=float)
                    sigma_mk_hist = np.asarray(getattr(deepc_trial, "_sigma_min_Mk_history", []), dtype=float)

                    row = {
                        "run_tag": run_tag,
                        "N_loc": int(N_loc_trial),
                        "sigma_bar": float(sigma_bar_trial),
                        "success": int(success),
                        "rmse_ee": float(np.sqrt(safe_stat(err * err, np.mean, default=np.nan))),
                        "total_cost": float(cost),
                        "solve_time_mean_ms": float(np.nanmean(solve_hist)) if solve_hist.size else np.nan,
                        "solve_time_p95_ms": float(np.nanpercentile(solve_hist, 95)) if solve_hist.size else np.nan,
                        "slack_inf_max": float(np.nanmax(slack_hist)) if slack_hist.size else np.nan,
                        "slack_inf_mean": float(np.nanmean(slack_hist)) if slack_hist.size else np.nan,
                        "sigma_min_Hu_min": float(np.nanmin(sigma_hu_hist)) if sigma_hu_hist.size else np.nan,
                        "sigma_min_Hu_mean": float(np.nanmean(sigma_hu_hist)) if sigma_hu_hist.size else np.nan,
                        "sigma_min_Mk_min": float(np.nanmin(sigma_mk_hist)) if sigma_mk_hist.size else np.nan,
                        "sigma_min_Mk_mean": float(np.nanmean(sigma_mk_hist)) if sigma_mk_hist.size else np.nan,
                        "local_gate_fail": int(getattr(deepc_trial, "_local_gate_fail", 0)),
                        "cond_gate_fail": int(getattr(deepc_trial, "_cond_gate_fail", 0)),
                        "Kopt_mean": float(np.nanmean(Kopt_hist)) if Kopt_hist.size else np.nan,
                        "Kopt_p95": float(np.nanpercentile(Kopt_hist, 95)) if Kopt_hist.size else np.nan,
                    }
                except Exception as e:
                    row = {
                        "run_tag": run_tag,
                        "N_loc": int(N_loc_trial),
                        "sigma_bar": float(sigma_bar_trial),
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
                        "local_gate_fail": np.nan,
                        "cond_gate_fail": np.nan,
                        "Kopt_mean": np.nan,
                        "Kopt_p95": np.nan,
                    }
                    print(f"[{run_tag}] failed: {e}")
                finally:
                    env.close()

                trial_results.append(row)

                # Optional: print progress
                print(
                    f"[{run_tag}] N_loc={N_loc_trial:>6d}, sigma_bar={sigma_bar_trial:>4.2f} | "
                    f"succ={row['success']} rmse={row['rmse_ee']:.4g} "
                    f"t_mean={row['solve_time_mean_ms']:.3g}ms slack_max={row['slack_inf_max']:.3g} "
                    f"Kopt_mean={row['Kopt_mean']:.3g}"
                )

    if len(trial_results) == 0:
        print("No trials to plot.")
        return

    # Save once: CSV + NPZ snapshot.
    import csv
    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trial_results)
    np.savez_compressed(
        os.path.join(args.outdir, "greedy_trial_results.npz"),
        rows=np.array(trial_results, dtype=object),
    )

    # ---------------------------
    # Quick plots (scatter views)
    # ---------------------------
    sorted_results = sorted(trial_results, key=lambda x: (x["success"], -x["rmse_ee"]))
    costs = np.array([tr["total_cost"] for tr in trial_results], dtype=float)
    rmses = np.array([tr["rmse_ee"] for tr in trial_results], dtype=float)
    successes = np.array([tr["success"] for tr in trial_results], dtype=float)
    N_locs = np.array([tr["N_loc"] for tr in trial_results], dtype=float)
    sigma_bars = np.array([tr["sigma_bar"] for tr in trial_results], dtype=float)
    solve_time_means = np.array([tr["solve_time_mean_ms"] for tr in trial_results], dtype=float)
    slack_inf_maxs = np.array([tr["slack_inf_max"] for tr in trial_results], dtype=float)
    local_gate_fails = np.array([tr["local_gate_fail"] for tr in trial_results], dtype=float)
    cond_gate_fails = np.array([tr["cond_gate_fail"] for tr in trial_results], dtype=float)
    Kopt_means = np.array([tr["Kopt_mean"] for tr in trial_results], dtype=float)

    # 1) cost scatter
    plt.figure(figsize=(10, 4))
    plt.scatter(N_locs, costs)
    plt.xlabel("N_loc")
    plt.ylabel("Total Cost")
    plt.title("Total Cost vs N_loc")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "cost_vs_Nloc.png"), dpi=200)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.scatter(sigma_bars, costs)
    plt.xlabel("sigma_bar")
    plt.ylabel("Total Cost")
    plt.title("Total Cost vs sigma_bar")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "cost_vs_sigma_bar.png"), dpi=200)
    plt.show()

    # 2) RMSE vs cost
    plt.figure(figsize=(7, 5))
    plt.scatter(costs, rmses)
    plt.xlabel("Total Cost")
    plt.ylabel("RMSE (ee)")
    plt.title("RMSE vs Total Cost")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "rmse_vs_cost.png"), dpi=200)
    plt.show()

    # 3) Solve time vs cost
    plt.figure(figsize=(7, 5))
    plt.scatter(costs, solve_time_means)
    plt.xlabel("Total Cost")
    plt.ylabel("Mean Solve Time (ms)")
    plt.title("Solve Time vs Total Cost")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "solve_time_vs_cost.png"), dpi=200)
    plt.show()

    # 4) Slack max vs cost
    plt.figure(figsize=(7, 5))
    plt.scatter(costs, slack_inf_maxs)
    plt.xlabel("Total Cost")
    plt.ylabel("Max ||slack||_inf")
    plt.title("Slack Max vs Total Cost")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "slack_max_vs_cost.png"), dpi=200)
    plt.show()

    # 5) Success by trial index
    plt.figure(figsize=(10, 3))
    plt.plot(np.arange(len(successes)), successes, marker="o", linewidth=0)
    plt.ylim([-0.1, 1.1])
    plt.xlabel("Trial idx")
    plt.ylabel("Success (0/1)")
    plt.title("Success per Trial")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "success_per_trial.png"), dpi=200)
    plt.show()

    # 6) Gate fails vs cost
    plt.figure(figsize=(7, 5))
    plt.scatter(local_gate_fails, costs, label="local_gate_fail")
    plt.scatter(cond_gate_fails, costs, label="cond_gate_fail")
    plt.xlabel("Gate fail count")
    plt.ylabel("Total Cost")
    plt.title("Gate Fails vs Total Cost")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "gate_fails_vs_cost.png"), dpi=200)
    plt.show()

    # 7) Kopt_mean vs (N_loc, sigma_bar) as scatter
    plt.figure(figsize=(7, 5))
    plt.scatter(N_locs, Kopt_means)
    plt.xlabel("N_loc")
    plt.ylabel("Kopt_mean")
    plt.title("Kopt_mean vs N_loc")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "Kopt_mean_vs_Nloc.png"), dpi=200)
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.scatter(sigma_bars, Kopt_means)
    plt.xlabel("sigma_bar")
    plt.ylabel("Kopt_mean")
    plt.title("Kopt_mean vs sigma_bar")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "Kopt_mean_vs_sigma_bar.png"), dpi=200)
    plt.show()

    print(f"Saved CSV: {csv_filename}")
    print(f"Saved plots to: {args.outdir}")


if __name__ == "__main__":
    main()
