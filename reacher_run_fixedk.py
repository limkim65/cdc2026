import argparse
import os
import sys

import numpy as np
import pandas as pd


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
from select_deepc.deepc_controller import AdaptiveSelectDeePC  # noqa: E402
from select_deepc.deepc_dataclasses import DeePCControllerArgs, DeePCDims, DeePCConstraints, DeePCCost
from select_deepc.deepc_utils import (  # noqa: E402
    DeePCCostAccumulator,
    IntegralAbsoluteError,
    IntegralSquareError,
    PerformanceAccumulatorWrapper,
    get_reacher_simulator,
    load_data_from_folder,
    run_reacher_simulator,
)




def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed-k Select-DeePC sweep.")
    parser.add_argument("--outdir", type=str, default="logs/reacher/fixed_k")
    parser.add_argument(
        "--dataset",
        type=str,
        default="iid",
        choices=["iid", "random_walk", "mixed"],
    )
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seedlist", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--k_start", type=int, default=10)
    parser.add_argument("--k_stop", type=int, default=100)
    parser.add_argument("--k_step", type=int, default=10)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--enable_measurement_constraint", action="store_true")
    parser.add_argument("--num_extra_targets", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    outdir = str(args.outdir)
    dataset = str(args.dataset)
    max_steps = int(args.max_steps)
    seedlist = list(args.seedlist)
    klist = list(range(int(args.k_start), int(args.k_stop), int(args.k_step)))
    record_video = bool(args.record_video)
    enable_measurement_constraint = bool(args.enable_measurement_constraint)
    results = []
    num_extra_targets = int(args.num_extra_targets)

    # Resolve dataset-aware output directory.
    # 1) If user passed a template path containing "{dataset}", format it.
    # 2) Otherwise append "dataset_<name>" once.
    if "{dataset}" in outdir:
        outdir = outdir.format(dataset=dataset)
    else:
        dataset_tag = f"dataset_{dataset}"
        norm_tail = os.path.basename(os.path.normpath(outdir))
        if norm_tail != dataset_tag:
            outdir = os.path.join(outdir, dataset_tag)

    os.makedirs(outdir, exist_ok=True)

    iid = load_data_from_folder(os.path.join(REPO_ROOT, "data", "reacher", "dataset_iid"), "iid")
    rw = load_data_from_folder(
        os.path.join(REPO_ROOT, "data", "reacher", "dataset_random_walk"), "random_walk"
    )
    for ds in [iid, rw]:
        fix_dataset(ds)

    if dataset == "iid":
        data = iid
    elif dataset == "random_walk":
        data = rw
    else:
        data = iid + rw

    #deepc arguments setup
    p = 8
    m = 2
    n = 75 # estimated model size
    t_past = 2
    t_fut = 15
    t_hankel = (m + 1) * (t_past + t_fut) + n - 1

    q = np.diag([0, 0, 0, 0, 40000, 40000, 10, 10])
    r = np.diag([10, 10])
    # controller_costs = DeePCCost(q, r, 5000000.0, 10.0, 10000.0)
    controller_costs = DeePCCost(q, r, 5000000.0, 10.0, 0.0)
    enable_measurement_constraint=bool(enable_measurement_constraint)

    #controller constraints setup
    a_u = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b_u = np.array([0.5, 0.5, 0.5, 0.5])
    a_y = None
    b_y = None
    if enable_measurement_constraint:
        a_y = np.array([[0, 0, 0, 0, 0, 1, 0, 0]])
        b_y = np.array([[0.1]])

    controller_constraints = DeePCConstraints(a_u, b_u, a_y, b_y)

    controller_args=DeePCControllerArgs(
        data,
        DeePCDims(t_past, t_fut, p, m),
        t_hankel,
        controller_costs,
        controller_constraints,
        [0, 0],
        False,
    )

   
    for seed in seedlist:
        for k in klist:
            env = get_reacher_simulator(
                num_steps=int(max_steps),
                video_folder=os.path.join(outdir, "video"),
                video_title="reacher_run",
                record_video=bool(record_video),
                render_mode="rgb_array" if bool(record_video) else None,
            )
            deepc = AdaptiveSelectDeePC(
                controller_args,
                selector_callback=LkSelector(order=2),
                num_hankel_cols=int(k),
                n_iter=1,
                d_gate_enabled=False,
                cond_gate_enabled=False                
            )

            scheduler = ReacherSetpointScheduler(
                env, controller_args.deepc_dims, num_extra_targets=int(num_extra_targets)
            )
            x_traj, u_traj, _, _, cost_obj, _ = run_reacher_simulator(
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

            print(f"Seed: {seed}, K: {k}, cost: {float(get_cost_dict(cost_obj).get("cost", np.nan))}, Sigma_min_all: {deepc.get_sigma_min_all()}, sigma_min_Mk: {np.mean(deepc.get_sigma_min_Mk_history())}")
            
            #설정 : seed, 실험 종류, dataset, max_steps
            #구조 : K, sigma min all, sigma min Mk
            #성능 : cost, rmse, success, solve_ms, local_gate_fail, cond_gate_fail, slack inf norm, status, trajectory
            results.append({
                "seed": seed,
                "experiment_type": "fixed_k",
                "dataset": dataset,
                "max_steps": max_steps,
                "K": k,
                "sigma_min_all": deepc.get_sigma_min_all(),
                "sigma_min_Mk": deepc.get_sigma_min_Mk_history(),
                "cost": float(get_cost_dict(cost_obj).get("cost", np.nan)),
                "rmse": float(np.sqrt(np.mean(err * err))),
                "success": int(compute_success(err, eps=0.01, window=20)),
                "solve_ms": deepc.get_solve_time_ms_history(),
                "local_gate_fail": int(deepc._local_gate_fail),
                "cond_gate_fail": int(deepc._cond_gate_fail),
                "slack_inf_norm": deepc.get_slack_norm_inf_history(),
                "status": deepc.get_status_history(),
                "trajectory": {  
                    "x_traj": np.asarray(x_traj, dtype=float),
                    "u_traj": np.asarray(u_traj, dtype=float),
                    "target_traj": np.asarray(target_traj, dtype=float),
                    "err": np.asarray(err, dtype=float),
                },
            })
          
            print(f"Seed: {seed}, K: {k}, cost: {float(get_cost_dict(cost_obj).get("cost", np.nan))}, Sigma_min_all: {deepc.get_sigma_min_all()}, sigma_min_Mk: {np.mean(deepc.get_sigma_min_Mk_history())}")
            env.close()

    #save results to csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(outdir, "results.csv"), index=False)

    plot_results(results_df, outdir)


import matplotlib.pyplot as plt

METRICS_7 = [
    ("sigma_mk_min", "sigma_min(Mk)"),
    ("K_opt_mean", "K_opt Mean"),
    ("cost", "Total Cost"),
    ("rmse", "RMSE"),
    ("solve_mean", "Mean Solve Time (ms)"),
    ("slack_max", "Slack Max"),
    ("solver_opt_rate", "Solver Optimal Rate"),
]


def _safe_hist(value, reducer):
    arr = np.asarray(value, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(reducer(arr))


def _build_run_level_df(results_df):
    rows = []
    for r in results_df.to_dict("records"):
        status_hist = np.asarray(r.get("status", []), dtype=object).reshape(-1)
        status_l = np.array([str(x).lower() for x in status_hist], dtype=object)
        opt_rate = float(np.mean([("optimal" in s) for s in status_l])) if status_l.size else np.nan

        rows.append(
            {
                "K": int(r["K"]),
                "sigma_mk_min": _safe_hist(r.get("sigma_min_Mk", []), np.min),
                "K_opt_mean": float(r["K"]),  # fixed-k path
                "cost": float(r.get("cost", np.nan)),
                "rmse": float(r.get("rmse", np.nan)),
                "solve_mean": _safe_hist(r.get("solve_ms", []), np.mean),
                "slack_max": _safe_hist(r.get("slack_inf_norm", []), np.max),
                "solver_opt_rate": opt_rate,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_by_k(run_df, metric_cols):
    rep_rows = []
    for k, sub in run_df.groupby("K"):
        row = {"K": int(k)}
        for m in metric_cols:
            v = sub[m].to_numpy(dtype=float)
            row[f"{m}_mean"] = float(np.nanmean(v))
            row[f"{m}_median"] = float(np.nanmedian(v))
            row[f"{m}_p10"] = float(np.nanpercentile(v, 10))
            row[f"{m}_p90"] = float(np.nanpercentile(v, 90))
            row[f"{m}_min"] = float(np.nanmin(v))
            row[f"{m}_max"] = float(np.nanmax(v))
        rep_rows.append(row)
    return pd.DataFrame(rep_rows).sort_values("K")


def _plot_one(ax, rep_df, metric, label, center):
    x = rep_df["K"].to_numpy(dtype=float)
    main = rep_df[f"{metric}_{center}"].to_numpy(dtype=float)
    p10 = rep_df[f"{metric}_p10"].to_numpy(dtype=float)
    p90 = rep_df[f"{metric}_p90"].to_numpy(dtype=float)
    mn = rep_df[f"{metric}_min"].to_numpy(dtype=float)
    mx = rep_df[f"{metric}_max"].to_numpy(dtype=float)

    ax.plot(x, main, "-o", color="#2ca02c", label=center)
    ax.fill_between(x, p10, p90, color="#2ca02c", alpha=0.2, label="p10-p90")
    ax.plot(x, mn, "--", color="#7f7f7f", linewidth=1, label="min")
    ax.plot(x, mx, "--", color="#9467bd", linewidth=1, label="max")
    ax.set_title(f"{label} vs K")
    ax.set_xlabel("K")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def plot_kopt_sigma(results_df, out_png, center="median"):
    run_df = _build_run_level_df(results_df)
    rep = _aggregate_by_k(run_df, ["sigma_mk_min", "K_opt_mean"])
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), dpi=180, sharex=True)
    _plot_one(axs[0], rep, "sigma_mk_min", "sigma_min(Mk)", center=center)
    _plot_one(axs[1], rep, "K_opt_mean", "K_opt Mean", center=center)
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_7x1(results_df, out_png, center="median"):
    run_df = _build_run_level_df(results_df)
    rep = _aggregate_by_k(run_df, [m for m, _ in METRICS_7])
    rep.to_csv(
        os.path.join(os.path.dirname(out_png), f"K_stats_representative_7x1_{center}.csv"),
        index=False,
    )
    fig, axs = plt.subplots(7, 1, figsize=(14, 24), dpi=180, sharex=True)
    for i, (metric, label) in enumerate(METRICS_7):
        _plot_one(axs[i], rep, metric, label, center=center)
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_results(results_df, outdir):
    # requested outputs: mean/median versions
    plot_kopt_sigma(
        results_df,
        os.path.join(outdir, "fig_fixedk_kopt_sigma_median.png"),
        center="median",
    )
    plot_kopt_sigma(
        results_df,
        os.path.join(outdir, "fig_fixedk_kopt_sigma_mean.png"),
        center="mean",
    )
    plot_7x1(
        results_df,
        os.path.join(outdir, "fig_fixedk_representative_7x1_median.png"),
        center="median",
    )
    plot_7x1(
        results_df,
        os.path.join(outdir, "fig_fixedk_representative_7x1_mean.png"),
        center="mean",
    )


if __name__ == "__main__":
    main()
