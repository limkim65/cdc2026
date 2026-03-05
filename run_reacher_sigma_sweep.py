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


def safe_stat(values, fn, default=np.nan):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return float(default)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(fn(arr))


def plot_results(results, outdir):
    import matplotlib.pyplot as plt

    if len(results) == 0:
        print("No results to plot.")
        return

    sigma_bars = np.array([r["sigma_bar"] for r in results], dtype=float)
    total_costs = np.array([r["total_cost"] for r in results], dtype=float)
    rmse_ee = np.array([r["rmse_ee"] for r in results], dtype=float)
    solve_time_ms = np.array(
        [
            np.mean(r["solve_time_ms"]) if np.size(r["solve_time_ms"]) > 0 else np.nan
            for r in results
        ],
        dtype=float,
    )
    ises = np.array([r["ISE"] for r in results], dtype=float)
    iaes = np.array([r["IAE"] for r in results], dtype=float)

    def stochastic_plot(x, y, xlabel, ylabel, title, ax=None):
        ax = ax if ax is not None else plt.gca()
        uniq_sigma_bar = np.unique(x)
        y_per_sigma_bar = [y[x == sigma_bar] for sigma_bar in uniq_sigma_bar]

        ax.violinplot(y_per_sigma_bar, positions=uniq_sigma_bar, showmedians=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        for i, sigma_bar in enumerate(uniq_sigma_bar):
            jitter = np.random.normal(loc=0.0, scale=0.07, size=len(y_per_sigma_bar[i]))
            ax.scatter(
                np.ones_like(y_per_sigma_bar[i]) * sigma_bar + jitter,
                y_per_sigma_bar[i],
                alpha=0.6,
                s=12,
                color="tab:blue",
                edgecolor="gray",
            )

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Stochastic Visualization of Results (per K over Seeds)")

    stochastic_plot(sigma_bars, total_costs, "sigma_bar", "Total Cost", "Total Cost vs sigma_bar", ax=axs[0, 0])
    stochastic_plot(sigma_bars, rmse_ee, "sigma_bar", "RMSE (end-effector)", "RMSE vs sigma_bar", ax=axs[0, 1])
    stochastic_plot(sigma_bars, solve_time_ms, "sigma_bar", "Solve Time (ms)", "Solve Time vs sigma_bar", ax=axs[1, 0])

    uniq_sigma_bar = np.unique(sigma_bars)
    for name, arr, color in [("ISE", ises, "tab:olive"), ("IAE", iaes, "tab:orange")]:
        y_grouped = [arr[sigma_bars == sigma_bar] for sigma_bar in uniq_sigma_bar]
        parts = axs[1, 1].violinplot(
            y_grouped, positions=uniq_sigma_bar, showmeans=False, showmedians=True, widths=0.7
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.3 if name == "IAE" else 0.6)
        axs[1, 1].scatter(
            uniq_sigma_bar,
            [np.median(g) for g in y_grouped],
            label=f"{name} median",
            color=color,
            marker="o",
            alpha=0.9,
        )

    axs[1, 1].set_xlabel("K")
    axs[1, 1].set_ylabel("Metric Value")
    axs[1, 1].set_title("ISE & IAE vs K (Stochastic)")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png = os.path.join(outdir, "fixedk_stochastic_summary.png")
    plt.savefig(out_png, dpi=180)
    plt.show()
    print(f"Saved plot: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Simple Reacher runner.")
    
    parser.add_argument("--dataset", choices=["iid", "random_walk", "both"], default="iid")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=80, help="Fixed K, or initial K for adaptive.")
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--adaptive_k", action="store_true")
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--Kmin", type=int, default=40)
    parser.add_argument("--Kmax", type=int, default=320)
    parser.add_argument("--Kstep", type=int, default=1)
    parser.add_argument("--gamma_min", type=float, default=1e-6)
    parser.add_argument("--eps_sigma", type=float, default=1e-3)
    parser.add_argument("--success_eps", type=float, default=0.01)
    parser.add_argument("--success_window", type=int, default=20)
    parser.add_argument("--num_extra_targets", type=int, default=0)
    parser.add_argument("--enable_measurement_constraint", action="store_true")
    parser.add_argument("--N_loc", type=int, default=1000)
    parser.add_argument("--d_max", type=float, default=10.0)
    parser.add_argument("--sigma_bar", type=float, default=1e-6)

    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("logs", "reacher", "fixed_k", "dataset_%s" % ("{dataset}")),
    )

    args = parser.parse_args()

    # Resolve dataset-aware output directory.
    # 1) If user passed a template path containing "{dataset}", format it.
    # 2) Otherwise append "dataset_<name>" once.
    if "{dataset}" in args.outdir:
        args.outdir = args.outdir.format(dataset=args.dataset)
    else:
        dataset_tag = f"dataset_{args.dataset}"
        norm_tail = os.path.basename(os.path.normpath(args.outdir))
        if norm_tail != dataset_tag:
            args.outdir = os.path.join(args.outdir, dataset_tag)

    os.makedirs(args.outdir, exist_ok=True)

    iid = load_data_from_folder(os.path.join(REPO_ROOT, "data", "reacher", "dataset_iid"), "iid")
    rw = load_data_from_folder(
        os.path.join(REPO_ROOT, "data", "reacher", "dataset_random_walk"), "random_walk"
    )
    for ds in [iid, rw]:
        fix_dataset(ds)

    if args.dataset == "iid":
        data = iid
    elif args.dataset == "random_walk":
        data = rw
    else:
        data = iid + rw

    #deepc arguments setup
    p = 8
    m = 2
    n = 75
    t_past = 2
    t_fut = 15
    t_hankel = (m + 1) * (t_past + t_fut) + n - 1

    q = np.diag([0, 0, 0, 0, 40000, 40000, 10, 10])
    r = np.diag([10, 10])
    # controller_costs = DeePCCost(q, r, 5000000.0, 10.0, 10000.0)
    controller_costs = DeePCCost(q, r, 5000000.0, 10.0, 0.0)
    enable_measurement_constraint=bool(args.enable_measurement_constraint)

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

   
    
    sigma_bar_list = list(np.arange(0.1, 1.0, 0.1))
    seedlist = [0]
    results = []
    

    for seed in seedlist:
        for sigma_bar in sigma_bar_list:

            env = get_reacher_simulator(
                num_steps=int(args.max_steps),
                video_folder=os.path.join(args.outdir, "video"),
                video_title="reacher_run",
            )

            deepc = AdaptiveSelectDeePC(
                controller_args,
                selector_callback=LkSelector(order=2),
                sigma_bar=sigma_bar,
                N_loc=int(args.N_loc),
                d_max=float(args.d_max),
                n_iter=1
            )
            deepc._eps_sigma = float(args.eps_sigma)

            scheduler = ReacherSetpointScheduler(
                env, controller_args.deepc_dims, num_extra_targets=int(args.num_extra_targets)
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

            out_npz = os.path.join(
                args.outdir,
                f"run_fixedk_K{int(k):03d}_seed{int(seed):03d}.npz",
            )
            deepc.save_history_npz(
                out_npz,
                extra={
                    "seed": int(seed),
                    "K": int(k),
                    "sigma_min_Mk":np.asarray(deepc.get_sigma_min_Mk_history(), dtype=float).reshape(-1),
                    "sigma_min_all":deepc.get_sigma_min_all(),
                    "total_cost": float(get_cost_dict(cost_obj).get("cost", np.nan)),
                    "rmse_ee": float(np.sqrt(safe_stat(err * err, np.mean, default=np.nan))),
                    "solve_time_ms":np.asarray(deepc.get_solve_time_ms_history(), dtype=float).reshape(-1),
                    "slack_norm_inf":np.asarray(deepc.get_slack_norm_inf_history(), dtype=float).reshape(-1),
                    "slack_violation":np.asarray(deepc.get_slack_violation_history(), dtype=int).reshape(-1),
                    "status":np.asarray(deepc.get_status_history(), dtype=object).reshape(-1),
                    "ISE":float(get_cost_dict(cost_obj).get("ISE", np.nan)),
                    "IAE":float(get_cost_dict(cost_obj).get("IAE", np.nan)),
                    "trajectory":{
                        "x_traj":np.asarray(x_traj, dtype=float),
                        "u_traj":np.asarray(u_traj, dtype=float),
                        "target_traj":np.asarray(target_traj, dtype=float),
                        "err":np.asarray(err, dtype=float),
                    }
                },
            )
            results.append({
                "seed": int(seed),
                    "K": int(k),
                    "sigma_min_Mk":np.asarray(deepc.get_sigma_min_Mk_history(), dtype=float).reshape(-1),
                    "sigma_min_all":deepc.get_sigma_min_all(),
                    "total_cost": float(get_cost_dict(cost_obj).get("cost", np.nan)),
                    "rmse_ee": float(np.sqrt(safe_stat(err * err, np.mean, default=np.nan))),
                    "solve_time_ms":np.asarray(deepc.get_solve_time_ms_history(), dtype=float).reshape(-1),
                    "slack_norm_inf":np.asarray(deepc.get_slack_norm_inf_history(), dtype=float).reshape(-1),
                    "slack_violation":np.asarray(deepc.get_slack_violation_history(), dtype=int).reshape(-1),
                    "status":np.asarray(deepc.get_status_history(), dtype=object).reshape(-1),
                    "ISE":float(get_cost_dict(cost_obj).get("ISE", np.nan)),
                    "IAE":float(get_cost_dict(cost_obj).get("IAE", np.nan)),
                    "trajectory":{
                        "x_traj":np.asarray(x_traj, dtype=float),
                        "u_traj":np.asarray(u_traj, dtype=float),
                        "target_traj":np.asarray(target_traj, dtype=float),
                        "err":np.asarray(err, dtype=float),
                    }
            })
            env.close()

    plot_results(results, args.outdir)


if __name__ == "__main__":
    main()
