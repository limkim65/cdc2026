import argparse
import csv
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from select_deepc.data_selectors import LkSelector  # noqa: E402
from select_deepc.deepc_controller import SelectDeePC  # noqa: E402
from select_deepc.deepc_dataclasses import (  # noqa: E402
    DeePCConstraints,
    DeePCCost,
    DeePCControllerArgs,
    DeePCDims,
    TrajectoryDataSet,
)
from select_deepc.deepc_rocket_utils import SetPointScheduler  # noqa: E402
from select_deepc.deepc_utils import (  # noqa: E402
    DeePCCostAccumulator,
    IntegralAbsoluteError,
    IntegralSquareError,
    PerformanceAccumulatorWrapper,
    get_reacher_simulator,
    load_data_from_folder,
    run_reacher_simulator,
)


def fix_dataset(dataset: TrajectoryDataSet):
    for traj in dataset.dataset:
        traj.state_trajectory[:, [4, 5]] += traj.state_trajectory[:, [8, 9]]
        traj.state_trajectory = traj.state_trajectory[:, [0, 1, 2, 3, 4, 5, 6, 7]]


def setup_deepc_reacher(
    trajectory_data: TrajectoryDataSet, enable_measurement_constraint: bool = False
) -> DeePCControllerArgs:
    p = 8
    m = 2
    n = 75
    t_past = 2
    t_fut = 15
    t_hankel = (m + 1) * (t_past + t_fut) + n - 1

    q = np.diag([0, 0, 0, 0, 40000, 40000, 10, 10])
    r = np.diag([10, 10])
    controller_costs = DeePCCost(q, r, 5000000.0, 10.0, 10000.0)

    a_u = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b_u = np.array([0.5, 0.5, 0.5, 0.5])
    a_y = None
    b_y = None
    if enable_measurement_constraint:
        a_y = np.array([[0, 0, 0, 0, 0, 1, 0, 0]])
        b_y = np.array([[0.1]])

    controller_constraints = DeePCConstraints(a_u, b_u, a_y, b_y)
    return DeePCControllerArgs(
        trajectory_data,
        DeePCDims(t_past, t_fut, p, m),
        t_hankel,
        controller_costs,
        controller_constraints,
        [0, 0],
        False,
    )


class ReacherTargets:
    def __init__(self, num_steps: int = 0):
        num_steps = int(max(0, num_steps))
        angles = np.linspace(3 * np.pi / 4, 2 * np.pi, max(1, num_steps))
        np.random.seed(42)
        dists = np.random.uniform(0.04, 0.15, max(1, num_steps))

        self.targets = [[0.043, 0.092]]
        for angle, dist in zip(angles[:num_steps], dists[:num_steps]):
            self.targets.append([dist * np.cos(angle), dist * np.sin(angle)])
        for target in self.targets:
            target[1] = np.minimum(target[1], 0.09)


class ReacherSetpointScheduler(SetPointScheduler):
    def __init__(self, env, deepc_dims: DeePCDims, num_extra_targets: int = 0):
        super().__init__(env)
        self._dims = deepc_dims
        self._targets = ReacherTargets(num_steps=int(num_extra_targets))
        self._target_idx = 0

    def __call__(self, state, **kwargs):
        if (
            np.linalg.norm(
                state[4:6]
                + state[8:10]
                - np.array(self._targets.targets[self._target_idx])
            )
            < 0.005
        ):
            self._target_idx = (self._target_idx + 1) % len(self._targets.targets)
        target_pos = self._targets.targets[self._target_idx]
        return [0, 0, 0, 0, target_pos[0], target_pos[1], 0, 0]

    def is_successful(self, state):
        return True


def safe_stat(values, fn, default=np.nan):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return float(default)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(fn(finite))


def get_cost_dict(cost_obj) -> Dict[str, float]:
    if isinstance(cost_obj, dict):
        return {k: float(v) for k, v in cost_obj.items()}
    return {"cost": float(cost_obj)}


def compute_reacher_tracking_errors(x_traj: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x_traj, dtype=float)
    if x_arr.size == 0:
        return np.array([], dtype=float)
    ee_xy = x_arr[:, 4:6] + x_arr[:, 8:10]
    tgt = np.asarray(target_xy, dtype=float).reshape(1, 2)
    err = np.linalg.norm(ee_xy - tgt, axis=1)
    return np.asarray(err, dtype=float)


def summarize_by_rho(rows: List[dict]) -> List[dict]:
    out = []
    for rho in sorted(set(float(r["rho"]) for r in rows)):
        subset = [r for r in rows if float(r["rho"]) == rho]
        out.append(
            {
                "rho": float(rho),
                "runs": int(len(subset)),
                "median_k_opt_mean": safe_stat([r["k_opt_mean"] for r in subset], np.median),
                "median_k_opt_var": safe_stat([r["k_opt_var"] for r in subset], np.median),
                "median_sigma_min_Ag": safe_stat([r["sigma_min_Ag"] for r in subset], np.median),
                "median_sigma_min_Hu": safe_stat([r["sigma_min_Hu"] for r in subset], np.median),
                "median_slack_max": safe_stat([r["slack_max"] for r in subset], np.median),
                "median_slack_mean": safe_stat([r["slack_mean"] for r in subset], np.median),
                "median_slack_fail_rate": safe_stat(
                    [r["slack_fail_rate"] for r in subset], np.median
                ),
                "median_total_cost": safe_stat([r["total_cost"] for r in subset], np.median),
                "median_mean_solve_time_ms": safe_stat(
                    [r["mean_solve_time_ms"] for r in subset], np.median
                ),
                "median_total_solve_time_ms": safe_stat(
                    [r["total_solve_time_ms"] for r in subset], np.median
                ),
                "median_final_tracking_error": safe_stat(
                    [r["final_tracking_error"] for r in subset], np.median
                ),
                "median_steady_state_error_mean": safe_stat(
                    [r["steady_state_error_mean"] for r in subset], np.median
                ),
                "p10_total_cost": safe_stat(
                    [r["total_cost"] for r in subset], lambda x: np.quantile(x, 0.1)
                ),
                "p90_total_cost": safe_stat(
                    [r["total_cost"] for r in subset], lambda x: np.quantile(x, 0.9)
                ),
            }
        )
    return out


def save_plots(summary_by_rho: List[dict], outdir: str):
    if not summary_by_rho:
        return

    xs = np.array([r["rho"] for r in summary_by_rho], dtype=float)
    med_k_mean = np.array([r["median_k_opt_mean"] for r in summary_by_rho], dtype=float)
    med_k_var = np.array([r["median_k_opt_var"] for r in summary_by_rho], dtype=float)
    med_sig = np.array([r["median_sigma_min_Ag"] for r in summary_by_rho], dtype=float)
    med_slack = np.array([r["median_slack_max"] for r in summary_by_rho], dtype=float)
    med_cost = np.array([r["median_total_cost"] for r in summary_by_rho], dtype=float)
    med_solve = np.array([r["median_mean_solve_time_ms"] for r in summary_by_rho], dtype=float)

    fig, axs = plt.subplots(3, 2, figsize=(10, 8), dpi=140)
    axs = axs.reshape(-1)
    series = [
        (med_k_mean, "median K_opt mean"),
        (med_k_var, "median K_opt var"),
        (med_sig, "median sigma_min_Ag (episode min)"),
        (med_slack, "median slack max"),
        (med_cost, "median total cost"),
        (med_solve, "median mean solve time [ms]"),
    ]
    for ax, (vals, name) in zip(axs, series):
        ax.plot(xs, vals, marker="o", lw=1.4)
        ax.set_xlabel("rho")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        if np.all(vals > 0):
            ax.set_xscale("log")
    fig.suptitle("Adaptive-K Rho Sweep (Reacher)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rho_sweep_metrics.png"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.2), dpi=140)
    sc = ax.scatter(med_solve, med_cost, c=xs, cmap="viridis", s=70)
    ax.set_xlabel("median mean solve time [ms]")
    ax.set_ylabel("median total cost")
    if np.all(med_cost > 0):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("rho")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "tradeoff_cost_vs_solve.png"), bbox_inches="tight")
    plt.close(fig)


def save_selected_rows_plot(
    out_path: str,
    k_hist: np.ndarray,
    selected_hist: List[np.ndarray],
    show_plot: bool = True,
):
    k_arr = np.asarray(k_hist, dtype=float).reshape(-1)
    steps = np.arange(1, k_arr.size + 1)

    sel_step = []
    sel_idx = []
    for i, idcs in enumerate(selected_hist):
        idcs_arr = np.asarray(idcs, dtype=int).reshape(-1)
        if idcs_arr.size == 0:
            continue
        sel_step.extend([i + 1] * int(idcs_arr.size))
        sel_idx.extend(idcs_arr.tolist())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=130, sharex=True)
    ax1.plot(steps, k_arr, color="tab:blue", lw=1.4)
    ax1.set_ylabel("K_opt")
    ax1.set_title("Adaptive-K Selection Diagnostics")
    ax1.grid(True, alpha=0.25)

    if len(sel_step) > 0:
        ax2.scatter(sel_step, sel_idx, s=5, alpha=0.35, color="tab:orange")
    ax2.set_xlabel("step")
    ax2.set_ylabel("selected row index")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive-K rho-sweep runner for Reacher SelectDeePC."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(REPO_ROOT, "logs", "reacher", "adaptive_k_dir"),
    )
    parser.add_argument("--rho", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--Kmin", type=int, default=40)
    parser.add_argument("--Kmax", type=int, default=200)
    parser.add_argument("--Kstep", type=int, default=5)
    parser.add_argument("--gamma_min", type=float, default=1e-6)
    parser.add_argument("--eps_sigma", type=float, default=1e-3)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["iid", "random_walk", "both"],
        default="iid",
    )
    parser.add_argument("--enable_measurement_constraint", action="store_true")
    parser.add_argument("--num_extra_targets", type=int, default=0)
    parser.add_argument(
        "--sse_window",
        type=int,
        default=20,
        help="Window size for steady-state error mean over last steps.",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    iid_trajectories = load_data_from_folder(
        os.path.join(REPO_ROOT, "data", "reacher", "dataset_iid"), "iid"
    )
    random_walk_trajectories = load_data_from_folder(
        os.path.join(REPO_ROOT, "data", "reacher", "dataset_random_walk"), "random_walk"
    )
    for dataset in [iid_trajectories, random_walk_trajectories]:
        fix_dataset(dataset)

    if args.dataset == "iid":
        trajectory_data = iid_trajectories
    elif args.dataset == "random_walk":
        trajectory_data = random_walk_trajectories
    else:
        trajectory_data = iid_trajectories + random_walk_trajectories

    controller_args = setup_deepc_reacher(
        trajectory_data,
        enable_measurement_constraint=bool(args.enable_measurement_constraint),
    )

    rows = []
    for rho in args.rho:
        for seed in range(int(args.seeds)):
            env = get_reacher_simulator(
                num_steps=int(args.max_steps),
                video_folder=os.path.join(args.outdir, "video"),
                video_title=f"rho{float(rho):.3g}_seed{seed:03d}",
            )
            deepc = SelectDeePC(
                controller_args,
                selector_callback=LkSelector(),
                num_hankel_cols=int(args.Kmin),
                n_iter=int(args.n_iter),
                adaptive_k=True,
                K_min=int(args.Kmin),
                K_max=int(args.Kmax),
                K_step=int(args.Kstep),
                rho=float(rho),
                gamma_min=float(args.gamma_min),
            )
            deepc._eps_sigma = float(args.eps_sigma)

            scheduler = ReacherSetpointScheduler(
                env,
                controller_args.deepc_dims,
                num_extra_targets=int(args.num_extra_targets),
            )

            (
                x_traj,
                _u_traj,
                _x_preds,
                _u_preds,
                cost_obj,
                _solve_avg,
            ) = run_reacher_simulator(
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

            cost_dict = get_cost_dict(cost_obj)
            k_hist = np.asarray(deepc.get_K_opt_history(), dtype=float)
            sigma_min_ag = safe_stat(deepc.get_sigma_min_A_history(), np.min)
            sigma_min_hu = safe_stat(deepc.get_min_selected_singular_values(), np.min)
            slack_hist = np.asarray(deepc.get_slack_norm_inf_history(), dtype=float)
            slack_max = safe_stat(slack_hist, np.max, default=0.0)
            slack_mean = safe_stat(slack_hist, np.mean, default=0.0)
            slack_fail_rate = safe_stat(deepc.get_slack_violation_history(), np.mean, default=0.0)
            total_cost = float(cost_dict.get("cost", np.nan))
            mean_solve_time_ms = safe_stat(deepc.get_solve_time_ms_history(), np.mean)
            total_solve_time_ms = safe_stat(deepc.get_solve_time_ms_history(), np.sum, default=0.0)
            steps = int(np.asarray(x_traj).shape[0])
            k_opt_mean = safe_stat(k_hist, np.mean)
            k_opt_var = safe_stat(k_hist, np.var, default=0.0)
            final_target = np.asarray(
                scheduler._targets.targets[scheduler._target_idx], dtype=float
            )
            tracking_err = compute_reacher_tracking_errors(x_traj, final_target)
            final_tracking_error = safe_stat(tracking_err[-1:], np.mean)
            w = int(max(1, args.sse_window))
            steady_state_error_mean = safe_stat(tracking_err[-w:], np.mean)

            tag = (
                f"Kmin{int(args.Kmin)}_Kmax{int(args.Kmax)}_Kstep{int(args.Kstep)}_"
                f"rho{float(rho):.3g}_gmin{float(args.gamma_min):.3g}"
            )
            npz_name = f"run_{tag}_seed{seed:03d}.npz"
            npz_path = os.path.join(args.outdir, npz_name)
            deepc.save_history_npz(
                npz_path,
                extra={
                    "rho": float(rho),
                    "seed": int(seed),
                    "steps": int(steps),
                    "Kmin": int(args.Kmin),
                    "Kmax": int(args.Kmax),
                    "Kstep": int(args.Kstep),
                    "gamma_min": float(args.gamma_min),
                    "k_opt_mean": float(k_opt_mean),
                    "k_opt_var": float(k_opt_var),
                    "sigma_min_Ag_episode_min": float(sigma_min_ag),
                    "sigma_min_Hu_episode_min": float(sigma_min_hu),
                    "slack_max": float(slack_max),
                    "slack_mean": float(slack_mean),
                    "slack_fail_rate": float(slack_fail_rate),
                    "total_cost": float(total_cost),
                    "mean_solve_time_ms": float(mean_solve_time_ms),
                    "total_solve_time_ms": float(total_solve_time_ms),
                    "final_tracking_error": float(final_tracking_error),
                    "steady_state_error_mean": float(steady_state_error_mean),
                    "sse_window": int(w),
                    "executed_state_traj": np.asarray(x_traj, dtype=float),
                    "executed_input_traj": np.asarray(_u_traj, dtype=float),
                },
            )

            row = {
                "rho": float(rho),
                "seed": int(seed),
                "steps": int(steps),
                "k_opt_mean": float(k_opt_mean),
                "k_opt_var": float(k_opt_var),
                "sigma_min_Ag": float(sigma_min_ag),
                "sigma_min_Hu": float(sigma_min_hu),
                "slack_max": float(slack_max),
                "slack_mean": float(slack_mean),
                "slack_fail_rate": float(slack_fail_rate),
                "total_cost": float(total_cost),
                "ISE": float(cost_dict.get("ISE", np.nan)),
                "IAE": float(cost_dict.get("IAE", np.nan)),
                "mean_solve_time_ms": float(mean_solve_time_ms),
                "total_solve_time_ms": float(total_solve_time_ms),
                "final_tracking_error": float(final_tracking_error),
                "steady_state_error_mean": float(steady_state_error_mean),
                "npz_path": npz_path,
            }
            rows.append(row)

            selected_hist = [np.asarray(v, dtype=int) for v in deepc.get_selected_idcs_history()]
            diag_png = os.path.join(args.outdir, f"selected_rows_{tag}_seed{seed:03d}.png")
            save_selected_rows_plot(
                out_path=diag_png,
                k_hist=k_hist,
                selected_hist=selected_hist,
                show_plot=True,
            )

            print(
                f"[run] rho={float(rho):.3g} seed={seed:3d} steps={steps:4d} "
                f"Kopt_mean={row['k_opt_mean']:.2f} Kopt_var={row['k_opt_var']:.2f} "
                f"sigma_min_Ag(min_epi)={row['sigma_min_Ag']:.3e} sigma_min_Hu(min_epi)={row['sigma_min_Hu']:.3e} "
                f"slack_max={row['slack_max']:.3e} slack_mean={row['slack_mean']:.3e} "
                f"slack_fail={row['slack_fail_rate']:.3f} "
                f"cost={row['total_cost']:.3e} solve_ms={row['mean_solve_time_ms']:.2f} total_solve_ms={row['total_solve_time_ms']:.2f} "
                f"e_final={row['final_tracking_error']:.3e} "
                f"sse_mean={row['steady_state_error_mean']:.3e}"
            )

    summary_path = os.path.join(args.outdir, "summary.csv")
    fieldnames = [
        "rho",
        "seed",
        "steps",
        "k_opt_mean",
        "k_opt_var",
        "sigma_min_Ag",
        "sigma_min_Hu",
        "slack_max",
        "slack_mean",
        "slack_fail_rate",
        "total_cost",
        "ISE",
        "IAE",
        "mean_solve_time_ms",
        "total_solve_time_ms",
        "final_tracking_error",
        "steady_state_error_mean",
        "npz_path",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rho = summarize_by_rho(rows)
    summary_rho_path = os.path.join(args.outdir, "summary_by_rho.csv")
    fieldnames_rho = list(summary_rho[0].keys()) if summary_rho else []
    if fieldnames_rho:
        with open(summary_rho_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_rho)
            writer.writeheader()
            writer.writerows(summary_rho)

    save_plots(summary_rho, args.outdir)

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved by-rho summary: {summary_rho_path}")
    print(f"Saved plots under: {args.outdir}")


if __name__ == "__main__":
    main()
