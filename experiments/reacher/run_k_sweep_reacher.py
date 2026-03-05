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
    g_regularization_1 = 10.0
    g_regularization_pi = 10000.0
    slack_cost = 5000000.0
    controller_costs = DeePCCost(
        q, r, slack_cost, g_regularization_1, g_regularization_pi
    )

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


def summarize_by_k(rows: List[dict]) -> List[dict]:
    out = []
    for k in sorted(set(int(r["K"]) for r in rows)):
        subset = [r for r in rows if int(r["K"]) == k]
        out.append(
            {
                "K": int(k),
                "runs": int(len(subset)),
                "median_sigma_min_Ag": safe_stat(
                    [r["sigma_min_Ag"] for r in subset], np.median
                ),
                "median_total_cost": safe_stat(
                    [r["total_cost"] for r in subset], np.median
                ),
                "median_slack_norm_max": safe_stat(
                    [r["slack_norm_max"] for r in subset], np.median
                ),
                "median_slack_norm_mean": safe_stat(
                    [r["slack_norm_mean"] for r in subset], np.median
                ),
                "p10_sigma_min_Ag": safe_stat(
                    [r["sigma_min_Ag"] for r in subset], lambda x: np.quantile(x, 0.1)
                ),
                "p90_sigma_min_Ag": safe_stat(
                    [r["sigma_min_Ag"] for r in subset], lambda x: np.quantile(x, 0.9)
                ),
                "p10_total_cost": safe_stat(
                    [r["total_cost"] for r in subset], lambda x: np.quantile(x, 0.1)
                ),
                "p90_total_cost": safe_stat(
                    [r["total_cost"] for r in subset], lambda x: np.quantile(x, 0.9)
                ),
                "p10_slack_norm_max": safe_stat(
                    [r["slack_norm_max"] for r in subset], lambda x: np.quantile(x, 0.1)
                ),
                "p90_slack_norm_max": safe_stat(
                    [r["slack_norm_max"] for r in subset], lambda x: np.quantile(x, 0.9)
                ),
            }
        )
    return out


def plot_metric_with_band(
    xs,
    med,
    p10,
    p90,
    title: str,
    ylabel: str,
    out_path: str,
    ylog: bool = False,
):
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.0), dpi=140)
    x = np.asarray(xs, dtype=float)
    m = np.asarray(med, dtype=float)
    lo = np.asarray(p10, dtype=float)
    hi = np.asarray(p90, dtype=float)

    ax.plot(x, m, marker="o", lw=1.4, label="median")
    ax.fill_between(x, lo, hi, alpha=0.22, label="p10-p90")
    ax.set_xlabel("K (N_cols)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if ylog and np.all(m > 0):
        ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Reacher K-sweep runner for SelectDeePC.")
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(REPO_ROOT, "logs", "reacher", "fixed_k_sweep_dir"),
    )
    parser.add_argument("--Kmin", type=int, default=40)
    parser.add_argument("--Kmax", type=int, default=200)
    parser.add_argument("--Kstep", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--n_iter", type=int, default=5)
    parser.add_argument(
        "--num_extra_targets",
        type=int,
        default=0,
        help="Number of extra targets after the first one. 0 means single-target.",
    )
    parser.add_argument("--eps_sigma", type=float, default=1e-3)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["iid", "random_walk", "both"],
        default="iid",
    )
    parser.add_argument("--enable_measurement_constraint", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    iid_trajectories = load_data_from_folder(
        os.path.join(REPO_ROOT, "data", "reacher", "dataset_iid"), "iid"
    )
    random_walk_trajectories = load_data_from_folder(
        os.path.join(REPO_ROOT, "data", "reacher", "dataset_random_walk"),
        "random_walk",
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
    k_list = list(range(int(args.Kmin), int(args.Kmax) + 1, int(args.Kstep)))
    rows = []

    for k in k_list:
        for seed in range(int(args.seeds)):
            env = get_reacher_simulator(
                num_steps=int(args.max_steps),
                video_folder=os.path.join(args.outdir, "video"),
                video_title=f"k{k:03d}_seed{seed:03d}",
            )
            selector = LkSelector()
            deepc = SelectDeePC(
                controller_args,
                selector_callback=selector,
                num_hankel_cols=int(k),
                n_iter=int(args.n_iter),
            )
            deepc._eps_sigma = float(args.eps_sigma)

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
                ReacherSetpointScheduler(
                    env,
                    controller_args.deepc_dims,
                    num_extra_targets=int(args.num_extra_targets),
                ),
                seed=int(seed),
                deepc_cost_accumulator=PerformanceAccumulatorWrapper(
                    DeePCCostAccumulator(controller_args.controller_costs),
                    IntegralSquareError(mask=np.array([0, 0, 0, 0, 1, 1, 0, 0])),
                    IntegralAbsoluteError(mask=np.array([0, 0, 0, 0, 1, 1, 0, 0])),
                ),
            )

            cost_dict = get_cost_dict(cost_obj)
            sigma_min_ag = safe_stat(deepc.get_sigma_min_A_history(), np.min)
            sigma_min_hu = safe_stat(deepc.get_min_selected_singular_values(), np.min)
            slack_hist = deepc.get_slack_norm_inf_history()
            slack_norm_max = safe_stat(slack_hist, np.max, default=0.0)
            slack_norm_mean = safe_stat(slack_hist, np.mean, default=0.0)
            mean_solve_time_ms = safe_stat(deepc.get_solve_time_ms_history(), np.mean)
            steps = int(np.asarray(x_traj).shape[0])

            npz_name = f"run_K{k:03d}_seed{seed:03d}.npz"
            npz_path = os.path.join(args.outdir, npz_name)
            deepc.save_history_npz(
                npz_path,
                extra={
                    "K": int(k),
                    "seed": int(seed),
                    "steps": int(steps),
                    "total_cost": float(cost_dict.get("cost", np.nan)),
                    "ISE": float(cost_dict.get("ISE", np.nan)),
                    "IAE": float(cost_dict.get("IAE", np.nan)),
                    "slack_norm_max": float(slack_norm_max),
                    "slack_norm_mean": float(slack_norm_mean),
                },
            )

            row = {
                "K": int(k),
                "seed": int(seed),
                "steps": int(steps),
                "sigma_min_Ag": float(sigma_min_ag),
                "sigma_min_Hu": float(sigma_min_hu),
                "slack_norm_max": float(slack_norm_max),
                "slack_norm_mean": float(slack_norm_mean),
                "total_cost": float(cost_dict.get("cost", np.nan)),
                "ISE": float(cost_dict.get("ISE", np.nan)),
                "IAE": float(cost_dict.get("IAE", np.nan)),
                "mean_solve_time_ms": float(mean_solve_time_ms),
                "npz_path": npz_path,
            }
            rows.append(row)

            print(
                f"[run] K={k:3d} seed={seed:3d} steps={steps:4d} "
                f"sigma_min_Ag={row['sigma_min_Ag']:.3e} "
                f"cost={row['total_cost']:.3e} "
                f"slack_max={row['slack_norm_max']:.3e} "
                f"slack_mean={row['slack_norm_mean']:.3e}"
            )

    summary_path = os.path.join(args.outdir, "summary.csv")
    fieldnames = [
        "K",
        "seed",
        "steps",
        "sigma_min_Ag",
        "sigma_min_Hu",
        "slack_norm_max",
        "slack_norm_mean",
        "total_cost",
        "ISE",
        "IAE",
        "mean_solve_time_ms",
        "npz_path",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_k = summarize_by_k(rows)
    by_k_path = os.path.join(args.outdir, "summary_by_k.csv")
    fieldnames_by_k = list(by_k[0].keys()) if by_k else []
    if fieldnames_by_k:
        with open(by_k_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_by_k)
            writer.writeheader()
            writer.writerows(by_k)

    if by_k:
        ks = [r["K"] for r in by_k]
        plot_metric_with_band(
            ks,
            [r["median_sigma_min_Ag"] for r in by_k],
            [r["p10_sigma_min_Ag"] for r in by_k],
            [r["p90_sigma_min_Ag"] for r in by_k],
            title="K vs Sigma Min (A_g) - Reacher",
            ylabel="sigma_min(A_g)",
            out_path=os.path.join(args.outdir, "k_vs_sigma_min_Ag.png"),
            ylog=True,
        )
        plot_metric_with_band(
            ks,
            [r["median_total_cost"] for r in by_k],
            [r["p10_total_cost"] for r in by_k],
            [r["p90_total_cost"] for r in by_k],
            title="K vs Total Cost - Reacher",
            ylabel="total_cost",
            out_path=os.path.join(args.outdir, "k_vs_total_cost.png"),
            ylog=True,
        )
        plot_metric_with_band(
            ks,
            [r["median_slack_norm_max"] for r in by_k],
            [r["p10_slack_norm_max"] for r in by_k],
            [r["p90_slack_norm_max"] for r in by_k],
            title="K vs Slack Norm (max over episode) - Reacher",
            ylabel="max_t ||sigma_t||_inf",
            out_path=os.path.join(args.outdir, "k_vs_slack_norm.png"),
            ylog=True,
        )

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved by-K summary: {by_k_path}")
    print(f"Saved plots under: {args.outdir}")


if __name__ == "__main__":
    main()
