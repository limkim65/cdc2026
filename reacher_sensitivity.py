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
from select_deepc.deepc_dataclasses import DeePCControllerArgs, DeePCDims, DeePCConstraints, DeePCCost
from select_deepc.deepc_utils import (  # noqa: E402
    DeePCCostAccumulator,
    IntegralAbsoluteError,
    IntegralSquareError,
    PerformanceAccumulatorWrapper,
    get_reacher_simulator,
    load_data_from_folder,
    run_reacher_simulator,
    run_reacher_simulator_sensitivity_analysis,
)


def safe_stat(values, fn, default=np.nan):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return float(default)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(fn(arr))


def main():
    parser = argparse.ArgumentParser(description="Simple Reacher runner.")
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("logs", "reacher", "adaptive_k_dir"),
    )
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
    args = parser.parse_args()

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
        ds,
        DeePCDims(t_past, t_fut, p, m),
        t_hankel,
        controller_costs,
        controller_constraints,
        [0, 0],
        False,
    )
    
    env = get_reacher_simulator(
        num_steps=int(args.max_steps),
        video_folder=os.path.join(args.outdir, "video"),
        video_title="reacher_run",
    )

    controller_args.controller_costs.g_regularization_pi = 0

    deepc = SelectDeePC(
        controller_args,
        selector_callback=AdaptiveLkSelector(order=2),
        sigma_bar=1,
        N_loc=1000,
        d_max=float(args.d_max),
        num_hankel_cols=int(args.K),
        n_iter=int(args.n_iter),
        K_min=int(args.Kmin),
        K_max=int(args.Kmax),
        K_step=int(args.Kstep),
    )
    deepc._eps_sigma = float(args.eps_sigma)

    noisy_deepc = SelectDeePC(
        controller_args,
        selector_callback=AdaptiveLkSelector(order=2),
        sigma_bar=1,
        N_loc=1000,
        d_max=float(args.d_max),
        num_hankel_cols=int(args.K),
        n_iter=int(args.n_iter),
        K_min=int(args.Kmin),
        K_max=int(args.Kmax),
        K_step=int(args.Kstep),
    )
    deepc._eps_sigma = float(args.eps_sigma)
    scheduler = ReacherSetpointScheduler(
        env, controller_args.deepc_dims, num_extra_targets=int(args.num_extra_targets)
    )
    x_traj, u_traj, _, _, cost_obj, _ = run_reacher_simulator_sensitivity_analysis(
        env,
        deepc,
        noisy_deepc,
        scheduler,
        seed=int(args.seed),
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
    success = compute_success(err, eps=float(args.success_eps), window=int(args.success_window))

    mode_tag = "adaptive" if args.adaptive_k else "fixed"
    out_npz = os.path.join(
        args.outdir,
        f"run_{mode_tag}_K{int(args.K):03d}_rho{float(args.rho):.3g}_seed{int(args.seed):03d}.npz",
    )
    deepc.save_history_npz(
        out_npz,
        extra={
            "seed": int(args.seed),
            "K": int(args.K),
            "adaptive_k": int(bool(args.adaptive_k)),
            "rho": float(args.rho),
            "total_cost": float(get_cost_dict(cost_obj).get("cost", np.nan)),
            "rmse_ee": float(np.sqrt(safe_stat(err * err, np.mean, default=np.nan))),
            "success": int(success),
            "N_loc": int(args.N_loc),
            "d_max": float(args.d_max),
            "sigma_bar": float(args.sigma_bar),
            "local_gate_fail": int(deepc._local_gate_fail),
            "cond_gate_fail": int(deepc._cond_gate_fail),
            "d_k_history": np.asarray(deepc._d_k_history, dtype=float),
            "selected_idcs_history": np.array(deepc._selected_idcs_history, dtype=object),
            "executed_state_traj": np.asarray(x_traj, dtype=float),
            "tracking_error": np.asarray(err, dtype=float),
            "K_opt_history": np.asarray(deepc._K_opt_history, dtype=int),
        },
    )

    K_opts=np.asarray(deepc._K_opt_history, dtype=int)
    print(f"K_opts: {K_opts}")
    print(f"K_opts_mean: {np.mean(K_opts)}")
    print(f"K_opts_std: {np.std(K_opts)}")
    print(f"K_opts_min: {np.min(K_opts)}")
    print(f"K_opts_max: {np.max(K_opts)}")
    print(f"K_opts_median: {np.median(K_opts)}")

    print(
        f"cost={get_cost_dict(cost_obj).get('cost', np.nan):.3e} "
        f"rmse={np.sqrt(safe_stat(err * err, np.mean, default=np.nan)):.3e} "
        f"success={int(success)} "
        f"solve_ms={safe_stat(deepc.get_solve_time_ms_history(), np.mean):.2f}"
        f"local_gate_fail={deepc._local_gate_fail} "
        f"cond_gate_fail={deepc._cond_gate_fail} "
    )


if __name__ == "__main__":
    main()
