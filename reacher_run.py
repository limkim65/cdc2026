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
    replay_targets,
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


def history_or_default(controller, method_name, length, default=np.nan, dtype=float):
    method = getattr(controller, method_name, None)
    if callable(method):
        try:
            arr = np.asarray(method())
            if arr.size > 0:
                return arr
        except Exception:
            pass
    return np.full(max(1, int(length)), default, dtype=dtype)


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def plot_xy_trajectory_with_goal(
    x_traj,
    goal_xy_traj,
    outdir,
    filename="fig_xy_trajectory_goal.png",
    show=True,
):
    import matplotlib.pyplot as plt

    x_arr = np.asarray(x_traj, dtype=float)
    g_arr = np.asarray(goal_xy_traj, dtype=float)
    if x_arr.ndim != 2 or x_arr.shape[0] == 0:
        return
    if x_arr.shape[1] < 10:
        return

    ee_xy = x_arr[:, 4:6] + x_arr[:, 8:10]
    n = min(ee_xy.shape[0], g_arr.shape[0] if g_arr.ndim == 2 else 0)
    if n <= 0:
        return
    ee_xy = ee_xy[:n]
    g_arr = g_arr[:n]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.plot(ee_xy[:, 0], ee_xy[:, 1], color="#1f77b4", linewidth=2.0, label="trajectory")
    ax.plot(g_arr[:, 0], g_arr[:, 1], color="#d62728", linewidth=1.8, linestyle="--", label="goal")
    ax.scatter(ee_xy[0, 0], ee_xy[0, 1], color="#1f77b4", marker="o", s=36, label="start")
    ax.scatter(ee_xy[-1, 0], ee_xy[-1, 1], color="#1f77b4", marker="x", s=42, label="end")
    ax.scatter(g_arr[-1, 0], g_arr[-1, 1], color="#d62728", marker="*", s=85, label="goal(final)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("End-Effector Trajectory vs Goal (XY)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, filename))
    if show:
        plt.show()


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
    parser.add_argument("--Kmin", type=int, default=60)
    parser.add_argument("--Kmax", type=int, default=10000)
    parser.add_argument("--Kstep", type=int, default=1)
    parser.add_argument("--success_eps", type=float, default=0.01)
    parser.add_argument("--success_window", type=int, default=20)
    parser.add_argument("--num_extra_targets", type=int, default=0)
    parser.add_argument("--enable_measurement_constraint", action="store_true")
    parser.add_argument("--N_loc", type=int, default=1000)
    parser.add_argument("--d_max", type=float, default=10.0)
    parser.add_argument("--d_gate", type=parse_bool_arg, nargs="?", const=True, default=False)
    parser.add_argument("--cond_gate", type=parse_bool_arg, nargs="?", const=True, default=False)
    parser.add_argument("--sigma_bar", type=float, default=1e-6)

    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Enable MuJoCo rendering and RecordVideo output.",
    )
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
        data,
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
        record_video=bool(args.record_video),
    )


    use_select_baseline = (not bool(args.d_gate)) and (not bool(args.cond_gate))
    if use_select_baseline:
        deepc = SelectDeePC(
            controller_args,
            selector_callback=LkSelector(order=2),
            num_hankel_cols=int(args.K),
            n_iter=int(args.n_iter),
        )
    else:
        deepc = AdaptiveSelectDeePC(
            controller_args,
            selector_callback=AdaptiveLkSelector(order=2),
            sigma_bar=float(args.sigma_bar),
            N_loc=int(args.N_loc),
            d_max=float(args.d_max),
            num_hankel_cols=int(args.K),
            n_iter=int(args.n_iter),
            K_min=int(args.Kmin),
            K_max=int(args.Kmax),
            K_step=int(args.Kstep),
            d_gate_enabled=bool(args.d_gate),
            cond_gate_enabled=bool(args.cond_gate),
        )

    scheduler = ReacherSetpointScheduler(
        env, controller_args.deepc_dims, num_extra_targets=int(args.num_extra_targets)
    )
    x_traj, u_traj, _, _, cost_obj, _ = run_reacher_simulator(
        env,
        deepc,
        scheduler,
        seed=int(args.seed),
        deepc_cost_accumulator=PerformanceAccumulatorWrapper(
            DeePCCostAccumulator(controller_args.controller_costs),
            IntegralSquareError(mask=np.array([0, 0, 0, 0, 1, 1, 0, 0])),
            IntegralAbsoluteError(mask=np.array([0, 0, 0, 0, 1, 1, 0, 0])),
        ),
    )

    target_traj = replay_targets(
        np.asarray(x_traj, dtype=float),
        target_mode="setpoint",
        num_extra_targets=int(args.num_extra_targets),
        circle_center_x=0.03,
        circle_center_y=0.06,
        circle_radius=0.02,
        circle_omega=0.05,
        circle_phase=0.0,
        circle_clip_y_max=0.09,
    )
    err = compute_tracking_error(np.asarray(x_traj, dtype=float), target_traj)
    success = compute_success(err, eps=float(args.success_eps), window=int(args.success_window))
    plot_xy_trajectory_with_goal(x_traj, target_traj, args.outdir)

    mode_tag = "adaptive" if args.adaptive_k else "fixed"
    out_npz = os.path.join(
        args.outdir,
        f"run_{mode_tag}_K{int(args.K):03d}_rho{float(args.rho):.3g}_seed{int(args.seed):03d}.npz",
    )
    if hasattr(deepc, "save_history_npz"):
        deepc.save_history_npz(
            out_npz,
            extra={
                "seed": int(args.seed),
                "K": int(args.K),
                "total_cost": float(get_cost_dict(cost_obj).get("cost", np.nan)),
                "rmse_ee": float(np.sqrt(safe_stat(err * err, np.mean, default=np.nan))),
                "success": int(success),
                "N_loc": int(args.N_loc),
                "d_max": float(args.d_max),
                "sigma_bar": float(args.sigma_bar),
                "d_gate_flag": bool(args.d_gate),
                "cond_gate_flag": bool(args.cond_gate),
            },
        )
    else:
        np.savez_compressed(
            out_npz,
            seed=int(args.seed),
            K=int(args.K),
            total_cost=float(get_cost_dict(cost_obj).get("cost", np.nan)),
            rmse_ee=float(np.sqrt(safe_stat(err * err, np.mean, default=np.nan))),
            success=int(success),
            N_loc=int(args.N_loc),
            d_max=float(args.d_max),
            sigma_bar=float(args.sigma_bar),
            d_gate_flag=bool(args.d_gate),
            cond_gate_flag=bool(args.cond_gate),
        )

    if hasattr(deepc, "_K_opt_history"):
        K_opts = np.asarray(deepc._K_opt_history, dtype=int)
    else:
        K_opts = np.full(max(1, len(x_traj)), int(args.K), dtype=int)
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
        f"solve_ms={safe_stat(history_or_default(deepc, 'get_solve_time_ms_history', len(x_traj)), np.mean):.2f}"
        f"local_gate_fail={int(getattr(deepc, '_local_gate_fail', 0))} "
        f"cond_gate_fail={int(getattr(deepc, '_cond_gate_fail', 0))} "
    )
    import matplotlib.pyplot as plt

    sigma_min_mk = np.asarray(
        history_or_default(deepc, "get_sigma_min_Mk_history", len(x_traj)), dtype=float
    ).reshape(-1)
    solve_time_ms = np.asarray(
        history_or_default(deepc, "get_solve_time_ms_history", len(x_traj)), dtype=float
    ).reshape(-1)
    slack_norm_inf = np.asarray(
        history_or_default(deepc, "get_slack_norm_inf_history", len(x_traj)), dtype=float
    ).reshape(-1)
    slack_violation = np.asarray(
        history_or_default(deepc, "get_slack_violation_history", len(x_traj), default=0.0), dtype=float
    ).reshape(-1)
    input_hist = np.asarray(
        history_or_default(deepc, "get_input_history", len(x_traj), default=0.0), dtype=float
    )
    if input_hist.ndim == 1:
        input_hist = input_hist.reshape(-1, 1)

    # K_opt can be appended twice per step in current controller path.
    n_step = int(sigma_min_mk.size)
    k_opts_plot = np.asarray(K_opts, dtype=float).reshape(-1)
    if k_opts_plot.size == 2 * n_step:
        k_opts_plot = k_opts_plot.reshape(n_step, 2)[:, 1]
    elif k_opts_plot.size > n_step:
        k_opts_plot = k_opts_plot[:n_step]

    status_raw = np.asarray(
        history_or_default(deepc, "get_status_history", len(x_traj), default="unknown", dtype=object),
        dtype=object,
    ).reshape(-1)
    if status_raw.size > n_step:
        status_raw = status_raw[:n_step]
    if status_raw.size > 0 and isinstance(status_raw[0], str):
        uniq = list(dict.fromkeys(status_raw.tolist()))
        status_map = {s: i for i, s in enumerate(uniq)}
        status_num = np.array([status_map[s] for s in status_raw], dtype=float)
    else:
        status_num = status_raw.astype(float) if status_raw.size else np.array([], dtype=float)

    # Align all traces to common length.
    n = min(
        n_step,
        k_opts_plot.size,
        solve_time_ms.size,
        slack_norm_inf.size,
        slack_violation.size,
        input_hist.shape[0],
        status_num.size if status_num.size > 0 else n_step,
    )
    t = np.arange(n, dtype=int)
    k_opts_plot = k_opts_plot[:n]
    sigma_min_mk = sigma_min_mk[:n]
    solve_time_ms = solve_time_ms[:n]
    slack_norm_inf = slack_norm_inf[:n]
    slack_violation = slack_violation[:n]
    input_hist = input_hist[:n, :]
    status_num = status_num[:n] if status_num.size else np.zeros(n, dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), dpi=160, sharex=True)
    axes = axes.flatten()

    axes[0].plot(t, k_opts_plot, label="K_opt", color="#1f77b4")
    axes[0].set_ylabel("K_opt")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, sigma_min_mk, label="sigma_min_Mk", color="#ff7f0e")
    axes[1].set_ylabel("sigma_min_Mk")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for j in range(input_hist.shape[1]):
        axes[2].plot(t, input_hist[:, j], label=f"u{j}")
    axes[2].set_ylabel("input")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=min(2, input_hist.shape[1]))

    axes[3].plot(t, solve_time_ms, label="solve_time_ms", color="#2ca02c")
    axes[3].set_ylabel("solve_time_ms")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    axes[4].plot(t, status_num, label="status(code)", color="#9467bd")
    axes[4].set_ylabel("status")
    axes[4].set_xlabel("step k")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    axes[5].plot(t, slack_norm_inf, label="slack_norm_inf", color="#8c564b")
    axes[5].plot(t, slack_violation, label="slack_violation", color="#d62728", alpha=0.85)
    axes[5].set_ylabel("slack")
    axes[5].set_xlabel("step k")
    axes[5].grid(True, alpha=0.3)
    axes[5].legend()

    fig.suptitle("Controller Diagnostics by Time Step", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "fig_diagnostics_subplot.png"))
    # Backward compatibility: keep legacy single-figure outputs too.
    plt.figure(dpi=160)
    plt.plot(t, k_opts_plot, label="K_opt")
    plt.xlabel("step k")
    plt.ylabel("K_opt")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_K_opts.png"))

    plt.figure(dpi=160)
    plt.plot(t, sigma_min_mk, label="sigma_min_Mk")
    plt.xlabel("step k")
    plt.ylabel("sigma_min_Mk")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_sigma_min_Mk.png"))

    plt.figure(dpi=160)
    for j in range(input_hist.shape[1]):
        plt.plot(t, input_hist[:, j], label=f"u{j}")
    plt.xlabel("step k")
    plt.ylabel("input")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=min(2, input_hist.shape[1]))
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_input.png"))

    plt.figure(dpi=160)
    plt.plot(t, solve_time_ms, label="solve_time_ms")
    plt.xlabel("step k")
    plt.ylabel("solve_time_ms")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_solve_time_ms.png"))

    plt.figure(dpi=160)
    plt.plot(t, status_num, label="status(code)")
    plt.xlabel("step k")
    plt.ylabel("status")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_status.png"))

    plt.figure(dpi=160)
    plt.plot(t, slack_norm_inf, label="slack_norm_inf")
    plt.plot(t, slack_violation, label="slack_violation")
    plt.xlabel("step k")
    plt.ylabel("slack (norm_inf & violation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_slack_norm_inf_and_violation.png"))
    plt.show()

if __name__ == "__main__":
    main()
