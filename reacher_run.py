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


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def configure_matplotlib(show_plots: bool):
    import matplotlib

    backend = str(matplotlib.get_backend()).lower()
    if (not show_plots) or ("agg" not in backend):
        return
    if os.name == "nt":
        try:
            matplotlib.use("TkAgg", force=True)
        except Exception:
            pass


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
    if x_arr.shape[1] >= 10:
        ee_xy = x_arr[:, 4:6] + x_arr[:, 8:10]
    elif x_arr.shape[1] >= 6:
        ee_xy = x_arr[:, 4:6]
    else:
        return
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
    out_path = os.path.join(outdir, filename)
    fig.savefig(out_path)
    print(f"[plot] saved: {out_path}")
    if show:
        plt.show(block=False)
        plt.pause(0.1)


def plot_xy_components_vs_time(
    x_traj,
    goal_xy_traj,
    outdir,
    filename="fig_xy_components_vs_time.png",
    show=False,
):
    import matplotlib.pyplot as plt

    x_arr = np.asarray(x_traj, dtype=float)
    g_arr = np.asarray(goal_xy_traj, dtype=float)
    if x_arr.ndim != 2 or x_arr.shape[0] == 0 or g_arr.ndim != 2:
        return

    if x_arr.shape[1] >= 10:
        ee_xy = x_arr[:, 4:6] + x_arr[:, 8:10]
    elif x_arr.shape[1] >= 6:
        ee_xy = x_arr[:, 4:6]
    else:
        return

    n = min(ee_xy.shape[0], g_arr.shape[0])
    if n <= 0:
        return

    ee_xy = ee_xy[:n]
    g_arr = g_arr[:n]
    t = np.arange(n, dtype=int)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=160, sharex=True)

    axes[0].plot(t, ee_xy[:, 0], color="#1f77b4", linewidth=1.8, label="x(t) traj")
    axes[0].plot(t, g_arr[:, 0], color="#d62728", linewidth=1.5, linestyle="--", label="x(t) ref")
    axes[0].set_ylabel("x")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, ee_xy[:, 1], color="#1f77b4", linewidth=1.8, label="y(t) traj")
    axes[1].plot(t, g_arr[:, 1], color="#d62728", linewidth=1.5, linestyle="--", label="y(t) ref")
    axes[1].set_xlabel("step k")
    axes[1].set_ylabel("y")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("End-Effector Components vs Reference")
    fig.tight_layout()
    out_path = os.path.join(outdir, filename)
    fig.savefig(out_path)
    print(f"[plot] saved: {out_path}")
    if show:
        plt.show(block=False)
        plt.pause(0.1)


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
    parser.add_argument(
        "--show_plots",
        type=parse_bool_arg,
        nargs="?",
        const=True,
        default=True,
        help="Show plot windows interactively (default: True).",
    )
    args = parser.parse_args()
    configure_matplotlib(bool(args.show_plots))

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
    # Do not block here; all diagnostics are shown together at the end.
    plot_xy_trajectory_with_goal(x_traj, target_traj, args.outdir, show=False)
    plot_xy_components_vs_time(x_traj, target_traj, args.outdir, show=False)

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
            "total_cost": float(get_cost_dict(cost_obj).get("cost", np.nan)),
            "rmse_ee": float(np.sqrt(safe_stat(err * err, np.mean, default=np.nan))),
            "success": int(success),
            "N_loc": int(args.N_loc),
            "d_max": float(args.d_max),
            "sigma_bar": float(args.sigma_bar),
            "sigma_min_all": (
                float(deepc.get_sigma_min_all())
                if hasattr(deepc, "get_sigma_min_all")
                else np.nan
            ),
            "d_gate_flag": bool(args.d_gate),
            "cond_gate_flag": bool(args.cond_gate),
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
    import matplotlib.pyplot as plt

    sigma_min_mk = np.asarray(deepc.get_sigma_min_Mk_history(), dtype=float).reshape(-1)
    solve_time_ms = np.asarray(deepc.get_solve_time_ms_history(), dtype=float).reshape(-1)
    slack_norm_inf = np.asarray(deepc.get_slack_norm_inf_history(), dtype=float).reshape(-1)
    slack_violation = np.asarray(deepc.get_slack_violation_history(), dtype=float).reshape(-1)
    input_hist = np.asarray(deepc.get_input_history(), dtype=float)
    if input_hist.ndim == 1:
        input_hist = input_hist.reshape(-1, 1)

    # K_opt can be appended twice per step in some controller paths.
    n_step = int(sigma_min_mk.size)
    k_opts_plot = np.asarray(K_opts, dtype=float).reshape(-1)
    if n_step > 0 and k_opts_plot.size == 2 * n_step:
        k_opts_plot = k_opts_plot.reshape(n_step, 2)[:, 1]

    status_raw = np.asarray(deepc.get_status_history(), dtype=object).reshape(-1)
    if status_raw.size > n_step:
        status_raw = status_raw[:n_step]
    if status_raw.size > 0 and isinstance(status_raw[0], str):
        uniq = list(dict.fromkeys(status_raw.tolist()))
        status_map = {s: i for i, s in enumerate(uniq)}
        status_num = np.array([status_map[s] for s in status_raw], dtype=float)
    else:
        status_num = status_raw.astype(float) if status_raw.size else np.array([], dtype=float)

    # Robust alignment:
    # do NOT anchor everything to sigma_min_mk, because some controller variants
    # may not populate that history while other metrics are valid.
    n = max(
        k_opts_plot.size,
        sigma_min_mk.size,
        solve_time_ms.size,
        slack_norm_inf.size,
        slack_violation.size,
        input_hist.shape[0],
        status_num.size,
    )
    if n <= 0:
        print("[warn] no diagnostic history available to plot.")
        return

    def _pad_1d(arr, n_out, fill=np.nan):
        out = np.full(int(n_out), float(fill), dtype=float)
        m = min(int(n_out), int(arr.size))
        if m > 0:
            out[:m] = arr[:m]
        return out

    t = np.arange(n, dtype=int)
    k_opts_plot = _pad_1d(k_opts_plot, n)
    sigma_min_mk = _pad_1d(sigma_min_mk, n)
    solve_time_ms = _pad_1d(solve_time_ms, n)
    slack_norm_inf = _pad_1d(slack_norm_inf, n)
    slack_violation = _pad_1d(slack_violation, n, fill=0.0)
    status_num = _pad_1d(status_num, n, fill=0.0)

    if input_hist.shape[0] < n:
        pad_rows = n - input_hist.shape[0]
        input_hist = np.vstack(
            [input_hist, np.full((pad_rows, input_hist.shape[1]), np.nan, dtype=float)]
        )
    else:
        input_hist = input_hist[:n, :]

    print(
        "[diag] lengths "
        f"K={k_opts_plot.size} sigmaMk={sigma_min_mk.size} solve={solve_time_ms.size} "
        f"slack={slack_norm_inf.size} status={status_num.size} input={input_hist.shape[0]}"
    )

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
    out_fig_diag = os.path.join(args.outdir, "fig_diagnostics_subplot.png")
    fig.savefig(out_fig_diag)
    print(f"[plot] saved: {out_fig_diag}")
    # Backward compatibility: keep legacy single-figure outputs too.
    plt.figure(dpi=160)
    plt.plot(t, k_opts_plot, label="K_opt")
    plt.xlabel("step k")
    plt.ylabel("K_opt")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_fig_k = os.path.join(args.outdir, "fig_K_opts.png")
    plt.savefig(out_fig_k)
    print(f"[plot] saved: {out_fig_k}")

    plt.figure(dpi=160)
    plt.plot(t, sigma_min_mk, label="sigma_min_Mk")
    plt.xlabel("step k")
    plt.ylabel("sigma_min_Mk")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_fig_sigma = os.path.join(args.outdir, "fig_sigma_min_Mk.png")
    plt.savefig(out_fig_sigma)
    print(f"[plot] saved: {out_fig_sigma}")

    plt.figure(dpi=160)
    for j in range(input_hist.shape[1]):
        plt.plot(t, input_hist[:, j], label=f"u{j}")
    plt.xlabel("step k")
    plt.ylabel("input")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=min(2, input_hist.shape[1]))
    plt.tight_layout()
    out_fig_input = os.path.join(args.outdir, "fig_input.png")
    plt.savefig(out_fig_input)
    print(f"[plot] saved: {out_fig_input}")

    plt.figure(dpi=160)
    plt.plot(t, solve_time_ms, label="solve_time_ms")
    plt.xlabel("step k")
    plt.ylabel("solve_time_ms")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_fig_solve = os.path.join(args.outdir, "fig_solve_time_ms.png")
    plt.savefig(out_fig_solve)
    print(f"[plot] saved: {out_fig_solve}")

    plt.figure(dpi=160)
    plt.plot(t, status_num, label="status(code)")
    plt.xlabel("step k")
    plt.ylabel("status")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_fig_status = os.path.join(args.outdir, "fig_status.png")
    plt.savefig(out_fig_status)
    print(f"[plot] saved: {out_fig_status}")

    plt.figure(dpi=160)
    plt.plot(t, slack_norm_inf, label="slack_norm_inf")
    plt.plot(t, slack_violation, label="slack_violation")
    plt.xlabel("step k")
    plt.ylabel("slack (norm_inf & violation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_fig_slack = os.path.join(args.outdir, "fig_slack_norm_inf_and_violation.png")
    plt.savefig(out_fig_slack)
    print(f"[plot] saved: {out_fig_slack}")
    if bool(args.show_plots):
        plt.show()
    else:
        plt.close("all")

if __name__ == "__main__":
    main()
