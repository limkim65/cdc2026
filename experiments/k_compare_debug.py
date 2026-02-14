import argparse
import importlib
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.registration import registry


# =========================
# Experiment configuration
# =========================
SIGMA_Y_DEFAULT = 0.05
K_LIST_DEFAULT = [40, 50]
TRIALS_DEFAULT = 10
T_SIM_DEFAULT = 300
REF_SWITCH_STEP_DEFAULT = 100
REF_BEFORE_DEFAULT = 0.0
REF_AFTER_DEFAULT = 0.5
OFFLINE_CACHE_DEFAULT = os.path.join("results", "offline_dataset_cache.pkl")


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
from custom.setup_deepc import selector_cb, setup_DeePC  # noqa: E402
from select_deepc.data_selectors import LkSelector  # noqa: E402
from select_deepc.deepc_controller import SelectDeePC  # noqa: E402
from select_deepc.deepc_utils import DeePCCostAccumulator  # noqa: E402


def ensure_env_registered():
    env_id = "InvertedPendulum-v4-swingup"
    if env_id in registry:
        return
    gym.envs.register(
        env_id,
        entry_point="inverted_pendulum_v4_swingup:InvertedPendulumSwingupEnv",
        max_episode_steps=1000,
        reward_threshold=950.0,
    )


def check_runtime_dependencies():
    try:
        importlib.import_module("mujoco")
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: `mujoco`.\n"
            "Install with: pip install \"gymnasium[mujoco]\"\n"
            "If needed on Windows: pip install mujoco"
        ) from exc


def featurize_obs(obs_raw: np.ndarray) -> np.ndarray:
    return np.array(
        [obs_raw[0], np.sin(obs_raw[1]), np.cos(obs_raw[1]), obs_raw[2], obs_raw[3]],
        dtype=float,
    )


def get_dt(env) -> float:
    if hasattr(env.unwrapped, "dt"):
        return float(env.unwrapped.dt)
    try:
        return float(env.unwrapped.model.opt.timestep * env.unwrapped.frame_skip)
    except Exception:
        return 0.02


def wrapped_angle_error_to_pi(theta: float) -> float:
    return abs((theta - np.pi + np.pi) % (2 * np.pi) - np.pi)


class DebugSelector:
    """Wrap existing selector and log sigma_min(U_ini_selected) per step."""

    def __init__(self, base_selector, k_fixed: int):
        self.base_selector = base_selector
        self.k_fixed = int(k_fixed)
        self.sigma_min_hist = []

    def __call__(self, input_traj, state_traj, H_u, H_y, reference):
        idcs, norms = self.base_selector(input_traj, state_traj, H_u, H_y, reference)
        used_idcs = idcs[: self.k_fixed]

        sigma_min = np.nan
        if used_idcs.size > 0:
            # U_ini_selected here is selected input Hankel block used by controller.
            U_ini_selected = H_u[:, used_idcs]
            G = U_ini_selected @ U_ini_selected.T
            eig_min = float(np.min(np.linalg.eigvalsh(G)))
            sigma_min = float(np.sqrt(max(eig_min, 0.0)))
        self.sigma_min_hist.append(sigma_min)
        return idcs, norms


def build_controller(controller_args, k: int):
    base_selector = LkSelector(
        1,
        controller_args.deepc_dims,
        custom_callback=selector_cb,
        forgetting_factor=0.8,
    )
    debug_selector = DebugSelector(base_selector, k_fixed=k)
    controller = SelectDeePC(
        controller_args,
        debug_selector,
        num_hankel_cols=k,
        n_iter=1,
        debug=False,
    )
    return controller


def run_trial(
    controller,
    k_val: int,
    sigma_y: float,
    reset_seed: int,
    noise_seed: int,
    t_sim: int,
    ref_switch_step: int,
    ref_before: float,
    ref_after: float,
):
    env = gym.make("InvertedPendulum-v4-swingup")
    obs_raw, _ = env.reset(seed=reset_seed)
    dt = get_dt(env)
    rng_noise = np.random.default_rng(noise_seed)

    settle_steps = int(np.ceil(0.5 / dt))
    consecutive_ok = 0
    success = False
    time_to_upright = t_sim * dt

    theta_traj = []
    cos_traj = []
    x_traj = []
    solver_status = []
    solve_times = []

    acc = DeePCCostAccumulator(controller._controller_args[0])
    _ = acc

    for step in range(t_sim):
        obs_true = featurize_obs(np.asarray(obs_raw, dtype=float))
        obs_meas = obs_true + rng_noise.normal(0.0, sigma_y, size=obs_true.shape)

        x_ref = ref_before if step <= ref_switch_step else ref_after
        ref = np.array([x_ref, 0.0, 1.0, 0.0, 0.0], dtype=float)

        t0 = time.perf_counter()
        action = controller.compute_action(obs_meas, ref)
        solve_times.append(time.perf_counter() - t0)

        st = "unknown"
        try:
            st = str(controller._deepc._problem.status)
        except Exception:
            pass
        solver_status.append(st)

        obs_raw, _, terminated, truncated, _ = env.step(action)
        theta = float(obs_raw[1])
        theta_dot = float(obs_raw[3])
        theta_traj.append(theta)
        cos_traj.append(float(np.cos(theta)))
        x_traj.append(float(obs_raw[0]))

        if (
            wrapped_angle_error_to_pi(theta) <= np.deg2rad(15.0)
            and abs(theta_dot) <= 1.0
        ):
            consecutive_ok += 1
            if (not success) and consecutive_ok >= settle_steps:
                success = True
                time_to_upright = (step + 1) * dt
        else:
            consecutive_ok = 0

        if terminated or truncated:
            break

    env.close()

    sigma_min_hist = np.asarray(controller._selector_callback.sigma_min_hist, dtype=float)
    solver_status_arr = np.asarray(solver_status, dtype=object)

    collapse_step = None
    for i, st in enumerate(solver_status_arr):
        if "optimal" not in str(st):
            collapse_step = i
            break

    return {
        "K": int(k_val),
        "success": bool(success),
        "time_to_upright": float(time_to_upright),
        "avg_solve_time": float(np.mean(solve_times)) if solve_times else np.nan,
        "theta_traj": np.asarray(theta_traj, dtype=float),
        "cos_theta": np.asarray(cos_traj, dtype=float),
        "x_traj": np.asarray(x_traj, dtype=float),
        "sigma_min_traj": sigma_min_hist,
        "solver_status": solver_status_arr,
        "collapse_step": collapse_step,
    }


def save_trial_npz(path: str, run: dict, seed_value: int):
    np.savez(
        path,
        theta=run["theta_traj"],
        cos_theta=run["cos_theta"],
        x=run["x_traj"],
        sigma_min=run["sigma_min_traj"],
        solver_status=run["solver_status"],
        K=np.array(run["K"], dtype=int),
        seed=np.array(seed_value, dtype=int),
    )


def main():
    parser = argparse.ArgumentParser(description="K=40 vs K=50 debug run with shared seeds.")
    parser.add_argument("--trials", type=int, default=TRIALS_DEFAULT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sigma-y", type=float, default=SIGMA_Y_DEFAULT)
    parser.add_argument("--k-list", type=int, nargs="+", default=K_LIST_DEFAULT)
    parser.add_argument("--t-sim", type=int, default=T_SIM_DEFAULT)
    parser.add_argument("--ref-switch-step", type=int, default=REF_SWITCH_STEP_DEFAULT)
    parser.add_argument("--ref-before", type=float, default=REF_BEFORE_DEFAULT)
    parser.add_argument("--ref-after", type=float, default=REF_AFTER_DEFAULT)
    parser.add_argument(
        "--offline-cache",
        type=str,
        default=os.path.join(REPO_ROOT, OFFLINE_CACHE_DEFAULT),
    )
    parser.add_argument("--rebuild-offline", action="store_true")
    parser.add_argument("--plot-trial", type=int, default=0)
    parser.add_argument(
        "--exp-tag",
        type=str,
        default="",
        help="Optional custom experiment tag used in output filenames.",
    )
    args = parser.parse_args()

    check_runtime_dependencies()
    ensure_env_registered()
    os.makedirs(os.path.join(REPO_ROOT, "results"), exist_ok=True)

    if (not args.rebuild_offline) and os.path.exists(args.offline_cache):
        with open(args.offline_cache, "rb") as f:
            trajectory_data = pickle.load(f)
    else:
        _, reset_time, u_off, y_off = collect_closedloop_controller_data(animation=False)
        trajectory_data = create_trajectory_dataset(reset_time, u_off, y_off, "swingup_data")
        with open(args.offline_cache, "wb") as f:
            pickle.dump(trajectory_data, f)

    controller_args = setup_DeePC(trajectory_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_tag = (
        f"sigma{args.sigma_y}_seed{args.seed}_tr{args.trials}_"
        f"k{'-'.join([str(k) for k in args.k_list])}_{timestamp}"
    )
    exp_tag = args.exp_tag if args.exp_tag else auto_tag

    rng_master = np.random.default_rng(args.seed)
    reset_seeds = rng_master.integers(0, 10_000_000, size=args.trials, endpoint=False)
    noise_seeds = rng_master.integers(0, 10_000_000, size=args.trials, endpoint=False)

    # results_by_k[K][trial] = run dict
    results_by_k = defaultdict(list)

    for trial_idx in range(args.trials):
        rs = int(reset_seeds[trial_idx])
        ns = int(noise_seeds[trial_idx])
        for k_val in args.k_list:
            controller = build_controller(controller_args, k_val)
            run = run_trial(
                controller=controller,
                k_val=int(k_val),
                sigma_y=float(args.sigma_y),
                reset_seed=rs,
                noise_seed=ns,
                t_sim=int(args.t_sim),
                ref_switch_step=int(args.ref_switch_step),
                ref_before=float(args.ref_before),
                ref_after=float(args.ref_after),
            )
            results_by_k[int(k_val)].append(run)

            out_npz = os.path.join(
                REPO_ROOT,
                "results",
                f"k_compare_debug_{exp_tag}_trial{trial_idx}_K{k_val}.npz",
            )
            save_trial_npz(out_npz, run, seed_value=rs)

    print("K | success_rate | mean_time_to_upright | avg_solve_time")
    print("-" * 62)
    for k_val in args.k_list:
        runs = results_by_k[int(k_val)]
        success_rate = float(np.mean([1.0 if r["success"] else 0.0 for r in runs]))
        mean_ttu = float(np.mean([r["time_to_upright"] for r in runs]))
        avg_solve = float(np.mean([r["avg_solve_time"] for r in runs]))
        print(f"{k_val:2d} | {success_rate:12.3f} | {mean_ttu:20.3f} | {avg_solve:14.6f}")

        collapse_steps = [
            (idx, r["collapse_step"])
            for idx, r in enumerate(runs)
            if r["collapse_step"] is not None
        ]
        if collapse_steps:
            steps_str = ", ".join([f"trial{ti}:step{st}" for ti, st in collapse_steps])
            print(f"  collapsed: {steps_str}")
        else:
            print("  collapsed: none")

    # Plot all trajectories (all trials) so full experiment behavior is visible.
    fig_all, axes_all = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    fig_all.suptitle(
        f"All trajectories across trials (sigma_y={args.sigma_y}, trials={args.trials})",
        fontsize=11,
    )
    color_map = {int(k): c for k, c in zip(args.k_list, ["tab:blue", "tab:orange", "tab:green", "tab:red"])}

    for k_val in args.k_list:
        runs = results_by_k[int(k_val)]
        c = color_map[int(k_val)]

        max_len = max(len(r["x_traj"]) for r in runs)
        X = np.full((len(runs), max_len), np.nan)
        TH = np.full((len(runs), max_len), np.nan)
        COS = np.full((len(runs), max_len), np.nan)
        SIG = np.full((len(runs), max_len), np.nan)

        for i, r in enumerate(runs):
            n_x = len(r["x_traj"])
            n_th = len(r["theta_traj"])
            n_cos = len(r["cos_theta"])
            n_sig = len(r["sigma_min_traj"])
            X[i, :n_x] = r["x_traj"]
            TH[i, :n_th] = r["theta_traj"]
            COS[i, :n_cos] = r["cos_theta"]
            SIG[i, :n_sig] = r["sigma_min_traj"]

            axes_all[0].plot(np.arange(n_x), r["x_traj"], color=c, alpha=0.2, linewidth=1)
            axes_all[1].plot(np.arange(n_th), r["theta_traj"], color=c, alpha=0.2, linewidth=1)
            axes_all[2].plot(np.arange(n_cos), r["cos_theta"], color=c, alpha=0.2, linewidth=1)
            axes_all[3].plot(np.arange(n_sig), r["sigma_min_traj"], color=c, alpha=0.2, linewidth=1)

        axes_all[0].plot(np.nanmean(X, axis=0), color=c, linewidth=2.0, label=f"K={k_val} mean")
        axes_all[1].plot(np.nanmean(TH, axis=0), color=c, linewidth=2.0, label=f"K={k_val} mean")
        axes_all[2].plot(np.nanmean(COS, axis=0), color=c, linewidth=2.0, label=f"K={k_val} mean")
        axes_all[3].plot(np.nanmean(SIG, axis=0), color=c, linewidth=2.0, label=f"K={k_val} mean")

    ref_line = np.where(
        np.arange(args.t_sim) <= args.ref_switch_step, args.ref_before, args.ref_after
    )
    axes_all[0].plot(ref_line, "k--", linewidth=1, label="x reference")
    axes_all[1].axhline(np.pi, color="k", linestyle="--", linewidth=1, label="upright pi")
    axes_all[2].axhline(1.0, color="k", linestyle="--", linewidth=1, label="upright cos(theta)=1")

    axes_all[0].set_ylabel("x")
    axes_all[0].set_title("Position trajectories (all trials)")
    axes_all[1].set_ylabel("theta [rad]")
    axes_all[1].set_title("Theta trajectories (all trials)")
    axes_all[2].set_ylabel("cos(theta)")
    axes_all[2].set_title("Cos(theta) trajectories (all trials)")
    axes_all[3].set_ylabel("sigma_min")
    axes_all[3].set_title("Sigma-min trajectories (all trials)")
    axes_all[3].set_xlabel("step")

    for ax in axes_all:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig_all.tight_layout()
    out_all = os.path.join(REPO_ROOT, "results", f"k_compare_debug_all_trajectories_{exp_tag}.png")
    fig_all.savefig(out_all, dpi=200, bbox_inches="tight")

    # Plot one matched trial (same seeds) for K=40 vs K=50 comparison.
    trial_for_plot = int(np.clip(args.plot_trial, 0, args.trials - 1))
    if 40 in results_by_k and 50 in results_by_k:
        run40 = results_by_k[40][trial_for_plot]
        run50 = results_by_k[50][trial_for_plot]
    else:
        k0, k1 = int(args.k_list[0]), int(args.k_list[1])
        run40 = results_by_k[k0][trial_for_plot]
        run50 = results_by_k[k1][trial_for_plot]

    # Detailed multi-view plot (state + sigma_min).
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    fig.suptitle(
        f"K comparison debug (trial {trial_for_plot}, sigma_y={args.sigma_y})",
        fontsize=11,
    )
    t40 = np.arange(run40["theta_traj"].shape[0])
    t50 = np.arange(run50["theta_traj"].shape[0])
    ref40 = np.where(t40 <= args.ref_switch_step, args.ref_before, args.ref_after)
    ref50 = np.where(t50 <= args.ref_switch_step, args.ref_before, args.ref_after)

    # Position trajectory is often the most intuitive instability indicator.
    axes[0].plot(t40, run40["x_traj"], label=f"K={run40['K']} x")
    axes[0].plot(t50, run50["x_traj"], label=f"K={run50['K']} x")
    axes[0].plot(t40, ref40, "k--", linewidth=1, label="x reference")
    axes[0].set_ylabel("x")
    axes[0].set_title("Position trajectory")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t40, run40["theta_traj"], label=f"K={run40['K']}")
    axes[1].plot(t50, run50["theta_traj"], label=f"K={run50['K']}")
    axes[1].axhline(np.pi, color="k", linestyle="--", linewidth=1, label="upright pi")
    axes[1].axhline(np.pi + np.deg2rad(15), color="gray", linestyle=":", linewidth=1)
    axes[1].axhline(np.pi - np.deg2rad(15), color="gray", linestyle=":", linewidth=1)
    axes[1].set_ylabel("theta [rad]")
    axes[1].set_title("Theta trajectory")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t40, run40["cos_theta"], label=f"K={run40['K']} cos(theta)")
    axes[2].plot(t50, run50["cos_theta"], label=f"K={run50['K']} cos(theta)")
    axes[2].axhline(1.0, color="k", linestyle="--", linewidth=1, label="upright cos(theta)=1")
    axes[2].set_ylabel("cos(theta)")
    axes[2].set_title("Cos(theta) trajectory")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    s40 = run40["sigma_min_traj"]
    s50 = run50["sigma_min_traj"]
    axes[3].plot(np.arange(s40.shape[0]), s40, label=f"K={run40['K']}")
    axes[3].plot(np.arange(s50.shape[0]), s50, label=f"K={run50['K']}")
    axes[3].set_xlabel("step")
    axes[3].set_ylabel("sigma_min(U_ini_selected)")
    axes[3].set_title("Sigma-min trajectory")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    # Mark first non-optimal step to visually connect collapse and state divergence.
    if run40["collapse_step"] is not None:
        for ax in axes:
            ax.axvline(run40["collapse_step"], color="tab:blue", linestyle=":", alpha=0.7)
    if run50["collapse_step"] is not None:
        for ax in axes:
            ax.axvline(run50["collapse_step"], color="tab:orange", linestyle=":", alpha=0.7)

    fig.tight_layout()
    out_plot = os.path.join(REPO_ROOT, "results", f"k40_vs_k50_sigma_debug_{exp_tag}.png")
    fig.savefig(out_plot, dpi=200, bbox_inches="tight")

    # Keep previous compact plot style too.
    fig_legacy, axes_legacy = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes_legacy[0].plot(t40, run40["theta_traj"], label=f"K={run40['K']}")
    axes_legacy[0].plot(t50, run50["theta_traj"], label=f"K={run50['K']}")
    axes_legacy[0].axhline(np.pi, color="k", linestyle="--", linewidth=1, label="upright pi")
    axes_legacy[0].set_ylabel("theta [rad]")
    axes_legacy[0].set_title(
        f"Theta comparison (trial {trial_for_plot}, sigma_y={args.sigma_y})"
    )
    axes_legacy[0].grid(True, alpha=0.3)
    axes_legacy[0].legend()
    axes_legacy[1].plot(np.arange(s40.shape[0]), s40, label=f"K={run40['K']}")
    axes_legacy[1].plot(np.arange(s50.shape[0]), s50, label=f"K={run50['K']}")
    axes_legacy[1].set_xlabel("step")
    axes_legacy[1].set_ylabel("sigma_min(U_ini_selected)")
    axes_legacy[1].set_title("Sigma-min trajectory")
    axes_legacy[1].grid(True, alpha=0.3)
    axes_legacy[1].legend()
    fig_legacy.tight_layout()
    out_legacy = os.path.join(
        REPO_ROOT, "results", f"k40_vs_k50_sigma_debug_legacy_{exp_tag}.png"
    )
    fig_legacy.savefig(out_legacy, dpi=200, bbox_inches="tight")

    # Additional view 1: phase portrait.
    theta_dot_40 = np.gradient(run40["theta_traj"])
    theta_dot_50 = np.gradient(run50["theta_traj"])
    fig_phase, ax_phase = plt.subplots(1, 1, figsize=(6, 5))
    ax_phase.plot(run40["theta_traj"], theta_dot_40, label=f"K={run40['K']}", alpha=0.9)
    ax_phase.plot(run50["theta_traj"], theta_dot_50, label=f"K={run50['K']}", alpha=0.9)
    ax_phase.set_xlabel("theta [rad]")
    ax_phase.set_ylabel("d(theta)/d(step)")
    ax_phase.set_title("Phase-style view of angular dynamics")
    ax_phase.grid(True, alpha=0.3)
    ax_phase.legend()
    fig_phase.tight_layout()
    out_phase = os.path.join(REPO_ROOT, "results", f"k40_vs_k50_phase_debug_{exp_tag}.png")
    fig_phase.savefig(out_phase, dpi=200, bbox_inches="tight")

    # Additional view 2: solver optimal/non-optimal timeline.
    status40 = np.array([1 if "optimal" in str(s) else 0 for s in run40["solver_status"]])
    status50 = np.array([1 if "optimal" in str(s) else 0 for s in run50["solver_status"]])
    fig_status, ax_status = plt.subplots(1, 1, figsize=(9, 3))
    ax_status.step(np.arange(status40.size), status40, where="post", label=f"K={run40['K']}")
    ax_status.step(np.arange(status50.size), status50, where="post", label=f"K={run50['K']}")
    ax_status.set_ylim(-0.1, 1.1)
    ax_status.set_yticks([0, 1])
    ax_status.set_yticklabels(["non-optimal", "optimal"])
    ax_status.set_xlabel("step")
    ax_status.set_title("Solver status timeline")
    ax_status.grid(True, alpha=0.3)
    ax_status.legend()
    fig_status.tight_layout()
    out_status = os.path.join(
        REPO_ROOT, "results", f"k40_vs_k50_solver_status_debug_{exp_tag}.png"
    )
    fig_status.savefig(out_status, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
    # run:
    # python experiments/k_compare_debug.py --trials 10
