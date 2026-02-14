import argparse
import importlib
import os
import pickle
import sys
import time
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.registration import registry


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


def featurize_obs(obs_raw: np.ndarray) -> np.ndarray:
    return np.array(
        [obs_raw[0], np.sin(obs_raw[1]), np.cos(obs_raw[1]), obs_raw[2], obs_raw[3]],
        dtype=float,
    )


def angle_err_to_pi(theta: float) -> float:
    return abs((theta - np.pi + np.pi) % (2 * np.pi) - np.pi)


def get_dt(env) -> float:
    if hasattr(env.unwrapped, "dt"):
        return float(env.unwrapped.dt)
    try:
        return float(env.unwrapped.model.opt.timestep * env.unwrapped.frame_skip)
    except Exception:
        return 0.02


def check_runtime_dependencies():
    try:
        importlib.import_module("mujoco")
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: `mujoco`.\n"
            "Install with: pip install \"gymnasium[mujoco]\"\n"
            "If needed on Windows: pip install mujoco"
        ) from exc


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


def run_experiment_with_noise(
    controller,
    sigma_y: float,
    seed: int,
    t_sim: int,
    ref_switch_step: int,
    ref_before: float,
    ref_after: float,
):
    env = gym.make("InvertedPendulum-v4-swingup")
    obs_raw, _ = env.reset(seed=seed)
    dt = get_dt(env)

    rng = np.random.default_rng(seed + 1000)
    traj_true = []
    traj_meas = []
    refs = []
    solve_times = []
    solver_statuses = []

    settle_steps = int(np.ceil(0.5 / dt))
    consecutive_ok = 0
    success = False
    time_to_upright = t_sim * dt

    total_cost = DeePCCostAccumulator(controller._controller_args[0])

    for i in range(t_sim):
        obs_true = featurize_obs(np.asarray(obs_raw, dtype=float))
        obs_meas = obs_true + rng.normal(0.0, sigma_y, size=obs_true.shape)

        x_ref = ref_before if i <= ref_switch_step else ref_after
        ref = np.array([x_ref, 0.0, 1.0, 0.0, 0.0], dtype=float)

        t0 = time.perf_counter()
        action = controller.compute_action(obs_meas, ref)
        solve_times.append(time.perf_counter() - t0)
        status = "unknown"
        try:
            status = str(controller._deepc._problem.status)
        except Exception:
            pass
        solver_statuses.append(status)

        total_cost.update_cost(obs_true, ref, action)
        obs_raw, _, terminated, truncated, _ = env.step(action)

        theta = float(obs_raw[1])
        theta_dot = float(obs_raw[3])
        is_ok = angle_err_to_pi(theta) <= np.deg2rad(15.0) and abs(theta_dot) <= 1.0
        if is_ok:
            consecutive_ok += 1
            if (not success) and consecutive_ok >= settle_steps:
                success = True
                time_to_upright = (i + 1) * dt
        else:
            consecutive_ok = 0

        traj_true.append(obs_true)
        traj_meas.append(obs_meas)
        refs.append(ref)

        if terminated or truncated:
            break

    env.close()
    collapse_step = None
    collapse_status = None
    for idx, st in enumerate(solver_statuses):
        if "optimal" not in st:
            collapse_step = idx
            collapse_status = st
            break
    return {
        "traj_true": np.array(traj_true),
        "traj_meas": np.array(traj_meas),
        "refs": np.array(refs),
        "success": success,
        "time_to_upright": float(time_to_upright),
        "mean_solve_time": float(np.mean(solve_times)) if solve_times else np.nan,
        "cost": total_cost.cost,
        "solver_statuses": solver_statuses,
        "collapse_step": collapse_step,
        "collapse_status": collapse_status,
        "sigma_min_u_hist": np.asarray(
            controller.get_min_selected_singular_values(), dtype=float
        ),
    }


def build_controller(controller_args, k: int):
    return SelectDeePC(
        controller_args,
        LkSelector(
            1,
            controller_args.deepc_dims,
            custom_callback=selector_cb,
            forgetting_factor=0.8,
        ),
        num_hankel_cols=k,
        n_iter=1,
        debug=False,
    )


def write_k_comparison_diagnostic(
    path: str,
    run_k40: dict,
    run_k50: dict,
    k40_label: str = "K=40",
    k50_label: str = "K=50",
):
    collapse = run_k40["collapse_step"]
    if collapse is None:
        collapse = min(
            len(run_k40["solver_statuses"]),
            len(run_k40["sigma_min_u_hist"]),
            len(run_k50["solver_statuses"]),
            len(run_k50["sigma_min_u_hist"]),
        ) - 1

    start = max(0, collapse - 10)
    end = min(
        collapse + 10,
        len(run_k40["solver_statuses"]) - 1,
        len(run_k50["solver_statuses"]) - 1,
        len(run_k40["sigma_min_u_hist"]) - 1,
        len(run_k50["sigma_min_u_hist"]) - 1,
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("Diagnostic around K=40 collapse step\n")
        f.write(f"{k40_label} collapse_step={run_k40['collapse_step']}, collapse_status={run_k40['collapse_status']}\n")
        f.write(f"{k50_label} collapse_step={run_k50['collapse_step']}, collapse_status={run_k50['collapse_status']}\n")
        f.write("\n")
        f.write("step, sigma_min_u_k40, status_k40, sigma_min_u_k50, status_k50\n")
        for step in range(start, end + 1):
            s40 = run_k40["sigma_min_u_hist"][step]
            st40 = run_k40["solver_statuses"][step]
            s50 = run_k50["sigma_min_u_hist"][step]
            st50 = run_k50["solver_statuses"][step]
            f.write(f"{step}, {s40:.8e}, {st40}, {s50:.8e}, {st50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Simple verification script: does measurement noise break performance?"
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--t-sim", type=int, default=300)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--sigma-clean", type=float, default=0.0)
    parser.add_argument("--sigma-noisy", type=float, default=0.02)
    parser.add_argument("--ref-switch-step", type=int, default=100)
    parser.add_argument("--ref-before", type=float, default=0.0)
    parser.add_argument("--ref-after", type=float, default=0.5)
    parser.add_argument("--diag-k40", type=int, default=40)
    parser.add_argument("--diag-k50", type=int, default=50)
    parser.add_argument(
        "--offline-cache",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "offline_dataset_cache.pkl"),
    )
    parser.add_argument(
        "--rebuild-offline",
        action="store_true",
        help="Ignore cache and rebuild offline dataset.",
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
        trajectory_data = create_trajectory_dataset(
            reset_time, u_off, y_off, "swingup_data"
        )
        with open(args.offline_cache, "wb") as f:
            pickle.dump(trajectory_data, f)

    controller_args = setup_DeePC(trajectory_data)

    controller_clean = build_controller(controller_args, args.k)
    controller_noisy = build_controller(controller_args, args.k)

    clean = run_experiment_with_noise(
        controller_clean,
        sigma_y=args.sigma_clean,
        seed=args.seed,
        t_sim=args.t_sim,
        ref_switch_step=args.ref_switch_step,
        ref_before=args.ref_before,
        ref_after=args.ref_after,
    )
    noisy = run_experiment_with_noise(
        controller_noisy,
        sigma_y=args.sigma_noisy,
        seed=args.seed,
        t_sim=args.t_sim,
        ref_switch_step=args.ref_switch_step,
        ref_before=args.ref_before,
        ref_after=args.ref_after,
    )

    # Same-seed/noise K comparison diagnostic (default K=40 vs K=50).
    controller_diag_40 = build_controller(controller_args, args.diag_k40)
    controller_diag_50 = build_controller(controller_args, args.diag_k50)
    run_diag_40 = run_experiment_with_noise(
        controller_diag_40,
        sigma_y=args.sigma_noisy,
        seed=args.seed,
        t_sim=args.t_sim,
        ref_switch_step=args.ref_switch_step,
        ref_before=args.ref_before,
        ref_after=args.ref_after,
    )
    run_diag_50 = run_experiment_with_noise(
        controller_diag_50,
        sigma_y=args.sigma_noisy,
        seed=args.seed,
        t_sim=args.t_sim,
        ref_switch_step=args.ref_switch_step,
        ref_before=args.ref_before,
        ref_after=args.ref_after,
    )

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.suptitle(
        f"Measurement-noise verification (clean={args.sigma_clean}, noisy={args.sigma_noisy})",
        fontsize=11,
    )

    t_clean = np.arange(clean["traj_true"].shape[0])
    t_noisy = np.arange(noisy["traj_true"].shape[0])

    axes[0].plot(t_clean, clean["traj_true"][:, 0], label=f"clean sigma={args.sigma_clean}")
    axes[0].plot(t_noisy, noisy["traj_true"][:, 0], label=f"noisy sigma={args.sigma_noisy}")
    axes[0].plot(t_clean, clean["refs"][:, 0], "k--", linewidth=1, label="reference")
    axes[0].set_ylabel("cart position x")
    axes[0].set_title(
        f"Position tracking (clean={args.sigma_clean}, noisy={args.sigma_noisy})"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    theta_clean = np.arctan2(clean["traj_true"][:, 1], clean["traj_true"][:, 2])
    theta_noisy = np.arctan2(noisy["traj_true"][:, 1], noisy["traj_true"][:, 2])
    axes[1].plot(t_clean, theta_clean, label="clean theta")
    axes[1].plot(t_noisy, theta_noisy, label="noisy theta")
    axes[1].axhline(np.pi, color="k", linestyle="--", linewidth=1, label="upright pi")
    axes[1].set_ylabel("theta [rad]")
    axes[1].set_title(f"Angle response (noisy sigma={args.sigma_noisy})")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    cos_clean = clean["traj_true"][:, 2]
    cos_noisy = noisy["traj_true"][:, 2]
    axes[2].plot(t_clean, cos_clean, label="clean cos(theta)")
    axes[2].plot(t_noisy, cos_noisy, label="noisy cos(theta)")
    axes[2].axhline(1.0, color="k", linestyle="--", linewidth=1, label="upright cos(theta)=1")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("cos(theta)")
    axes[2].set_title(f"Cos(theta) response (noisy sigma={args.sigma_noisy})")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    def annotate_collapse(run_data, label, color):
        step = run_data["collapse_step"]
        status = run_data["collapse_status"]
        if step is None:
            return
        for ax in axes:
            ax.axvline(step, color=color, linestyle=":", linewidth=1.5, alpha=0.8)
        axes[0].text(
            step,
            axes[0].get_ylim()[1],
            f"{label} COLLAPSED @ {step} ({status})",
            color=color,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor=color),
        )

    annotate_collapse(clean, "clean", "tab:blue")
    annotate_collapse(noisy, "noisy", "tab:orange")

    clean_status_summary = (
        f"OK ({len(clean['solver_statuses'])} steps)"
        if clean["collapse_step"] is None
        else f"COLLAPSED @ {clean['collapse_step']} [{clean['collapse_status']}]"
    )
    noisy_status_summary = (
        f"OK ({len(noisy['solver_statuses'])} steps)"
        if noisy["collapse_step"] is None
        else f"COLLAPSED @ {noisy['collapse_step']} [{noisy['collapse_status']}]"
    )
    axes[2].text(
        0.01,
        0.02,
        f"solver clean: {clean_status_summary}\nsolver noisy: {noisy_status_summary}",
        transform=axes[2].transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    fig.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        REPO_ROOT,
        "results",
        f"verify_measurement_noise_clean{args.sigma_clean}_noisy{args.sigma_noisy}_{timestamp}.png",
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    diag_path = os.path.join(
        REPO_ROOT,
        "results",
        f"diag_k{args.diag_k40}_vs_k{args.diag_k50}_seed{args.seed}_sigma{args.sigma_noisy}_{timestamp}.txt",
    )
    write_k_comparison_diagnostic(
        diag_path,
        run_diag_40,
        run_diag_50,
        k40_label=f"K={args.diag_k40}",
        k50_label=f"K={args.diag_k50}",
    )


if __name__ == "__main__":
    main()
    # Example:
    # python experiments/verify_measurement_noise.py --sigma-noisy 0.02 --k 40 --t-sim 300
    # Fast repeated run (reuse offline cache):
    # python experiments/verify_measurement_noise.py --t-sim 120 --k 40 --sigma-noisy 0.02
