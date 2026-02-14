import argparse
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

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
from custom.setup_deepc import setup_DeePC, selector_cb  # noqa: E402
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


def featurize_obs(obs_raw: np.ndarray) -> np.ndarray:
    return np.array(
        [obs_raw[0], np.sin(obs_raw[1]), np.cos(obs_raw[1]), obs_raw[2], obs_raw[3]],
        dtype=float,
    )


def wrapped_angle_error_to_pi(theta: float) -> float:
    diff = (theta - np.pi + np.pi) % (2 * np.pi) - np.pi
    return abs(diff)


class TrackingSelector:
    def __init__(self, base_selector, k_fixed: int):
        self.base_selector = base_selector
        self.k_fixed = k_fixed
        self.sigma_min_history = []

    def __call__(self, input_traj, state_traj, H_u, H_y, reference):
        idcs, norms = self.base_selector(input_traj, state_traj, H_u, H_y, reference)
        used_idcs = idcs[: self.k_fixed]

        sigma_min = np.nan
        if used_idcs.size > 0:
            H_u_sel = H_u[:, used_idcs]
            gram = H_u_sel @ H_u_sel.T
            min_eig = float(np.min(np.linalg.eigvalsh(gram)))
            sigma_min = float(np.sqrt(max(min_eig, 0.0)))
        self.sigma_min_history.append(sigma_min)
        return idcs, norms


@dataclass
class EpisodeResult:
    success: bool
    time_to_upright: float
    mean_solve_time: float
    min_sigma_min_u_selected: float
    status_counts: dict


def get_dt(env) -> float:
    if hasattr(env.unwrapped, "dt"):
        return float(env.unwrapped.dt)
    try:
        return float(env.unwrapped.model.opt.timestep * env.unwrapped.frame_skip)
    except Exception:
        return 0.02


def run_single_episode(
    controller,
    sigma_y: float,
    noise_seed: int,
    reset_seed: int,
    tmax: float,
    ref_switch_step: int,
    ref_before: float,
    ref_after: float,
) -> EpisodeResult:
    env = gym.make("InvertedPendulum-v4-swingup")
    obs_raw, _ = env.reset(seed=reset_seed)
    dt = get_dt(env)
    max_steps = int(np.ceil(tmax / dt))
    settle_steps = int(np.ceil(0.5 / dt))

    rng_noise = np.random.default_rng(noise_seed)
    consecutive_ok = 0
    success = False
    time_to_upright = tmax

    solve_times = []
    status_counts = defaultdict(int)

    for step in range(max_steps):
        obs_true_feat = featurize_obs(np.asarray(obs_raw, dtype=float))
        obs_meas = obs_true_feat + rng_noise.normal(0.0, sigma_y, size=obs_true_feat.shape)

        x_ref = ref_before if step <= ref_switch_step else ref_after
        ref = np.array([x_ref, 0.0, 1.0, 0.0, 0.0], dtype=float)

        t0 = time.perf_counter()
        action = controller.compute_action(obs_meas, ref)
        solve_times.append(time.perf_counter() - t0)

        status = "unknown"
        try:
            status = str(controller._deepc._problem.status)
        except Exception:
            pass
        status_counts[status] += 1

        obs_raw, _, terminated, truncated, _ = env.step(action)

        theta = float(obs_raw[1])
        theta_dot = float(obs_raw[3])
        is_ok = (
            wrapped_angle_error_to_pi(theta) <= np.deg2rad(15.0)
            and abs(theta_dot) <= 1.0
        )
        if is_ok:
            consecutive_ok += 1
            if (not success) and consecutive_ok >= settle_steps:
                success = True
                time_to_upright = min((step + 1) * dt, tmax)
        else:
            consecutive_ok = 0

        if terminated or truncated:
            break

    env.close()

    min_sigma = np.nan
    try:
        sigma_hist = np.asarray(controller._selector_callback.sigma_min_history, dtype=float)
        if sigma_hist.size > 0 and np.any(np.isfinite(sigma_hist)):
            min_sigma = float(np.nanmin(sigma_hist))
    except Exception:
        pass

    return EpisodeResult(
        success=success,
        time_to_upright=float(time_to_upright),
        mean_solve_time=float(np.mean(solve_times)) if len(solve_times) > 0 else np.nan,
        min_sigma_min_u_selected=min_sigma,
        status_counts=dict(status_counts),
    )


def aggregate_results(episode_results):
    success_rate = float(np.mean([1.0 if r.success else 0.0 for r in episode_results]))
    mean_ttu = float(np.mean([r.time_to_upright for r in episode_results]))
    mean_solve = float(np.mean([r.mean_solve_time for r in episode_results]))
    min_sigma = float(np.nanmin([r.min_sigma_min_u_selected for r in episode_results]))

    status_total = defaultdict(int)
    for r in episode_results:
        for key, val in r.status_counts.items():
            status_total[key] += int(val)

    return {
        "success_rate": success_rate,
        "mean_time_to_upright": mean_ttu,
        "mean_solve_time_per_step": mean_solve,
        "min_k_sigma_min_u_selected": min_sigma,
        "status_counts": dict(status_total),
    }


def main():
    parser = argparse.ArgumentParser(description="Noise sweep with fixed-K Select-DeePC.")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--tmax", type=float, default=6.0, help="Episode horizon in seconds.")
    parser.add_argument("--seed", type=int, default=7, help="Master seed.")
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.0, 0.005, 0.01, 0.02],
        help="Measurement noise std sweep.",
    )
    parser.add_argument(
        "--k-list",
        type=int,
        nargs="+",
        default=[40, 80, 120, 160],
        help="Fixed K values for column selection.",
    )
    parser.add_argument("--ref-switch-step", type=int, default=100)
    parser.add_argument("--ref-before", type=float, default=0.0)
    parser.add_argument("--ref-after", type=float, default=0.5)
    args = parser.parse_args()

    ensure_env_registered()
    os.makedirs(os.path.join(REPO_ROOT, "results"), exist_ok=True)

    print("Collecting offline dataset once...")
    _, reset_time, u_off, y_off = collect_closedloop_controller_data(animation=False)
    trajectory_data = create_trajectory_dataset(reset_time, u_off, y_off, "swingup_data")
    controller_args = setup_DeePC(trajectory_data)

    rng_master = np.random.default_rng(args.seed)
    reset_seeds = rng_master.integers(0, 10_000_000, size=args.trials, endpoint=False)
    noise_seeds = rng_master.integers(0, 10_000_000, size=args.trials, endpoint=False)

    results = {}
    for k_val in args.k_list:
        for sigma_y in args.sigmas:
            combo_key = (float(sigma_y), int(k_val))
            ep_results = []

            for tr in range(args.trials):
                base_selector = LkSelector(
                    1,
                    controller_args.deepc_dims,
                    custom_callback=selector_cb,
                    forgetting_factor=0.8,
                )
                tracking_selector = TrackingSelector(base_selector, k_fixed=int(k_val))
                controller = SelectDeePC(
                    controller_args,
                    tracking_selector,
                    num_hankel_cols=int(k_val),
                    n_iter=1,
                    debug=False,
                )

                _ = DeePCCostAccumulator(controller_args.controller_costs)
                ep = run_single_episode(
                    controller=controller,
                    sigma_y=float(sigma_y),
                    noise_seed=int(noise_seeds[tr]),
                    reset_seed=int(reset_seeds[tr]),
                    tmax=float(args.tmax),
                    ref_switch_step=int(args.ref_switch_step),
                    ref_before=float(args.ref_before),
                    ref_after=float(args.ref_after),
                )
                ep_results.append(ep)

            results[combo_key] = aggregate_results(ep_results)

    print("\nResults table")
    print(
        "sigma_y | K | success_rate | mean_ttu[s] | mean_solve_time[s/step] | min_k_sigma_min(U_sel)"
    )
    print("-" * 98)
    for sigma_y in args.sigmas:
        for k_val in args.k_list:
            row = results[(float(sigma_y), int(k_val))]
            print(
                f"{sigma_y:7.4f} | {k_val:3d} | "
                f"{row['success_rate']:11.3f} | "
                f"{row['mean_time_to_upright']:11.3f} | "
                f"{row['mean_solve_time_per_step']:22.6f} | "
                f"{row['min_k_sigma_min_u_selected']:20.6e}"
            )
            print(f"  status_counts={row['status_counts']}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for k_val in args.k_list:
        success_vals = [results[(float(s), int(k_val))]["success_rate"] for s in args.sigmas]
        solve_vals = [
            results[(float(s), int(k_val))]["mean_solve_time_per_step"] for s in args.sigmas
        ]
        axes[0].plot(args.sigmas, success_vals, marker="o", label=f"K={k_val}")
        axes[1].plot(args.sigmas, solve_vals, marker="o", label=f"K={k_val}")

    axes[0].set_xlabel("sigma_y")
    axes[0].set_ylabel("success rate")
    axes[0].set_title("Success rate vs noise")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("sigma_y")
    axes[1].set_ylabel("mean solve time [s/step]")
    axes[1].set_title("Solve time vs noise")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_path = os.path.join(REPO_ROOT, "results", "noise_sweep_K.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
    # Example:
    # python experiments/noise_sweep_k.py --trials 20 --tmax 6.0 --seed 7
    # Optional:
    # python experiments/noise_sweep_k.py --trials 20 --sigmas 0 0.005 0.01 0.02 0.05 --k-list 40 80 120 160
