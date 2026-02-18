import argparse
import csv
import os
import pickle
import sys
from typing import Tuple

import gymnasium as gym
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


def load_or_build_offline_dataset(cache_path: str, rebuild: bool):
    if (not rebuild) and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    _, reset_time, u_off, y_off = collect_closedloop_controller_data(animation=False)
    trajectory_data = create_trajectory_dataset(reset_time, u_off, y_off, "swingup_data")
    with open(cache_path, "wb") as f:
        pickle.dump(trajectory_data, f)
    return trajectory_data


def build_controller(controller_args, k: int, debug: bool, eps_sigma: float):
    selector = LkSelector(
        1,
        controller_args.deepc_dims,
        custom_callback=selector_cb,
        forgetting_factor=1,
    )
    controller = SelectDeePC(
        controller_args,
        selector_callback=selector,
        num_hankel_cols=int(k),
        n_iter=1,
        debug=debug,
    )
    controller._eps_sigma = float(eps_sigma)
    return controller


def reset_env(env, seed: int):
    reset_out = env.reset(seed=seed)
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def step_env(env, action) -> Tuple[np.ndarray, bool, bool]:
    step_out = env.step(action)
    if len(step_out) == 5:
        obs_raw, _, terminated, truncated, _ = step_out
        return obs_raw, bool(terminated), bool(truncated)
    obs_raw, _, done, _ = step_out
    done = bool(done)
    return obs_raw, done, False


def safe_stat(values, fn, default=np.nan):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float(default)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(fn(finite))


def run_episode(controller, max_steps: int, seed: int):
    env = gym.make("InvertedPendulum-v4-swingup")
    obs_raw = reset_env(env, seed=seed)

    steps = 0
    for _ in range(max_steps):
        obs_true = featurize_obs(np.asarray(obs_raw, dtype=float))
        reference = [0,0,1,0,0]
        action = controller.compute_action(obs_true, reference)
        obs_raw, terminated, truncated = step_env(env, action)
        steps += 1
        if terminated or truncated:
            break
    env.close()

    slack_fail_rate = safe_stat(controller._slack_violation_history, np.mean, default=0.0)
    min_sigma_min_Ag = safe_stat(controller._sigma_min_A_history, np.min)
    min_sigma_min_Hu = safe_stat(controller._min_selected_sigma_history, np.min)
    total_cost = safe_stat(controller._stage_cost_history, np.sum, default=0.0)
    mean_solve_time_ms = safe_stat(controller._solve_time_ms_history, np.mean)

    return {
        "steps": int(steps),
        "slack_fail_rate": slack_fail_rate,
        "min_sigma_min_Ag": min_sigma_min_Ag,
        "min_sigma_min_Hu": min_sigma_min_Hu,
        "total_cost": total_cost,
        "mean_solve_time_ms": mean_solve_time_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="K-sweep runner for SelectDeePC.")
    parser.add_argument("--outdir", type=str, default=os.path.join(REPO_ROOT, "logs", "k_sweep"))
    parser.add_argument("--Kmin", type=int, default=10)
    parser.add_argument("--Kmax", type=int, default=200)
    parser.add_argument("--Kstep", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--eps_sigma", type=float, default=1e-3)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--offline-cache",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "offline_dataset_cache.pkl"),
    )
    parser.add_argument("--rebuild-offline", action="store_true")
    args = parser.parse_args()

    ensure_env_registered()
    os.makedirs(args.outdir, exist_ok=True)

    trajectory_data = load_or_build_offline_dataset(args.offline_cache, args.rebuild_offline)
    controller_args = setup_DeePC(trajectory_data)

    k_list = list(range(int(args.Kmin), int(args.Kmax) + 1, int(args.Kstep)))
    rows = []

    for k in k_list:
        for seed in range(int(args.seeds)):
            controller = build_controller(
                controller_args=controller_args,
                k=int(k),
                debug=bool(args.debug),
                eps_sigma=float(args.eps_sigma),
            )
            metrics = run_episode(
                controller=controller,
                max_steps=int(args.max_steps),
                seed=int(seed),
            )

            npz_name = f"run_K{k:03d}_seed{seed:03d}.npz"
            npz_path = os.path.join(args.outdir, npz_name)
            controller.save_history_npz(
                npz_path,
                extra={"K": int(k), "seed": int(seed), "steps": int(metrics["steps"])},
            )

            rows.append(
                {
                    "K": int(k),
                    "seed": int(seed),
                    "steps": int(metrics["steps"]),
                    "slack_fail_rate": float(metrics["slack_fail_rate"]),
                    "min_sigma_min_Ag": float(metrics["min_sigma_min_Ag"]),
                    "min_sigma_min_Hu": float(metrics["min_sigma_min_Hu"]),
                    "total_cost": float(metrics["total_cost"]),
                    "mean_solve_time_ms": float(metrics["mean_solve_time_ms"]),
                    "npz_path": npz_path,
                }
            )
            print(
                f"[run] K={k:3d} seed={seed:3d} steps={metrics['steps']:4d} "
                f"slack_fail={metrics['slack_fail_rate']:.3f} "
                f"min_sigma_Ag={metrics['min_sigma_min_Ag']:.3e}"
            )

    summary_path = os.path.join(args.outdir, "summary.csv")
    fieldnames = [
        "K",
        "seed",
        "steps",
        "slack_fail_rate",
        "min_sigma_min_Ag",
        "min_sigma_min_Hu",
        "total_cost",
        "mean_solve_time_ms",
        "npz_path",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved run npz files under: {args.outdir}")


if __name__ == "__main__":
    main()
    # Example:
    # python cdc2026/experiments/run_k_sweep_selectdeepc.py --outdir logs/k_sweep --Kmin 10 --Kmax 200 --Kstep 10 --seeds 10 --max_steps 600 --eps_sigma 1e-3
