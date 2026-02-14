import argparse
import importlib
import os
import pickle
import sys

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import registry


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from custom.offline_hankel_collection import (  # noqa: E402
    collect_closedloop_controller_data,
    create_trajectory_dataset,
)
from custom.setup_deepc import selector_cb, setup_DeePC  # noqa: E402
from experiments.online_buffer import OnlineTrajectoryBuffer  # noqa: E402
from select_deepc.data_selectors import LkSelector  # noqa: E402
from select_deepc.deepc_controller import SelectDeePC  # noqa: E402
from utils.debug_visualizer import OnlineBufferDebugVisualizer  # noqa: E402


def check_runtime_dependencies():
    try:
        importlib.import_module("mujoco")
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: `mujoco`. Install with: pip install \"gymnasium[mujoco]\""
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


def featurize_obs(obs_raw: np.ndarray) -> np.ndarray:
    return np.array(
        [obs_raw[0], np.sin(obs_raw[1]), np.cos(obs_raw[1]), obs_raw[2], obs_raw[3]],
        dtype=float,
    )


def main():
    parser = argparse.ArgumentParser(description="Simple test for OnlineTrajectoryBuffer.")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--L", type=int, default=32, help="segment_len and Hankel depth")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--sigma-y", type=float, default=0.0)
    parser.add_argument("--debug_online_buffer", action="store_true")
    parser.add_argument("--debug_pause", type=float, default=0.02)
    parser.add_argument("--debug_gamma", type=float, default=None)
    parser.add_argument("--debug_plot_every", type=int, default=20)
    parser.add_argument(
        "--offline-cache",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "offline_dataset_cache.pkl"),
    )
    parser.add_argument("--rebuild-offline", action="store_true")
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
    controller = SelectDeePC(
        controller_args,
        LkSelector(
            1,
            controller_args.deepc_dims,
            custom_callback=selector_cb,
            forgetting_factor=0.8,
        ),
        num_hankel_cols=args.k,
        n_iter=1,
        debug=False,
    )

    p = int(controller_args.deepc_dims.p)
    m = int(controller_args.deepc_dims.m)
    t_ini = int(controller_args.deepc_dims.T_past)
    L = int(controller_args.deepc_dims.T_past + controller_args.deepc_dims.T_fut)

    buf = OnlineTrajectoryBuffer(
        m=m,
        p=p,
        max_steps=args.max_steps,
        segment_len=args.L,
        stride=args.stride,
    )
    vis = None
    if args.debug_online_buffer:
        vis = OnlineBufferDebugVisualizer(
            max_steps=args.max_steps,
            pause_s=args.debug_pause,
            save_dir=os.path.join(REPO_ROOT, "results"),
            gamma=args.debug_gamma,
        )

    env = gym.make("InvertedPendulum-v4-swingup")
    obs_raw, _ = env.reset(seed=args.seed)
    rng_noise = np.random.default_rng(args.seed + 1000)

    first_can_extract_step = None
    for step in range(args.steps):
        y_true = featurize_obs(np.asarray(obs_raw, dtype=float))
        y_meas = y_true + rng_noise.normal(0.0, args.sigma_y, size=y_true.shape)
        ref = np.array([0.5 if step > 100 else 0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
        action = controller.compute_action(y_meas, ref)
        buf.append(np.asarray(action).reshape(-1), y_meas, t_t=float(step))

        if first_can_extract_step is None and buf.can_extract():
            first_can_extract_step = step

        selected_idcs = None
        try:
            selected_idcs = controller.get_last_selected_idcs()
        except Exception:
            selected_idcs = None

        snap = buf.debug_snapshot(
            step_idx=step,
            u_t=np.asarray(action).reshape(-1),
            y_t=y_meas,
            T_ini=t_ini,
            L=L,
            selected_indices=selected_idcs,
        )

        solver_status = "unknown"
        try:
            solver_status = str(controller._deepc._problem.status)
        except Exception:
            pass

        if args.debug_online_buffer and step % 20 == 0:
            print(f"Step {step}:")
            print(f"  buffer_size = {snap['buffer_size']}")
            print(f"  num_segments = {snap['num_segments']}")
            print(f"  sigma_min = {snap['sigma_min']}")
            print(f"  sigma_max = {snap['sigma_max']}")
            print(f"  condition_number = {snap['condition_number']}")
            print(
                f"  selected_K = {0 if snap['selected_indices'] is None else snap['selected_indices'].size}"
            )
            print(f"  solver_status = {solver_status}")

        if args.debug_online_buffer and (
            (not np.isfinite(snap["sigma_min"]))
            or (not np.isfinite(snap["sigma_max"]))
            or (not np.isfinite(snap["condition_number"]))
        ):
            print(
                f"\033[91mWARNING step {step}: NaN/Inf in sigma/condition metrics.\033[0m"
            )

        if (
            args.debug_online_buffer
            and vis is not None
            and (step % max(1, args.debug_plot_every) == 0)
        ):
            vis.update(
                snap,
                selected_k=0 if snap["selected_indices"] is None else snap["selected_indices"].size,
            )

        obs_raw, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs_raw, _ = env.reset(seed=args.seed + step + 1)

    env.close()

    print(f"buffer size: {buf.size()}")
    print(f"can_extract: {buf.can_extract()}")
    print(f"first can_extract step: {first_can_extract_step}")

    U_segs, Y_segs = buf.extract_segments(mode="rolling")
    print(f"U_segs shape: {U_segs.shape}")
    print(f"Y_segs shape: {Y_segs.shape}")
    print(f"num segments: {U_segs.shape[0]}")

    H_u, H_y = buf.build_hankel(args.L)
    print(f"H_u shape: {H_u.shape}")
    print(f"H_y shape: {H_y.shape}")

    try:
        u_ini, y_ini = buf.get_Uini_Yini(T_ini=t_ini)
        print(f"u_ini shape: {u_ini.shape}")
        print(f"y_ini shape: {y_ini.shape}")
    except ValueError as e:
        print(f"get_Uini_Yini failed: {e}")


if __name__ == "__main__":
    main()
