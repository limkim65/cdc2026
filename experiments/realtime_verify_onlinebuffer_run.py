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
from experiments.online_buffer import OnlineTrajectoryBuffer  # noqa: E402
from select_deepc.data_selectors import LkSelector  # noqa: E402
from select_deepc.deepc_controller import SelectDeePC  # noqa: E402
from select_deepc.deepc_utils import DeePCCostAccumulator  # noqa: E402
from utils.debug_visualizer import OnlineBufferDebugVisualizer  # noqa: E402


def check_runtime_dependencies():
    try:
        importlib.import_module("mujoco")
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: `mujoco`.\nInstall with: pip install \"gymnasium[mujoco]\""
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


def build_controller(controller_args, k: int):
    return SelectDeePC(
        controller_args,
        LkSelector(
            1,
            controller_args.deepc_dims,
            custom_callback=selector_cb,
            forgetting_factor=0.8,
        ),
        num_hankel_cols=int(k),
        n_iter=1,
        debug=False,
    )


def draw_reference_marker(env, x_ref: float):
    try:
        viewer = env.unwrapped.mujoco_renderer.viewer
    except Exception:
        return
    if viewer is None:
        return
    try:
        for z in (0.0, 0.4, 0.8):
            viewer.add_marker(
                pos=np.array([x_ref, 0.0, z], dtype=float),
                size=np.array([0.005, 0.005, 0.02], dtype=float),
                rgba=np.array([1.0, 0.1, 0.1, 0.9], dtype=float),
                type=2,
                label="" if z > 0 else f"x_ref={x_ref:.2f}",
            )
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Realtime Select-DeePC with online-buffer Hankel updates."
    )
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--sigma-y", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--t-sim", type=int, default=300)
    parser.add_argument("--ref-switch-step", type=int, default=100)
    parser.add_argument("--ref-before", type=float, default=0.0)
    parser.add_argument("--ref-after", type=float, default=0.5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--show-ref-marker", action="store_true")
    parser.add_argument("--online-max-steps", type=int, default=4000)
    parser.add_argument("--online-stride", type=int, default=1)
    parser.add_argument("--debug_online_buffer", action="store_true")
    parser.add_argument("--debug_plot_every", type=int, default=20)
    parser.add_argument("--debug_pause", type=float, default=0.02)
    parser.add_argument(
        "--offline-cache",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "offline_dataset_cache.pkl"),
    )
    parser.add_argument("--rebuild-offline", action="store_true")
    parser.add_argument("--exp-tag", type=str, default="")
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
    controller = build_controller(controller_args, args.k)
    acc = DeePCCostAccumulator(controller_args.controller_costs)
    H_u_offline_base = controller._H_u.copy()
    H_y_offline_base = controller._H_y.copy()
    offline_base_cols = int(H_u_offline_base.shape[1])

    L = int(controller_args.deepc_dims.T_past + controller_args.deepc_dims.T_fut)
    online_buffer = OnlineTrajectoryBuffer(
        m=int(controller_args.deepc_dims.m),
        p=int(controller_args.deepc_dims.p),
        max_steps=int(args.online_max_steps),
        segment_len=L,
        stride=int(args.online_stride),
    )
    debug_vis = None
    if args.debug_online_buffer:
        debug_vis = OnlineBufferDebugVisualizer(
            max_steps=int(args.online_max_steps),
            pause_s=float(args.debug_pause),
            save_dir=os.path.join(REPO_ROOT, "results"),
            gamma=None,
        )

    env = gym.make(
        "InvertedPendulum-v4-swingup",
        render_mode="human" if args.render else None,
    )
    obs_raw, _ = env.reset(seed=args.seed)
    rng_noise = np.random.default_rng(args.seed + 1000)

    traj_true = []
    traj_meas = []
    theta_raw = []
    cos_theta = []
    x_ref_hist = []
    actions = []
    solve_times = []
    statuses = []
    sigma_min_hist = []
    online_hankel_cols_pre = []
    online_hankel_cols_post = []
    online_buffer_size_hist = []
    online_hankel_used = []
    combined_hankel_cols = []

    for step in range(args.t_sim):
        obs_true = featurize_obs(np.asarray(obs_raw, dtype=float))
        obs_meas = obs_true + rng_noise.normal(0.0, args.sigma_y, size=obs_true.shape)

        x_ref = args.ref_before if step <= args.ref_switch_step else args.ref_after
        ref = np.array([x_ref, 0.0, 1.0, 0.0, 0.0], dtype=float)

        H_u_online, H_y_online = online_buffer.build_hankel(L)
        online_hankel_cols_pre.append(int(H_u_online.shape[1]))
        if H_u_online.shape[1] > 0:
            controller._H_u = np.hstack((H_u_offline_base, H_u_online))
            controller._H_y = np.hstack((H_y_offline_base, H_y_online))
            online_hankel_used.append(1)
        else:
            controller._H_u = H_u_offline_base
            controller._H_y = H_y_offline_base
            online_hankel_used.append(0)
        combined_hankel_cols.append(int(controller._H_u.shape[1]))

        if args.render and args.show_ref_marker:
            draw_reference_marker(env, x_ref)

        t0 = time.perf_counter()
        action = controller.compute_action(obs_meas, ref)
        solve_times.append(time.perf_counter() - t0)

        st = "unknown"
        try:
            st = str(controller._deepc._problem.status)
        except Exception:
            pass
        statuses.append(st)

        try:
            sigma_min_hist.append(float(controller.get_min_selected_singular_values()[-1]))
        except Exception:
            sigma_min_hist.append(np.nan)

        acc.update_cost(obs_true, ref, action)
        obs_raw, _, terminated, truncated, _ = env.step(action)

        traj_true.append(obs_true)
        traj_meas.append(obs_meas)
        theta_raw.append(float(obs_raw[1]))
        cos_theta.append(float(np.cos(obs_raw[1])))
        x_ref_hist.append(float(x_ref))
        actions.append(np.array(action).reshape(-1))

        # Online data collection: push measurement that controller sees.
        online_buffer.append(np.asarray(action).reshape(-1), obs_meas, t_t=float(step))
        H_u_online_post, _ = online_buffer.build_hankel(L)
        online_hankel_cols_post.append(int(H_u_online_post.shape[1]))
        online_buffer_size_hist.append(int(online_buffer.size()))

        if (
            args.debug_online_buffer
            and debug_vis is not None
            and (step % max(1, int(args.debug_plot_every)) == 0)
        ):
            try:
                sel = controller.get_last_selected_idcs()
            except Exception:
                sel = None
            snap = online_buffer.debug_snapshot(
                step_idx=step,
                u_t=np.asarray(action).reshape(-1),
                y_t=obs_meas,
                T_ini=int(controller_args.deepc_dims.T_past),
                L=L,
                selected_indices=sel,
            )
            debug_vis.update(
                snap,
                selected_k=0 if snap["selected_indices"] is None else snap["selected_indices"].size,
            )

        if args.render and args.sleep > 0:
            time.sleep(args.sleep)
        if terminated or truncated:
            break

    env.close()

    traj_true = np.asarray(traj_true, dtype=float)
    traj_meas = np.asarray(traj_meas, dtype=float)
    theta_raw = np.asarray(theta_raw, dtype=float)
    cos_theta = np.asarray(cos_theta, dtype=float)
    x_ref_hist = np.asarray(x_ref_hist, dtype=float)
    actions = np.asarray(actions, dtype=float)
    solve_times = np.asarray(solve_times, dtype=float)
    status_arr = np.asarray(statuses, dtype=object)
    sigma_min_hist = np.asarray(sigma_min_hist, dtype=float)
    online_hankel_cols_pre = np.asarray(online_hankel_cols_pre, dtype=int)
    online_hankel_cols_post = np.asarray(online_hankel_cols_post, dtype=int)
    online_buffer_size_hist = np.asarray(online_buffer_size_hist, dtype=int)
    online_hankel_used = np.asarray(online_hankel_used, dtype=int)
    combined_hankel_cols = np.asarray(combined_hankel_cols, dtype=int)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_tag = f"onlinebuf_k{args.k}_sigma{args.sigma_y}_seed{args.seed}_{timestamp}"
    tag = args.exp_tag if args.exp_tag else auto_tag

    npz_path = os.path.join(REPO_ROOT, "results", f"realtime_verify_onlinebuffer_{tag}.npz")
    np.savez(
        npz_path,
        traj_true=traj_true,
        traj_meas=traj_meas,
        theta=theta_raw,
        cos_theta=cos_theta,
        x=traj_true[:, 0] if traj_true.size > 0 else np.array([]),
        x_ref=x_ref_hist,
        action=actions,
        sigma_min=sigma_min_hist,
        solver_status=status_arr,
        solve_time=solve_times,
        K=np.array(args.k, dtype=int),
        sigma_y=np.array(args.sigma_y, dtype=float),
        seed=np.array(args.seed, dtype=int),
        offline_base_cols=np.array(offline_base_cols, dtype=int),
        online_hankel_cols_pre=online_hankel_cols_pre,
        online_hankel_cols_post=online_hankel_cols_post,
        online_buffer_size=online_buffer_size_hist,
        online_hankel_used=online_hankel_used,
        combined_hankel_cols=combined_hankel_cols,
        total_cost=np.array(acc.cost["cost"], dtype=float),
    )

    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True)
    fig.suptitle(
        f"Realtime verify (online Hankel) K={args.k}, sigma_y={args.sigma_y}, seed={args.seed}",
        fontsize=11,
    )
    t = np.arange(theta_raw.shape[0])

    if traj_true.shape[0] > 0:
        axes[0].plot(t, traj_true[:, 0], label="x")
    axes[0].plot(t, x_ref_hist, "k--", linewidth=1, label="x_ref")
    axes[0].set_ylabel("x")
    axes[0].set_title("Position")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, theta_raw, label="theta")
    axes[1].axhline(np.pi, color="k", linestyle="--", linewidth=1, label="upright pi")
    axes[1].set_ylabel("theta [rad]")
    axes[1].set_title("Theta")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t, cos_theta, label="cos(theta)")
    axes[2].axhline(1.0, color="k", linestyle="--", linewidth=1, label="upright=1")
    axes[2].set_ylabel("cos(theta)")
    axes[2].set_title("Cos(theta)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(np.arange(sigma_min_hist.shape[0]), sigma_min_hist, label="sigma_min(U_selected)")
    axes[3].set_ylabel("sigma_min")
    axes[3].set_title("Selection conditioning")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    axes[4].plot(
        np.arange(online_hankel_cols_pre.shape[0]),
        online_hankel_cols_pre,
        label="online Hankel cols (pre-append)",
        alpha=0.6,
    )
    axes[4].plot(
        np.arange(online_hankel_cols_post.shape[0]),
        online_hankel_cols_post,
        label="online Hankel cols (post-append)",
        linewidth=2.0,
    )
    axes[4].plot(
        np.arange(online_buffer_size_hist.shape[0]),
        online_buffer_size_hist,
        label="online buffer size",
        linestyle=":",
    )
    combined_delta_cols = combined_hankel_cols - offline_base_cols
    axes[4].plot(
        np.arange(combined_delta_cols.shape[0]),
        combined_delta_cols,
        label="combined - offline_base",
        linewidth=2.0,
    )
    axes[4].set_xlabel("step")
    axes[4].set_ylabel("delta cols")
    axes[4].set_title("Online Hankel growth (delta view)")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    fig.tight_layout()
    plot_path = os.path.join(REPO_ROOT, "results", f"realtime_verify_onlinebuffer_{tag}.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")

    print(f"Saved data: {npz_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
    # Example:
    # python experiments/realtime_verify_onlinebuffer_run.py --render --k 40 --sigma-y 0.02 --t-sim 300
