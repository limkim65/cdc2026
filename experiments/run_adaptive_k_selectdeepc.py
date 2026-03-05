import argparse
import csv
import os
import pickle
import sys
from typing import Tuple

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
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


def reset_env(env, seed: int):
    reset_out = env.reset(seed=seed)
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def step_env(env, action) -> Tuple[np.ndarray, bool, bool, dict]:
    step_out = env.step(action)
    if len(step_out) == 5:
        obs_raw, _, terminated, truncated, info = step_out
        return obs_raw, bool(terminated), bool(truncated), dict(info)
    obs_raw, _, done, info = step_out
    done = bool(done)
    return obs_raw, done, False, dict(info) if info is not None else {}


def safe_stat(values, fn, default=np.nan):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float(default)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(fn(finite))


def angle_error_to_upright(theta: float) -> float:
    # Upright reference is pi for this swing-up setup.
    return float(np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi)))


def save_trajectory_plot(
    out_path: str,
    state_traj: np.ndarray,
    action_traj: np.ndarray,
    show_plot: bool = True,
):
    state_arr = np.asarray(state_traj, dtype=float)
    action_arr = np.asarray(action_traj, dtype=float)
    if state_arr.size == 0:
        return

    t = np.arange(state_arr.shape[0])
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), dpi=120, sharex=True)
    axs[0].plot(t, state_arr[:, 0], label="cart position x")
    axs[0].plot(t, np.cos(state_arr[:, 1]), label="cos(theta)")
    axs[0].set_ylabel("State")
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(loc="upper right", fontsize=8, frameon=False)

    if action_arr.ndim == 1:
        action_arr = action_arr.reshape(-1, 1)
    for i in range(action_arr.shape[1]):
        axs[1].plot(np.arange(action_arr.shape[0]), action_arr[:, i], label=f"u[{i}]")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Action")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


class LiveKViz:
    def __init__(
        self,
        max_steps: int,
        max_cols: int,
        k_max_scale: int,
        pause_s: float = 0.001,
    ):
        self._pause_s = float(max(0.0, pause_s))
        self._steps = []
        self._k_vals = []
        self._cost_vals = []
        self._inst_cost_vals = []
        self._sel_step = []
        self._sel_idx = []

        plt.ion()
        self._fig, (self._ax_k, self._ax_cost, self._ax_idx) = plt.subplots(
            3, 1, figsize=(8, 8), dpi=110
        )
        self._line_k, = self._ax_k.plot([], [], color="tab:blue", lw=1.5)
        self._line_cost, = self._ax_cost.plot([], [], color="tab:green", lw=1.5, label="cumulative")
        self._line_inst_cost, = self._ax_cost.plot(
            [], [], color="tab:red", lw=1.2, alpha=0.9, label="instantaneous"
        )
        self._sc = self._ax_idx.scatter([], [], s=6, alpha=0.4, color="tab:orange")

        self._ax_k.set_xlim(0, max_steps + 1)
        self._ax_k.set_ylim(0, max(1, int(k_max_scale)) + 5)
        self._ax_k.set_ylabel("K selected")
        self._ax_k.grid(True, alpha=0.25)

        self._ax_cost.set_xlim(0, max_steps + 1)
        self._ax_cost.set_ylabel("Cost")
        self._ax_cost.grid(True, alpha=0.25)
        self._ax_cost.legend(loc="upper left", frameon=False, fontsize=8)

        self._ax_idx.set_xlim(0, max_steps + 1)
        self._ax_idx.set_ylim(0, max(1, int(max_cols)) + 5)
        self._ax_idx.set_xlabel("Step")
        self._ax_idx.set_ylabel("Selected index")
        self._ax_idx.grid(True, alpha=0.25)

        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def update(
        self,
        step: int,
        k_selected: int,
        selected_idcs: np.ndarray,
        cumulative_cost: float,
        instantaneous_cost: float,
    ):
        self._steps.append(int(step))
        self._k_vals.append(int(k_selected))
        self._cost_vals.append(float(cumulative_cost))
        self._inst_cost_vals.append(float(instantaneous_cost))

        if selected_idcs.size > 0:
            self._sel_step.extend([int(step)] * int(selected_idcs.size))
            self._sel_idx.extend(selected_idcs.astype(int).tolist())

        self._line_k.set_data(self._steps, self._k_vals)
        self._line_cost.set_data(self._steps, self._cost_vals)
        self._line_inst_cost.set_data(self._steps, self._inst_cost_vals)
        if len(self._cost_vals) > 0 and len(self._inst_cost_vals) > 0:
            ymax = float(
                max(
                    1.0,
                    np.nanmax(self._cost_vals) * 1.05,
                    np.nanmax(self._inst_cost_vals) * 1.05,
                )
            )
            self._ax_cost.set_ylim(0.0, ymax)
        if len(self._sel_step) > 0:
            offsets = np.column_stack([np.asarray(self._sel_step), np.asarray(self._sel_idx)])
        else:
            offsets = np.empty((0, 2), dtype=float)
        self._sc.set_offsets(offsets)

        self._ax_k.set_title(f"Live Adaptive-K (step={step}, K={k_selected})")
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        if self._pause_s > 0.0:
            plt.pause(self._pause_s)

    def close(self):
        try:
            plt.ioff()
            plt.close(self._fig)
        except Exception:
            pass


def save_selection_diagnostics_plot(
    out_path: str,
    k_hist: np.ndarray,
    stage_cost_hist: np.ndarray,
    selected_hist: list[np.ndarray],
    max_cols: int,
    show_plot: bool = True,
):
    k_hist = np.asarray(k_hist, dtype=float).reshape(-1)
    stage_cost_hist = np.asarray(stage_cost_hist, dtype=float).reshape(-1)
    n = int(max(k_hist.size, stage_cost_hist.size))
    if n == 0:
        return

    steps = np.arange(1, n + 1)
    cum_cost = np.nancumsum(stage_cost_hist) if stage_cost_hist.size > 0 else np.zeros(n)

    sel_step = []
    sel_idx = []
    for i, idcs in enumerate(selected_hist):
        idcs_arr = np.asarray(idcs, dtype=int).reshape(-1)
        if idcs_arr.size == 0:
            continue
        sel_step.extend([i + 1] * int(idcs_arr.size))
        sel_idx.extend(idcs_arr.tolist())

    fig, (ax_k, ax_cost, ax_idx) = plt.subplots(3, 1, figsize=(8, 8), dpi=120)
    ax_k.plot(np.arange(1, k_hist.size + 1), k_hist, color="tab:blue", lw=1.5)
    ax_k.set_xlim(0, n + 1)
    ax_k.set_ylim(0, max(1, int(max_cols)) + 5)
    ax_k.set_ylabel("K selected")
    ax_k.grid(True, alpha=0.25)

    ax_cost.plot(np.arange(1, stage_cost_hist.size + 1), stage_cost_hist, color="tab:red", lw=1.2, label="instantaneous")
    ax_cost.plot(np.arange(1, cum_cost.size + 1), cum_cost, color="tab:green", lw=1.5, label="cumulative")
    ax_cost.set_xlim(0, n + 1)
    ax_cost.set_ylabel("Cost")
    ax_cost.grid(True, alpha=0.25)
    ax_cost.legend(loc="upper left", frameon=False, fontsize=8)

    if len(sel_step) > 0:
        ax_idx.scatter(sel_step, sel_idx, s=6, alpha=0.4, color="tab:orange")
    ax_idx.set_xlim(0, n + 1)
    ax_idx.set_ylim(0, max(1, int(max_cols)) + 5)
    ax_idx.set_xlabel("Step")
    ax_idx.set_ylabel("Selected index")
    ax_idx.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def build_controller(
    controller_args,
    debug: bool,
    eps_sigma: float,
    adaptive_k: bool,
    k_min: int,
    k_max: int,
    k_step: int,
    rho: float,
    gamma_min: float,
):
    selector = LkSelector(
        1,
        controller_args.deepc_dims,
        custom_callback=selector_cb,
        forgetting_factor=1,
    )
    controller = SelectDeePC(
        controller_args,
        selector_callback=selector,
        num_hankel_cols=int(k_min),
        n_iter=1,
        debug=debug,
        adaptive_k=bool(adaptive_k),
        K_min=int(k_min),
        K_max=int(k_max),
        K_step=int(k_step),
        rho=float(rho),
        gamma_min=float(gamma_min),
    )
    controller._eps_sigma = float(eps_sigma)
    return controller


def run_episode(
    controller,
    max_steps: int,
    seed: int,
    render_human: bool = False,
    live_k_viz: bool = False,
    live_viz_pause: float = 0.001,
    k_max_scale: int = 200,
    upright_angle_threshold: float = 0.2,
):
    render_mode = "human" if render_human else None
    env = gym.make("InvertedPendulum-v4-swingup", render_mode=render_mode)
    obs_raw = reset_env(env, seed=seed)

    steps = 0
    state_hist = []
    action_hist = []
    reached_upright = False
    lost_after_reach = False
    first_upright_step = -1
    final_angle_err = np.nan
    ended_terminated = False
    ended_truncated = False
    for _ in range(max_steps):
        obs_true = featurize_obs(np.asarray(obs_raw, dtype=float))
        reference = [0, 0, 1, 0, 0]
        action = controller.compute_action(obs_true, reference)
        state_hist.append(np.asarray(obs_raw, dtype=float).copy())
        action_hist.append(np.asarray(action, dtype=float).copy())
        obs_raw, terminated, truncated, info = step_env(env, action)
        steps += 1
        theta = float(np.asarray(obs_raw, dtype=float)[1])
        ang_err = abs(angle_error_to_upright(theta))
        final_angle_err = ang_err
        is_upright = bool(ang_err <= float(upright_angle_threshold))
        if is_upright and not reached_upright:
            reached_upright = True
            first_upright_step = int(steps)
        if reached_upright and (not is_upright):
            lost_after_reach = True
        if terminated or truncated:
            ended_terminated = bool(terminated)
            ended_truncated = bool(truncated)
            action_norm = float(np.linalg.norm(np.asarray(action, dtype=float)))
            state_norm = float(np.linalg.norm(np.asarray(obs_raw, dtype=float)))
            ob1 = float(np.asarray(obs_raw, dtype=float)[1])
            print(
                f"[term] step={steps} terminated={terminated} truncated={truncated} "
                f"action_norm={action_norm:.6g} state_norm={state_norm:.6g} "
                f"ob1={ob1:.6g} info={info}"
            )
            break

    env.close()

    sigma_hist = np.asarray(controller._slack_norm_inf_history, dtype=float)
    k_hist = np.asarray(controller._K_opt_history, dtype=float)
    success_upright_hold = bool(
        (steps >= int(max_steps))
        and reached_upright
        and (not lost_after_reach)
        and (not ended_terminated)
    )

    return {
        "steps": int(steps),
        "sigma_inf_max": safe_stat(sigma_hist, np.max),
        "sigma_inf_mean": safe_stat(sigma_hist, np.mean),
        "k_opt_median": safe_stat(k_hist, np.median),
        "slack_fail_rate": safe_stat(controller._slack_violation_history, np.mean, default=0.0),
        "min_sigma_min_Ag": safe_stat(controller._sigma_min_A_history, np.min),
        "min_sigma_min_Hu": safe_stat(controller._min_selected_sigma_history, np.min),
        "total_cost": safe_stat(controller._stage_cost_history, np.sum, default=0.0),
        "mean_solve_time_ms": safe_stat(controller._solve_time_ms_history, np.mean),
        "total_solve_time_ms": safe_stat(controller._solve_time_ms_history, np.sum, default=0.0),
        "state_traj": np.asarray(state_hist, dtype=float),
        "action_traj": np.asarray(action_hist, dtype=float),
        "k_hist": np.asarray(controller._K_opt_history, dtype=float),
        "stage_cost_hist": np.asarray(controller._stage_cost_history, dtype=float),
        "selected_hist": list(controller._selected_idcs_history),
        "reached_upright": int(reached_upright),
        "lost_after_reach": int(lost_after_reach),
        "first_upright_step": int(first_upright_step),
        "final_angle_err": float(final_angle_err),
        "success_upright_hold": int(success_upright_hold),
        "terminated": int(ended_terminated),
        "truncated": int(ended_truncated),
    }


def summarize_by_rho(rows):
    if not rows:
        return []

    out = []
    rho_values = sorted(set(float(r["rho"]) for r in rows))
    for rho in rho_values:
        subset = [r for r in rows if float(r["rho"]) == rho]
        success_count = int(sum(int(r["success_upright_hold"]) for r in subset))
        runs = int(len(subset))
        out.append(
            {
                "rho": float(rho),
                "runs": runs,
                "success_count_upright_hold": success_count,
                "failure_count_upright_hold": int(runs - success_count),
                "median_sigma_inf_max": safe_stat([r["sigma_inf_max"] for r in subset], np.median),
                "median_sigma_inf_mean": safe_stat([r["sigma_inf_mean"] for r in subset], np.median),
                "median_k_opt": safe_stat([r["k_opt_median"] for r in subset], np.median),
                "median_total_cost": safe_stat([r["total_cost"] for r in subset], np.median),
                "median_mean_solve_time_ms": safe_stat([r["mean_solve_time_ms"] for r in subset], np.median),
                "median_total_solve_time_ms": safe_stat([r["total_solve_time_ms"] for r in subset], np.median),
                "success_rate_upright_hold": safe_stat([r["success_upright_hold"] for r in subset], np.mean, default=0.0),
                "failure_rate_upright_hold": 1.0
                - safe_stat([r["success_upright_hold"] for r in subset], np.mean, default=0.0),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Adaptive-K runner for SelectDeePC.")
    parser.add_argument("--outdir", type=str, default=os.path.join(REPO_ROOT, "logs", "adaptive_k"))
    parser.add_argument(
        "--adaptive-k",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable internal adaptive-K in SelectDeePC.",
    )
    parser.add_argument("--Kmin", type=int, default=20)
    parser.add_argument("--Kmax", type=int, default=200)
    parser.add_argument("--Kstep", type=int, default=10)
    parser.add_argument("--rho", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--gamma_min", type=float, default=1e-6)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument(
        "--render-human",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render env in real time with render_mode='human'.",
    )
    parser.add_argument(
        "--live-k-viz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show one diagnostics plot at episode end (K/cost/selected indices).",
    )
    parser.add_argument(
        "--live-viz-pause",
        type=float,
        default=0.001,
        help="Pause seconds per live-viz update.",
    )
    parser.add_argument("--eps_sigma", type=float, default=1e-3)
    parser.add_argument(
        "--upright-angle-threshold",
        type=float,
        default=0.2,
        help="Upright condition threshold in |wrap(theta-pi)| [rad].",
    )
    parser.add_argument(
        "--save-trajectory-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save state/action trajectory plot after each run.",
    )
    parser.add_argument(
        "--show-trajectory-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display trajectory plot window after each run.",
    )
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

    if int(args.Kstep) <= 0:
        raise ValueError("Kstep must be positive")
    if int(args.Kmin) > int(args.Kmax):
        raise ValueError("Kmin must be <= Kmax")

    rows = []
    common_tag = (
        f"Kmin{int(args.Kmin)}_Kmax{int(args.Kmax)}_"
        f"Kstep{int(args.Kstep)}_gmin{float(args.gamma_min):.3g}"
    )

    for rho in args.rho:
        rho_tag = f"rho{float(rho):.3g}"
        for seed in range(int(args.seeds)):
            controller = build_controller(
                controller_args=controller_args,
                debug=bool(args.debug),
                eps_sigma=float(args.eps_sigma),
                adaptive_k=bool(args.adaptive_k),
                k_min=int(args.Kmin),
                k_max=int(args.Kmax),
                k_step=int(args.Kstep),
                rho=float(rho),
                gamma_min=float(args.gamma_min),
            )

            metrics = run_episode(
                controller=controller,
                max_steps=int(args.max_steps),
                seed=int(seed),
                render_human=bool(args.render_human),
                live_k_viz=bool(args.live_k_viz),
                live_viz_pause=float(args.live_viz_pause),
                k_max_scale=int(args.Kmax),
                upright_angle_threshold=float(args.upright_angle_threshold),
            )

            if bool(args.save_trajectory_plot):
                traj_png = os.path.join(
                    args.outdir, f"traj_{common_tag}_{rho_tag}_seed{seed:03d}.png"
                )
                save_trajectory_plot(
                    out_path=traj_png,
                    state_traj=metrics["state_traj"],
                    action_traj=metrics["action_traj"],
                    show_plot=bool(args.show_trajectory_plot),
                )
            if bool(args.live_k_viz):
                diag_png = os.path.join(
                    args.outdir, f"diag_{common_tag}_{rho_tag}_seed{seed:03d}.png"
                )
                save_selection_diagnostics_plot(
                    out_path=diag_png,
                    k_hist=metrics["k_hist"],
                    stage_cost_hist=metrics["stage_cost_hist"],
                    selected_hist=metrics["selected_hist"],
                    max_cols=int(args.Kmax),
                    show_plot=True,
                )

            npz_name = f"run_{common_tag}_{rho_tag}_seed{seed:03d}.npz"
            npz_path = os.path.join(args.outdir, npz_name)
            controller.save_history_npz(
                npz_path,
                extra={
                    "adaptive_k": bool(args.adaptive_k),
                    "rho": float(rho),
                    "gamma_min": float(args.gamma_min),
                    "Kmin": int(args.Kmin),
                    "Kmax": int(args.Kmax),
                    "Kstep": int(args.Kstep),
                    "seed": int(seed),
                    "steps": int(metrics["steps"]),
                    "upright_angle_threshold": float(args.upright_angle_threshold),
                    "success_upright_hold": int(metrics["success_upright_hold"]),
                },
            )

            row = {
                "adaptive_k": bool(args.adaptive_k),
                "rho": float(rho),
                "gamma_min": float(args.gamma_min),
                "Kmin": int(args.Kmin),
                "Kmax": int(args.Kmax),
                "Kstep": int(args.Kstep),
                "seed": int(seed),
                "steps": int(metrics["steps"]),
                "sigma_inf_max": float(metrics["sigma_inf_max"]),
                "sigma_inf_mean": float(metrics["sigma_inf_mean"]),
                "k_opt_median": float(metrics["k_opt_median"]),
                "slack_fail_rate": float(metrics["slack_fail_rate"]),
                "min_sigma_min_Ag": float(metrics["min_sigma_min_Ag"]),
                "min_sigma_min_Hu": float(metrics["min_sigma_min_Hu"]),
                "total_cost": float(metrics["total_cost"]),
                "mean_solve_time_ms": float(metrics["mean_solve_time_ms"]),
                "total_solve_time_ms": float(metrics["total_solve_time_ms"]),
                "reached_upright": int(metrics["reached_upright"]),
                "lost_after_reach": int(metrics["lost_after_reach"]),
                "first_upright_step": int(metrics["first_upright_step"]),
                "final_angle_err": float(metrics["final_angle_err"]),
                "success_upright_hold": int(metrics["success_upright_hold"]),
                "terminated": int(metrics["terminated"]),
                "truncated": int(metrics["truncated"]),
                "npz_path": npz_path,
            }
            rows.append(row)

            print(
                f"[run] rho={float(rho):.3f} seed={seed:3d} steps={metrics['steps']:4d} "
                f"sigma_ref={float(getattr(controller, '_sigma_ref', np.nan)):.3e} "
                f"sigma_max={metrics['sigma_inf_max']:.3e} "
                f"sigma_mean={metrics['sigma_inf_mean']:.3e} "
                f"K_med={metrics['k_opt_median']:.1f} "
                f"success={int(metrics['success_upright_hold'])}"
            )

    summary_path = os.path.join(args.outdir, f"summary_{common_tag}.csv")
    fieldnames = [
        "adaptive_k",
        "rho",
        "gamma_min",
        "Kmin",
        "Kmax",
        "Kstep",
        "seed",
        "steps",
        "sigma_inf_max",
        "sigma_inf_mean",
        "k_opt_median",
        "slack_fail_rate",
        "min_sigma_min_Ag",
        "min_sigma_min_Hu",
        "total_cost",
        "mean_solve_time_ms",
        "total_solve_time_ms",
        "reached_upright",
        "lost_after_reach",
        "first_upright_step",
        "final_angle_err",
        "success_upright_hold",
        "terminated",
        "truncated",
        "npz_path",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_by_rho_path = os.path.join(args.outdir, f"summary_by_rho_{common_tag}.csv")
    rows_by_rho = summarize_by_rho(rows)
    fieldnames_by_rho = [
        "rho",
        "runs",
        "success_count_upright_hold",
        "failure_count_upright_hold",
        "median_sigma_inf_max",
        "median_sigma_inf_mean",
        "median_k_opt",
        "median_total_cost",
        "median_mean_solve_time_ms",
        "median_total_solve_time_ms",
        "success_rate_upright_hold",
        "failure_rate_upright_hold",
    ]
    with open(summary_by_rho_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_by_rho)
        writer.writeheader()
        writer.writerows(rows_by_rho)

    overall_success = int(sum(int(r["success_upright_hold"]) for r in rows))
    overall_runs = int(len(rows))
    print(
        f"Success count (upright-hold): {overall_success}/{overall_runs} "
        f"(rate={overall_success / max(1, overall_runs):.3f})"
    )
    for rr in rows_by_rho:
        print(
            f"[rho={rr['rho']:.3g}] success={int(rr['success_count_upright_hold'])}/"
            f"{int(rr['runs'])} fail={int(rr['failure_count_upright_hold'])}"
        )

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved rho-median summary: {summary_by_rho_path}")
    print(f"Saved run npz files under: {args.outdir}")


if __name__ == "__main__":
    main()
    # Example:
    # python experiments/run_adaptive_k_selectdeepc.py --outdir logs/adaptive_k --adaptive-k --Kmin 20 --Kmax 200 --Kstep 10 --rho 0.05 0.1 0.2 --gamma_min 1e-6 --seeds 10 --max_steps 600 --eps_sigma 1e-3
