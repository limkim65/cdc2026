import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from select_deepc.data_selectors import LkSelector  # noqa: E402
from select_deepc.deepc_controller import SelectDeePC  # noqa: E402
from select_deepc.deepc_dataclasses import (  # noqa: E402
    DeePCConstraints,
    DeePCCost,
    DeePCControllerArgs,
    DeePCDims,
    TrajectoryDataSet,
)
from select_deepc.deepc_rocket_utils import SetPointScheduler  # noqa: E402
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
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(fn(finite))


def fix_dataset(dataset: TrajectoryDataSet):
    for traj in dataset.dataset:
        traj.state_trajectory[:, [4, 5]] += traj.state_trajectory[:, [8, 9]]
        traj.state_trajectory = traj.state_trajectory[:, [0, 1, 2, 3, 4, 5, 6, 7]]


def setup_deepc_reacher(
    trajectory_data: TrajectoryDataSet, enable_measurement_constraint: bool = False
) -> DeePCControllerArgs:
    p = 8
    m = 2
    n = 75
    t_past = 2
    t_fut = 15
    t_hankel = (m + 1) * (t_past + t_fut) + n - 1

    q = np.diag([0, 0, 0, 0, 40000, 40000, 10, 10])
    r = np.diag([10, 10])
    controller_costs = DeePCCost(q, r, 5000000.0, 10.0, 10000.0)

    a_u = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b_u = np.array([0.5, 0.5, 0.5, 0.5])
    a_y = None
    b_y = None
    if enable_measurement_constraint:
        a_y = np.array([[0, 0, 0, 0, 0, 1, 0, 0]])
        b_y = np.array([[0.1]])

    controller_constraints = DeePCConstraints(a_u, b_u, a_y, b_y)
    return DeePCControllerArgs(
        trajectory_data,
        DeePCDims(t_past, t_fut, p, m),
        t_hankel,
        controller_costs,
        controller_constraints,
        [0, 0],
        False,
    )


class ReacherTargets:
    def __init__(self, num_steps: int = 0):
        num_steps = int(max(0, num_steps))
        angles = np.linspace(3 * np.pi / 4, 2 * np.pi, max(1, num_steps))
        np.random.seed(42)
        dists = np.random.uniform(0.04, 0.15, max(1, num_steps))

        self.targets = [[0.043, 0.092]]
        for angle, dist in zip(angles[:num_steps], dists[:num_steps]):
            self.targets.append([dist * np.cos(angle), dist * np.sin(angle)])
        for target in self.targets:
            target[1] = np.minimum(target[1], 0.09)


class ReacherSetpointScheduler(SetPointScheduler):
    def __init__(self, env, deepc_dims: DeePCDims, num_extra_targets: int = 0):
        super().__init__(env)
        self._dims = deepc_dims
        self._targets = ReacherTargets(num_steps=int(num_extra_targets))
        self._target_idx = 0

    def __call__(self, state, **kwargs):
        if (
            np.linalg.norm(
                state[4:6]
                + state[8:10]
                - np.array(self._targets.targets[self._target_idx])
            )
            < 0.005
        ):
            self._target_idx = (self._target_idx + 1) % len(self._targets.targets)
        target_pos = self._targets.targets[self._target_idx]
        return [0, 0, 0, 0, target_pos[0], target_pos[1], 0, 0]

    def is_successful(self, state):
        return True


class ReacherCircleSetpointScheduler(SetPointScheduler):
    def __init__(
        self,
        env,
        center_x: float = 0.03,
        center_y: float = 0.06,
        radius: float = 0.02,
        omega: float = 0.05,
        phase: float = 0.0,
        clip_y_max: float = 0.09,
    ):
        super().__init__(env)
        self._cx = float(center_x)
        self._cy = float(center_y)
        self._r = float(max(1e-6, radius))
        self._omega = float(omega)
        self._phase = float(phase)
        self._clip_y_max = float(clip_y_max)
        self._t = 0

    def _target_xy(self, t: int) -> np.ndarray:
        th = self._phase + self._omega * float(t)
        x = self._cx + self._r * np.cos(th)
        y = self._cy + self._r * np.sin(th)
        y = min(y, self._clip_y_max)
        return np.array([x, y], dtype=float)

    def __call__(self, state, **kwargs):
        target_pos = self._target_xy(self._t)
        self._t += 1
        return [0, 0, 0, 0, float(target_pos[0]), float(target_pos[1]), 0, 0]

    def is_successful(self, state):
        return True


def get_cost_dict(cost_obj) -> Dict[str, float]:
    if isinstance(cost_obj, dict):
        return {k: float(v) for k, v in cost_obj.items()}
    return {"cost": float(cost_obj)}


def transform_obs(obs: np.ndarray) -> np.ndarray:
    o = np.asarray(obs, dtype=float).copy()
    o[4:6] += o[8:10]
    return o[[0, 1, 2, 3, 4, 5, 6, 7]]


def replay_targets(
    x_traj: np.ndarray,
    target_mode: str,
    num_extra_targets: int,
    circle_center_x: float,
    circle_center_y: float,
    circle_radius: float,
    circle_omega: float,
    circle_phase: float,
    circle_clip_y_max: float,
) -> np.ndarray:
    x_arr = np.asarray(x_traj, dtype=float)
    n = int(x_arr.shape[0]) if x_arr.ndim >= 1 else 0
    if n <= 0:
        return np.zeros((0, 2), dtype=float)

    if str(target_mode) == "circle":
        t = np.arange(n, dtype=float)
        th = float(circle_phase) + float(circle_omega) * t
        xx = float(circle_center_x) + float(circle_radius) * np.cos(th)
        yy = float(circle_center_y) + float(circle_radius) * np.sin(th)
        yy = np.minimum(yy, float(circle_clip_y_max))
        return np.vstack([xx, yy]).T

    targets = ReacherTargets(num_steps=int(num_extra_targets)).targets
    idx = 0
    out = []
    for obs in x_arr:
        if np.linalg.norm(obs[4:6] + obs[8:10] - np.asarray(targets[idx], dtype=float)) < 0.005:
            idx = (idx + 1) % len(targets)
        out.append(np.asarray(targets[idx], dtype=float))
    return np.vstack(out) if out else np.zeros((0, 2), dtype=float)


def compute_tracking_error(x_traj: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x_traj, dtype=float)
    if x_arr.size == 0:
        return np.zeros(0, dtype=float)
    ee_xy = x_arr[:, 4:6] + x_arr[:, 8:10]
    tgt = np.asarray(target_xy, dtype=float)
    return np.linalg.norm(ee_xy - tgt, axis=1)


def _corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    m = np.isfinite(aa) & np.isfinite(bb)
    if np.sum(m) < 3:
        return -np.inf
    aa = aa[m] - np.mean(aa[m])
    bb = bb[m] - np.mean(bb[m])
    na = np.linalg.norm(aa)
    nb = np.linalg.norm(bb)
    if na <= 1e-12 or nb <= 1e-12:
        return -np.inf
    return float(np.dot(aa, bb) / (na * nb))


def estimate_phase_lag_steps(target_xy: np.ndarray, ee_xy: np.ndarray, max_lag: int = 80) -> Tuple[int, float]:
    n = int(min(len(target_xy), len(ee_xy)))
    if n <= 5:
        return 0, np.nan
    tx = np.asarray(target_xy[:n, 0], dtype=float)
    ty = np.asarray(target_xy[:n, 1], dtype=float)
    ex = np.asarray(ee_xy[:n, 0], dtype=float)
    ey = np.asarray(ee_xy[:n, 1], dtype=float)
    max_lag = int(max(1, min(max_lag, n // 2)))

    scored = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            txx, tyy = tx[-lag:], ty[-lag:]
            exx, eyy = ex[: n + lag], ey[: n + lag]
        elif lag > 0:
            txx, tyy = tx[: n - lag], ty[: n - lag]
            exx, eyy = ex[lag:], ey[lag:]
        else:
            txx, tyy = tx, ty
            exx, eyy = ex, ey
        score = 0.5 * (_corr_1d(txx, exx) + _corr_1d(tyy, eyy))
        scored.append((int(lag), float(score)))
    best_score = max(s for _, s in scored)
    # Avoid periodic aliasing: among near-best correlations, choose smallest |lag|.
    tol = 0.02
    cand = [x for x in scored if x[1] >= best_score - tol]
    best_lag = min(cand, key=lambda x: abs(x[0]))[0]
    return int(best_lag), float(best_score)


def compute_tracking_metrics(target_xy: np.ndarray, ee_xy: np.ndarray, circle_omega: float) -> Dict[str, float]:
    n = int(min(len(target_xy), len(ee_xy)))
    if n <= 0:
        return {
            "tracking_rmse": np.nan,
            "tracking_phase_lag_steps": np.nan,
            "tracking_phase_lag_rad": np.nan,
            "tracking_phase_corr": np.nan,
            "tracking_amp_ratio": np.nan,
        }
    tgt = np.asarray(target_xy[:n], dtype=float)
    ee = np.asarray(ee_xy[:n], dtype=float)
    err = np.linalg.norm(ee - tgt, axis=1)
    max_lag = min(80, max(1, n // 3))
    if np.isfinite(circle_omega) and abs(float(circle_omega)) > 1e-9:
        half_period_steps = int(np.pi / abs(float(circle_omega)))
        max_lag = min(max_lag, max(5, half_period_steps))
    lag_steps, lag_score = estimate_phase_lag_steps(tgt, ee, max_lag=max_lag)
    lag_rad = float(circle_omega) * float(lag_steps)

    tgt_centered = tgt - np.mean(tgt, axis=0, keepdims=True)
    ee_centered = ee - np.mean(ee, axis=0, keepdims=True)
    tgt_r = np.linalg.norm(tgt_centered, axis=1)
    ee_r = np.linalg.norm(ee_centered, axis=1)
    amp_ratio = safe_stat(ee_r, np.mean) / max(safe_stat(tgt_r, np.mean), 1e-12)

    return {
        "tracking_rmse": float(np.sqrt(safe_stat(err * err, np.mean, default=np.nan))),
        "tracking_phase_lag_steps": float(lag_steps),
        "tracking_phase_lag_rad": float(lag_rad),
        "tracking_phase_corr": float(lag_score),
        "tracking_amp_ratio": float(amp_ratio),
    }


def compute_success(error: np.ndarray, eps: float, window: int) -> int:
    if error.size == 0:
        return 0
    ok = (np.asarray(error, dtype=float) <= float(eps)).astype(int)
    w = int(max(1, window))
    if ok.size < w:
        return int(np.all(ok == 1))
    run = np.convolve(ok, np.ones(w, dtype=int), mode="valid")
    return int(np.any(run >= w))


def compute_action_metrics(u_traj: np.ndarray, u_limit: float = 0.5) -> Dict[str, float]:
    u = np.asarray(u_traj, dtype=float)
    if u.size == 0:
        return {
            "effort_sum_sq": np.nan,
            "jerk_sum_sq": np.nan,
            "jerk_max": np.nan,
            "u_violation_rate": np.nan,
        }
    effort = np.sum(np.sum(u * u, axis=1))
    du = np.diff(u, axis=0)
    if du.size == 0:
        jerk_sum_sq = 0.0
        jerk_max = 0.0
    else:
        du_norm = np.linalg.norm(du, axis=1)
        jerk_sum_sq = float(np.sum(du_norm * du_norm))
        jerk_max = float(np.max(du_norm))
    viol = np.any(np.abs(u) > float(u_limit) + 1e-12, axis=1)
    return {
        "effort_sum_sq": float(effort),
        "jerk_sum_sq": float(jerk_sum_sq),
        "jerk_max": float(jerk_max),
        "u_violation_rate": float(np.mean(viol.astype(float))),
    }


def compute_localness_and_conditioning(
    deepc: SelectDeePC,
    x_traj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    selected_hist = deepc.get_selected_idcs_history()
    if len(selected_hist) == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    p = int(deepc._dims.p)
    tp = int(deepc._T_past)
    Y_p = np.asarray(deepc._H_y[: p * tp, :], dtype=float)
    H_u = np.asarray(deepc._H_u, dtype=float)

    x = np.asarray(x_traj, dtype=float)
    x_t = np.vstack([transform_obs(v) for v in x]) if x.size > 0 else np.zeros((0, p))

    y_past_hist = []
    y_past = None
    for state in x_t:
        if y_past is None:
            y_past = np.tile(state, tp)
        else:
            y_past = np.append(y_past[p:], state)
        y_past_hist.append(y_past.copy())

    n = min(len(selected_hist), len(y_past_hist))
    loc = np.full(n, np.nan, dtype=float)
    kappa_ag = np.full(n, np.nan, dtype=float)
    for t in range(n):
        idcs = np.asarray(selected_hist[t], dtype=int).reshape(-1)
        if idcs.size == 0:
            continue
        yp = y_past_hist[t]
        d = np.linalg.norm(Y_p[:, idcs] - yp[:, None], axis=0)
        loc[t] = float(np.mean(d))
        try:
            A_g = np.vstack([H_u[:, idcs], Y_p[:, idcs]])
            s = np.linalg.svd(A_g, compute_uv=False)
            if s.size > 0 and np.isfinite(s[-1]) and s[-1] > 0:
                kappa_ag[t] = float(s[0] / s[-1])
        except np.linalg.LinAlgError:
            kappa_ag[t] = np.nan
    return loc, kappa_ag


def summarize_group(rows: List[dict], x_key: str) -> List[dict]:
    out = []
    x_vals = sorted(set(float(r[x_key]) for r in rows))
    for xv in x_vals:
        subset = [r for r in rows if float(r[x_key]) == xv]
        out.append(
            {
                x_key: float(xv),
                "runs": int(len(subset)),
                "median_sigma_min_Ag": safe_stat([r["sigma_min_Ag"] for r in subset], np.median),
                "median_kappa_Ag": safe_stat([r["kappa_Ag_mean"] for r in subset], np.median),
                "median_localness": safe_stat([r["localness_mean"] for r in subset], np.median),
                "median_slack_max": safe_stat([r["slack_max"] for r in subset], np.median),
                "median_kkt_max": safe_stat([r["kkt_res_max"] for r in subset], np.median),
                "median_rmse_ee": safe_stat([r["rmse_ee"] for r in subset], np.median),
                "success_rate": safe_stat([r["success"] for r in subset], np.mean, default=0.0),
                "median_total_cost": safe_stat([r["total_cost"] for r in subset], np.median),
                "median_jerk_sum_sq": safe_stat([r["jerk_sum_sq"] for r in subset], np.median),
                "median_solve_ms": safe_stat([r["solve_ms_mean"] for r in subset], np.median),
                "median_k_opt_mean": safe_stat([r["k_opt_mean"] for r in subset], np.median),
            }
        )
    return out


def plot_group(summary: List[dict], outdir: str, x_key: str, prefix: str):
    if not summary:
        return
    xs = np.asarray([r[x_key] for r in summary], dtype=float)

    def arr(name):
        return np.asarray([r[name] for r in summary], dtype=float)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7), dpi=140)
    axs = axs.reshape(-1)
    metrics = [
        ("median_sigma_min_Ag", r"$\sigma_{\min}(A_g)$ (episode min)"),
        ("median_localness", "Localness mean"),
        ("median_slack_max", r"Slack $\|\sigma\|_{\infty}$ max"),
        ("median_kkt_max", "KKT proxy max"),
    ]
    for ax, (m, title) in zip(axs, metrics):
        y = arr(m)
        ax.plot(xs, y, marker="o", lw=1.3)
        ax.set_xlabel(x_key)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if np.all(y > 0):
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{prefix}_fig1_conditioning_solver.png"), bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7), dpi=140)
    axs = axs.reshape(-1)
    metrics = [
        ("median_rmse_ee", "End-effector RMSE"),
        ("success_rate", "Success rate"),
        ("median_total_cost", "Total cost"),
        ("median_jerk_sum_sq", "Jerk sum sq"),
    ]
    for ax, (m, title) in zip(axs, metrics):
        y = arr(m)
        ax.plot(xs, y, marker="o", lw=1.3)
        ax.set_xlabel(x_key)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if m != "success_rate" and np.all(y > 0):
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{prefix}_fig2_performance.png"), bbox_inches="tight")
    plt.close(fig)


def run_once(
    args,
    controller_args: DeePCControllerArgs,
    seed: int,
    mode: str,
    K: int,
    rho: float,
) -> dict:
    env = get_reacher_simulator(
        num_steps=int(args.max_steps),
        video_folder=os.path.join(args.outdir, "video"),
        video_title=f"{mode}_{str(args.target_mode)}_K{int(K)}_rho{float(rho):.3g}_seed{int(seed):03d}",
    )
    adaptive = bool(mode == "adaptive_rho")
    deepc = SelectDeePC(
        controller_args,
        selector_callback=LkSelector(),
        num_hankel_cols=int(K),
        n_iter=int(args.n_iter),
        adaptive_k=adaptive,
        K_min=int(args.Kmin),
        K_max=int(args.Kmax),
        K_step=int(args.Kstep),
        rho=float(rho),
        gamma_min=float(args.gamma_min),
    )
    deepc._eps_sigma = float(args.eps_sigma)

    if str(args.target_mode) == "circle":
        scheduler = ReacherCircleSetpointScheduler(
            env,
            center_x=float(args.circle_center_x),
            center_y=float(args.circle_center_y),
            radius=float(args.circle_radius),
            omega=float(args.circle_omega),
            phase=float(args.circle_phase),
            clip_y_max=float(args.circle_clip_y_max),
        )
    else:
        scheduler = ReacherSetpointScheduler(
            env, controller_args.deepc_dims, num_extra_targets=int(args.num_extra_targets)
        )
    x_traj, u_traj, _, _, cost_obj, _ = run_reacher_simulator(
        env,
        deepc,
        scheduler,
        seed=int(seed),
        deepc_cost_accumulator=PerformanceAccumulatorWrapper(
            DeePCCostAccumulator(controller_args.controller_costs),
            IntegralSquareError(mask=np.array([0, 0, 0, 0, 1, 1, 0, 0])),
            IntegralAbsoluteError(mask=np.array([0, 0, 0, 0, 1, 1, 0, 0])),
        ),
    )

    cost = get_cost_dict(cost_obj)
    k_hist = np.asarray(deepc.get_K_opt_history(), dtype=float)
    sigma_ag_hist = np.asarray(deepc.get_sigma_min_A_history(), dtype=float)
    slack_hist = np.asarray(deepc.get_slack_norm_inf_history(), dtype=float)
    kkt_hist = np.asarray(deepc.get_kkt_residual_history(), dtype=float)
    eq_hist = np.asarray(deepc.get_eq_residual_history(), dtype=float)
    solve_hist = np.asarray(deepc.get_solve_time_ms_history(), dtype=float)
    status_hist = np.asarray(deepc.get_status_history(), dtype=object)

    loc_hist, kappa_hist = compute_localness_and_conditioning(deepc, x_traj)
    tgt_hist = replay_targets(
        x_traj,
        target_mode=str(args.target_mode),
        num_extra_targets=int(args.num_extra_targets),
        circle_center_x=float(args.circle_center_x),
        circle_center_y=float(args.circle_center_y),
        circle_radius=float(args.circle_radius),
        circle_omega=float(args.circle_omega),
        circle_phase=float(args.circle_phase),
        circle_clip_y_max=float(args.circle_clip_y_max),
    )
    err = compute_tracking_error(x_traj, tgt_hist)
    act = compute_action_metrics(u_traj, u_limit=0.5)
    x_arr = np.asarray(x_traj, dtype=float)
    if x_arr.ndim == 2 and x_arr.shape[1] >= 10:
        ee_xy = x_arr[:, 4:6] + x_arr[:, 8:10]
    else:
        ee_xy = np.zeros((0, 2), dtype=float)
    tracking = compute_tracking_metrics(tgt_hist, ee_xy, circle_omega=float(args.circle_omega))

    if str(args.target_mode) == "circle":
        success = int(
            (tracking["tracking_rmse"] <= float(args.tracking_success_rmse))
            and (abs(tracking["tracking_phase_lag_steps"]) <= float(args.tracking_success_lag_steps))
            and (tracking["tracking_amp_ratio"] >= float(args.tracking_success_amp_min))
        )
    else:
        success = compute_success(err, eps=float(args.success_eps), window=int(args.success_window))
    fail_solver = np.array(
        [0 if ("optimal" in str(s)) else 1 for s in status_hist], dtype=float
    )

    tag = (
        f"{mode}_{str(args.target_mode)}_K{int(K):03d}_rho{float(rho):.3g}_"
        f"Kmin{int(args.Kmin)}_Kmax{int(args.Kmax)}_Kstep{int(args.Kstep)}_"
        f"gmin{float(args.gamma_min):.3g}"
    )
    npz_path = os.path.join(args.outdir, f"run_{tag}_seed{int(seed):03d}.npz")
    deepc.save_history_npz(
        npz_path,
        extra={
            "mode": mode,
            "seed": int(seed),
            "K": int(K),
            "rho": float(rho),
            "target_mode": str(args.target_mode),
            "steps": int(np.asarray(x_traj).shape[0]),
            "sigma_min_Ag_episode_min": float(safe_stat(sigma_ag_hist, np.min)),
            "kappa_Ag_mean": float(safe_stat(kappa_hist, np.mean)),
            "localness_mean": float(safe_stat(loc_hist, np.mean)),
            "localness_max": float(safe_stat(loc_hist, np.max)),
            "kkt_res_max": float(safe_stat(kkt_hist, np.max)),
            "eq_res_max": float(safe_stat(eq_hist, np.max)),
            "slack_max": float(safe_stat(slack_hist, np.max)),
            "slack_mean": float(safe_stat(slack_hist, np.mean)),
            "solver_fail_rate": float(safe_stat(fail_solver, np.mean, default=0.0)),
            "rmse_ee": float(np.sqrt(safe_stat(err * err, np.mean, default=np.nan))),
            "err_p95": float(safe_stat(err, lambda x: np.quantile(x, 0.95))),
            "err_max": float(safe_stat(err, np.max)),
            "err_final": float(safe_stat(err[-1:], np.mean)),
            "success": int(success),
            "tracking_rmse": float(tracking["tracking_rmse"]),
            "tracking_phase_lag_steps": float(tracking["tracking_phase_lag_steps"]),
            "tracking_phase_lag_rad": float(tracking["tracking_phase_lag_rad"]),
            "tracking_phase_corr": float(tracking["tracking_phase_corr"]),
            "tracking_amp_ratio": float(tracking["tracking_amp_ratio"]),
            "effort_sum_sq": float(act["effort_sum_sq"]),
            "jerk_sum_sq": float(act["jerk_sum_sq"]),
            "jerk_max": float(act["jerk_max"]),
            "u_violation_rate": float(act["u_violation_rate"]),
            "total_cost": float(cost.get("cost", np.nan)),
            "ISE": float(cost.get("ISE", np.nan)),
            "IAE": float(cost.get("IAE", np.nan)),
            "solve_ms_mean": float(safe_stat(solve_hist, np.mean)),
            "solve_ms_total": float(safe_stat(solve_hist, np.sum)),
            "k_opt_mean": float(safe_stat(k_hist, np.mean)),
            "k_opt_var": float(safe_stat(k_hist, np.var)),
            "executed_state_traj": np.asarray(x_traj, dtype=float),
            "executed_input_traj": np.asarray(u_traj, dtype=float),
            "tracking_error": np.asarray(err, dtype=float),
            "target_xy": np.asarray(tgt_hist, dtype=float),
            "localness_hist": np.asarray(loc_hist, dtype=float),
            "kappa_Ag_hist": np.asarray(kappa_hist, dtype=float),
        },
    )

    row = {
        "mode": mode,
        "seed": int(seed),
        "K": int(K),
        "rho": float(rho),
        "target_mode": str(args.target_mode),
        "steps": int(np.asarray(x_traj).shape[0]),
        "k_opt_mean": float(safe_stat(k_hist, np.mean)),
        "k_opt_var": float(safe_stat(k_hist, np.var)),
        "sigma_min_Ag": float(safe_stat(sigma_ag_hist, np.min)),
        "kappa_Ag_mean": float(safe_stat(kappa_hist, np.mean)),
        "localness_mean": float(safe_stat(loc_hist, np.mean)),
        "slack_max": float(safe_stat(slack_hist, np.max)),
        "slack_mean": float(safe_stat(slack_hist, np.mean)),
        "kkt_res_max": float(safe_stat(kkt_hist, np.max)),
        "eq_res_max": float(safe_stat(eq_hist, np.max)),
        "solver_fail_rate": float(safe_stat(fail_solver, np.mean, default=0.0)),
        "rmse_ee": float(np.sqrt(safe_stat(err * err, np.mean, default=np.nan))),
        "err_p95": float(safe_stat(err, lambda x: np.quantile(x, 0.95))),
        "err_max": float(safe_stat(err, np.max)),
        "err_final": float(safe_stat(err[-1:], np.mean)),
        "success": int(success),
        "tracking_rmse": float(tracking["tracking_rmse"]),
        "tracking_phase_lag_steps": float(tracking["tracking_phase_lag_steps"]),
        "tracking_phase_lag_rad": float(tracking["tracking_phase_lag_rad"]),
        "tracking_phase_corr": float(tracking["tracking_phase_corr"]),
        "tracking_amp_ratio": float(tracking["tracking_amp_ratio"]),
        "effort_sum_sq": float(act["effort_sum_sq"]),
        "jerk_sum_sq": float(act["jerk_sum_sq"]),
        "jerk_max": float(act["jerk_max"]),
        "u_violation_rate": float(act["u_violation_rate"]),
        "total_cost": float(cost.get("cost", np.nan)),
        "ISE": float(cost.get("ISE", np.nan)),
        "IAE": float(cost.get("IAE", np.nan)),
        "solve_ms_mean": float(safe_stat(solve_hist, np.mean)),
        "solve_ms_total": float(safe_stat(solve_hist, np.sum)),
        "npz_path": npz_path,
    }
    return row


def main():
    parser = argparse.ArgumentParser(
        description="CDC metric benchmark runner for Reacher SelectDeePC."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(REPO_ROOT, "logs", "reacher", "adaptive_k_dir"),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fixed_k", "adaptive_rho", "both"],
        default="both",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--Kmin", type=int, default=40)
    parser.add_argument("--Kmax", type=int, default=200)
    parser.add_argument("--Kstep", type=int, default=10)
    parser.add_argument("--rho", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--gamma_min", type=float, default=1e-6)
    parser.add_argument("--eps_sigma", type=float, default=1e-3)
    parser.add_argument("--success_eps", type=float, default=0.01)
    parser.add_argument("--success_window", type=int, default=20)
    parser.add_argument("--num_extra_targets", type=int, default=0)
    parser.add_argument(
        "--target_mode",
        type=str,
        choices=["waypoint", "circle"],
        default="waypoint",
    )
    parser.add_argument("--circle_center_x", type=float, default=0.03)
    parser.add_argument("--circle_center_y", type=float, default=0.06)
    parser.add_argument("--circle_radius", type=float, default=0.02)
    parser.add_argument("--circle_omega", type=float, default=0.05)
    parser.add_argument("--circle_phase", type=float, default=0.0)
    parser.add_argument("--circle_clip_y_max", type=float, default=0.09)
    parser.add_argument("--tracking_success_rmse", type=float, default=0.06)
    parser.add_argument("--tracking_success_lag_steps", type=float, default=15.0)
    parser.add_argument("--tracking_success_amp_min", type=float, default=0.70)
    parser.add_argument(
        "--dataset", type=str, choices=["iid", "random_walk", "both"], default="iid"
    )
    parser.add_argument("--enable_measurement_constraint", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    iid = load_data_from_folder(
        os.path.join(REPO_ROOT, "data", "reacher", "dataset_iid"), "iid"
    )
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

    controller_args = setup_deepc_reacher(
        data, enable_measurement_constraint=bool(args.enable_measurement_constraint)
    )

    rows: List[dict] = []
    run_fixed = args.mode in ["fixed_k", "both"]
    run_adapt = args.mode in ["adaptive_rho", "both"]

    if run_fixed:
        for K in range(int(args.Kmin), int(args.Kmax) + 1, int(args.Kstep)):
            for seed in range(int(args.seeds)):
                row = run_once(
                    args=args,
                    controller_args=controller_args,
                    seed=seed,
                    mode="fixed_k",
                    K=K,
                    rho=float(args.rho[0]),
                )
                rows.append(row)
                print(
                    f"[fixed_k] K={int(K):3d} seed={int(seed):3d} "
                    f"sigma_min_Ag={row['sigma_min_Ag']:.3e} local={row['localness_mean']:.3e} "
                    f"slack_max={row['slack_max']:.3e} kkt_max={row['kkt_res_max']:.3e} "
                    f"rmse={row['rmse_ee']:.3e} succ={int(row['success'])} "
                    f"cost={row['total_cost']:.3e} solve_ms={row['solve_ms_mean']:.2f}"
                )

    if run_adapt:
        for rho in args.rho:
            for seed in range(int(args.seeds)):
                row = run_once(
                    args=args,
                    controller_args=controller_args,
                    seed=seed,
                    mode="adaptive_rho",
                    K=int(args.Kmin),
                    rho=float(rho),
                )
                rows.append(row)
                print(
                    f"[adaptive_rho] rho={float(rho):.3g} seed={int(seed):3d} "
                    f"Kopt_mean={row['k_opt_mean']:.2f} sigma_min_Ag={row['sigma_min_Ag']:.3e} "
                    f"local={row['localness_mean']:.3e} slack_max={row['slack_max']:.3e} "
                    f"kkt_max={row['kkt_res_max']:.3e} rmse={row['rmse_ee']:.3e} "
                    f"succ={int(row['success'])} cost={row['total_cost']:.3e}"
                )

    summary_path = os.path.join(args.outdir, "summary.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(summary_path, "w", newline="") as f:
            f.write("")

    fixed_rows = [r for r in rows if r["mode"] == "fixed_k"]
    adapt_rows = [r for r in rows if r["mode"] == "adaptive_rho"]

    if fixed_rows:
        s_fixed = summarize_group(fixed_rows, x_key="K")
        p = os.path.join(args.outdir, "summary_by_K.csv")
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(s_fixed[0].keys()))
            writer.writeheader()
            writer.writerows(s_fixed)
        plot_group(s_fixed, args.outdir, x_key="K", prefix="fixed_k")

    if adapt_rows:
        s_adapt = summarize_group(adapt_rows, x_key="rho")
        p = os.path.join(args.outdir, "summary_by_rho.csv")
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(s_adapt[0].keys()))
            writer.writeheader()
            writer.writerows(s_adapt)
        plot_group(s_adapt, args.outdir, x_key="rho", prefix="adaptive_rho")

    print(f"\nSaved run summary: {summary_path}")
    print(f"Saved plots under: {args.outdir}")


if __name__ == "__main__":
    main()
