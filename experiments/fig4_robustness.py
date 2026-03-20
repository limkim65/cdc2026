import argparse
import os
import sys
from copy import deepcopy
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from experiments.reacher.run_cdc_reacher_benchmark import (  # noqa: E402
    ReacherSetpointScheduler,
    compute_success,
    fix_dataset,
    setup_deepc_reacher,
)
from select_deepc.data_selectors import AdaptiveLkSelector, LkSelector  # noqa: E402
from select_deepc.deepc_controller import AdaptiveSelectDeePC  # noqa: E402
from select_deepc.deepc_utils import load_data_from_folder  # noqa: E402


DEFAULT_SUCCESS_EPS = 0.01
DEFAULT_SUCCESS_WINDOW = 20


def _safe_stat(values: Iterable[float], fn, default=np.nan) -> float:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if arr.size == 0:
        return float(default)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(fn(arr))


def _safe_quantile(values: Iterable[float], q: float, default=np.nan) -> float:
    return _safe_stat(values, lambda x: np.quantile(x, q), default=default)


def _parse_float_csv(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip() != ""]


def _parse_int_csv(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip() != ""]


def _make_reacher_env_no_video(max_steps: int):
    from gymnasium.envs.registration import register, registry

    env_id = f"Reacher-v4-fig4-{int(max_steps)}"
    if env_id not in registry:
        register(
            id=env_id,
            entry_point="gymnasium.envs.mujoco.reacher_v4:ReacherEnv",
            max_episode_steps=int(max_steps),
            reward_threshold=-3.75,
        )
    return gym.make(env_id, render_mode="rgb_array")


def _obs_to_y(obs: np.ndarray) -> np.ndarray:
    """Transform env observation to controller measurement y_k.

    For Reacher-v4, this matches the existing project convention:
    y = [obs[0], obs[1], obs[2], obs[3], obs[4]+obs[8], obs[5]+obs[9], obs[6], obs[7]]
    """
    arr = np.asarray(obs, dtype=float).reshape(-1)
    if arr.size >= 10:
        a = arr.copy()
        a[4:6] += a[8:10]
        return a[[0, 1, 2, 3, 4, 5, 6, 7]]
    return arr.copy()


def _to_reference(reference: Any, y_dim: int) -> np.ndarray:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    if ref.size == y_dim:
        return ref
    if ref.size > y_dim:
        return ref[:y_dim]
    out = np.zeros(int(y_dim), dtype=float)
    out[: ref.size] = ref
    return out


def _tracking_error_from_y(y: np.ndarray, ref: np.ndarray) -> float:
    yv = np.asarray(y, dtype=float).reshape(-1)
    rv = np.asarray(ref, dtype=float).reshape(-1)
    n = min(yv.size, rv.size)
    if n <= 0:
        return np.nan
    yv = yv[:n]
    rv = rv[:n]
    if n >= 6:
        return float(np.linalg.norm(yv[4:6] - rv[4:6]))
    return float(np.linalg.norm(yv - rv))


def _extract_last_from_inner_controller(controller: Any) -> Tuple[Optional[str], float]:
    """Fallback metric extraction for controllers without explicit histories."""
    inner = getattr(controller, "_deepc", None)
    status = None
    slack = np.nan
    if inner is None:
        return status, slack
    if hasattr(inner, "get_last_solver_status"):
        try:
            status = inner.get_last_solver_status()
        except Exception:
            status = None
    if hasattr(inner, "get_last_slack_inf"):
        try:
            slack = float(inner.get_last_slack_inf())
        except Exception:
            slack = np.nan
    return status, slack


def _get_history_or_fallback(controller: Any, getter: str, fallback: Sequence, dtype=None):
    if hasattr(controller, getter):
        try:
            values = getattr(controller, getter)()
            arr = np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
            if arr.size > 0:
                return arr
        except Exception:
            pass
    return np.asarray(fallback, dtype=dtype) if dtype is not None else np.asarray(fallback)


def _instantiate_env(env: Any) -> Tuple[Any, bool]:
    if callable(env):
        return env(), True
    return env, False


def _instantiate_scheduler(scheduler: Any, env_obj: Any, controller_obj: Any):
    if callable(scheduler):
        # Support scheduler factories with signatures: (), (env), or (env, controller)
        for args in [(env_obj, controller_obj), (env_obj,), tuple()]:
            try:
                return scheduler(*args)
            except TypeError:
                continue
    try:
        sch = deepcopy(scheduler)
    except Exception:
        sch = scheduler
    if hasattr(sch, "_env"):
        try:
            sch._env = env_obj
        except Exception:
            pass
    return sch


def rollout_with_noise(env, controller, scheduler, sigma_n, seed, max_steps) -> dict:
    """Run one noisy rollout.

    Noise model:
        y_meas = y + sigma_n * N(0, I)
    """
    sigma_n = float(sigma_n)
    rng = np.random.default_rng(int(seed))

    obs, _ = env.reset(seed=int(seed))
    status_step_hist: List[Optional[str]] = []
    slack_step_hist: List[float] = []
    solve_step_hist: List[float] = []
    err_hist: List[float] = []
    y_hist: List[np.ndarray] = []
    ref_hist: List[np.ndarray] = []

    done = False
    truncated = False
    info_last: Dict[str, Any] = {}

    for _ in range(int(max_steps)):
        y_true = _obs_to_y(obs)
        if scheduler is not None:
            reference = scheduler(obs, env=env)
        else:
            reference = np.zeros_like(y_true)
        reference = _to_reference(reference, y_true.size)
        y_meas = y_true + sigma_n * rng.normal(0.0, 1.0, size=y_true.shape)

        t0 = perf_counter()
        action = controller.compute_action(y_meas, reference)
        t1 = perf_counter()
        solve_step_hist.append(1000.0 * (t1 - t0))

        st, sl = _extract_last_from_inner_controller(controller)
        status_step_hist.append(st)
        slack_step_hist.append(sl)

        if action is None:
            break

        err_hist.append(_tracking_error_from_y(y_true, reference))
        y_hist.append(y_true)
        ref_hist.append(reference)

        obs, _, done, truncated, info_last = env.step(action)
        if done or truncated:
            break

    status_hist = _get_history_or_fallback(controller, "get_status_history", status_step_hist, dtype=object)
    slack_hist = _get_history_or_fallback(controller, "get_slack_norm_inf_history", slack_step_hist, dtype=float)
    solve_hist = _get_history_or_fallback(controller, "get_solve_time_ms_history", solve_step_hist, dtype=float)

    status_str = np.array([str(s).lower() for s in np.asarray(status_hist, dtype=object).reshape(-1)], dtype=object)
    optimal_rate = float(np.mean([("optimal" in s) for s in status_str])) if status_str.size else np.nan

    rmse_info = info_last.get("rmse", np.nan) if isinstance(info_last, dict) else np.nan
    if np.isfinite(rmse_info):
        rmse = float(rmse_info)
    else:
        err_arr = np.asarray(err_hist, dtype=float)
        rmse = float(np.sqrt(_safe_stat(err_arr * err_arr, np.mean, default=np.nan)))

    success_info = np.nan
    if isinstance(info_last, dict) and "success" in info_last:
        try:
            success_info = float(info_last["success"])
        except Exception:
            success_info = np.nan

    success = np.nan
    if np.isfinite(success_info):
        success = float(success_info)
    else:
        err_arr = np.asarray(err_hist, dtype=float)
        if err_arr.size > 0:
            success = float(compute_success(err_arr, eps=DEFAULT_SUCCESS_EPS, window=DEFAULT_SUCCESS_WINDOW))

    kopt_hist = _get_history_or_fallback(controller, "get_K_opt_history", [], dtype=float)
    sigma_mk_hist = _get_history_or_fallback(controller, "get_sigma_min_Mk_history", [], dtype=float)

    return {
        "sigma_n": float(sigma_n),
        "seed": int(seed),
        "steps": int(len(err_hist)),
        "optimal_rate": float(optimal_rate),
        "slack_q95": _safe_quantile(slack_hist, 0.95, default=np.nan),
        "solve_time_q95": _safe_quantile(solve_hist, 0.95, default=np.nan),
        "rmse": float(rmse),
        "success": float(success) if np.isfinite(success) else np.nan,
        "k_opt_mean": _safe_stat(kopt_hist, np.mean, default=np.nan),
        "k_opt_q95": _safe_quantile(kopt_hist, 0.95, default=np.nan),
        "sigma_min_Mk_min": _safe_stat(sigma_mk_hist, np.min, default=np.nan),
        "sigma_min_Mk_repr": _safe_stat(sigma_mk_hist, np.median, default=np.nan),
    }


def robustness_sweep(env, scheduler, make_controller_fn, sigmas, seeds, max_steps) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sigmas = [float(s) for s in sigmas]
    seeds = [int(s) for s in seeds]
    total = len(sigmas) * len(seeds)
    ridx = 0

    for sigma_n in sigmas:
        for seed in seeds:
            ridx += 1
            controller_raw = make_controller_fn()
            method = getattr(make_controller_fn, "method_name", None)
            if isinstance(controller_raw, tuple) and len(controller_raw) == 2:
                method = str(controller_raw[0])
                controller = controller_raw[1]
            else:
                controller = controller_raw
                if method is None:
                    method = controller.__class__.__name__

            env_run, close_after = _instantiate_env(env)
            scheduler_run = _instantiate_scheduler(scheduler, env_run, controller)
            try:
                metrics = rollout_with_noise(
                    env=env_run,
                    controller=controller,
                    scheduler=scheduler_run,
                    sigma_n=sigma_n,
                    seed=seed,
                    max_steps=max_steps,
                )
                metrics["method"] = method
                rows.append(metrics)
                print(
                    f"[{ridx}/{total}] method={method} sigma={sigma_n:.4g} seed={seed:03d} "
                    f"opt_rate={metrics['optimal_rate']:.3f} rmse={metrics['rmse']:.4g} "
                    f"slack_q95={metrics['slack_q95']:.3g} solve_q95={metrics['solve_time_q95']:.2f}ms"
                )
            finally:
                if close_after and hasattr(env_run, "close"):
                    try:
                        env_run.close()
                    except Exception:
                        pass

    return pd.DataFrame(rows)


def _aggregate_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "optimal_rate",
        "slack_q95",
        "solve_time_q95",
        "rmse",
        "success",
        "k_opt_mean",
        "sigma_min_Mk_min",
        "sigma_min_Mk_repr",
    ]
    use_cols = [c for c in metric_cols if c in df.columns]
    grouped = df.groupby(["method", "sigma_n"], as_index=False)[use_cols].agg(["mean", "std"])
    grouped.columns = ["_".join(c).strip("_") for c in grouped.columns.to_flat_index()]
    return grouped


def plot_fig4(df: pd.DataFrame, out_png: str) -> None:
    if df.empty:
        raise ValueError("plot_fig4: empty dataframe")

    agg = _aggregate_for_plot(df)
    methods = sorted(agg["method"].unique().tolist())
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(methods)}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=180)
    ax_opt, ax_stab, ax_rmse = axes
    ax_stab2 = ax_stab.twinx()

    # (a) optimal_rate
    for method in methods:
        sub = agg[agg["method"] == method].sort_values("sigma_n")
        x = sub["sigma_n"].to_numpy(dtype=float)
        y = sub["optimal_rate_mean"].to_numpy(dtype=float)
        s = np.nan_to_num(sub["optimal_rate_std"].to_numpy(dtype=float), nan=0.0)
        ax_opt.plot(x, y, marker="o", lw=2, color=colors[method], label=method)
        ax_opt.fill_between(x, y - s, y + s, alpha=0.18, color=colors[method])
    ax_opt.set_title("(a) Optimal Solver Rate")
    ax_opt.set_xlabel("measurement noise sigma_n")
    ax_opt.set_ylabel("optimal_rate (higher better)")
    ax_opt.grid(alpha=0.25)
    ax_opt.set_ylim(-0.02, 1.02)
    ax_opt.legend(loc="best", fontsize=8)

    # (b) robustness/stability: slack_q95 + solve_time_q95
    for method in methods:
        sub = agg[agg["method"] == method].sort_values("sigma_n")
        x = sub["sigma_n"].to_numpy(dtype=float)
        y_slack = sub["slack_q95_mean"].to_numpy(dtype=float)
        s_slack = np.nan_to_num(sub["slack_q95_std"].to_numpy(dtype=float), nan=0.0)
        y_solve = sub["solve_time_q95_mean"].to_numpy(dtype=float)
        s_solve = np.nan_to_num(sub["solve_time_q95_std"].to_numpy(dtype=float), nan=0.0)

        ax_stab.plot(
            x, y_slack, marker="o", lw=2, color=colors[method], label=f"{method} | slack_q95"
        )
        ax_stab.fill_between(x, y_slack - s_slack, y_slack + s_slack, alpha=0.14, color=colors[method])

        ax_stab2.plot(
            x,
            y_solve,
            marker="s",
            lw=1.6,
            ls="--",
            color=colors[method],
            label=f"{method} | solve_q95",
        )
        ax_stab2.fill_between(x, y_solve - s_solve, y_solve + s_solve, alpha=0.10, color=colors[method])

    ax_stab.set_title("(b) Slack/Time Robustness")
    ax_stab.set_xlabel("measurement noise sigma_n")
    ax_stab.set_ylabel("slack_q95 (lower better)")
    ax_stab2.set_ylabel("solve_time_q95 [ms] (lower better)")
    ax_stab.grid(alpha=0.25)
    h1, l1 = ax_stab.get_legend_handles_labels()
    h2, l2 = ax_stab2.get_legend_handles_labels()
    ax_stab.legend(h1 + h2, l1 + l2, loc="best", fontsize=7)

    # (c) rmse
    for method in methods:
        sub = agg[agg["method"] == method].sort_values("sigma_n")
        x = sub["sigma_n"].to_numpy(dtype=float)
        y = sub["rmse_mean"].to_numpy(dtype=float)
        s = np.nan_to_num(sub["rmse_std"].to_numpy(dtype=float), nan=0.0)
        ax_rmse.plot(x, y, marker="o", lw=2, color=colors[method], label=method)
        ax_rmse.fill_between(x, y - s, y + s, alpha=0.18, color=colors[method])
    ax_rmse.set_title("(c) Tracking RMSE")
    ax_rmse.set_xlabel("measurement noise sigma_n")
    ax_rmse.set_ylabel("rmse (lower better)")
    ax_rmse.grid(alpha=0.25)
    ax_rmse.legend(loc="best", fontsize=8)

    fig.suptitle("Fig.4 Robustness Under Measurement Noise", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _save_df_npz(df: pd.DataFrame, out_npz: str) -> None:
    payload = {}
    for col in df.columns:
        vals = df[col].to_numpy()
        if vals.dtype == object:
            vals = vals.astype(str)
        payload[str(col)] = vals
    np.savez(out_npz, **payload)


def main():
    parser = argparse.ArgumentParser(description="Fig.4 robustness sweep for Reacher DeePC selection.")
    parser.add_argument("--outdir", type=str, default=os.path.join("logs", "reacher", "fig4_robustness"))
    parser.add_argument("--dataset", choices=["iid", "random_walk", "both"], default="iid")
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--sigmas", type=str, default="0,0.005,0.01,0.02,0.05")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--num_extra_targets", type=int, default=0)
    parser.add_argument("--enable_measurement_constraint", action="store_true")

    # Baselines
    parser.add_argument("--large_K", type=int, default=400)
    parser.add_argument("--small_K", type=int, default=80)
    parser.add_argument("--include_small_baseline", action="store_true")

    # Proposed SelectDeePC (CPQR + sigma gate)
    parser.add_argument("--sigma_bar", type=float, default=0.5)
    parser.add_argument("--N_loc", type=int, default=1000)
    parser.add_argument("--d_max", type=float, default=10.0)
    parser.add_argument("--K_min", type=int, default=40)
    parser.add_argument("--K_max", type=int, default=10000)
    parser.add_argument("--K_step", type=int, default=1)

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sigmas = _parse_float_csv(args.sigmas)
    seeds = _parse_int_csv(args.seeds)
    if len(sigmas) == 0 or len(seeds) == 0:
        raise ValueError("sigmas/seeds must be non-empty")

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

    controller_args = setup_deepc_reacher(
        data, enable_measurement_constraint=bool(args.enable_measurement_constraint)
    )

    env_factory = lambda: _make_reacher_env_no_video(args.max_steps)
    scheduler_factory = lambda env, *_: ReacherSetpointScheduler(
        env, controller_args.deepc_dims, num_extra_targets=int(args.num_extra_targets)
    )

    def make_large_baseline():
        return (
            "baseline_largeK",
            AdaptiveSelectDeePC(
                controller_args,
                selector_callback=LkSelector(),
                num_hankel_cols=int(args.large_K),
                n_iter=int(args.n_iter),
                adaptive_k=False,
                d_gate_enabled=False,
                cond_gate_enabled=False,
            ),
        )

    def make_proposed():
        return (
            "proposed_cpqr_gate",
            AdaptiveSelectDeePC(
                controller_args,
                selector_callback=AdaptiveLkSelector(order=2),
                num_hankel_cols=int(args.large_K),
                n_iter=int(args.n_iter),
                adaptive_k=True,
                K_min=int(args.K_min),
                K_max=int(args.K_max),
                K_step=int(args.K_step),
                sigma_bar=float(args.sigma_bar),
                N_loc=int(args.N_loc),
                d_max=float(args.d_max),
                d_gate_enabled=True,
                cond_gate_enabled=True,
            ),
        )

    def make_small_baseline():
        return (
            "baseline_smallK",
            AdaptiveSelectDeePC(
                controller_args,
                selector_callback=LkSelector(),
                num_hankel_cols=int(args.small_K),
                n_iter=int(args.n_iter),
                adaptive_k=False,
                d_gate_enabled=False,
                cond_gate_enabled=False,
            ),
        )

    dfs: List[pd.DataFrame] = []
    for make_fn in [make_large_baseline, make_proposed]:
        df = robustness_sweep(
            env=env_factory,
            scheduler=scheduler_factory,
            make_controller_fn=make_fn,
            sigmas=sigmas,
            seeds=seeds,
            max_steps=int(args.max_steps),
        )
        dfs.append(df)
    if bool(args.include_small_baseline):
        df = robustness_sweep(
            env=env_factory,
            scheduler=scheduler_factory,
            make_controller_fn=make_small_baseline,
            sigmas=sigmas,
            seeds=seeds,
            max_steps=int(args.max_steps),
        )
        dfs.append(df)

    raw_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    raw_csv = os.path.join(args.outdir, "fig4_raw_results.csv")
    raw_npz = os.path.join(args.outdir, "fig4_raw_results.npz")
    raw_df.to_csv(raw_csv, index=False)
    _save_df_npz(raw_df, raw_npz)

    agg_df = _aggregate_for_plot(raw_df) if not raw_df.empty else pd.DataFrame()
    agg_csv = os.path.join(args.outdir, "fig4_summary_by_sigma.csv")
    agg_npz = os.path.join(args.outdir, "fig4_summary_by_sigma.npz")
    agg_df.to_csv(agg_csv, index=False)
    _save_df_npz(agg_df, agg_npz)

    fig_path = os.path.join(args.outdir, "fig4_robustness.png")
    if not raw_df.empty:
        plot_fig4(raw_df, fig_path)

    print(f"Saved raw csv: {raw_csv}")
    print(f"Saved raw npz: {raw_npz}")
    print(f"Saved summary csv: {agg_csv}")
    print(f"Saved summary npz: {agg_npz}")
    if not raw_df.empty:
        print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
