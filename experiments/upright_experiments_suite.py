import argparse
import os
import sys
import time
from types import SimpleNamespace

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from custom.setup_deepc import setup_DeePC  # noqa: E402
from seed_sweep_sigma005 import (  # noqa: E402
    build_controller,
    check_runtime_dependencies,
    ensure_env_registered,
    evaluate_upright_and_fail,
    featurize_obs,
    load_or_build_offline_dataset,
    save_csv,
)
from wandb_utils import finish_wandb, log_rows_table, log_summary_dict, save_artifacts, setup_wandb  # noqa: E402


def parse_sigma_list(s: str):
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    if not vals:
        raise ValueError("sigma-list is empty")
    return vals


def sample_initial_state(seed, theta0_max, thetadot0_max, x0_max, xdot0_max):
    rng = np.random.default_rng(seed + 7777)
    return {
        "x": float(rng.uniform(-x0_max, x0_max)),
        "theta": float(rng.uniform(-theta0_max, theta0_max)),
        "xdot": float(rng.uniform(-xdot0_max, xdot0_max)),
        "thetadot": float(rng.uniform(-thetadot0_max, thetadot0_max)),
    }


def try_direct_set(env, init_state):
    qpos = np.array([init_state["x"], init_state["theta"]], dtype=float)
    qvel = np.array([init_state["xdot"], init_state["thetadot"]], dtype=float)
    if hasattr(env.unwrapped, "set_state"):
        try:
            env.unwrapped.set_state(qpos, qvel)
            return True, "env.unwrapped.set_state"
        except Exception:
            return False, "env.unwrapped.set_state"
    return False, "none"


def compute_upright_stats(cos_theta, c_up=0.95):
    c = np.asarray(cos_theta, dtype=float)
    if c.size == 0:
        return np.nan, 0
    up = c >= float(c_up)
    upright_fraction = float(np.mean(up.astype(float)))
    best = 0
    cur = 0
    for v in up:
        if bool(v):
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return upright_fraction, int(best)


def run_episode_upright(
    controller,
    seed,
    sigma_y,
    t_sim,
    ref_switch_step,
    init_state,
    sigma_bar,
):
    env = gym.make("InvertedPendulum-v4-swingup")
    obs_raw, _ = env.reset(seed=seed)
    direct_ok, direct_api = try_direct_set(env, init_state)
    if direct_ok:
        try:
            obs_raw = np.asarray(env.unwrapped._get_obs(), dtype=float)
        except Exception:
            obs_raw, _, _, _, _ = env.step(np.array([0.0], dtype=float))

    rng_noise = np.random.default_rng(seed + 1000)
    cos_theta = []
    u_hist = []
    solve_times = []
    bad_steps = []
    fallback_steps = []
    fallback_available = True
    warned_fallback_access = False
    lam_hist = []
    tau_hist = []
    d90_hist = []
    csize_hist = []
    statuses = []

    for step in range(int(t_sim)):
        obs_true = featurize_obs(np.asarray(obs_raw, dtype=float))
        obs_meas = obs_true + rng_noise.normal(0.0, sigma_y, size=obs_true.shape)
        x_ref = 0.0 if step <= ref_switch_step else 0.5
        ref = np.array([x_ref, 0.0, 1.0, 0.0, 0.0], dtype=float)

        t0 = time.perf_counter()
        action = controller.compute_action(obs_meas, ref)
        solve_times.append(time.perf_counter() - t0)
        u_hist.append(float(np.asarray(action, dtype=float).reshape(-1)[0]))

        status = "unknown"
        try:
            status = str(controller._deepc._problem.status)
        except Exception:
            pass
        statuses.append(status)

        fallback = False
        try:
            fallback = bool(controller._deepc._open_loop_index > 1)
        except Exception:
            fallback_available = False
            fallback = False
            if not warned_fallback_access:
                print("[warn] fallback_used flag unavailable; fallback metrics set to NaN.")
                warned_fallback_access = True

        slack_bad = False
        try:
            slack = controller._deepc._y_past_slack.value
            if slack is not None:
                slack_max = float(np.max(np.abs(np.asarray(slack, dtype=float))))
                if np.isfinite(slack_max) and slack_max > float(sigma_bar):
                    slack_bad = True
        except Exception:
            pass

        # practical step infeasibility: non-optimal OR fallback OR slack breach
        non_opt = "optimal" not in status
        bad_steps.append(bool(non_opt or fallback or slack_bad))
        fallback_steps.append(bool(fallback))

        try:
            lam_hist.append(float(controller.get_last_uini_lam_min()))
        except Exception:
            lam_hist.append(np.nan)
        try:
            tau_hist.append(float(controller.get_last_local_tau()))
        except Exception:
            tau_hist.append(np.nan)
        try:
            d90_hist.append(float(controller.get_dist_p90_history()[-1]))
        except Exception:
            d90_hist.append(np.nan)
        try:
            csize_hist.append(float(controller.get_last_local_pool_size()))
        except Exception:
            csize_hist.append(np.nan)

        obs_raw, _, terminated, truncated, _ = env.step(action)
        cos_theta.append(float(np.cos(obs_raw[1])))
        if terminated or truncated:
            break
    env.close()

    cos_theta = np.asarray(cos_theta, dtype=float)
    u_hist = np.asarray(u_hist, dtype=float)
    bad_steps = np.asarray(bad_steps, dtype=bool)
    fallback_steps = np.asarray(fallback_steps, dtype=bool)
    lam_hist = np.asarray(lam_hist, dtype=float)
    tau_hist = np.asarray(tau_hist, dtype=float)
    d90_hist = np.asarray(d90_hist, dtype=float)
    csize_hist = np.asarray(csize_hist, dtype=float)

    success, reached, upright_step, fail_step = evaluate_upright_and_fail(cos_theta, n_up=20, n_fail=20)
    fall_down = bool(reached and (fail_step is not None))
    upright_fraction, longest_streak = compute_upright_stats(cos_theta, c_up=0.95)
    k_hist = np.asarray(controller.get_k_used_history(), dtype=float)
    mean_k = float(np.nanmean(k_hist)) if k_hist.size > 0 else np.nan
    max_k = float(np.nanmax(k_hist)) if k_hist.size > 0 else np.nan
    fallback_episode = float(np.any(fallback_steps)) if fallback_available else np.nan

    return {
        "direct_set_ok": bool(direct_ok),
        "direct_set_api": str(direct_api),
        "fallback_available": bool(fallback_available),
        "success": int(success),
        "success_reached": int(reached),
        "fall_down": int(fall_down),
        "time_to_upright": int(upright_step) if upright_step is not None else int(t_sim),
        "fail_step": int(fail_step) if fail_step is not None else -1,
        "upright_fraction": float(upright_fraction),
        "longest_upright_streak": int(longest_streak),
        "fail_practical": int(np.any(bad_steps)),
        "infeasible_step_rate": float(np.mean(bad_steps.astype(float))) if bad_steps.size > 0 else np.nan,
        "fallback_episode": fallback_episode,
        "avg_solve_time": float(np.mean(solve_times)) if solve_times else np.nan,
        "mean_K_used": mean_k,
        "max_K_used": max_k,
        "mean_lambda_min": float(np.nanmean(lam_hist)) if lam_hist.size > 0 else np.nan,
        "mean_tau": float(np.nanmean(tau_hist)) if tau_hist.size > 0 else np.nan,
        "mean_d90": float(np.nanmean(d90_hist)) if d90_hist.size > 0 else np.nan,
        "mean_C_size": float(np.nanmean(csize_hist)) if csize_hist.size > 0 else np.nan,
        "cos_theta": cos_theta,
        "u": u_hist,
        "lambda_min_hist": lam_hist,
        "K_used_hist": k_hist,
        "statuses": statuses,
    }


def summarize_subset(rows, sigma, method):
    data = [r for r in rows if np.isclose(r["sigma_y"], sigma) and r["method"] == method]
    if not data:
        return {}
    arr = lambda key: np.array([r[key] for r in data], dtype=float)
    reached = arr("success_reached")
    fall = arr("fall_down")
    reached_mask = reached > 0.5
    return {
        "success_rate": float(np.nanmean(arr("success"))),
        "fail_practical_rate": float(np.nanmean(arr("fail_practical"))),
        "upright_fraction_mean": float(np.nanmean(arr("upright_fraction"))),
        "reached_rate": float(np.nanmean(reached)),
        "fall_down_rate": float(np.nanmean(fall[reached_mask])) if np.any(reached_mask) else np.nan,
        "mean_ttu_success": float(np.nanmean(arr("time_to_upright")[arr("success") > 0.5]))
        if np.any(arr("success") > 0.5)
        else np.nan,
        "mean_avg_solve_time": float(np.nanmean(arr("avg_solve_time"))),
        "mean_infeasible_step_rate": float(np.nanmean(arr("infeasible_step_rate"))),
        "fallback_episode_rate": float(np.nanmean(arr("fallback_episode"))),
        "mean_K_used": float(np.nanmean(arr("mean_K_used"))),
        "mean_maxK": float(np.nanmean(arr("max_K_used"))),
    }


def main():
    parser = argparse.ArgumentParser(description="Upright-start Locally-LTI experiment suite.")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--sigma-mode", type=str, choices=["single", "sweep"], default="single")
    parser.add_argument("--sigma-y", type=float, default=0.0)
    parser.add_argument("--sigma-list", type=str, default="0.02,0.03,0.04,0.05,0.06,0.07,0.08")
    parser.add_argument("--theta0", type=float, default=0.1)
    parser.add_argument("--thetadot0", type=float, default=0.2)
    parser.add_argument("--x0", type=float, default=0.05)
    parser.add_argument("--xdot0", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--local-quantile", type=float, default=0.05)
    parser.add_argument("--k-min", type=int, default=40)
    parser.add_argument("--k-max", type=int, default=200)
    parser.add_argument("--k-step", type=int, default=5)
    parser.add_argument("--local-min-cands", type=int, default=100)
    parser.add_argument("--local-max-cands", type=int, default=1000)
    parser.add_argument("--sigma-bar", type=float, default=1e-2)
    parser.add_argument("--t-sim", type=int, default=300)
    parser.add_argument("--ref-switch-step", type=int, default=100)
    parser.add_argument("--k-list", type=str, default="30,40,50,60,70,80")
    parser.add_argument("--outdir", type=str, default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument(
        "--offline-cache",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "cache", "offline_dataset_cache.pkl"),
    )
    parser.add_argument("--rebuild-offline", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cdc2026")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="upright_suite")
    parser.add_argument("--wandb-run-name", type=str, default="")
    args = parser.parse_args()

    check_runtime_dependencies()
    ensure_env_registered()
    os.makedirs(args.outdir, exist_ok=True)
    traj_dir = os.path.join(args.outdir, "upright_suite_traj")
    os.makedirs(traj_dir, exist_ok=True)
    fixed_k_list = [int(v.strip()) for v in args.k_list.split(",") if v.strip()]
    sigma_vals = [float(args.sigma_y)] if args.sigma_mode == "single" else parse_sigma_list(args.sigma_list)

    trajectory_data = load_or_build_offline_dataset(
        cache_path=args.offline_cache, rebuild_offline=bool(args.rebuild_offline)
    )
    controller_args = setup_DeePC(trajectory_data)
    wb_run = setup_wandb(args, "upright_experiments_suite")
    seeds = [int(args.seed0 + i) for i in range(int(args.seeds))]
    init_map = {
        sd: sample_initial_state(sd, args.theta0, args.thetadot0, args.x0, args.xdot0)
        for sd in seeds
    }

    method_cfgs = [{"method": f"fixedK{k}", "K": k, "adaptive_k": False, "local_gate": False} for k in fixed_k_list]
    method_cfgs.append({"method": "unified_a001", "K": int(args.k_min), "adaptive_k": True, "local_gate": True})

    rows = []
    traj_cache = {}
    direct_set_ok_count = 0
    total_runs = 0
    direct_api_used = None
    fallback_available_global = True

    for sigma in sigma_vals:
        for seed in seeds:
            init_state = init_map[seed]
            for m in method_cfgs:
                ctl_args = SimpleNamespace(
                    k=int(m["K"]),
                    adaptive_k=bool(m["adaptive_k"]),
                    k_min=int(args.k_min),
                    k_max=int(args.k_max),
                    k_step=int(args.k_step),
                    alpha=float(args.alpha),
                    local_gate=bool(m["local_gate"]),
                    local_quantile=float(args.local_quantile),
                    local_min_cands=int(args.local_min_cands),
                    local_max_cands=int(args.local_max_cands),
                )
                controller = build_controller(controller_args, ctl_args)
                ep = run_episode_upright(
                    controller=controller,
                    seed=int(seed),
                    sigma_y=float(sigma),
                    t_sim=int(args.t_sim),
                    ref_switch_step=int(args.ref_switch_step),
                    init_state=init_state,
                    sigma_bar=float(args.sigma_bar),
                )
                total_runs += 1
                if ep["direct_set_ok"]:
                    direct_set_ok_count += 1
                    if direct_api_used is None:
                        direct_api_used = ep["direct_set_api"]
                if not ep["fallback_available"]:
                    fallback_available_global = False
                row = {
                    "sigma_y": float(sigma),
                    "seed": int(seed),
                    "method": str(m["method"]),
                    "K": int(m["K"]),
                    "success": int(ep["success"]),
                    "success_reached": int(ep["success_reached"]),
                    "fall_down": int(ep["fall_down"]),
                    "time_to_upright": int(ep["time_to_upright"]),
                    "fail_step": int(ep["fail_step"]),
                    "upright_fraction": float(ep["upright_fraction"]),
                    "longest_upright_streak": int(ep["longest_upright_streak"]),
                    "fail_practical": int(ep["fail_practical"]),
                    "infeasible_step_rate": float(ep["infeasible_step_rate"]),
                    "fallback_episode": float(ep["fallback_episode"]),
                    "avg_solve_time": float(ep["avg_solve_time"]),
                    "mean_K_used": float(m["K"] if not m["adaptive_k"] else ep["mean_K_used"]),
                    "max_K_used": float(m["K"] if not m["adaptive_k"] else ep["max_K_used"]),
                    "mean_lambda_min": float(ep["mean_lambda_min"]),
                    "mean_tau": float(np.nan if not m["adaptive_k"] else ep["mean_tau"]),
                    "mean_d90": float(np.nan if not m["adaptive_k"] else ep["mean_d90"]),
                    "mean_C_size": float(np.nan if not m["adaptive_k"] else ep["mean_C_size"]),
                }
                rows.append(row)
                traj_npz = os.path.join(
                    traj_dir,
                    f"traj_sigma{float(sigma):.3f}_seed{int(seed):03d}_{str(m['method'])}.npz",
                )
                np.savez(
                    traj_npz,
                    sigma_y=np.array(float(sigma), dtype=float),
                    seed=np.array(int(seed), dtype=int),
                    method=np.array(str(m["method"]), dtype=object),
                    K=np.array(int(m["K"]), dtype=int),
                    success=np.array(int(ep["success"]), dtype=int),
                    success_reached=np.array(int(ep["success_reached"]), dtype=int),
                    fall_down=np.array(int(ep["fall_down"]), dtype=int),
                    time_to_upright=np.array(int(ep["time_to_upright"]), dtype=int),
                    fail_step=np.array(int(ep["fail_step"]), dtype=int),
                    fail_practical=np.array(int(ep["fail_practical"]), dtype=int),
                    infeasible_step_rate=np.array(float(ep["infeasible_step_rate"]), dtype=float),
                    fallback_episode=np.array(float(ep["fallback_episode"]), dtype=float),
                    avg_solve_time=np.array(float(ep["avg_solve_time"]), dtype=float),
                    mean_K_used=np.array(float(row["mean_K_used"]), dtype=float),
                    max_K_used=np.array(float(row["max_K_used"]), dtype=float),
                    mean_lambda_min=np.array(float(row["mean_lambda_min"]), dtype=float),
                    mean_tau=np.array(float(row["mean_tau"]), dtype=float),
                    mean_d90=np.array(float(row["mean_d90"]), dtype=float),
                    mean_C_size=np.array(float(row["mean_C_size"]), dtype=float),
                    cos_theta=np.asarray(ep["cos_theta"], dtype=float),
                    u=np.asarray(ep["u"], dtype=float),
                    lambda_min_hist=np.asarray(ep["lambda_min_hist"], dtype=float),
                    K_used_hist=np.asarray(ep["K_used_hist"], dtype=float),
                    solver_status=np.asarray(ep["statuses"], dtype=object),
                )
                if np.isclose(sigma, 0.0) and (m["method"] in ("fixedK40", "unified_a001")):
                    traj_cache[(float(sigma), m["method"], int(seed))] = ep
                print(
                    f"[sigma={sigma:.3f}][seed={seed:03d}][{m['method']}] "
                    f"succ={row['success']} failP={row['fail_practical']} UF={row['upright_fraction']:.3f}"
                )

    out_csv = os.path.join(args.outdir, "upright_suite.csv")
    cols = [
        "sigma_y",
        "seed",
        "method",
        "K",
        "success",
        "success_reached",
        "fall_down",
        "time_to_upright",
        "fail_step",
        "upright_fraction",
        "longest_upright_streak",
        "fail_practical",
        "infeasible_step_rate",
        "fallback_episode",
        "avg_solve_time",
        "mean_K_used",
        "max_K_used",
        "mean_lambda_min",
        "mean_tau",
        "mean_d90",
        "mean_C_size",
    ]
    save_csv(rows, out_csv, fieldnames=cols)

    print("\n[summary]")
    wb_summary_rows = []
    for sigma in sigma_vals:
        print(f"sigma={sigma:.3f}")
        for m in [cfg["method"] for cfg in method_cfgs]:
            s = summarize_subset(rows, sigma, m)
            wb_summary_rows.append(
                {
                    "sigma_y": float(sigma),
                    "method": str(m),
                    "success_rate": float(s["success_rate"]),
                    "fail_practical_rate": float(s["fail_practical_rate"]),
                    "upright_fraction_mean": float(s["upright_fraction_mean"]),
                    "reached_rate": float(s["reached_rate"]),
                    "fall_down_rate": float(s["fall_down_rate"]),
                    "mean_ttu_success": float(s["mean_ttu_success"]),
                    "mean_avg_solve_time": float(s["mean_avg_solve_time"]),
                    "mean_infeasible_step_rate": float(s["mean_infeasible_step_rate"]),
                    "fallback_episode_rate": float(s["fallback_episode_rate"]),
                    "mean_K_used": float(s["mean_K_used"]),
                    "mean_maxK": float(s["mean_maxK"]),
                }
            )
            print(
                f"  {m:12s} success={s['success_rate']:.3f} "
                f"failP={s['fail_practical_rate']:.3f} UF={s['upright_fraction_mean']:.3f} "
                f"reached={s['reached_rate']:.3f} fall={s['fall_down_rate']:.3f} "
                f"time={s['mean_avg_solve_time']:.4e}"
            )
        if np.isclose(sigma, 0.0):
            print("  [sigma=0 highlight] fail_practical_rate vs K:")
            for k in fixed_k_list:
                fk = summarize_subset(rows, sigma, f"fixedK{k}")
                print(f"    K={k:3d}: {fk['fail_practical_rate']:.3f}")

    out_paths = []

    # sigma=0 plots
    if any(np.isclose(np.array(sigma_vals), 0.0)):
        sigma0 = 0.0
        fixed_x = np.array(fixed_k_list, dtype=float)
        unified_x = float(np.max(fixed_x) + 5.0)
        y_fail = [summarize_subset(rows, sigma0, f"fixedK{k}")["fail_practical_rate"] for k in fixed_k_list]
        y_uf = [summarize_subset(rows, sigma0, f"fixedK{k}")["upright_fraction_mean"] for k in fixed_k_list]
        u_fail = summarize_subset(rows, sigma0, "unified_a001")["fail_practical_rate"]
        u_uf = summarize_subset(rows, sigma0, "unified_a001")["upright_fraction_mean"]

        fig_a, ax_a = plt.subplots(figsize=(8, 4))
        ax_a.plot(fixed_x, y_fail, marker="o", linewidth=2, label="fixed-K", color="tab:blue")
        ax_a.scatter([unified_x], [u_fail], s=80, label="unified", color="tab:green")
        ax_a.set_xlabel("K")
        ax_a.set_ylabel("practical_infeasible_rate")
        ax_a.set_ylim(0.0, 1.0)
        ax_a.grid(True, alpha=0.3)
        ax_a.legend()
        out_a = os.path.join(args.outdir, "upright_sigma0_infeasible_vsK.png")
        fig_a.tight_layout()
        fig_a.savefig(out_a, dpi=200, bbox_inches="tight")
        plt.close(fig_a)
        out_paths.append(out_a)

        fig_b, ax_b = plt.subplots(figsize=(8, 4))
        ax_b.plot(fixed_x, y_uf, marker="o", linewidth=2, label="fixed-K", color="tab:blue")
        ax_b.scatter([unified_x], [u_uf], s=80, label="unified", color="tab:green")
        ax_b.set_xlabel("K")
        ax_b.set_ylabel("upright_fraction_mean")
        ax_b.set_ylim(0.0, 1.0)
        ax_b.grid(True, alpha=0.3)
        ax_b.legend()
        out_b = os.path.join(args.outdir, "upright_sigma0_upright_fraction_vsK.png")
        fig_b.tight_layout()
        fig_b.savefig(out_b, dpi=200, bbox_inches="tight")
        plt.close(fig_b)
        out_paths.append(out_b)

        # representative trajectory plot
        rep_fixed_seed = None
        for sd in seeds:
            s = summarize_subset(
                [r for r in rows if r["seed"] == sd and np.isclose(r["sigma_y"], 0.0) and r["method"] == "fixedK40"],
                0.0,
                "fixedK40",
            )
            if s and (s["fail_practical_rate"] > 0.5 or s["success_rate"] < 0.5):
                rep_fixed_seed = sd
                break
        rep_unified_seed = None
        for sd in seeds:
            s = summarize_subset(
                [r for r in rows if r["seed"] == sd and np.isclose(r["sigma_y"], 0.0) and r["method"] == "unified_a001"],
                0.0,
                "unified_a001",
            )
            if s and (s["success_rate"] > 0.5 and s["fail_practical_rate"] < 0.5):
                rep_unified_seed = sd
                break

        if rep_fixed_seed is not None and rep_unified_seed is not None:
            d_fix = traj_cache.get((0.0, "fixedK40", rep_fixed_seed), None)
            d_uni = traj_cache.get((0.0, "unified_a001", rep_unified_seed), None)
            if d_fix is not None and d_uni is not None:
                fig_r, ax_r = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
                t_fix = np.arange(d_fix["cos_theta"].shape[0])
                t_uni = np.arange(d_uni["cos_theta"].shape[0])
                ax_r[0].plot(t_fix, d_fix["cos_theta"], label=f"fixedK40 seed={rep_fixed_seed}", color="tab:blue")
                ax_r[0].plot(
                    t_uni, d_uni["cos_theta"], label=f"unified seed={rep_unified_seed}", color="tab:green", alpha=0.9
                )
                ax_r[0].axhline(0.95, color="k", linestyle="--", linewidth=1)
                ax_r[0].set_ylabel("cos(theta)")
                ax_r[0].grid(True, alpha=0.3)
                ax_r[0].legend()
                ax_r[1].plot(np.arange(d_fix["u"].shape[0]), d_fix["u"], color="tab:blue")
                ax_r[1].plot(np.arange(d_uni["u"].shape[0]), d_uni["u"], color="tab:green", alpha=0.9)
                ax_r[1].set_ylabel("u")
                ax_r[1].grid(True, alpha=0.3)
                ax_r[2].plot(np.arange(d_fix["lambda_min_hist"].shape[0]), d_fix["lambda_min_hist"], color="tab:blue")
                ax_r[2].plot(np.arange(d_uni["lambda_min_hist"].shape[0]), d_uni["lambda_min_hist"], color="tab:green")
                ax_r[2].set_ylabel("lambda_min")
                ax_r[2].grid(True, alpha=0.3)
                ax_r[3].plot(np.arange(d_fix["K_used_hist"].shape[0]), d_fix["K_used_hist"], color="tab:blue")
                ax_r[3].plot(np.arange(d_uni["K_used_hist"].shape[0]), d_uni["K_used_hist"], color="tab:green")
                ax_r[3].set_ylabel("K_used")
                ax_r[3].set_xlabel("step")
                ax_r[3].grid(True, alpha=0.3)
                fig_r.tight_layout()
                out_r = os.path.join(args.outdir, "upright_representative_traj.png")
                fig_r.savefig(out_r, dpi=200, bbox_inches="tight")
                plt.close(fig_r)
                out_paths.append(out_r)

    # sigma sweep plots (fixedK40 vs unified)
    if args.sigma_mode == "sweep":
        sig = np.array(sigma_vals, dtype=float)
        f40_success = [summarize_subset(rows, s, "fixedK40")["success_rate"] for s in sig]
        uni_success = [summarize_subset(rows, s, "unified_a001")["success_rate"] for s in sig]
        f40_fall = [summarize_subset(rows, s, "fixedK40")["fall_down_rate"] for s in sig]
        uni_fall = [summarize_subset(rows, s, "unified_a001")["fall_down_rate"] for s in sig]
        f40_uf = [summarize_subset(rows, s, "fixedK40")["upright_fraction_mean"] for s in sig]
        uni_uf = [summarize_subset(rows, s, "unified_a001")["upright_fraction_mean"] for s in sig]

        fig_c, ax_c = plt.subplots(figsize=(7, 4))
        ax_c.plot(sig, f40_success, marker="o", label="fixedK40", color="tab:blue")
        ax_c.plot(sig, uni_success, marker="o", label="unified", color="tab:green")
        ax_c.set_xlabel("sigma_y")
        ax_c.set_ylabel("success_rate")
        ax_c.set_ylim(0.0, 1.0)
        ax_c.grid(True, alpha=0.3)
        ax_c.legend()
        out_c = os.path.join(args.outdir, "upright_sigma_sweep_success.png")
        fig_c.tight_layout()
        fig_c.savefig(out_c, dpi=200, bbox_inches="tight")
        plt.close(fig_c)
        out_paths.append(out_c)

        fig_d, ax_d = plt.subplots(figsize=(7, 4))
        ax_d.plot(sig, f40_fall, marker="o", label="fixedK40", color="tab:blue")
        ax_d.plot(sig, uni_fall, marker="o", label="unified", color="tab:green")
        ax_d.set_xlabel("sigma_y")
        ax_d.set_ylabel("fall_down_rate")
        ax_d.set_ylim(0.0, 1.0)
        ax_d.grid(True, alpha=0.3)
        ax_d.legend()
        out_d = os.path.join(args.outdir, "upright_sigma_sweep_falldown.png")
        fig_d.tight_layout()
        fig_d.savefig(out_d, dpi=200, bbox_inches="tight")
        plt.close(fig_d)
        out_paths.append(out_d)

        fig_e, ax_e = plt.subplots(figsize=(7, 4))
        ax_e.plot(sig, f40_uf, marker="o", label="fixedK40", color="tab:blue")
        ax_e.plot(sig, uni_uf, marker="o", label="unified", color="tab:green")
        ax_e.set_xlabel("sigma_y")
        ax_e.set_ylabel("upright_fraction")
        ax_e.set_ylim(0.0, 1.0)
        ax_e.grid(True, alpha=0.3)
        ax_e.legend()
        out_e = os.path.join(args.outdir, "upright_sigma_sweep_upright_fraction.png")
        fig_e.tight_layout()
        fig_e.savefig(out_e, dpi=200, bbox_inches="tight")
        plt.close(fig_e)
        out_paths.append(out_e)

    print("\n[init-mode]")
    print(f"direct set success: {direct_set_ok_count}/{total_runs}")
    print(f"direct set api: {direct_api_used if direct_api_used is not None else 'not_available'}")
    if not fallback_available_global:
        print("[warn] fallback_used unavailable in this environment; fallback_episode may be NaN.")

    print("\n[outputs]")
    print(f"CSV: {out_csv}")
    print(f"Trajectory dir: {traj_dir}")
    for p in out_paths:
        print(f"Plot: {p}")

    log_rows_table(wb_run, "episodes", rows)
    log_rows_table(wb_run, "summary", wb_summary_rows)
    log_summary_dict(
        wb_run,
        {
            "direct_set_ok_count": int(direct_set_ok_count),
            "total_runs": int(total_runs),
            "direct_set_ok_rate": float(direct_set_ok_count / total_runs) if total_runs > 0 else np.nan,
            "fallback_available_global": bool(fallback_available_global),
        },
    )
    traj_artifacts = []
    try:
        traj_artifacts = [
            os.path.join(traj_dir, f)
            for f in os.listdir(traj_dir)
            if f.endswith(".npz")
        ]
    except Exception:
        traj_artifacts = []
    save_artifacts(wb_run, [out_csv] + out_paths + traj_artifacts)
    finish_wandb(wb_run)


if __name__ == "__main__":
    main()
