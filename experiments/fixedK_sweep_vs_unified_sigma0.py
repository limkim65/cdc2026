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


def sample_upright_initial_state(seed, theta0=0.1, thetadot0=0.2):
    rng = np.random.default_rng(seed + 909)
    return {
        "x": 0.0,
        "theta": float(rng.uniform(-theta0, theta0)),
        "xdot": 0.0,
        "thetadot": float(rng.uniform(-thetadot0, thetadot0)),
    }


def inject_initial_state(env, init_state):
    qpos = np.array([init_state["x"], init_state["theta"]], dtype=float)
    qvel = np.array([init_state["xdot"], init_state["thetadot"]], dtype=float)
    try:
        env.unwrapped.set_state(qpos, qvel)
        return True
    except Exception:
        return False


def run_episode_sigma0(
    controller,
    seed,
    t_sim,
    ref_switch_step,
    init_state,
    sigma_bar=1e-2,
):
    env = gym.make("InvertedPendulum-v4-swingup")
    obs_raw, _ = env.reset(seed=seed)
    direct_ok = inject_initial_state(env, init_state)
    if direct_ok:
        try:
            obs_raw = np.asarray(env.unwrapped._get_obs(), dtype=float)
        except Exception:
            obs_raw, _, _, _, _ = env.step(np.array([0.0], dtype=float))

    cos_theta = []
    solve_times = []
    step_bad = []
    step_fallback = []
    statuses = []
    slack_viol = []
    for step in range(int(t_sim)):
        obs_true = featurize_obs(np.asarray(obs_raw, dtype=float))
        x_ref = 0.0 if step <= ref_switch_step else 0.5
        ref = np.array([x_ref, 0.0, 1.0, 0.0, 0.0], dtype=float)

        t0 = time.perf_counter()
        action = controller.compute_action(obs_true, ref)
        solve_times.append(time.perf_counter() - t0)

        status = "unknown"
        fallback = False
        slack_bad = False
        try:
            status = str(controller._deepc._problem.status)
        except Exception:
            pass
        # open_loop_index > 1 means open-loop fallback trajectory is being used.
        try:
            fallback = bool(controller._deepc._open_loop_index > 1)
        except Exception:
            fallback = False
        if "optimal" not in status:
            fallback = True

        try:
            slack_val = controller._deepc._y_past_slack.value
            if slack_val is not None:
                slack_max = float(np.max(np.abs(np.asarray(slack_val, dtype=float))))
                if np.isfinite(slack_max) and slack_max > float(sigma_bar):
                    slack_bad = True
        except Exception:
            pass

        bad = bool(("optimal" not in status) or fallback or slack_bad)
        step_bad.append(bad)
        step_fallback.append(fallback)
        statuses.append(status)
        slack_viol.append(slack_bad)

        obs_raw, _, terminated, truncated, _ = env.step(action)
        cos_theta.append(float(np.cos(obs_raw[1])))
        if terminated or truncated:
            break
    env.close()

    cos_theta = np.asarray(cos_theta, dtype=float)
    k_used = np.asarray(controller.get_k_used_history(), dtype=float)
    if k_used.size == 0:
        k_used = np.full(cos_theta.shape[0], np.nan, dtype=float)
    final_success, success_reached, upright_step, fail_step = evaluate_upright_and_fail(
        cos_theta=cos_theta,
        n_up=20,
        n_fail=20,
    )
    fall_down = bool(success_reached and (fail_step is not None))
    step_bad_arr = np.asarray(step_bad, dtype=bool)
    step_fb_arr = np.asarray(step_fallback, dtype=bool)
    slack_arr = np.asarray(slack_viol, dtype=bool)

    return {
        "direct_set_ok": bool(direct_ok),
        "fail_practical": int(np.any(step_bad_arr)),
        "infeasible_step_rate": float(np.mean(step_bad_arr.astype(float))) if step_bad_arr.size > 0 else np.nan,
        "fallback_episode": int(np.any(step_fb_arr)),
        "mean_solve_time": float(np.mean(solve_times)) if solve_times else np.nan,
        "mean_K_used": float(np.nanmean(k_used)) if k_used.size > 0 else np.nan,
        "max_K_used": float(np.nanmax(k_used)) if k_used.size > 0 else np.nan,
        "success": int(final_success),
        "reached": int(success_reached),
        "fall_down": int(fall_down),
        "slack_violation_episode": int(np.any(slack_arr)),
        "time_to_upright": int(upright_step) if upright_step is not None else int(t_sim),
        "solver_statuses": statuses,
    }


def summarize_method(rows, method):
    data = [r for r in rows if r["method"] == method]
    fail = np.array([r["fail_practical"] for r in data], dtype=float)
    inf_step = np.array([r["infeasible_step_rate"] for r in data], dtype=float)
    fallback = np.array([r["fallback_episode"] for r in data], dtype=float)
    solve = np.array([r["mean_solve_time"] for r in data], dtype=float)
    mean_k = np.array([r["mean_K_used"] for r in data], dtype=float)
    max_k = np.array([r["max_K_used"] for r in data], dtype=float)
    succ = np.array([r["success"] for r in data], dtype=float)
    return {
        "practical_infeasible_rate": float(np.nanmean(fail)) if fail.size > 0 else np.nan,
        "mean_infeasible_step_rate": float(np.nanmean(inf_step)) if inf_step.size > 0 else np.nan,
        "fallback_episode_rate": float(np.nanmean(fallback)) if fallback.size > 0 else np.nan,
        "mean_solve_time": float(np.nanmean(solve)) if solve.size > 0 else np.nan,
        "mean_K_used": float(np.nanmean(mean_k)) if mean_k.size > 0 else np.nan,
        "mean_maxK": float(np.nanmean(max_k)) if max_k.size > 0 else np.nan,
        "success_rate": float(np.nanmean(succ)) if succ.size > 0 else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(
        description="sigma=0 fixed-K sweep vs unified (practical infeasibility/fallback/time)."
    )
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--sigma-y", type=float, default=0.0)
    parser.add_argument("--fixed-k-list", type=str, default="30,40,50,60,70,80")
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
    parser.add_argument("--wandb-group", type=str, default="fixedK_vs_unified_sigma0")
    parser.add_argument("--wandb-run-name", type=str, default="")
    args = parser.parse_args()

    check_runtime_dependencies()
    ensure_env_registered()
    os.makedirs(args.outdir, exist_ok=True)
    fixed_k_list = [int(x.strip()) for x in args.fixed_k_list.split(",") if x.strip()]

    trajectory_data = load_or_build_offline_dataset(
        cache_path=args.offline_cache,
        rebuild_offline=bool(args.rebuild_offline),
    )
    controller_args = setup_DeePC(trajectory_data)
    wb_run = setup_wandb(args, "fixedK_sweep_vs_unified_sigma0")

    seeds = [int(args.seed0 + i) for i in range(int(args.seeds))]
    rows = []
    direct_set_success_cnt = 0
    method_cfgs = [{"method": f"fixedK{k}", "K": k, "adaptive_k": False, "local_gate": False} for k in fixed_k_list]
    method_cfgs.append({"method": "unified_a001", "K": int(args.k_min), "adaptive_k": True, "local_gate": True})

    for seed in seeds:
        init_state = sample_upright_initial_state(seed)
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
            ep = run_episode_sigma0(
                controller=controller,
                seed=int(seed),
                t_sim=int(args.t_sim),
                ref_switch_step=int(args.ref_switch_step),
                init_state=init_state,
                sigma_bar=float(args.sigma_bar),
            )
            if ep["direct_set_ok"]:
                direct_set_success_cnt += 1
            rows.append(
                {
                    "seed": int(seed),
                    "method": str(m["method"]),
                    "K": int(m["K"]),
                    "fail_practical": int(ep["fail_practical"]),
                    "infeasible_step_rate": float(ep["infeasible_step_rate"]),
                    "fallback_episode": int(ep["fallback_episode"]),
                    "mean_solve_time": float(ep["mean_solve_time"]),
                    "mean_K_used": float(m["K"] if not m["adaptive_k"] else ep["mean_K_used"]),
                    "max_K_used": float(m["K"] if not m["adaptive_k"] else ep["max_K_used"]),
                    "success": int(ep["success"]),
                    "reached": int(ep["reached"]),
                    "fall_down": int(ep["fall_down"]),
                }
            )
            print(
                f"[seed={seed:03d}][{m['method']}] practical_fail={int(ep['fail_practical'])} "
                f"fallback={int(ep['fallback_episode'])} inf_step={ep['infeasible_step_rate']:.3f}"
            )

    out_csv = os.path.join(args.outdir, "fixedK_sweep_vs_unified_sigma0.csv")
    cols = [
        "seed",
        "method",
        "K",
        "fail_practical",
        "infeasible_step_rate",
        "fallback_episode",
        "mean_solve_time",
        "mean_K_used",
        "max_K_used",
        "success",
        "reached",
        "fall_down",
    ]
    save_csv(rows, out_csv, fieldnames=cols)

    # Summaries
    print("\n[summary]")
    summary = {}
    summary_rows = []
    for m in [cfg["method"] for cfg in method_cfgs]:
        s = summarize_method(rows, m)
        summary[m] = s
        summary_rows.append({"method": str(m), **{k: float(v) for k, v in s.items()}})
        print(
            f"{m:12s} practical_infeasible_rate={s['practical_infeasible_rate']:.3f} "
            f"mean_infeasible_step_rate={s['mean_infeasible_step_rate']:.3f} "
            f"fallback_episode_rate={s['fallback_episode_rate']:.3f} "
            f"mean_solve_time={s['mean_solve_time']:.4e} "
            f"meanK={s['mean_K_used']:.2f} meanMaxK={s['mean_maxK']:.2f} "
            f"success_rate={s['success_rate']:.3f}"
        )

    fixed_methods = [f"fixedK{k}" for k in fixed_k_list]
    fixed_x = np.array(fixed_k_list, dtype=float)
    unified_x = float(np.max(fixed_x) + 5.0)
    unified = summary["unified_a001"]

    # Plot 1: practical infeasible rate vs K + unified marker
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    y1 = [summary[m]["practical_infeasible_rate"] for m in fixed_methods]
    ax1.plot(fixed_x, y1, marker="o", color="tab:blue", linewidth=2, label="fixed-K")
    ax1.scatter([unified_x], [unified["practical_infeasible_rate"]], color="tab:green", s=80, label="unified")
    ax1.set_xlabel("K")
    ax1.set_ylabel("practical_infeasible_rate")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    out1 = os.path.join(args.outdir, "sigma0_infeasible_vs_K.png")
    fig1.tight_layout()
    fig1.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: solve time vs K + unified marker
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    y2 = [summary[m]["mean_solve_time"] for m in fixed_methods]
    ax2.plot(fixed_x, y2, marker="o", color="tab:blue", linewidth=2, label="fixed-K")
    ax2.scatter([unified_x], [unified["mean_solve_time"]], color="tab:green", s=80, label="unified")
    ax2.set_xlabel("K")
    ax2.set_ylabel("mean solve time [s]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    out2 = os.path.join(args.outdir, "sigma0_time_vs_K.png")
    fig2.tight_layout()
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # Plot 3: success rate vs K + unified marker
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    y3 = [summary[m]["success_rate"] for m in fixed_methods]
    ax3.plot(fixed_x, y3, marker="o", color="tab:blue", linewidth=2, label="fixed-K")
    ax3.scatter([unified_x], [unified["success_rate"]], color="tab:green", s=80, label="unified")
    ax3.set_xlabel("K")
    ax3.set_ylabel("success_rate")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    out3 = os.path.join(args.outdir, "sigma0_success_vs_K.png")
    fig3.tight_layout()
    fig3.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close(fig3)

    # Plot 4: infeasible step rate vs K + unified marker
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    y4 = [summary[m]["mean_infeasible_step_rate"] for m in fixed_methods]
    ax4.plot(fixed_x, y4, marker="o", color="tab:blue", linewidth=2, label="fixed-K")
    ax4.scatter([unified_x], [unified["mean_infeasible_step_rate"]], color="tab:green", s=80, label="unified")
    ax4.set_xlabel("K")
    ax4.set_ylabel("infeasible_step_rate")
    ax4.set_ylim(0.0, 1.0)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    out4 = os.path.join(args.outdir, "sigma0_infeasible_step_rate_vs_K.png")
    fig4.tight_layout()
    fig4.savefig(out4, dpi=200, bbox_inches="tight")
    plt.close(fig4)

    print("\n[init-mode]")
    print(f"direct set_state success count: {direct_set_success_cnt} / {len(rows)}")
    print("\n[outputs]")
    print(f"CSV: {out_csv}")
    print(f"Plot(infeasible vs K): {out1}")
    print(f"Plot(time vs K): {out2}")
    print(f"Plot(success vs K): {out3}")
    print(f"Plot(infeasible_step_rate vs K): {out4}")

    log_rows_table(wb_run, "episodes", rows)
    log_rows_table(wb_run, "summary", summary_rows)
    log_summary_dict(
        wb_run,
        {
            "direct_set_success_count": int(direct_set_success_cnt),
            "num_rows": int(len(rows)),
            "num_methods": int(len(method_cfgs)),
        },
    )
    save_artifacts(wb_run, [out_csv, out1, out2, out3, out4])
    finish_wandb(wb_run)


if __name__ == "__main__":
    main()
