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


def _try_direct_set(env, init_state):
    qpos = np.array([init_state["x"], init_state["theta"]], dtype=float)
    qvel = np.array([init_state["xdot"], init_state["thetadot"]], dtype=float)
    targets = []
    try:
        targets.append(("env.set_state", env))
    except Exception:
        pass
    try:
        targets.append(("unwrapped.set_state", env.unwrapped))
    except Exception:
        pass
    for api_name, obj in targets:
        if hasattr(obj, "set_state"):
            try:
                obj.set_state(qpos, qvel)
                return True, api_name
            except Exception:
                continue
    return False, ""


def detect_direct_set_api():
    env = gym.make("InvertedPendulum-v4-swingup")
    obs, _ = env.reset(seed=0)
    sample = {"x": 0.0, "theta": float(obs[1]), "xdot": 0.0, "thetadot": float(obs[3])}
    ok, api_name = _try_direct_set(env, sample)
    env.close()
    return ok, api_name


def sample_init_direct(seed, theta_max, thetadot_max):
    rng = np.random.default_rng(seed + 2027)
    return {
        "x": 0.0,
        "theta": float(rng.uniform(-theta_max, theta_max)),
        "xdot": 0.0,
        "thetadot": float(rng.uniform(-thetadot_max, thetadot_max)),
    }


def sample_init_rejection(seed, theta_max, thetadot_max, attempts):
    env = gym.make("InvertedPendulum-v4-swingup")
    chosen = None
    for i in range(int(attempts)):
        obs, _ = env.reset(seed=seed + i)
        theta = float(obs[1])
        thetadot = float(obs[3])
        chosen = {"x": float(obs[0]), "theta": theta, "xdot": float(obs[2]), "thetadot": thetadot}
        if (-theta_max <= theta <= theta_max) and (-thetadot_max <= thetadot <= thetadot_max):
            env.close()
            return chosen, False
    env.close()
    return chosen, True


def run_episode_with_fixed_init(controller, sigma_y, seed, t_sim, ref_switch_step, init_state, use_direct_set):
    env = gym.make("InvertedPendulum-v4-swingup")
    obs_raw, _ = env.reset(seed=seed)
    if use_direct_set:
        ok, _ = _try_direct_set(env, init_state)
        if ok:
            try:
                obs_raw = np.asarray(env.unwrapped._get_obs(), dtype=float)
            except Exception:
                obs_raw, _, _, _, _ = env.step(np.array([0.0], dtype=float))

    rng_noise = np.random.default_rng(seed + 1000)
    solve_times = []
    cos_theta = []
    for step in range(int(t_sim)):
        obs_true = featurize_obs(np.asarray(obs_raw, dtype=float))
        obs_meas = obs_true + rng_noise.normal(0.0, sigma_y, size=obs_true.shape)
        x_ref = 0.0 if step <= ref_switch_step else 0.5
        ref = np.array([x_ref, 0.0, 1.0, 0.0, 0.0], dtype=float)

        t0 = time.perf_counter()
        action = controller.compute_action(obs_meas, ref)
        solve_times.append(time.perf_counter() - t0)

        obs_raw, _, terminated, truncated, _ = env.step(action)
        cos_theta.append(float(np.cos(obs_raw[1])))
        if terminated or truncated:
            break
    env.close()

    cos_theta = np.asarray(cos_theta, dtype=float)
    final_success, success_reached, upright_step, fail_step = evaluate_upright_and_fail(
        cos_theta=cos_theta,
        n_up=20,
        n_fail=20,
    )
    fall_down = bool(success_reached and (fail_step is not None))

    k_hist = np.asarray(controller.get_k_used_history(), dtype=float)
    return {
        "success": int(final_success),
        "success_reached": int(success_reached),
        "fall_down": int(fall_down),
        "time_to_upright": int(upright_step) if upright_step is not None else int(t_sim),
        "fail_step": int(fail_step) if fail_step is not None else -1,
        "avg_solve_time": float(np.mean(solve_times)) if solve_times else np.nan,
        "mean_K_used": float(np.nanmean(k_hist)) if k_hist.size > 0 else np.nan,
        "max_K_used": float(np.nanmax(k_hist)) if k_hist.size > 0 else np.nan,
    }


def summarize(rows, scenario, method):
    subset = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
    success = np.array([r["success"] for r in subset], dtype=float)
    reached = np.array([r["success_reached"] for r in subset], dtype=float)
    fall = np.array([r["fall_down"] for r in subset], dtype=float)
    ttu = np.array([r["time_to_upright"] for r in subset], dtype=float)
    solve = np.array([r["avg_solve_time"] for r in subset], dtype=float)
    reached_mask = reached > 0.5
    succ_mask = success > 0.5
    return {
        "success_rate": float(np.nanmean(success)) if success.size > 0 else np.nan,
        "reached_rate": float(np.nanmean(reached)) if reached.size > 0 else np.nan,
        "fall_down_rate": float(np.nanmean(fall[reached_mask])) if np.any(reached_mask) else np.nan,
        "mean_ttu_success": float(np.nanmean(ttu[succ_mask])) if np.any(succ_mask) else np.nan,
        "mean_avg_solve_time": float(np.nanmean(solve)) if solve.size > 0 else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="Offline-only unseen-init sweep: fixedK40 vs unified_a001.")
    parser.add_argument("--sigma-y", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--theta0", type=float, default=0.2)
    parser.add_argument("--thetadot0", type=float, default=0.5)
    parser.add_argument("--unseen-scale", type=float, default=1.5)
    parser.add_argument("--attempts", type=int, default=200)
    parser.add_argument("--t-sim", type=int, default=300)
    parser.add_argument("--ref-switch-step", type=int, default=100)
    parser.add_argument("--outdir", type=str, default=os.path.join(REPO_ROOT, "results"))

    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--local-quantile", type=float, default=0.05)
    parser.add_argument("--k-min", type=int, default=40)
    parser.add_argument("--k-max", type=int, default=200)
    parser.add_argument("--k-step", type=int, default=5)
    parser.add_argument("--local-min-cands", type=int, default=100)
    parser.add_argument("--local-max-cands", type=int, default=1000)
    parser.add_argument(
        "--offline-cache",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "cache", "offline_dataset_cache.pkl"),
    )
    parser.add_argument("--rebuild-offline", action="store_true")
    args = parser.parse_args()

    check_runtime_dependencies()
    ensure_env_registered()
    os.makedirs(args.outdir, exist_ok=True)

    trajectory_data = load_or_build_offline_dataset(
        cache_path=args.offline_cache,
        rebuild_offline=bool(args.rebuild_offline),
    )
    controller_args = setup_DeePC(trajectory_data)

    direct_ok, direct_api = detect_direct_set_api()
    rejection_used = False
    rejection_fallback_last_count = 0

    methods = [
        {"name": "fixedK40", "k": 40, "adaptive_k": False, "local_gate": False},
        {"name": "unified_a001", "k": int(args.k_min), "adaptive_k": True, "local_gate": True},
    ]
    scenarios = [
        ("nominal", float(args.theta0), float(args.thetadot0)),
        (
            "unseen",
            float(args.unseen_scale) * float(args.theta0),
            float(args.unseen_scale) * float(args.thetadot0),
        ),
    ]

    rows = []
    seed_list = [int(args.seed0 + i) for i in range(int(args.seeds))]
    for scenario_name, theta_max, thetadot_max in scenarios:
        for seed in seed_list:
            if direct_ok:
                init_state = sample_init_direct(seed, theta_max, thetadot_max)
            else:
                rejection_used = True
                init_state, used_last = sample_init_rejection(
                    seed=seed,
                    theta_max=theta_max,
                    thetadot_max=thetadot_max,
                    attempts=int(args.attempts),
                )
                if used_last:
                    rejection_fallback_last_count += 1

            for m in methods:
                ctl_args = SimpleNamespace(
                    k=int(m["k"]),
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
                ep = run_episode_with_fixed_init(
                    controller=controller,
                    sigma_y=float(args.sigma_y),
                    seed=int(seed),
                    t_sim=int(args.t_sim),
                    ref_switch_step=int(args.ref_switch_step),
                    init_state=init_state,
                    use_direct_set=bool(direct_ok),
                )
                rows.append(
                    {
                        "scenario": scenario_name,
                        "seed": int(seed),
                        "method": m["name"],
                        "success": int(ep["success"]),
                        "success_reached": int(ep["success_reached"]),
                        "fall_down": int(ep["fall_down"]),
                        "time_to_upright": int(ep["time_to_upright"]),
                        "fail_step": int(ep["fail_step"]),
                        "avg_solve_time": float(ep["avg_solve_time"]),
                        "mean_K_used": float(40.0 if m["name"] == "fixedK40" else ep["mean_K_used"]),
                        "max_K_used": float(40.0 if m["name"] == "fixedK40" else ep["max_K_used"]),
                    }
                )

    out_csv = os.path.join(args.outdir, "unseen_init_offline_only.csv")
    cols = [
        "scenario",
        "seed",
        "method",
        "success",
        "success_reached",
        "fall_down",
        "time_to_upright",
        "fail_step",
        "avg_solve_time",
        "mean_K_used",
        "max_K_used",
    ]
    save_csv(rows, out_csv, fieldnames=cols)

    print("\n[summary]")
    header = (
        f"{'scenario':>8} | {'method':>12} | {'success_rate':>12} | {'reached_rate':>12} | "
        f"{'fall_down_rate':>14} | {'mean_ttu_success':>16} | {'mean_avg_solve':>14}"
    )
    print(header)
    print("-" * len(header))
    for scenario_name, _, _ in scenarios:
        for m in methods:
            s = summarize(rows, scenario_name, m["name"])
            print(
                f"{scenario_name:8s} | {m['name']:12s} | {s['success_rate']:12.3f} | {s['reached_rate']:12.3f} | "
                f"{s['fall_down_rate']:14.3f} | {s['mean_ttu_success']:16.2f} | {s['mean_avg_solve_time']:14.4e}"
            )

    scen_labels = ["nominal", "unseen"]
    x = np.arange(len(scen_labels))
    w = 0.35
    succ_fixed = [summarize(rows, s, "fixedK40")["success_rate"] for s in scen_labels]
    succ_unif = [summarize(rows, s, "unified_a001")["success_rate"] for s in scen_labels]

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(x - w / 2, succ_fixed, width=w, label="fixedK40", color="tab:blue")
    ax1.bar(x + w / 2, succ_unif, width=w, label="unified_a001", color="tab:green")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scen_labels)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("success_rate")
    ax1.set_title("Success Rate by Scenario (Offline-only)")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    out_success = os.path.join(args.outdir, "unseen_init_success.png")
    fig1.savefig(out_success, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    reached_fixed = [summarize(rows, s, "fixedK40")["reached_rate"] for s in scen_labels]
    reached_unif = [summarize(rows, s, "unified_a001")["reached_rate"] for s in scen_labels]
    fall_fixed = [summarize(rows, s, "fixedK40")["fall_down_rate"] for s in scen_labels]
    fall_unif = [summarize(rows, s, "unified_a001")["fall_down_rate"] for s in scen_labels]

    fig2, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax[0].bar(x - w / 2, reached_fixed, width=w, label="fixedK40", color="tab:blue")
    ax[0].bar(x + w / 2, reached_unif, width=w, label="unified_a001", color="tab:green")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(scen_labels)
    ax[0].set_ylim(0.0, 1.0)
    ax[0].set_ylabel("rate")
    ax[0].set_title("Reached Rate")
    ax[0].grid(True, axis="y", alpha=0.3)
    ax[0].legend()

    ax[1].bar(x - w / 2, fall_fixed, width=w, label="fixedK40", color="tab:blue")
    ax[1].bar(x + w / 2, fall_unif, width=w, label="unified_a001", color="tab:green")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(scen_labels)
    ax[1].set_ylim(0.0, 1.0)
    ax[1].set_title("Fall-down Rate | Reached")
    ax[1].grid(True, axis="y", alpha=0.3)
    ax[1].legend()
    fig2.tight_layout()
    out_rf = os.path.join(args.outdir, "unseen_init_reached_falldown.png")
    fig2.savefig(out_rf, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print("\n[init-mode]")
    if direct_ok:
        print(f"Direct set: success (API={direct_api})")
        print("Rejection sampling used: no")
    else:
        print("Direct set: failed")
        print(
            f"Rejection sampling used: yes (attempts={args.attempts}, "
            f"fallback_last_count={rejection_fallback_last_count})"
        )
    print("\n[outputs]")
    print(f"CSV: {out_csv}")
    print(f"Plot(success): {out_success}")
    print(f"Plot(reached/fall-down): {out_rf}")


if __name__ == "__main__":
    main()
