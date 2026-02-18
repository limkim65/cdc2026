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


def sample_initial_state(seed: int, scenario: str, theta0_max: float, thetadot0_max: float, unseen_scale: float):
    if scenario not in ("nominal", "unseen"):
        raise ValueError(f"Unknown scenario: {scenario}")
    scale = float(unseen_scale) if scenario == "unseen" else 1.0
    rng = np.random.default_rng(seed + 31415)
    theta0 = float(rng.uniform(-scale * theta0_max, scale * theta0_max))
    thetadot0 = float(rng.uniform(-scale * thetadot0_max, scale * thetadot0_max))
    return {
        "x": 0.0,
        "theta": theta0,
        "xdot": 0.0,
        "thetadot": thetadot0,
    }


def inject_initial_state(env, init_state):
    qpos = np.array([init_state["x"], init_state["theta"]], dtype=float)
    qvel = np.array([init_state["xdot"], init_state["thetadot"]], dtype=float)
    try:
        env.unwrapped.set_state(qpos, qvel)
    except Exception as exc:
        raise RuntimeError("Failed direct set via env.unwrapped.set_state") from exc


def run_episode_with_state(controller, seed, sigma_y, t_sim, ref_switch_step, init_state):
    env = gym.make("InvertedPendulum-v4-swingup")
    obs_raw, _ = env.reset(seed=seed)
    inject_initial_state(env, init_state)
    try:
        obs_raw = np.asarray(env.unwrapped._get_obs(), dtype=float)
    except Exception:
        obs_raw, _, _, _, _ = env.step(np.array([0.0], dtype=float))

    rng_noise = np.random.default_rng(seed + 1000)
    cos_theta = []
    u_hist = []
    solve_times = []
    for step in range(int(t_sim)):
        obs_true = featurize_obs(np.asarray(obs_raw, dtype=float))
        obs_meas = obs_true + rng_noise.normal(0.0, sigma_y, size=obs_true.shape)
        x_ref = 0.0 if step <= ref_switch_step else 0.5
        ref = np.array([x_ref, 0.0, 1.0, 0.0, 0.0], dtype=float)

        t0 = time.perf_counter()
        action = controller.compute_action(obs_meas, ref)
        solve_times.append(time.perf_counter() - t0)
        u_hist.append(float(np.asarray(action, dtype=float).reshape(-1)[0]))

        obs_raw, _, terminated, truncated, _ = env.step(action)
        cos_theta.append(float(np.cos(obs_raw[1])))
        if terminated or truncated:
            break
    env.close()

    cos_theta = np.asarray(cos_theta, dtype=float)
    u_hist = np.asarray(u_hist, dtype=float)
    final_success, success_reached, upright_step, fail_step = evaluate_upright_and_fail(
        cos_theta=cos_theta,
        n_up=20,
        n_fail=20,
    )
    fall_down = bool(success_reached and (fail_step is not None))

    k_used = np.asarray(controller.get_k_used_history(), dtype=float)
    lam_min = np.asarray(controller.get_selected_uini_lam_min_values(), dtype=float)
    mean_k = float(np.nanmean(k_used)) if k_used.size > 0 else np.nan
    return {
        "success": int(final_success),
        "reached": int(success_reached),
        "fall_down": int(fall_down),
        "time_to_upright": int(upright_step) if upright_step is not None else int(t_sim),
        "avg_solve_time": float(np.mean(solve_times)) if solve_times else np.nan,
        "mean_K_used": mean_k,
        "cos_theta": cos_theta,
        "u": u_hist,
        "K_used": k_used,
        "lambda_min": lam_min,
    }


def summarize(rows, scenario, method):
    data = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
    success = np.array([r["success"] for r in data], dtype=float)
    reached = np.array([r["reached"] for r in data], dtype=float)
    fall = np.array([r["fall_down"] for r in data], dtype=float)
    ttu = np.array([r["time_to_upright"] for r in data], dtype=float)
    solve = np.array([r["avg_solve_time"] for r in data], dtype=float)
    reached_mask = reached > 0.5
    success_mask = success > 0.5
    return {
        "success_rate": float(np.nanmean(success)) if success.size > 0 else np.nan,
        "reached_rate": float(np.nanmean(reached)) if reached.size > 0 else np.nan,
        "fall_rate": float(np.nanmean(fall[reached_mask])) if np.any(reached_mask) else np.nan,
        "mean_ttu_success": float(np.nanmean(ttu[success_mask])) if np.any(success_mask) else np.nan,
        "mean_solve": float(np.nanmean(solve)) if solve.size > 0 else np.nan,
    }


def _pick_seed(rows, scenario, method, want_success):
    for r in rows:
        if r["scenario"] != scenario or r["method"] != method:
            continue
        if int(r["success"]) == int(want_success):
            return int(r["seed"])
    return None


def plot_representative(rows, traj_map, scenario, out_path):
    seed_fixed_fail = _pick_seed(rows, scenario, "fixedK40", want_success=0)
    seed_unified_succ = _pick_seed(rows, scenario, "unified_a001", want_success=1)

    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"Representative Trajectories ({scenario})", fontsize=11)

    plotted = False
    if seed_fixed_fail is not None:
        d = traj_map[(scenario, "fixedK40", seed_fixed_fail)]
        t = np.arange(d["cos_theta"].shape[0])
        ax[0].plot(t, d["cos_theta"], label=f"fixed fail seed={seed_fixed_fail}", color="tab:blue")
        ax[1].plot(np.arange(d["u"].shape[0]), d["u"], color="tab:blue")
        ax[2].plot(np.arange(d["K_used"].shape[0]), d["K_used"], color="tab:blue")
        ax[3].plot(np.arange(d["lambda_min"].shape[0]), d["lambda_min"], color="tab:blue")
        plotted = True

    if seed_unified_succ is not None:
        d = traj_map[(scenario, "unified_a001", seed_unified_succ)]
        t = np.arange(d["cos_theta"].shape[0])
        ax[0].plot(t, d["cos_theta"], label=f"unified success seed={seed_unified_succ}", color="tab:green")
        ax[1].plot(np.arange(d["u"].shape[0]), d["u"], color="tab:green")
        ax[2].plot(np.arange(d["K_used"].shape[0]), d["K_used"], color="tab:green")
        ax[3].plot(np.arange(d["lambda_min"].shape[0]), d["lambda_min"], color="tab:green")
        plotted = True

    ax[0].axhline(0.5, color="r", linestyle="--", linewidth=1, alpha=0.7)
    ax[0].set_ylabel("cos(theta)")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()
    ax[1].set_ylabel("u")
    ax[1].grid(True, alpha=0.3)
    ax[2].set_ylabel("K_used")
    ax[2].grid(True, alpha=0.3)
    ax[3].set_ylabel("lambda_min")
    ax[3].set_xlabel("step")
    ax[3].grid(True, alpha=0.3)
    if not plotted:
        ax[0].text(0.5, 0.5, "No representative pair found", transform=ax[0].transAxes, ha="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Reproducible unseen-init compare: fixedK40 vs unified_a001.")
    parser.add_argument("--sigma-y", type=float, default=0.05)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--theta0", type=float, default=0.2)
    parser.add_argument("--thetadot0", type=float, default=0.5)
    parser.add_argument("--unseen-scale", type=float, default=1.5)
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

    methods = [
        {"name": "fixedK40", "k": 40, "adaptive_k": False, "local_gate": False},
        {"name": "unified_a001", "k": int(args.k_min), "adaptive_k": True, "local_gate": True},
    ]
    scenarios = ["nominal", "unseen"]
    seeds = [int(args.seed0 + i) for i in range(int(args.seeds))]

    init_map = {}
    for sc in scenarios:
        for sd in seeds:
            init_map[(sc, sd)] = sample_initial_state(
                seed=sd,
                scenario=sc,
                theta0_max=float(args.theta0),
                thetadot0_max=float(args.thetadot0),
                unseen_scale=float(args.unseen_scale),
            )

    rows = []
    traj_map = {}
    for sc in scenarios:
        for sd in seeds:
            init_state = init_map[(sc, sd)]
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
                out = run_episode_with_state(
                    controller=controller,
                    seed=sd,
                    sigma_y=float(args.sigma_y),
                    t_sim=int(args.t_sim),
                    ref_switch_step=int(args.ref_switch_step),
                    init_state=init_state,
                )
                row = {
                    "seed": int(sd),
                    "scenario": sc,
                    "method": m["name"],
                    "theta0": float(init_state["theta"]),
                    "thetadot0": float(init_state["thetadot"]),
                    "success": int(out["success"]),
                    "reached": int(out["reached"]),
                    "fall_down": int(out["fall_down"]),
                    "time_to_upright": int(out["time_to_upright"]),
                    "avg_solve_time": float(out["avg_solve_time"]),
                    "mean_K_used": float(40.0 if m["name"] == "fixedK40" else out["mean_K_used"]),
                }
                rows.append(row)
                traj_map[(sc, m["name"], sd)] = out
                print(
                    f"[{sc}][seed={sd:02d}][{m['name']}] "
                    f"succ={row['success']} reach={row['reached']} fall={row['fall_down']}"
                )

    out_csv = os.path.join(args.outdir, "unseen_repro.csv")
    save_csv(
        rows,
        out_csv,
        fieldnames=[
            "seed",
            "scenario",
            "method",
            "theta0",
            "thetadot0",
            "success",
            "reached",
            "fall_down",
            "time_to_upright",
            "avg_solve_time",
            "mean_K_used",
        ],
    )

    # Stats plots
    scenarios_labels = ["nominal", "unseen"]
    x = np.arange(len(scenarios_labels))
    w = 0.35
    succ_fixed = [summarize(rows, s, "fixedK40")["success_rate"] for s in scenarios_labels]
    succ_unif = [summarize(rows, s, "unified_a001")["success_rate"] for s in scenarios_labels]

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(x - w / 2, succ_fixed, width=w, label="fixedK40", color="tab:blue")
    ax1.bar(x + w / 2, succ_unif, width=w, label="unified_a001", color="tab:green")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios_labels)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("success_rate")
    ax1.set_title("Unseen Repro: Success Rate")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    out_success = os.path.join(args.outdir, "unseen_repro_success.png")
    fig1.savefig(out_success, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    reached_fixed = [summarize(rows, s, "fixedK40")["reached_rate"] for s in scenarios_labels]
    reached_unif = [summarize(rows, s, "unified_a001")["reached_rate"] for s in scenarios_labels]
    fall_fixed = [summarize(rows, s, "fixedK40")["fall_rate"] for s in scenarios_labels]
    fall_unif = [summarize(rows, s, "unified_a001")["fall_rate"] for s in scenarios_labels]

    fig2, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax[0].bar(x - w / 2, reached_fixed, width=w, label="fixedK40", color="tab:blue")
    ax[0].bar(x + w / 2, reached_unif, width=w, label="unified_a001", color="tab:green")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(scenarios_labels)
    ax[0].set_ylim(0.0, 1.0)
    ax[0].set_title("Reached Rate")
    ax[0].set_ylabel("rate")
    ax[0].grid(True, axis="y", alpha=0.3)
    ax[0].legend()

    ax[1].bar(x - w / 2, fall_fixed, width=w, label="fixedK40", color="tab:blue")
    ax[1].bar(x + w / 2, fall_unif, width=w, label="unified_a001", color="tab:green")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(scenarios_labels)
    ax[1].set_ylim(0.0, 1.0)
    ax[1].set_title("Fall-down Rate | Reached")
    ax[1].grid(True, axis="y", alpha=0.3)
    ax[1].legend()
    fig2.tight_layout()
    out_rf = os.path.join(args.outdir, "unseen_repro_reached_fall.png")
    fig2.savefig(out_rf, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # Representative trajectories
    out_nom = os.path.join(args.outdir, "unseen_repro_traj_nominal.png")
    out_uns = os.path.join(args.outdir, "unseen_repro_traj_unseen.png")
    plot_representative(rows, traj_map, "nominal", out_nom)
    plot_representative(rows, traj_map, "unseen", out_uns)

    print("\n[outputs]")
    print(f"CSV: {out_csv}")
    print(f"Plot(success): {out_success}")
    print(f"Plot(reached/fall): {out_rf}")
    print(f"Plot(traj nominal): {out_nom}")
    print(f"Plot(traj unseen): {out_uns}")


if __name__ == "__main__":
    main()
