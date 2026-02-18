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
    featurize_obs,
    load_or_build_offline_dataset,
    save_csv,
)


def parse_k_list(s):
    vals = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not vals:
        raise ValueError("k-list is empty")
    return vals


def sample_upright_init(seed, theta0=0.1, thetadot0=0.2, x0=0.05, xdot0=0.1):
    rng = np.random.default_rng(seed + 4242)
    return {
        "x": float(rng.uniform(-x0, x0)),
        "theta": float(rng.uniform(-theta0, theta0)),
        "xdot": float(rng.uniform(-xdot0, xdot0)),
        "thetadot": float(rng.uniform(-thetadot0, thetadot0)),
    }


def inject_state(env, init_state):
    qpos = np.array([init_state["x"], init_state["theta"]], dtype=float)
    qvel = np.array([init_state["xdot"], init_state["thetadot"]], dtype=float)
    try:
        env.unwrapped.set_state(qpos, qvel)
        return True
    except Exception:
        return False


def run_once(controller, sigma_y, seed, t_sim, ref_switch_step, init_state):
    env = gym.make("InvertedPendulum-v4-swingup")
    obs_raw, _ = env.reset(seed=seed)
    direct_ok = inject_state(env, init_state)
    if direct_ok:
        try:
            obs_raw = np.asarray(env.unwrapped._get_obs(), dtype=float)
        except Exception:
            obs_raw, _, _, _, _ = env.step(np.array([0.0], dtype=float))

    rng_noise = np.random.default_rng(seed + 1000)
    solve_times = []
    for step in range(int(t_sim)):
        obs_true = featurize_obs(np.asarray(obs_raw, dtype=float))
        obs_meas = obs_true + rng_noise.normal(0.0, sigma_y, size=obs_true.shape)
        x_ref = 0.0 if step <= ref_switch_step else 0.5
        ref = np.array([x_ref, 0.0, 1.0, 0.0, 0.0], dtype=float)

        t0 = time.perf_counter()
        action = controller.compute_action(obs_meas, ref)
        solve_times.append(time.perf_counter() - t0)
        obs_raw, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()

    sigma_hist = np.asarray(controller.get_selected_uini_sigma_values(), dtype=float)
    lam_hist = np.asarray(controller.get_selected_uini_lam_min_values(), dtype=float)
    return {
        "direct_set_ok": int(direct_ok),
        "mean_sigma_min": float(np.nanmean(sigma_hist)) if sigma_hist.size > 0 else np.nan,
        "min_sigma_min": float(np.nanmin(sigma_hist)) if sigma_hist.size > 0 else np.nan,
        "mean_lambda_min": float(np.nanmean(lam_hist)) if lam_hist.size > 0 else np.nan,
        "min_lambda_min": float(np.nanmin(lam_hist)) if lam_hist.size > 0 else np.nan,
        "mean_solve_time": float(np.nanmean(np.asarray(solve_times, dtype=float))) if solve_times else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot sigma_min(A) vs fixed K.")
    parser.add_argument("--k-list", type=str, default="30,40,50,60,70,80")
    parser.add_argument("--sigma-y", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--t-sim", type=int, default=300)
    parser.add_argument("--ref-switch-step", type=int, default=100)
    parser.add_argument("--theta0", type=float, default=0.1)
    parser.add_argument("--thetadot0", type=float, default=0.2)
    parser.add_argument("--x0", type=float, default=0.05)
    parser.add_argument("--xdot0", type=float, default=0.1)
    parser.add_argument("--outdir", type=str, default=os.path.join(REPO_ROOT, "results"))
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

    k_list = parse_k_list(args.k_list)
    seeds = [int(args.seed0 + i) for i in range(int(args.seeds))]
    init_map = {
        sd: sample_upright_init(sd, args.theta0, args.thetadot0, args.x0, args.xdot0)
        for sd in seeds
    }

    trajectory_data = load_or_build_offline_dataset(
        cache_path=args.offline_cache,
        rebuild_offline=bool(args.rebuild_offline),
    )
    controller_args = setup_DeePC(trajectory_data)

    rows = []
    for k in k_list:
        for sd in seeds:
            ctl_args = SimpleNamespace(
                k=int(k),
                adaptive_k=False,
                k_min=40,
                k_max=200,
                k_step=5,
                alpha=0.01,
                local_gate=False,
                local_quantile=0.05,
                local_min_cands=100,
                local_max_cands=1000,
            )
            controller = build_controller(controller_args, ctl_args)
            out = run_once(
                controller=controller,
                sigma_y=float(args.sigma_y),
                seed=int(sd),
                t_sim=int(args.t_sim),
                ref_switch_step=int(args.ref_switch_step),
                init_state=init_map[sd],
            )
            rows.append(
                {
                    "seed": int(sd),
                    "K": int(k),
                    "sigma_y": float(args.sigma_y),
                    "direct_set_ok": int(out["direct_set_ok"]),
                    "mean_sigma_min": float(out["mean_sigma_min"]),
                    "min_sigma_min": float(out["min_sigma_min"]),
                    "mean_lambda_min": float(out["mean_lambda_min"]),
                    "min_lambda_min": float(out["min_lambda_min"]),
                    "mean_solve_time": float(out["mean_solve_time"]),
                }
            )
            print(
                f"[K={k:3d}][seed={sd:03d}] "
                f"mean_sigma_min={out['mean_sigma_min']:.4e} min_sigma_min={out['min_sigma_min']:.4e}"
            )

    out_csv = os.path.join(args.outdir, "sigma_min_vs_k.csv")
    save_csv(
        rows,
        out_csv,
        fieldnames=[
            "seed",
            "K",
            "sigma_y",
            "direct_set_ok",
            "mean_sigma_min",
            "min_sigma_min",
            "mean_lambda_min",
            "min_lambda_min",
            "mean_solve_time",
        ],
    )

    mean_mu = []
    mean_std = []
    min_mu = []
    min_std = []
    for k in k_list:
        rk = [r for r in rows if r["K"] == k]
        mean_vals = np.array([r["mean_sigma_min"] for r in rk], dtype=float)
        min_vals = np.array([r["min_sigma_min"] for r in rk], dtype=float)
        mean_mu.append(float(np.nanmean(mean_vals)))
        mean_std.append(float(np.nanstd(mean_vals)))
        min_mu.append(float(np.nanmean(min_vals)))
        min_std.append(float(np.nanstd(min_vals)))

    x = np.array(k_list, dtype=float)
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax[0].errorbar(x, mean_mu, yerr=mean_std, marker="o", capsize=3, color="tab:blue")
    ax[0].set_ylabel("mean sigma_min(U_ini_sel)")
    ax[0].set_title("sigma_min vs K (mean over episode, across seeds)")
    ax[0].grid(True, alpha=0.3)
    ax[1].errorbar(x, min_mu, yerr=min_std, marker="o", capsize=3, color="tab:orange")
    ax[1].set_ylabel("min sigma_min(U_ini_sel)")
    ax[1].set_xlabel("K")
    ax[1].set_title("sigma_min vs K (min over episode, across seeds)")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(args.outdir, "sigma_min_vs_k.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n[outputs]")
    print(f"CSV: {out_csv}")
    print(f"Plot: {out_png}")


if __name__ == "__main__":
    main()
