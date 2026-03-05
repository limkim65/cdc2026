import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.reacher.run_cdc_reacher_benchmark import (
    ReacherSetpointScheduler,
    compute_success,
    compute_tracking_error,
    fix_dataset,
    get_cost_dict,
    setup_deepc_reacher,
)
from select_deepc.data_selectors import AdaptiveLkSelector
from select_deepc.deepc_controller import SelectDeePC
from select_deepc.deepc_utils import (
    DeePCCostAccumulator,
    IntegralAbsoluteError,
    IntegralSquareError,
    PerformanceAccumulatorWrapper,
    load_data_from_folder,
    run_reacher_simulator,
)


def safe_stat(values, fn, default=np.nan):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return float(default)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(fn(arr))


def align_kopt(k_hist: np.ndarray, n_steps: int) -> np.ndarray:
    k = np.asarray(k_hist, dtype=float).reshape(-1)
    if k.size == n_steps:
        return k
    if k.size == 2 * n_steps:
        return k.reshape(n_steps, 2)[:, 1]
    if k.size > n_steps:
        return k[:n_steps]
    out = np.full(n_steps, np.nan, dtype=float)
    out[: k.size] = k
    return out


def load_reacher_dataset(repo_root: Path, dataset_mode: str):
    iid = load_data_from_folder(str(repo_root / "data" / "reacher" / "dataset_iid"), "iid")
    rw = load_data_from_folder(str(repo_root / "data" / "reacher" / "dataset_random_walk"), "random_walk")
    for ds in [iid, rw]:
        fix_dataset(ds)
    if dataset_mode == "iid":
        return iid
    if dataset_mode == "random_walk":
        return rw
    return iid + rw


def make_env(outdir: Path, run_tag: str, max_steps: int, record_video: bool):
    from gymnasium.envs.registration import register

    try:
        register(
            id="Reacher-v4-paper-validation",
            entry_point="gymnasium.envs.mujoco.reacher_v4:ReacherEnv",
            max_episode_steps=int(max_steps),
            reward_threshold=-3.75,
        )
    except Exception:
        pass

    env = gym.make("Reacher-v4-paper-validation", render_mode="rgb_array" if record_video else None)
    if record_video:
        video_dir = outdir / "video" / run_tag
        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda _: True,
            name_prefix=f"paper_validation_{run_tag}",
        )
    return env


def run_single_trial(
    repo_root: Path,
    outdir: Path,
    dataset_mode: str,
    seed: int,
    max_steps: int,
    num_extra_targets: int,
    success_eps: float,
    success_window: int,
    mode: str,
    K: int,
    Kmin: int,
    Kmax: int,
    sigma_bar: float,
    n_loc: int,
    d_max: float,
    record_video: bool,
) -> Dict:
    data = load_reacher_dataset(repo_root, dataset_mode)
    controller_args = setup_deepc_reacher(data, enable_measurement_constraint=False)
    run_tag = (
        f"{mode}_seed{seed:03d}_K{K:03d}_Kmin{Kmin}_Kmax{Kmax}_"
        f"sigma{sigma_bar:.3f}_Nloc{n_loc}"
    ).replace(".", "p")

    env = make_env(outdir, run_tag, max_steps=max_steps, record_video=record_video)
    try:
        deepc = SelectDeePC(
            controller_args,
            selector_callback=AdaptiveLkSelector(order=2),
            num_hankel_cols=int(max(K, 1)),
            adaptive_k=bool(mode == "adaptive"),
            K_min=int(Kmin),
            K_max=int(Kmax),
            sigma_bar=float(sigma_bar),
            N_loc=int(n_loc),
            d_max=float(d_max),
            n_iter=1,
        )
        scheduler = ReacherSetpointScheduler(
            env, controller_args.deepc_dims, num_extra_targets=int(num_extra_targets)
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

        target_xy = np.asarray(scheduler._targets.targets[scheduler._target_idx], dtype=float).reshape(1, 2)
        target_traj = np.repeat(target_xy, repeats=max(1, len(x_traj)), axis=0)
        err = compute_tracking_error(np.asarray(x_traj, dtype=float), target_traj)
        success = compute_success(err, eps=float(success_eps), window=int(success_window))

        solve_hist = np.asarray(deepc.get_solve_time_ms_history(), dtype=float)
        slack_hist = np.asarray(deepc.get_slack_norm_inf_history(), dtype=float)
        sigma_hu_hist = np.asarray(deepc.get_min_selected_singular_values(), dtype=float)
        sigma_mk_hist = np.asarray(deepc.get_sigma_min_Mk_history(), dtype=float)
        kopt_hist_raw = np.asarray(deepc.get_K_opt_history(), dtype=float)
        n_steps = int(solve_hist.size)
        kopt_hist = align_kopt(kopt_hist_raw, n_steps=n_steps)

        npz_path = outdir / "runs" / f"{run_tag}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            run_tag=run_tag,
            mode=mode,
            seed=int(seed),
            K=int(K),
            Kmin=int(Kmin),
            Kmax=int(Kmax),
            sigma_bar=float(sigma_bar),
            N_loc=int(n_loc),
            d_max=float(d_max),
            tracking_error=np.asarray(err, dtype=float),
            solve_time_ms=solve_hist,
            slack_inf=slack_hist,
            sigma_min_Hu=sigma_hu_hist,
            sigma_min_Mk=sigma_mk_hist,
            K_opt=kopt_hist,
        )

        row = {
            "run_tag": run_tag,
            "mode": mode,
            "seed": int(seed),
            "K": int(K),
            "Kmin": int(Kmin),
            "Kmax": int(Kmax),
            "sigma_bar": float(sigma_bar),
            "N_loc": int(n_loc),
            "d_max": float(d_max),
            "success": int(success),
            "rmse_ee": float(np.sqrt(safe_stat(err * err, np.mean, default=np.nan))),
            "total_cost": float(get_cost_dict(cost_obj).get("cost", np.nan)),
            "solve_time_mean_ms": safe_stat(solve_hist, np.mean, default=np.nan),
            "solve_time_p95_ms": safe_stat(solve_hist, lambda x: np.percentile(x, 95), default=np.nan),
            "slack_inf_max": safe_stat(slack_hist, np.max, default=np.nan),
            "slack_inf_mean": safe_stat(slack_hist, np.mean, default=np.nan),
            "sigma_min_Hu_min": safe_stat(sigma_hu_hist, np.min, default=np.nan),
            "sigma_min_Hu_mean": safe_stat(sigma_hu_hist, np.mean, default=np.nan),
            "sigma_min_Mk_min": safe_stat(sigma_mk_hist, np.min, default=np.nan),
            "Kopt_mean": safe_stat(kopt_hist, np.mean, default=np.nan),
            "Kopt_p90": safe_stat(kopt_hist, lambda x: np.percentile(x, 90), default=np.nan),
            "npz_path": str(npz_path),
        }
        return row
    except Exception:
        # If failure happens before run_reacher_simulator closes env, close once here.
        try:
            env.close()
        except Exception:
            pass
        raise


def aggregate_fixed(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("K", as_index=False)
        .agg(
            sigma_min_Hu_min=("sigma_min_Hu_min", "median"),
            rmse_ee=("rmse_ee", "median"),
            slack_inf_max=("slack_inf_max", "median"),
            solve_time_mean_ms=("solve_time_mean_ms", "median"),
            success=("success", "mean"),
        )
        .sort_values("K")
    )
    return g


def aggregate_adaptive(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("sigma_bar", as_index=False)
        .agg(
            rmse_ee=("rmse_ee", "mean"),
            solve_time_mean_ms=("solve_time_mean_ms", "mean"),
            slack_inf_max=("slack_inf_max", "mean"),
            Kopt_mean=("Kopt_mean", "mean"),
            sigma_min_Hu_min=("sigma_min_Hu_min", "mean"),
            success=("success", "mean"),
        )
        .sort_values("sigma_bar")
    )
    return g


def make_figures(outdir: Path, df: pd.DataFrame):
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df_fixed = df[df["mode"] == "fixed"].copy()
    df_adapt = df[df["mode"] == "adaptive"].copy()
    fixed_agg = aggregate_fixed(df_fixed)
    adapt_agg = aggregate_adaptive(df_adapt)

    # Figure 1: 3-way relation in fixed-K sweep.
    fig, ax1 = plt.subplots(figsize=(9.5, 6))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    l1 = ax1.plot(fixed_agg["K"], fixed_agg["sigma_min_Hu_min"], "-o", color="#1f77b4", label="sigma_min(Hu)")
    l2 = ax2.plot(fixed_agg["K"], fixed_agg["rmse_ee"], "-s", color="#d62728", label="tracking RMSE")
    l3 = ax3.plot(fixed_agg["K"], fixed_agg["slack_inf_max"], "-^", color="#2ca02c", label="slack max")
    ax1.set_xlabel("K")
    ax1.set_ylabel("sigma_min(Hu)", color="#1f77b4")
    ax2.set_ylabel("tracking RMSE", color="#d62728")
    ax3.set_ylabel("slack max", color="#2ca02c")
    ax1.grid(alpha=0.3)
    lines = l1 + l2 + l3
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="best")
    plt.title("Fig1: Conditioning vs Performance vs Instability (Fixed-K)")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_threeway_fixedk.png", dpi=220)
    plt.close(fig)

    # Figure 2: recovery example from adaptive run.
    tau = float(np.nanpercentile(df_fixed["sigma_min_Hu_min"].to_numpy(dtype=float), 20))
    pick = df_adapt.sort_values(["slack_inf_max", "solve_time_mean_ms"], ascending=[False, False]).head(1)
    if len(pick) > 0:
        p = np.load(pick.iloc[0]["npz_path"], allow_pickle=True)
        sigma = np.asarray(p["sigma_min_Hu"], dtype=float).reshape(-1)
        solve = np.asarray(p["solve_time_ms"], dtype=float).reshape(-1)
        kopt = np.asarray(p["K_opt"], dtype=float).reshape(-1)
        n = min(len(sigma), len(solve), len(kopt))
        t = np.arange(n)
        sigma = sigma[:n]
        solve = solve[:n]
        kopt = kopt[:n]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t, sigma, color="#1f77b4", label="sigma_min(Hu)")
        axes[0].axhline(tau, color="red", linestyle="--", label="tau")
        axes[0].set_ylabel("sigma")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(t, kopt, color="#ff7f0e", label="K_opt")
        axes[1].set_ylabel("K_opt")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[2].plot(t, solve, color="#2ca02c", label="solve time (ms)")
        axes[2].set_ylabel("solve(ms)")
        axes[2].set_xlabel("time step")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        plt.suptitle(f"Fig2: Adaptive Recovery Example ({pick.iloc[0]['run_tag']})")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig2_recovery_adaptive_example.png", dpi=220)
        plt.close(fig)

    # Figure 3: sigma_bar trade-off in adaptive mode.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    x = adapt_agg["sigma_bar"].to_numpy(dtype=float)
    axes = axes.flatten()
    axes[0].plot(x, adapt_agg["rmse_ee"], "-o")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.3)
    axes[1].plot(x, adapt_agg["solve_time_mean_ms"], "-o")
    axes[1].set_ylabel("solve time (ms)")
    axes[1].grid(alpha=0.3)
    axes[2].plot(x, adapt_agg["Kopt_mean"], "-o")
    axes[2].set_ylabel("Kopt mean")
    axes[2].set_xlabel("sigma_bar")
    axes[2].grid(alpha=0.3)
    axes[3].plot(x, adapt_agg["slack_inf_max"], "-o")
    axes[3].set_ylabel("slack max")
    axes[3].set_xlabel("sigma_bar")
    axes[3].grid(alpha=0.3)
    plt.suptitle("Fig3: sigma_bar Sweep Trade-off (Adaptive)")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_sigma_bar_tradeoff.png", dpi=220)
    plt.close(fig)

    # Figure 4: Large-K fixed vs Adaptive best tradeoff.
    fixed_large = df_fixed.sort_values("K").tail(1)
    score = (
        (df_adapt["rmse_ee"] - df_adapt["rmse_ee"].min()) / (df_adapt["rmse_ee"].max() - df_adapt["rmse_ee"].min() + 1e-12)
        + (df_adapt["solve_time_mean_ms"] - df_adapt["solve_time_mean_ms"].min())
        / (df_adapt["solve_time_mean_ms"].max() - df_adapt["solve_time_mean_ms"].min() + 1e-12)
    )
    adapt_best = df_adapt.iloc[[int(np.nanargmin(score.to_numpy(dtype=float)))]]
    comp = pd.concat([fixed_large.assign(label="fixed_largeK"), adapt_best.assign(label="adaptive_best")], ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(comp["label"], comp["rmse_ee"], color=["#1f77b4", "#ff7f0e"])
    axes[0].set_title("RMSE")
    axes[1].bar(comp["label"], comp["solve_time_mean_ms"], color=["#1f77b4", "#ff7f0e"])
    axes[1].set_title("Solve Time (ms)")
    axes[2].bar(comp["label"], comp["slack_inf_max"], color=["#1f77b4", "#ff7f0e"])
    axes[2].set_title("Slack Max")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Fig4: Fixed Large-K vs Proposed Adaptive")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_fixed_largek_vs_adaptive.png", dpi=220)
    plt.close(fig)

    fixed_agg.to_csv(outdir / "summary_fixed_by_K.csv", index=False)
    adapt_agg.to_csv(outdir / "summary_adaptive_by_sigma.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Paper validation runner on Reacher (new runner only).")
    parser.add_argument("--outdir", type=str, default=os.path.join("logs", "reacher", "paper_validation_deepc4"))
    parser.add_argument("--dataset", choices=["iid", "random_walk", "both"], default="iid")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--num_extra_targets", type=int, default=0)
    parser.add_argument("--success_eps", type=float, default=0.01)
    parser.add_argument("--success_window", type=int, default=20)
    parser.add_argument("--fixed_K_values", type=int, nargs="+", default=[40, 60, 80, 100, 140, 180])
    parser.add_argument("--adaptive_sigma_values", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--adaptive_Kmin", type=int, default=40)
    parser.add_argument("--adaptive_Kmax", type=int, default=300)
    parser.add_argument("--N_loc", type=int, default=500)
    parser.add_argument("--d_max", type=float, default=10.0)
    parser.add_argument("--record_video", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    total = len(args.seeds) * (len(args.fixed_K_values) + len(args.adaptive_sigma_values))
    idx = 0

    for seed in args.seeds:
        for K in args.fixed_K_values:
            idx += 1
            print(f"[{idx}/{total}] fixed K={K} seed={seed}")
            row = run_single_trial(
                repo_root=repo_root,
                outdir=outdir,
                dataset_mode=args.dataset,
                seed=seed,
                max_steps=args.max_steps,
                num_extra_targets=args.num_extra_targets,
                success_eps=args.success_eps,
                success_window=args.success_window,
                mode="fixed",
                K=int(K),
                Kmin=int(K),
                Kmax=int(K),
                sigma_bar=1e-9,
                n_loc=int(args.N_loc),
                d_max=float(args.d_max),
                record_video=bool(args.record_video),
            )
            rows.append(row)

        for sigma in args.adaptive_sigma_values:
            idx += 1
            print(f"[{idx}/{total}] adaptive sigma={sigma} seed={seed}")
            row = run_single_trial(
                repo_root=repo_root,
                outdir=outdir,
                dataset_mode=args.dataset,
                seed=seed,
                max_steps=args.max_steps,
                num_extra_targets=args.num_extra_targets,
                success_eps=args.success_eps,
                success_window=args.success_window,
                mode="adaptive",
                K=int(args.adaptive_Kmax),
                Kmin=int(args.adaptive_Kmin),
                Kmax=int(args.adaptive_Kmax),
                sigma_bar=float(sigma),
                n_loc=int(args.N_loc),
                d_max=float(args.d_max),
                record_video=bool(args.record_video),
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "paper_validation_runs.csv", index=False)
    make_figures(outdir=outdir, df=df)

    print(f"Saved runs: {outdir / 'paper_validation_runs.csv'}")
    print(f"Saved figures under: {outdir / 'figures'}")
    print(f"Saved summaries: {outdir / 'summary_fixed_by_K.csv'}, {outdir / 'summary_adaptive_by_sigma.csv'}")


if __name__ == "__main__":
    main()
