import argparse
import os
import sys
from types import SimpleNamespace

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
    load_or_build_offline_dataset,
    run_episode,
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


def wilson_ci(k, n, z=1.96):
    if n <= 0:
        return np.nan, np.nan
    p = float(k) / float(n)
    den = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / den
    radius = (z / den) * np.sqrt((p * (1.0 - p) / n) + (z**2) / (4.0 * n * n))
    return max(0.0, center - radius), min(1.0, center + radius)


def bootstrap_ci(values, b=1000, alpha=0.05, seed=0):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        v = float(x[0])
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(b), n))
    means = np.mean(x[idx], axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _longest_streak(mask):
    best = 0
    cur = 0
    for v in mask:
        if bool(v):
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _fall_event_count(mask_up):
    up = np.asarray(mask_up, dtype=bool)
    if up.size < 2:
        return 0
    return int(np.sum(np.logical_and(up[:-1], np.logical_not(up[1:]))))


def compute_episode_upright_metrics(cos_theta, c_up=0.95):
    c = np.asarray(cos_theta, dtype=float)
    if c.size == 0:
        return np.nan, 0, 0
    up = c >= float(c_up)
    upright_fraction = float(np.mean(up.astype(float)))
    longest = _longest_streak(up)
    fall_events = _fall_event_count(up)
    return upright_fraction, longest, fall_events


def summarize_sigma_method(rows, sigma, method, bootstrap_b=1000):
    subset = [r for r in rows if np.isclose(r["sigma_y"], sigma) and r["method"] == method]
    if len(subset) == 0:
        return {}
    success = np.array([r["success"] for r in subset], dtype=int)
    reached = np.array([r["success_reached"] for r in subset], dtype=int)
    fall = np.array([r["fall_down"] for r in subset], dtype=int)
    ttu = np.array([r["time_to_upright"] for r in subset], dtype=float)
    solve = np.array([r["avg_solve_time"] for r in subset], dtype=float)
    mean_k = np.array([r["mean_K_used"] for r in subset], dtype=float)
    max_k = np.array([r["max_K_used"] for r in subset], dtype=float)
    uf = np.array([r["upright_fraction"] for r in subset], dtype=float)
    streak = np.array([r["longest_upright_streak"] for r in subset], dtype=float)

    n = int(success.size)
    k_success = int(np.sum(success))
    k_reached = int(np.sum(reached))
    k_fall = int(np.sum(fall))
    reached_mask = reached > 0
    n_reached = int(np.sum(reached_mask))
    k_fall_cond = int(np.sum(fall[reached_mask])) if n_reached > 0 else 0

    p_success = float(k_success / n) if n > 0 else np.nan
    p_reached = float(k_reached / n) if n > 0 else np.nan
    p_fall_cond = float(k_fall_cond / n_reached) if n_reached > 0 else np.nan

    s_low, s_high = wilson_ci(k_success, n)
    r_low, r_high = wilson_ci(k_reached, n)
    f_low, f_high = wilson_ci(k_fall_cond, n_reached) if n_reached > 0 else (np.nan, np.nan)

    uf_mean = float(np.nanmean(uf)) if uf.size > 0 else np.nan
    uf_low, uf_high = bootstrap_ci(uf, b=bootstrap_b, alpha=0.05, seed=123)
    st_mean = float(np.nanmean(streak)) if streak.size > 0 else np.nan
    st_low, st_high = bootstrap_ci(streak, b=bootstrap_b, alpha=0.05, seed=456)

    success_mask = success > 0
    return {
        "success_rate": p_success,
        "success_low": s_low,
        "success_high": s_high,
        "reached_rate": p_reached,
        "reached_low": r_low,
        "reached_high": r_high,
        "fall_down_rate": p_fall_cond,
        "fall_low": f_low,
        "fall_high": f_high,
        "reached_count": n_reached,
        "fall_down_count": k_fall,
        "mean_ttu_success": float(np.nanmean(ttu[success_mask])) if np.any(success_mask) else np.nan,
        "mean_avg_solve_time": float(np.nanmean(solve)) if solve.size > 0 else np.nan,
        "mean_of_mean_k": float(np.nanmean(mean_k)) if mean_k.size > 0 else np.nan,
        "mean_of_max_k": float(np.nanmean(max_k)) if max_k.size > 0 else np.nan,
        "upright_fraction_mean": uf_mean,
        "upright_fraction_low": uf_low,
        "upright_fraction_high": uf_high,
        "longest_streak_mean": st_mean,
        "longest_streak_low": st_low,
        "longest_streak_high": st_high,
    }


def print_sigma_summary(summary_map, sigma_vals, methods):
    print("\n[sigma-summary]")
    header = (
        f"{'sigma':>7} | {'method':>12} | {'success_rate':>12} | {'reached_rate(count)':>20} | "
        f"{'fall_down_rate(count)':>22} | {'UF_mean[95%CI]':>22} | {'mean_ttu_success':>16} | "
        f"{'mean_avg_solve':>14} | {'meanK':>8} | {'meanMaxK':>10}"
    )
    print(header)
    print("-" * len(header))
    for sigma in sigma_vals:
        for method in methods:
            s = summary_map[(sigma, method)]
            reached_field = f"{s['reached_rate']:.3f} ({s['reached_count']})"
            fall_field = f"{s['fall_down_rate']:.3f} ({s['fall_down_count']})"
            uf_field = f"{s['upright_fraction_mean']:.3f}[{s['upright_fraction_low']:.3f},{s['upright_fraction_high']:.3f}]"
            print(
                f"{sigma:7.3f} | {method:12s} | {s['success_rate']:12.3f} | {reached_field:20s} | "
                f"{fall_field:22s} | {uf_field:22s} | {s['mean_ttu_success']:16.2f} | {s['mean_avg_solve_time']:14.4e} | "
                f"{s['mean_of_mean_k']:8.2f} | {s['mean_of_max_k']:10.2f}"
            )
            print(
                f"  sigma={sigma:.2f} {method}: success={s['success_rate']:.3f} "
                f"[{s['success_low']:.3f},{s['success_high']:.3f}], "
                f"UF={s['upright_fraction_mean']:.3f} "
                f"[{s['upright_fraction_low']:.3f},{s['upright_fraction_high']:.3f}]"
            )


def style_ax(ax, xvals, ylabel, title):
    ax.set_xlabel("sigma_y")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(xvals)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
    ax.legend(frameon=True)


def _yerr_from_bounds(mean_vals, lows, highs):
    mean = np.asarray(mean_vals, dtype=float)
    low = np.asarray(lows, dtype=float)
    high = np.asarray(highs, dtype=float)
    return np.vstack([mean - low, high - mean])


def main():
    parser = argparse.ArgumentParser(
        description="Wide sigma sweep for fixedK40 vs unified_a001 with reached/fall-down decomposition."
    )
    parser.add_argument("--sigma-list", type=str, default="0.02,0.03,0.04,0.05,0.06,0.07,0.08")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--t-sim", type=int, default=300)
    parser.add_argument("--ref-switch-step", type=int, default=100)
    parser.add_argument("--outdir", type=str, default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--plot-tradeoff", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-b", type=int, default=1000)

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
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cdc2026")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="sigma_sweep_wide")
    parser.add_argument("--wandb-run-name", type=str, default="")
    args = parser.parse_args()

    sigma_vals = parse_sigma_list(args.sigma_list)
    seed_list = [int(args.seed0 + i) for i in range(int(args.seeds))]
    methods = ["fixedK40", "unified_a001"]

    check_runtime_dependencies()
    ensure_env_registered()
    os.makedirs(args.outdir, exist_ok=True)

    trajectory_data = load_or_build_offline_dataset(
        cache_path=args.offline_cache,
        rebuild_offline=bool(args.rebuild_offline),
    )
    controller_args = setup_DeePC(trajectory_data)
    wb_run = setup_wandb(args, "sigma_sweep_wide_fixed_vs_unified")

    print(
        "[config] "
        f"sigma_list={sigma_vals} seeds={len(seed_list)} seed0={args.seed0} "
        f"alpha={args.alpha} local_quantile={args.local_quantile} "
        f"local_min_cands={args.local_min_cands} local_max_cands={args.local_max_cands} "
        f"k_min={args.k_min} k_max={args.k_max} k_step={args.k_step}"
    )

    rows = []
    for sigma in sigma_vals:
        for seed in seed_list:
            configs = [
                {"name": "fixedK40", "k": 40, "adaptive_k": False, "local_gate": False},
                {"name": "unified_a001", "k": int(args.k_min), "adaptive_k": True, "local_gate": True},
            ]
            for cfg in configs:
                ctl_args = SimpleNamespace(
                    k=int(cfg["k"]),
                    adaptive_k=bool(cfg["adaptive_k"]),
                    k_min=int(args.k_min),
                    k_max=int(args.k_max),
                    k_step=int(args.k_step),
                    alpha=float(args.alpha),
                    local_gate=bool(cfg["local_gate"]),
                    local_quantile=float(args.local_quantile),
                    local_min_cands=int(args.local_min_cands),
                    local_max_cands=int(args.local_max_cands),
                )
                controller = build_controller(controller_args, ctl_args)
                ep = run_episode(
                    controller=controller,
                    sigma_y=float(sigma),
                    seed=int(seed),
                    t_sim=int(args.t_sim),
                    ref_switch_step=int(args.ref_switch_step),
                )
                upright_fraction, longest_streak, fall_events = compute_episode_upright_metrics(
                    ep["cos_theta"], c_up=0.95
                )
                rows.append(
                    {
                        "sigma_y": float(sigma),
                        "seed": int(seed),
                        "method": cfg["name"],
                        "success": int(ep["success"]),
                        "success_reached": int(ep["success_reached"]),
                        "fall_down": int(ep["fall_down"]),
                        "time_to_upright": int(ep["time_to_upright"]),
                        "fail_step": int(ep["fail_step"]),
                        "avg_solve_time": float(ep["avg_solve_time"]),
                        "mean_K_used": float(40.0 if cfg["name"] == "fixedK40" else ep["mean_K_used"]),
                        "max_K_used": float(40.0 if cfg["name"] == "fixedK40" else ep["max_K_used"]),
                        "certificate_rate": float(np.nan if cfg["name"] == "fixedK40" else ep["certificate_rate"]),
                        "mean_tau": float(np.nan if cfg["name"] == "fixedK40" else ep["mean_tau"]),
                        "mean_d90": float(np.nan if cfg["name"] == "fixedK40" else ep["mean_d90"]),
                        "mean_C_size": float(np.nan if cfg["name"] == "fixedK40" else ep["mean_C_size"]),
                        "upright_fraction": float(upright_fraction),
                        "longest_upright_streak": int(longest_streak),
                        "fall_event_count": int(fall_events),
                        "local_min_cands": int(args.local_min_cands),
                        "local_max_cands": int(args.local_max_cands),
                    }
                )
                print(
                    f"[sigma={sigma:.3f}][seed={seed:03d}][{cfg['name']}] "
                    f"success={int(ep['success'])} reached={int(ep['success_reached'])} "
                    f"fall={int(ep['fall_down'])} ttu={int(ep['time_to_upright'])}"
                )

    out_csv = os.path.join(args.outdir, "sigma_sweep_wide.csv")
    cols = [
        "sigma_y",
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
        "certificate_rate",
        "mean_tau",
        "mean_d90",
        "mean_C_size",
        "upright_fraction",
        "longest_upright_streak",
        "fall_event_count",
        "local_min_cands",
        "local_max_cands",
    ]
    save_csv(rows, out_csv, fieldnames=cols)

    summary = {
        (sigma, method): summarize_sigma_method(rows, sigma, method, bootstrap_b=int(args.bootstrap_b))
        for sigma in sigma_vals
        for method in methods
    }

    success_series = {m: [summary[(s, m)]["success_rate"] for s in sigma_vals] for m in methods}
    success_low = {m: [summary[(s, m)]["success_low"] for s in sigma_vals] for m in methods}
    success_high = {m: [summary[(s, m)]["success_high"] for s in sigma_vals] for m in methods}

    reached_series = {m: [summary[(s, m)]["reached_rate"] for s in sigma_vals] for m in methods}
    reached_low = {m: [summary[(s, m)]["reached_low"] for s in sigma_vals] for m in methods}
    reached_high = {m: [summary[(s, m)]["reached_high"] for s in sigma_vals] for m in methods}

    fall_series = {m: [summary[(s, m)]["fall_down_rate"] for s in sigma_vals] for m in methods}
    fall_low = {m: [summary[(s, m)]["fall_low"] for s in sigma_vals] for m in methods}
    fall_high = {m: [summary[(s, m)]["fall_high"] for s in sigma_vals] for m in methods}

    uf_series = {m: [summary[(s, m)]["upright_fraction_mean"] for s in sigma_vals] for m in methods}
    uf_low = {m: [summary[(s, m)]["upright_fraction_low"] for s in sigma_vals] for m in methods}
    uf_high = {m: [summary[(s, m)]["upright_fraction_high"] for s in sigma_vals] for m in methods}

    streak_series = {m: [summary[(s, m)]["longest_streak_mean"] for s in sigma_vals] for m in methods}
    streak_low = {m: [summary[(s, m)]["longest_streak_low"] for s in sigma_vals] for m in methods}
    streak_high = {m: [summary[(s, m)]["longest_streak_high"] for s in sigma_vals] for m in methods}

    time_series = {m: [summary[(s, m)]["mean_avg_solve_time"] for s in sigma_vals] for m in methods}

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.errorbar(
        sigma_vals,
        success_series["fixedK40"],
        yerr=_yerr_from_bounds(success_series["fixedK40"], success_low["fixedK40"], success_high["fixedK40"]),
        marker="o",
        linewidth=2,
        capsize=3,
        label="fixedK40",
        color="tab:blue",
    )
    ax1.errorbar(
        sigma_vals,
        success_series["unified_a001"],
        yerr=_yerr_from_bounds(success_series["unified_a001"], success_low["unified_a001"], success_high["unified_a001"]),
        marker="o",
        linewidth=2,
        capsize=3,
        label="unified_a001",
        color="tab:green",
    )
    ax1.set_ylim(0.0, 1.0)
    style_ax(ax1, sigma_vals, "success_rate", "Success Rate vs sigma_y")
    fig1.tight_layout()
    out_success = os.path.join(args.outdir, "sigma_wide_success.png")
    fig1.savefig(out_success, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.errorbar(
        sigma_vals,
        reached_series["fixedK40"],
        yerr=_yerr_from_bounds(reached_series["fixedK40"], reached_low["fixedK40"], reached_high["fixedK40"]),
        marker="o",
        linewidth=2,
        capsize=3,
        label="fixedK40",
        color="tab:blue",
    )
    ax2.errorbar(
        sigma_vals,
        reached_series["unified_a001"],
        yerr=_yerr_from_bounds(reached_series["unified_a001"], reached_low["unified_a001"], reached_high["unified_a001"]),
        marker="o",
        linewidth=2,
        capsize=3,
        label="unified_a001",
        color="tab:green",
    )
    ax2.set_ylim(0.0, 1.0)
    style_ax(ax2, sigma_vals, "reached_rate", "Reached Rate vs sigma_y")
    fig2.tight_layout()
    out_reached = os.path.join(args.outdir, "sigma_wide_reached.png")
    fig2.savefig(out_reached, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.errorbar(
        sigma_vals,
        fall_series["fixedK40"],
        yerr=_yerr_from_bounds(fall_series["fixedK40"], fall_low["fixedK40"], fall_high["fixedK40"]),
        marker="o",
        linewidth=2,
        capsize=3,
        label="fixedK40",
        color="tab:blue",
    )
    ax3.errorbar(
        sigma_vals,
        fall_series["unified_a001"],
        yerr=_yerr_from_bounds(fall_series["unified_a001"], fall_low["unified_a001"], fall_high["unified_a001"]),
        marker="o",
        linewidth=2,
        capsize=3,
        label="unified_a001",
        color="tab:green",
    )
    ax3.set_ylim(0.0, 1.0)
    style_ax(ax3, sigma_vals, "fall_down_rate | reached", "Fall-Down Rate vs sigma_y")
    fig3.tight_layout()
    out_fall = os.path.join(args.outdir, "sigma_wide_falldown.png")
    fig3.savefig(out_fall, dpi=200, bbox_inches="tight")
    plt.close(fig3)

    fig_uf, ax_uf = plt.subplots(figsize=(7, 4))
    ax_uf.errorbar(
        sigma_vals,
        uf_series["fixedK40"],
        yerr=_yerr_from_bounds(uf_series["fixedK40"], uf_low["fixedK40"], uf_high["fixedK40"]),
        marker="o",
        linewidth=2,
        capsize=3,
        label="fixedK40",
        color="tab:blue",
    )
    ax_uf.errorbar(
        sigma_vals,
        uf_series["unified_a001"],
        yerr=_yerr_from_bounds(uf_series["unified_a001"], uf_low["unified_a001"], uf_high["unified_a001"]),
        marker="o",
        linewidth=2,
        capsize=3,
        label="unified_a001",
        color="tab:green",
    )
    ax_uf.set_ylim(0.0, 1.0)
    style_ax(ax_uf, sigma_vals, "upright_fraction", "Upright Fraction vs sigma_y")
    fig_uf.tight_layout()
    out_uf = os.path.join(args.outdir, "sigma_wide_upright_fraction.png")
    fig_uf.savefig(out_uf, dpi=200, bbox_inches="tight")
    plt.close(fig_uf)

    fig_st, ax_st = plt.subplots(figsize=(7, 4))
    ax_st.errorbar(
        sigma_vals,
        streak_series["fixedK40"],
        yerr=_yerr_from_bounds(streak_series["fixedK40"], streak_low["fixedK40"], streak_high["fixedK40"]),
        marker="o",
        linewidth=2,
        capsize=3,
        label="fixedK40",
        color="tab:blue",
    )
    ax_st.errorbar(
        sigma_vals,
        streak_series["unified_a001"],
        yerr=_yerr_from_bounds(streak_series["unified_a001"], streak_low["unified_a001"], streak_high["unified_a001"]),
        marker="o",
        linewidth=2,
        capsize=3,
        label="unified_a001",
        color="tab:green",
    )
    style_ax(ax_st, sigma_vals, "longest_upright_streak", "Longest Upright Streak vs sigma_y")
    fig_st.tight_layout()
    out_streak = os.path.join(args.outdir, "sigma_wide_longest_streak.png")
    fig_st.savefig(out_streak, dpi=200, bbox_inches="tight")
    plt.close(fig_st)

    out_tradeoff = None
    if args.plot_tradeoff:
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        for method, color in [("fixedK40", "tab:blue"), ("unified_a001", "tab:green")]:
            x = np.array(time_series[method], dtype=float)
            y = np.array(success_series[method], dtype=float)
            ax4.scatter(x, y, s=46, color=color, label=method, alpha=0.85)
            for i, sigma in enumerate(sigma_vals):
                if np.isfinite(x[i]) and np.isfinite(y[i]):
                    ax4.annotate(f"{sigma:.2f}", (x[i], y[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax4.set_xlabel("mean_avg_solve_time [s]")
        ax4.set_ylabel("success_rate")
        ax4.set_ylim(0.0, 1.0)
        ax4.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
        ax4.legend(frameon=True)
        ax4.set_title("Trade-off: Solve Time vs Success")
        fig4.tight_layout()
        out_tradeoff = os.path.join(args.outdir, "sigma_wide_tradeoff.png")
        fig4.savefig(out_tradeoff, dpi=200, bbox_inches="tight")
        plt.close(fig4)

    print_sigma_summary(summary, sigma_vals, methods)
    print("\n[outputs]")
    print(f"CSV: {out_csv}")
    print(f"Plot(success): {out_success}")
    print(f"Plot(reached): {out_reached}")
    print(f"Plot(fall-down): {out_fall}")
    print(f"Plot(upright_fraction): {out_uf}")
    print(f"Plot(longest_streak): {out_streak}")
    if out_tradeoff is not None:
        print(f"Plot(tradeoff): {out_tradeoff}")

    wb_summary_rows = []
    for sigma in sigma_vals:
        for method in methods:
            s = summary[(sigma, method)]
            wb_summary_rows.append(
                {
                    "sigma_y": float(sigma),
                    "method": str(method),
                    "success_rate": float(s["success_rate"]),
                    "success_low": float(s["success_low"]),
                    "success_high": float(s["success_high"]),
                    "reached_rate": float(s["reached_rate"]),
                    "fall_down_rate": float(s["fall_down_rate"]),
                    "upright_fraction_mean": float(s["upright_fraction_mean"]),
                    "longest_streak_mean": float(s["longest_streak_mean"]),
                    "mean_avg_solve_time": float(s["mean_avg_solve_time"]),
                }
            )
    log_rows_table(wb_run, "episodes", rows)
    log_rows_table(wb_run, "summary", wb_summary_rows)
    log_summary_dict(
        wb_run,
        {
            "num_rows": int(len(rows)),
            "num_sigmas": int(len(sigma_vals)),
            "num_methods": int(len(methods)),
        },
    )
    save_artifacts(
        wb_run,
        [
            out_csv,
            out_success,
            out_reached,
            out_fall,
            out_uf,
            out_streak,
            out_tradeoff,
        ],
    )
    finish_wandb(wb_run)


if __name__ == "__main__":
    main()
