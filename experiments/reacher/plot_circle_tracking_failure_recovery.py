import argparse
import csv
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _ee_xy(d: Dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(d.get("executed_state_traj", np.zeros((0, 10), dtype=float)), dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 10:
        return np.zeros((0, 2), dtype=float)
    return x[:, 4:6] + x[:, 8:10]


def _arr_float(d: Dict[str, np.ndarray], key: str, n: int, default=np.nan) -> np.ndarray:
    if key not in d:
        return np.full(n, default, dtype=float)
    a = np.asarray(d[key]).reshape(-1)
    if a.size == 0:
        return np.full(n, default, dtype=float)
    out = np.full(n, default, dtype=float)
    out[: min(n, a.size)] = np.asarray(a[: min(n, a.size)], dtype=float)
    return out


def _arr_obj(d: Dict[str, np.ndarray], key: str, n: int) -> np.ndarray:
    if key not in d:
        return np.asarray(["unknown"] * n, dtype=object)
    a = np.asarray(d[key]).reshape(-1)
    out = np.asarray(["unknown"] * n, dtype=object)
    out[: min(n, a.size)] = a[: min(n, a.size)]
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    m = np.isfinite(aa) & np.isfinite(bb)
    if np.sum(m) < 3:
        return -np.inf
    aa = aa[m] - np.mean(aa[m])
    bb = bb[m] - np.mean(bb[m])
    da = np.linalg.norm(aa)
    db = np.linalg.norm(bb)
    if da <= 1e-12 or db <= 1e-12:
        return -np.inf
    return float(np.dot(aa, bb) / (da * db))


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
        score = 0.5 * (_corr(txx, exx) + _corr(tyy, eyy))
        scored.append((int(lag), float(score)))
    best_score = max(s for _, s in scored)
    # Avoid periodic aliasing: among near-best correlations, choose smallest |lag|.
    tol = 0.02
    cand = [x for x in scored if x[1] >= best_score - tol]
    best_lag = min(cand, key=lambda x: abs(x[0]))[0]
    return int(best_lag), float(best_score)


def _tracking_amp_ratio(target_xy: np.ndarray, ee_xy: np.ndarray) -> float:
    n = int(min(len(target_xy), len(ee_xy)))
    if n <= 3:
        return np.nan
    tgt = np.asarray(target_xy[:n], dtype=float)
    ee = np.asarray(ee_xy[:n], dtype=float)
    tgt_c = tgt - np.mean(tgt, axis=0, keepdims=True)
    ee_c = ee - np.mean(ee, axis=0, keepdims=True)
    tgt_r = np.linalg.norm(tgt_c, axis=1)
    ee_r = np.linalg.norm(ee_c, axis=1)
    return float(np.mean(ee_r) / max(np.mean(tgt_r), 1e-12))


def _cond_index(slack: np.ndarray, sigma: np.ndarray, fail_mask: np.ndarray) -> np.ndarray:
    log_slack = np.log10(np.maximum(np.asarray(slack, dtype=float), 1e-12))
    log_sigma = np.log10(np.maximum(np.asarray(sigma, dtype=float), 1e-16))
    return log_slack - log_sigma + 0.5 * np.asarray(fail_mask, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot circle-tracking failure/recovery with tracking-first metrics."
    )
    parser.add_argument("--fixed40", type=str, required=True)
    parser.add_argument("--fixed100", type=str, required=True)
    parser.add_argument("--adaptive06", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--circle_omega", type=float, default=np.nan)
    parser.add_argument("--sigma_threshold", type=float, default=1e-6)
    parser.add_argument("--tracking_success_rmse", type=float, default=0.06)
    parser.add_argument("--tracking_success_lag_steps", type=float, default=15.0)
    parser.add_argument("--tracking_success_amp_min", type=float, default=0.70)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    runs = {
        "fixed K=40": _load_npz(args.fixed40),
        "fixed K=100": _load_npz(args.fixed100),
        "adaptive K (rho=0.6)": _load_npz(args.adaptive06),
    }
    colors = {
        "fixed K=40": "tab:blue",
        "fixed K=100": "tab:red",
        "adaptive K (rho=0.6)": "tab:green",
    }

    target_ref = np.asarray(runs["adaptive K (rho=0.6)"].get("target_xy", np.zeros((0, 2))), dtype=float)
    n_ref = int(target_ref.shape[0])
    if n_ref <= 0:
        raise ValueError("No target_xy in adaptive run npz.")

    metrics = {}
    rows = []
    for name, d in runs.items():
        ee = _ee_xy(d)
        n = int(min(n_ref, len(ee)))
        tgt = target_ref[:n]
        ee = ee[:n]
        err = np.linalg.norm(ee - tgt, axis=1) if n > 0 else np.zeros(0, dtype=float)
        slack = _arr_float(d, "slack_inf", n)
        sigma = _arr_float(d, "sigma_min_Ag", n)
        kopt = _arr_float(d, "K_opt", n)
        status = _arr_obj(d, "solver_status", n)
        fail_mask = np.asarray([("optimal" not in str(s).lower()) for s in status], dtype=bool)
        cond = _cond_index(slack, sigma, fail_mask)
        max_lag = min(80, max(1, n // 3))
        if np.isfinite(args.circle_omega) and abs(float(args.circle_omega)) > 1e-9:
            half_period_steps = int(np.pi / abs(float(args.circle_omega)))
            max_lag = min(max_lag, max(5, half_period_steps))
        lag_steps, lag_score = estimate_phase_lag_steps(tgt, ee, max_lag=max_lag)
        lag_rad = float(args.circle_omega) * float(lag_steps) if np.isfinite(args.circle_omega) else np.nan
        amp_ratio = _tracking_amp_ratio(tgt, ee)
        rmse = float(np.sqrt(np.mean(err * err))) if err.size > 0 else np.nan
        t_success = int(
            (rmse <= float(args.tracking_success_rmse))
            and (abs(lag_steps) <= float(args.tracking_success_lag_steps))
            and (amp_ratio >= float(args.tracking_success_amp_min))
        )

        metrics[name] = {
            "target": tgt,
            "ee": ee,
            "err": err,
            "slack": slack,
            "sigma": sigma,
            "kopt": kopt,
            "fail_mask": fail_mask,
            "cond": cond,
            "lag_steps": lag_steps,
            "lag_rad": lag_rad,
            "lag_score": lag_score,
            "amp_ratio": amp_ratio,
            "rmse": rmse,
            "tracking_success": t_success,
        }
        rows.append(
            {
                "method": name,
                "tracking_success": int(t_success),
                "lag_steps": int(lag_steps),
                "lag_radian_est": float(lag_rad),
                "lag_corr_score": float(lag_score),
                "amp_ratio": float(amp_ratio),
                "err_rmse": float(rmse),
                "err_final": float(err[-1]) if err.size > 0 else np.nan,
                "slack_max": float(np.nanmax(slack)) if slack.size > 0 else np.nan,
                "sigma_min": float(np.nanmin(sigma)) if sigma.size > 0 else np.nan,
                "solver_fail_rate": float(np.mean(fail_mask.astype(float))) if fail_mask.size > 0 else np.nan,
                "cond_index_mean": float(np.nanmean(cond)) if cond.size > 0 else np.nan,
            }
        )

    fig, axs = plt.subplots(6, 1, figsize=(12, 16), dpi=150, sharex=False)

    # Step 1 - failure exposure
    for name, m in metrics.items():
        axs[0].plot(np.arange(len(m["cond"])), m["cond"], lw=1.5, color=colors[name], label=name)
    axs[0].set_title("Step 1 - Conditioning Failure Exposure (composite index)")
    axs[0].set_ylabel("log10(slack)-log10(sigma)+fail")
    axs[0].set_xlabel("Step")
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(loc="best")

    for name, m in metrics.items():
        axs[1].plot(np.arange(len(m["sigma"])), m["sigma"], lw=1.5, color=colors[name], label=name)
    axs[1].axhline(float(args.sigma_threshold), color="k", ls="--", lw=1.0, alpha=0.7)
    axs[1].set_title("Step-wise Minimum Sigma")
    axs[1].set_ylabel(r"$\sigma_{\min}(A_g)$")
    axs[1].set_xlabel("Step")
    axs[1].set_yscale("log")
    axs[1].grid(True, alpha=0.25)

    for name, m in metrics.items():
        axs[2].plot(np.arange(len(m["slack"])), m["slack"], lw=1.5, color=colors[name], label=name)
    axs[2].set_title("Step-wise Slack Norm")
    axs[2].set_ylabel(r"$\|\sigma_y\|_\infty$")
    axs[2].set_xlabel("Step")
    axs[2].set_yscale("log")
    axs[2].grid(True, alpha=0.25)

    # Step 2 - recovery evidence
    axs[3].plot(target_ref[:, 0], target_ref[:, 1], color="k", lw=2.0, ls="--", label="target (circle)")
    for name, m in metrics.items():
        lbl = f"{name} (lag={m['lag_steps']}, amp={m['amp_ratio']:.2f})"
        axs[3].plot(m["ee"][:, 0], m["ee"][:, 1], lw=1.5, color=colors[name], label=lbl)
    axs[3].set_title("Step 2 - Recovery on Circle Tracking (Path + Phase)")
    axs[3].set_xlabel("x")
    axs[3].set_ylabel("y")
    axs[3].grid(True, alpha=0.25)
    axs[3].axis("equal")
    axs[3].legend(loc="best")

    for name, m in metrics.items():
        axs[4].plot(np.arange(len(m["err"])), m["err"], lw=1.5, color=colors[name], label=name)
    axs[4].axhline(float(args.tracking_success_rmse), color="k", ls="--", lw=1.0, alpha=0.7)
    axs[4].set_title("Tracking Error by Step (state status)")
    axs[4].set_ylabel(r"$\|e_{ee}\|$")
    axs[4].set_xlabel("Step")
    axs[4].grid(True, alpha=0.25)

    for name, m in metrics.items():
        axs[5].plot(np.arange(len(m["kopt"])), m["kopt"], lw=1.5, color=colors[name], label=name)
    axs[5].set_title("K Allocation (Minimal-K vs Adaptive Recovery)")
    axs[5].set_ylabel("K_opt")
    axs[5].set_xlabel("Step")
    axs[5].grid(True, alpha=0.25)

    fig.tight_layout()
    fig_path = os.path.join(args.outdir, "circle_failure_recovery_phase_lag.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    summary_path = os.path.join(args.outdir, "phase_lag_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved figure: {fig_path}")
    print(f"Saved table : {summary_path}")


if __name__ == "__main__":
    main()
