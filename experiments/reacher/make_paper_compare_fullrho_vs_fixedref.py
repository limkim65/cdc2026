import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


RHO_TARGETS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5]


def _f(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _read_csv(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: List[dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)


def _is_fail_status(v) -> bool:
    s = str(v).strip().lower()
    return ("optimal" not in s) and (len(s) > 0)


def _safe_arr(d: dict, key: str, n_default: int = 0) -> np.ndarray:
    if key not in d:
        return np.full(int(max(0, n_default)), np.nan, dtype=float)
    arr = np.asarray(d[key])
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _trim(*arrs: np.ndarray) -> Tuple[np.ndarray, ...]:
    n = min((len(a) for a in arrs if a is not None), default=0)
    if n <= 0:
        return tuple(np.asarray([]) for _ in arrs)
    return tuple(np.asarray(a)[:n] for a in arrs)


def _fmt_rho(r: float) -> str:
    return f"{r:g}"


@dataclass
class RecoveryCandidate:
    path: str
    rho: float
    seed: int
    score: float
    collapse_idx: int
    kup_idx: int
    rec_idx: int


def _select_fixed_reference(by_k_rows: List[dict]) -> dict:
    rows = []
    for r in by_k_rows:
        rows.append(
            {
                "K": _f(r.get("K")),
                "success_rate": _f(r.get("success_rate")),
                "solve_ms": _f(r.get("median_solve_ms")),
                "sigma": _f(r.get("median_sigma_min_Ag")),
            }
        )
    rows = [r for r in rows if np.isfinite(r["K"]) and np.isfinite(r["success_rate"]) and np.isfinite(r["solve_ms"])]
    if not rows:
        raise RuntimeError("No valid fixed-K rows in summary_by_K.csv")

    max_succ = max(r["success_rate"] for r in rows)
    cands = [r for r in rows if abs(r["success_rate"] - max_succ) <= 1e-12]
    kmin = min(r["K"] for r in cands)
    cands2 = [r for r in cands if abs(r["K"] - kmin) <= 1e-12]
    best = min(cands2, key=lambda z: z["solve_ms"]) if cands2 else min(cands, key=lambda z: z["solve_ms"])
    return best


def _compute_tau_turning(by_k_rows: List[dict]) -> Tuple[float, float]:
    items = []
    for r in by_k_rows:
        k = _f(r.get("K"))
        s = _f(r.get("median_sigma_min_Ag"))
        if np.isfinite(k) and np.isfinite(s):
            items.append((k, s))
    if not items:
        return np.nan, np.nan
    items.sort(key=lambda t: t[0])
    ks = np.array([x[0] for x in items], dtype=float)
    ss = np.array([x[1] for x in items], dtype=float)

    # Turning point: first local minimum where trend changes from down to up.
    k_turn = np.nan
    tau = np.nan
    if len(ss) >= 3:
        d = np.diff(ss)
        for i in range(1, len(ss) - 1):
            if np.isfinite(d[i - 1]) and np.isfinite(d[i]) and d[i - 1] < 0 and d[i] > 0:
                k_turn = ks[i]
                tau = ss[i]
                break
    if not np.isfinite(tau):
        i = int(np.nanargmin(ss))
        k_turn = ks[i]
        tau = ss[i]
    return float(tau), float(k_turn)


def _build_adaptive_full(low_rows: List[dict], high_rows: List[dict]) -> List[dict]:
    merged: Dict[float, dict] = {}
    for r in low_rows:
        rho = _f(r.get("rho"))
        if not np.isfinite(rho):
            continue
        merged[rho] = {
            "rho": rho,
            "runs": _f(r.get("runs")),
            "adaptive_success": _f(r.get("success_rate")),
            "adaptive_solve_ms": _f(r.get("median_solve_ms")),
            "adaptive_kopt_median": _f(r.get("median_k_opt_mean")),
            "adaptive_kopt_p90": _f(r.get("median_k_opt_p90")),
            "adaptive_sigma_min": _f(r.get("median_sigma_min_Ag")),
            "adaptive_infeasible_rate": _f(r.get("median_infeasible_rate")),
            "source": "waypoint_reconstructed",
        }

    for r in high_rows:
        rho = _f(r.get("rho"))
        if not np.isfinite(rho):
            continue
        merged[rho] = {
            "rho": rho,
            "runs": _f(r.get("runs")),
            "adaptive_success": _f(r.get("success_rate")),
            "adaptive_solve_ms": _f(r.get("median_solve_ms")),
            "adaptive_kopt_median": _f(r.get("median_k_opt_mean")),
            "adaptive_kopt_p90": _f(r.get("median_k_opt_p90")),
            "adaptive_sigma_min": _f(r.get("median_sigma_min_Ag")),
            "adaptive_infeasible_rate": _f(r.get("median_infeasible_rate")),
            "source": "adaptive_summary",
        }

    out = []
    for rho in sorted(merged.keys()):
        out.append(merged[rho])
    return out


def _collect_fixed_solver_fail_by_k(fixed_summary_rows: List[dict]) -> Dict[float, float]:
    bucket: Dict[float, List[float]] = {}
    for r in fixed_summary_rows:
        k = _f(r.get("K"))
        v = _f(r.get("solver_fail_rate"))
        if np.isfinite(k) and np.isfinite(v):
            bucket.setdefault(k, []).append(v)
    return {k: float(np.median(vs)) for k, vs in bucket.items() if len(vs) > 0}


def _find_recovery_candidates(npz_paths: List[str], tau: float, k_step_min: float = 10.0) -> List[RecoveryCandidate]:
    out: List[RecoveryCandidate] = []
    for p in npz_paths:
        try:
            d0 = np.load(p, allow_pickle=True)
            d = {k: d0[k] for k in d0.files}
        except Exception:
            continue

        rho = _f(d.get("rho", np.nan))
        seed = int(_f(d.get("seed", np.nan), default=-1))

        sigma = _safe_arr(d, "sigma_min_Ag")
        kopt = _safe_arr(d, "K_opt", n_default=len(sigma))
        solve = _safe_arr(d, "solve_time_ms", n_default=len(sigma))
        status = _safe_arr(d, "solver_status", n_default=len(sigma))
        sigma, kopt, solve, status = _trim(sigma, kopt, solve, status)

        n = len(sigma)
        if n < 40:
            continue

        fail = np.array([_is_fail_status(x) for x in status], dtype=bool)
        sf = np.asarray(sigma, dtype=float)
        kf = np.asarray(kopt, dtype=float)

        # Relative collapse detection: large drop from early baseline to local minimum.
        head = max(10, n // 6)
        base = float(np.nanmedian(sf[:head])) if np.any(np.isfinite(sf[:head])) else np.nan
        if not np.isfinite(base):
            continue

        tail_limit = max(head + 5, int(0.85 * n))
        sf_local = np.asarray(sf[:tail_limit], dtype=float)
        if not np.any(np.isfinite(sf_local)):
            continue

        # Prefer early collapse crossing (more room for K-up/recovery), fallback to local minimum.
        cross = np.where(np.isfinite(sf_local) & (sf_local <= base * 0.5))[0]
        if cross.size > 0:
            c0 = int(cross[0])
        else:
            c0 = int(np.nanargmin(sf_local))

        s_c = float(sf[c0])
        if not np.isfinite(s_c) or s_c <= 0:
            continue

        drop_ratio = base / s_c
        if drop_ratio < 1.1:
            continue

        w_up_end = min(n, c0 + 80)
        if w_up_end <= c0 + 1:
            continue
        k_base = float(kf[c0]) if np.isfinite(kf[c0]) else np.nan
        if not np.isfinite(k_base):
            continue
        post_k = np.asarray(kf[c0:w_up_end], dtype=float)
        up_rel = np.where(np.isfinite(post_k) & (post_k >= k_base + k_step_min))[0]
        if up_rel.size == 0:
            continue
        kup_idx = int(c0 + up_rel[0])

        # Recovery: sigma rebounds after K increase.
        w_rec_end = min(n, kup_idx + 120)
        post_s = np.asarray(sf[kup_idx:w_rec_end], dtype=float)
        if not np.any(np.isfinite(post_s)):
            continue
        rec_target = max(s_c * 1.3, base * 0.25)
        rec_rel = np.where(np.isfinite(post_s) & (post_s >= rec_target))[0]
        if rec_rel.size == 0:
            continue
        rec_idx = int(kup_idx + rec_rel[0])

        # Optional improvement signals (used for ranking, not hard filter).
        b0 = max(0, c0 - 30)
        b1 = c0
        a0 = rec_idx
        a1 = min(n, rec_idx + 30)
        fail_before = float(np.mean(fail[b0:b1])) if b1 > b0 else 0.0
        fail_after = float(np.mean(fail[a0:a1])) if a1 > a0 else 0.0
        fail_gain = max(0.0, fail_before - fail_after)

        s_b0 = max(0, c0 - 20)
        s_b1 = min(n, c0 + 20)
        s_a0 = rec_idx
        s_a1 = min(n, rec_idx + 30)
        solve_before = float(np.nanmedian(np.asarray(solve[s_b0:s_b1], dtype=float))) if s_b1 > s_b0 else np.nan
        solve_after = float(np.nanmedian(np.asarray(solve[s_a0:s_a1], dtype=float))) if s_a1 > s_a0 else np.nan
        solve_gain = 0.0
        if np.isfinite(solve_before) and np.isfinite(solve_after) and solve_before > 1e-9:
            solve_gain = max(0.0, (solve_before - solve_after) / solve_before)

        k_gain = float(np.nanmax(post_k) - k_base) if np.any(np.isfinite(post_k)) else 0.0
        rec_gain = float(np.nanmax(post_s) / s_c) if s_c > 0 and np.any(np.isfinite(post_s)) else 1.0

        # Score prioritizes clear collapse/recovery + practical gains.
        score = 1.0 * drop_ratio + 0.4 * rec_gain + 0.003 * max(0.0, k_gain) + 2.0 * fail_gain + 0.6 * solve_gain

        out.append(
            RecoveryCandidate(
                path=p,
                rho=float(rho),
                seed=int(seed),
                score=float(score),
                collapse_idx=c0,
                kup_idx=kup_idx,
                rec_idx=rec_idx,
            )
        )

    out.sort(key=lambda z: z.score, reverse=True)
    return out


def _choose_diverse_candidates(cands: List[RecoveryCandidate], n_pick: int = 4) -> List[RecoveryCandidate]:
    picks: List[RecoveryCandidate] = []
    used_rho: set = set()

    # First pass: unique rho.
    for c in cands:
        key = round(c.rho, 6)
        if key in used_rho:
            continue
        picks.append(c)
        used_rho.add(key)
        if len(picks) >= n_pick:
            return picks

    # Second pass: fill remaining slots.
    for c in cands:
        if c in picks:
            continue
        picks.append(c)
        if len(picks) >= n_pick:
            return picks
    return picks


def _plot_fig1_failure(
    out_path: str,
    by_k_rows: List[dict],
    fixed_fail_by_k: Dict[float, float],
    tau: float,
    k_turn: float,
    fixed_ref_k: float,
) -> None:
    rows = []
    for r in by_k_rows:
        k = _f(r.get("K"))
        sig = _f(r.get("median_sigma_min_Ag"))
        solv = _f(r.get("median_solve_ms"))
        succ = _f(r.get("success_rate"))
        if np.isfinite(k):
            rows.append((k, sig, solv, succ, fixed_fail_by_k.get(k, np.nan)))
    rows.sort(key=lambda t: t[0])
    if not rows:
        return

    k = np.array([r[0] for r in rows], dtype=float)
    sig = np.array([r[1] for r in rows], dtype=float)
    solv = np.array([r[2] for r in rows], dtype=float)
    succ = np.array([r[3] for r in rows], dtype=float)
    fail = np.array([r[4] for r in rows], dtype=float)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True, dpi=160)

    axs[0].plot(k, sig, marker="o", lw=1.6, color="tab:blue")
    if np.all(np.isfinite(sig)) and np.nanmin(sig) > 0:
        axs[0].set_yscale("log")
    if np.isfinite(tau):
        axs[0].axhline(tau, color="k", ls="--", lw=1.0, alpha=0.8, label=f"tau={tau:.2e}")
    if np.isfinite(k_turn):
        axs[0].axvline(k_turn, color="tab:red", ls="--", lw=1.1, alpha=0.9, label=f"collapse K~{k_turn:g}")
    axs[0].set_ylabel(r"median $\sigma_{\min}(A_g)$")
    axs[0].set_title("Figure 1. Failure mode (fixed-K reference)")
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(loc="best")

    axs[1].plot(k, solv, marker="o", lw=1.6, color="tab:orange", label="median solve ms")
    axs[1].set_ylabel("Solve time [ms]")
    axs[1].grid(True, alpha=0.25)
    if np.isfinite(k_turn):
        axs[1].axvline(k_turn, color="tab:red", ls="--", lw=1.1, alpha=0.9)

    axs[2].plot(k, 1.0 - succ, marker="o", lw=1.5, color="tab:red", label="failure rate = 1-success")
    m = np.isfinite(fail)
    if np.any(m):
        axs[2].plot(k[m], fail[m], marker="x", lw=1.2, color="tab:purple", label="median solver_fail_rate")
    if np.isfinite(k_turn):
        axs[2].axvline(k_turn, color="tab:red", ls="--", lw=1.1, alpha=0.9)
    if np.isfinite(fixed_ref_k):
        axs[2].axvline(fixed_ref_k, color="tab:green", ls=":", lw=1.2, alpha=0.9, label=f"fixed ref K*={fixed_ref_k:g}")
    axs[2].set_ylabel("Failure / infeasible proxy")
    axs[2].set_xlabel("K")
    axs[2].set_ylim(-0.02, 1.02)
    axs[2].grid(True, alpha=0.25)
    axs[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_fig2_recovery(out_path: str, picks: List[RecoveryCandidate], tau: float) -> None:
    if not picks:
        return
    ncol = len(picks)
    fig, axs = plt.subplots(3, ncol, figsize=(4.6 * ncol, 8.5), sharex="col", dpi=160)
    if ncol == 1:
        axs = np.asarray(axs).reshape(3, 1)

    for j, c in enumerate(picks):
        d0 = np.load(c.path, allow_pickle=True)
        d = {k: d0[k] for k in d0.files}
        sigma = _safe_arr(d, "sigma_min_Ag")
        kopt = _safe_arr(d, "K_opt", n_default=len(sigma))
        solve = _safe_arr(d, "solve_time_ms", n_default=len(sigma))
        status = _safe_arr(d, "solver_status", n_default=len(sigma))
        sigma, kopt, solve, status = _trim(sigma, kopt, solve, status)
        t = np.arange(len(sigma))
        fail = np.array([_is_fail_status(x) for x in status], dtype=bool)

        ax0 = axs[0, j]
        ax1 = axs[1, j]
        ax2 = axs[2, j]

        ax0.plot(t, sigma, lw=1.4, color="tab:blue")
        if np.nanmin(sigma[np.isfinite(sigma)]) > 0:
            ax0.set_yscale("log")
        if np.isfinite(tau):
            ax0.axhline(tau, color="k", ls="--", lw=1.0, alpha=0.8)
        ax0.axvline(c.collapse_idx, color="tab:red", ls="--", lw=1.0)
        ax0.axvline(c.kup_idx, color="tab:green", ls="--", lw=1.0)
        ax0.axvline(c.rec_idx, color="tab:purple", ls="--", lw=1.0)
        ax0.set_title(f"rho={c.rho:g}, seed={c.seed}")
        if j == 0:
            ax0.set_ylabel(r"$\sigma_{\min}(A_g)$")
        ax0.grid(True, alpha=0.25)

        ax1.plot(t, kopt, lw=1.4, color="tab:orange")
        ax1.axvline(c.collapse_idx, color="tab:red", ls="--", lw=1.0)
        ax1.axvline(c.kup_idx, color="tab:green", ls="--", lw=1.0)
        ax1.axvline(c.rec_idx, color="tab:purple", ls="--", lw=1.0)
        if j == 0:
            ax1.set_ylabel("K(t)")
        ax1.grid(True, alpha=0.25)

        ax2.plot(t, solve, lw=1.2, color="tab:gray", label="solve ms")
        if np.any(fail):
            ax2.scatter(np.where(fail)[0], solve[fail], s=10, color="tab:red", alpha=0.6, label="solver fail")
        ax2.axvline(c.collapse_idx, color="tab:red", ls="--", lw=1.0)
        ax2.axvline(c.kup_idx, color="tab:green", ls="--", lw=1.0)
        ax2.axvline(c.rec_idx, color="tab:purple", ls="--", lw=1.0)
        if j == 0:
            ax2.set_ylabel("Solve time / fail")
            ax2.legend(loc="best")
        ax2.set_xlabel("t")
        ax2.grid(True, alpha=0.25)

    fig.suptitle("Figure 2. Recovery (adaptive-K): collapse -> K up -> sigma recovery", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_fig3_min_compute(
    out_path: str,
    by_k_rows: List[dict],
    adaptive_rows: List[dict],
    fixed_ref_k: float,
) -> None:
    rows = []
    for r in by_k_rows:
        k = _f(r.get("K"))
        succ = _f(r.get("success_rate"))
        solve = _f(r.get("median_solve_ms"))
        if np.isfinite(k) and np.isfinite(succ) and np.isfinite(solve):
            rows.append((k, succ, solve))
    rows.sort(key=lambda t: t[0])
    if not rows:
        return

    k = np.array([r[0] for r in rows], dtype=float)
    succ = np.array([r[1] for r in rows], dtype=float)
    solve = np.array([r[2] for r in rows], dtype=float)

    fig, ax1 = plt.subplots(1, 1, figsize=(11, 5), dpi=160)
    ax2 = ax1.twinx()

    l1 = ax1.plot(k, succ, color="tab:blue", lw=2.0, marker="o", label="fixed success rate")
    l2 = ax2.plot(k, solve, color="tab:orange", lw=2.0, marker="o", label="fixed solve ms")

    cmap = plt.cm.viridis
    rho_vals = [_f(r.get("rho")) for r in adaptive_rows]
    finite_rho = np.array([r for r in rho_vals if np.isfinite(r)], dtype=float)
    rmin = float(np.nanmin(finite_rho)) if finite_rho.size > 0 else 0.0
    rmax = float(np.nanmax(finite_rho)) if finite_rho.size > 0 else 1.0

    for r in adaptive_rows:
        rho = _f(r.get("rho"))
        km = _f(r.get("adaptive_kopt_median"))
        kp = _f(r.get("adaptive_kopt_p90"))
        if not np.isfinite(rho):
            continue
        c = cmap((rho - rmin) / max(1e-12, rmax - rmin))
        if np.isfinite(km):
            ax1.axvline(km, color=c, lw=1.0, alpha=0.35)
        if np.isfinite(kp):
            ax1.axvline(kp, color=c, lw=0.9, ls="--", alpha=0.25)

    if np.isfinite(fixed_ref_k):
        ax1.axvline(fixed_ref_k, color="tab:green", lw=1.5, ls=":", alpha=0.9, label=f"fixed ref K*={fixed_ref_k:g}")

    ax1.set_xlabel("K (fixed sweep)")
    ax1.set_ylabel("Success rate", color="tab:blue")
    ax2.set_ylabel("Solve time [ms]", color="tab:orange")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.25)
    ax1.set_title("Figure 3. Minimal computation: adaptive Kopt overlays on fixed-K sweep")

    # Compact legend
    handles = l1 + l2
    labels = [h.get_label() for h in handles]
    handles2, labels2 = ax1.get_legend_handles_labels()
    handles.extend(handles2)
    labels.extend(labels2)
    ax1.legend(handles, labels, loc="lower right")

    # colorbar for rho
    norm = plt.Normalize(vmin=rmin, vmax=rmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.025, pad=0.02)
    cbar.set_label("adaptive rho (median: solid, p90: dashed)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_notes(
    out_path: str,
    tau: float,
    k_turn: float,
    fixed_ref: dict,
    adaptive_rows: List[dict],
    selected_paths: List[RecoveryCandidate],
    rho_set_ok: bool,
    runs_ok: bool,
) -> None:
    lines: List[str] = []
    lines.append("# Analysis Notes: full-rho adaptive vs fixed reference")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Adaptive uses rho=0.001~1.5 combined from two summaries.")
    lines.append("- Fixed reference uses rho=0.005 summary_by_K sweep.")
    lines.append("")
    lines.append("## Data provenance warning")
    lines.append("- low-rho (0.001~0.2): reconstructed from waypoint-tagged runs.")
    lines.append("- high-rho (0.3~1.5): adaptive summary rows.")
    lines.append("- Interpretation of absolute levels should be conservative across these sources.")
    lines.append("")
    lines.append("## Fixed reference")
    lines.append(f"- K*: {fixed_ref.get('K', np.nan):g}")
    lines.append(f"- success_rate(K*): {fixed_ref.get('success_rate', np.nan):.4f}")
    lines.append(f"- median_solve_ms(K*): {fixed_ref.get('solve_ms', np.nan):.3f}")
    lines.append(f"- median_sigma_min_Ag(K*): {fixed_ref.get('sigma', np.nan):.3e}")
    lines.append("")
    lines.append("## Tau and collapse marker")
    lines.append(f"- tau (from fixed turning/min point): {tau:.3e}")
    lines.append(f"- K_turn: {k_turn:g}")
    lines.append("")
    lines.append("## Validation")
    lines.append(f"- rho set exact match: {rho_set_ok}")
    lines.append(f"- runs==10 for all rho: {runs_ok}")
    lines.append("")
    lines.append("## Selected recovery trajectories")
    if not selected_paths:
        lines.append("- none (criteria not met)")
    else:
        for c in selected_paths:
            lines.append(
                f"- rho={c.rho:g}, seed={c.seed}, collapse={c.collapse_idx}, K-up={c.kup_idx}, recover={c.rec_idx}, score={c.score:.3f}"
            )
    lines.append("")
    lines.append("## Paper-aligned takeaway template")
    lines.append("- Adaptive uses near-threshold K allocation over rho while avoiding unnecessary large fixed K.")
    lines.append("- Failures are concentrated near low sigma_min; K increase aligns with sigma_min recovery and reduced failures.")
    lines.append("- Mixed source condition (waypoint/non-waypoint) is explicitly reported.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper figures for adaptive(Kmax=2000) vs fixed reference.")
    parser.add_argument(
        "--adaptive_dir",
        type=str,
        default="logs/reacher_cdc_adaptive_k_noise0_Kmax2000_seed10",
    )
    parser.add_argument(
        "--fixed_dir",
        type=str,
        default="logs/reacher/fixed_k_sweep_dir/reacher_cdc_fixed_seed10",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="logs/reacher_cdc_adaptive_k_noise0_Kmax2000_seed10/paper_compare_fullrho_vs_fixedref",
    )
    parser.add_argument("--n_recovery", type=int, default=4)
    args = parser.parse_args()

    low_csv = os.path.join(args.adaptive_dir, "command_rho_results", "summary_by_rho_command.csv")
    high_csv = os.path.join(args.adaptive_dir, "summary_by_rho.csv")
    fixed_by_k_csv = os.path.join(args.fixed_dir, "summary_by_K.csv")
    fixed_summary_csv = os.path.join(args.fixed_dir, "summary.csv")

    low_rows = _read_csv(low_csv)
    high_rows = _read_csv(high_csv)
    fixed_by_k_rows = _read_csv(fixed_by_k_csv)
    fixed_summary_rows = _read_csv(fixed_summary_csv)

    if not low_rows:
        raise RuntimeError(f"Missing/empty low-rho summary: {low_csv}")
    if not high_rows:
        raise RuntimeError(f"Missing/empty high-rho summary: {high_csv}")
    if not fixed_by_k_rows:
        raise RuntimeError(f"Missing/empty fixed summary_by_K: {fixed_by_k_csv}")

    os.makedirs(args.outdir, exist_ok=True)

    fixed_ref = _select_fixed_reference(fixed_by_k_rows)
    tau, k_turn = _compute_tau_turning(fixed_by_k_rows)

    adaptive_rows = _build_adaptive_full(low_rows, high_rows)

    # Validation checks.
    got_rhos = sorted([round(_f(r.get("rho")), 12) for r in adaptive_rows if np.isfinite(_f(r.get("rho")))])
    tar_rhos = sorted([round(x, 12) for x in RHO_TARGETS])
    rho_set_ok = got_rhos == tar_rhos
    runs_ok = True
    for r in adaptive_rows:
        rv = _f(r.get("runs"))
        if np.isfinite(rv) and int(round(rv)) != 10:
            runs_ok = False

    out_table_rows = []
    for r in adaptive_rows:
        asucc = _f(r.get("adaptive_success"))
        asolve = _f(r.get("adaptive_solve_ms"))
        delta_s = asucc - fixed_ref["success_rate"] if np.isfinite(asucc) else np.nan
        delta_t = (
            (asolve / fixed_ref["solve_ms"] - 1.0) * 100.0
            if np.isfinite(asolve) and np.isfinite(fixed_ref["solve_ms"]) and fixed_ref["solve_ms"] > 1e-12
            else np.nan
        )
        notes = [str(r.get("source", ""))]
        runs = _f(r.get("runs"))
        if np.isfinite(runs):
            notes.append(f"n={int(round(runs))}")
        if not rho_set_ok:
            notes.append("rho_set_mismatch")

        out_table_rows.append(
            {
                "rho": _f(r.get("rho")),
                "adaptive_success": asucc,
                "adaptive_solve_ms": asolve,
                "adaptive_kopt_median": _f(r.get("adaptive_kopt_median")),
                "adaptive_kopt_p90": _f(r.get("adaptive_kopt_p90")),
                "adaptive_sigma_min": _f(r.get("adaptive_sigma_min")),
                "fixed_ref_success": fixed_ref["success_rate"],
                "fixed_ref_solve_ms": fixed_ref["solve_ms"],
                "fixed_ref_K_star": fixed_ref["K"],
                "delta_success": delta_s,
                "delta_solve_ms_pct": delta_t,
                "notes": ";".join(notes),
            }
        )

    table_path = os.path.join(args.outdir, "table_rho_full_vs_fixedref.csv")
    _write_csv(
        table_path,
        out_table_rows,
        fieldnames=[
            "rho",
            "adaptive_success",
            "adaptive_solve_ms",
            "adaptive_kopt_median",
            "adaptive_kopt_p90",
            "adaptive_sigma_min",
            "fixed_ref_success",
            "fixed_ref_solve_ms",
            "fixed_ref_K_star",
            "delta_success",
            "delta_solve_ms_pct",
            "notes",
        ],
    )

    # Figure 1.
    fixed_fail_by_k = _collect_fixed_solver_fail_by_k(fixed_summary_rows)
    fig1_path = os.path.join(args.outdir, "fig1_failure_fixed_reference.png")
    _plot_fig1_failure(
        fig1_path,
        fixed_by_k_rows,
        fixed_fail_by_k,
        tau=tau,
        k_turn=k_turn,
        fixed_ref_k=fixed_ref["K"],
    )

    # Figure 2.
    npz_paths = [
        os.path.join(args.adaptive_dir, x)
        for x in os.listdir(args.adaptive_dir)
        if x.endswith(".npz") and x.startswith("run_adaptive_rho")
    ]
    cands = _find_recovery_candidates(npz_paths, tau=tau, k_step_min=10.0)
    picks = _choose_diverse_candidates(cands, n_pick=max(1, int(args.n_recovery)))
    fig2_path = os.path.join(args.outdir, "fig2_recovery_adaptive_fullrho.png")
    _plot_fig2_recovery(fig2_path, picks, tau=tau)

    # Figure 3.
    fig3_path = os.path.join(args.outdir, "fig3_minimal_computation_fullrho.png")
    _plot_fig3_min_compute(fig3_path, fixed_by_k_rows, adaptive_rows, fixed_ref_k=fixed_ref["K"])

    # Notes.
    notes_path = os.path.join(args.outdir, "analysis_notes.md")
    _write_notes(
        notes_path,
        tau=tau,
        k_turn=k_turn,
        fixed_ref=fixed_ref,
        adaptive_rows=adaptive_rows,
        selected_paths=picks,
        rho_set_ok=rho_set_ok,
        runs_ok=runs_ok,
    )

    print(f"Saved table: {table_path}")
    print(f"Saved figure: {fig1_path}")
    print(f"Saved figure: {fig2_path}")
    print(f"Saved figure: {fig3_path}")
    print(f"Saved notes: {notes_path}")
    print(f"tau={tau:.6e}, k_turn={k_turn:g}, fixed_K_star={fixed_ref['K']:g}")
    print(f"rho_set_ok={rho_set_ok}, runs_ok={runs_ok}, recovery_picks={len(picks)}")


if __name__ == "__main__":
    main()
