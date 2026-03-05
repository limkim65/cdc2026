import argparse
import csv
import os
import pickle
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


K_SMALL = 50
K_LARGE = 200
RHO_LIST_DEFAULT = []
RHO_BEST = None
LAMBDA_G_LIST = []
BOOTSTRAP_N = 1000
SSE_WINDOW = 20
REACH_EPS = 0.01
REACH_HOLD = 20
SIGMA_THRESHOLD = None


ALIASES = {
    "method": [
        "method",
        "mode",
        "controller",
        "algo",
        "strategy",
    ],
    "run_id": [
        "run_id",
        "episode",
        "trial",
        "seed_run",
        "npz_path",
    ],
    "step": ["step", "t", "time_idx"],
    "K": ["K", "k", "num_cols", "n_cols"],
    "K_opt": ["K_opt", "k_opt", "Kopt"],
    "rho": ["rho", "RHO"],
    "lambda_g": ["lambda_g", "lambda", "lam_g", "regularizer_cost_g_1"],
    "sigma_min_Ag": ["sigma_min_Ag", "sigma_min_ag", "sigma_min_Ag_episode_min"],
    "loc": ["loc", "localness", "Loc", "LocK", "localness_mean"],
    "slack_norm": ["slack_norm", "slack_l2", "sigma_y_norm", "slack_mean"],
    "slack_max": ["slack_max", "slack_inf", "sigma_y_max", "slack_norm_max"],
    "kkt_residual": ["kkt_residual", "kkt_res", "kkt", "kkt_max", "kkt_res_max"],
    "cost_total": ["cost_total", "total_cost", "cost", "stage_cost_sum"],
    "ee_rmse": ["ee_rmse", "rmse_ee", "tracking_rmse", "final_tracking_error"],
    "tracking_error": ["tracking_error", "ee_error", "error_ee", "e_ee"],
    "success": ["success", "is_success", "success_flag"],
    "success_rate": ["success_rate"],
    "solve_time_ms": ["solve_time_ms", "mean_solve_time_ms", "solve_ms", "solver_time_ms"],
    "noise_level": ["noise_level", "disturbance_level", "noise", "disturbance"],
}


def _norm_key(k: str) -> str:
    return "".join(ch.lower() for ch in str(k) if ch.isalnum())


def _parse_float(v, default=np.nan) -> float:
    if v is None:
        return float(default)
    if isinstance(v, (list, tuple, dict)):
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _parse_method(v: Optional[str], row: dict) -> str:
    s = str(v).lower() if v is not None else ""
    src = str(row.get("_source_file", "")).lower()
    src_name = os.path.basename(src)
    if "adaptive" in s:
        return "adaptiveK"
    if "lambda" in s:
        return "fixedK_lambda"
    if "fixed" in s:
        return "fixedK"
    if ("adaptive" in src) or ("rho_sweep" in src) or ("run_adaptive" in src_name):
        return "adaptiveK"
    if ("lambda" in src) or ("fixedk_lambda" in src) or ("lam" in src_name):
        return "fixedK_lambda"
    if ("k_sweep" in src) or ("fixedk" in src) or ("fixed_k" in src) or ("cdc_fixed" in src):
        return "fixedK"
    rho = _parse_float(row.get("rho", np.nan))
    k_fixed = _parse_float(row.get("K", np.nan))
    kopt = _parse_float(row.get("K_opt", np.nan))
    lam = _parse_float(row.get("lambda_g", np.nan))
    if np.isfinite(lam):
        return "fixedK_lambda"
    if np.isfinite(k_fixed) and not np.isfinite(rho):
        return "fixedK"
    if np.isfinite(rho) or np.isfinite(kopt):
        return "adaptiveK"
    return "fixedK"


def _build_alias_map(keys: List[str]) -> Dict[str, str]:
    key_lookup = {_norm_key(k): k for k in keys}
    out = {}
    for std, alias_list in ALIASES.items():
        found = None
        for a in alias_list:
            na = _norm_key(a)
            if na in key_lookup:
                found = key_lookup[na]
                break
        if found is not None:
            out[std] = found
    return out


def _read_csv(path: str) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _records_from_dict_of_arrays(data: Dict, source_id: str) -> List[dict]:
    keys = list(data.keys())
    arrs = {}
    for k in keys:
        try:
            arrs[k] = np.asarray(data[k], dtype=object)
        except Exception:
            arrs[k] = data[k]

    T = 1
    for v in arrs.values():
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 1:
            T = max(T, int(v.size))

    records = []
    for t in range(T):
        row = {"run_id": source_id, "step": t}
        for k, v in arrs.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 0:
                    row[k] = v.item()
                elif v.ndim == 1:
                    row[k] = v[t] if t < v.size else v[-1]
                else:
                    # Keep only scalar-like metrics from >=2D arrays.
                    continue
            else:
                row[k] = v
        records.append(row)
    return records


def _read_npz(path: str) -> List[dict]:
    npz = np.load(path, allow_pickle=True)
    data = {k: npz[k] for k in npz.files}
    return _records_from_dict_of_arrays(data, source_id=os.path.splitext(os.path.basename(path))[0])


def _read_pkl(path: str) -> List[dict]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        if isinstance(obj[0], dict):
            return [dict(r) for r in obj]
    if isinstance(obj, dict):
        return _records_from_dict_of_arrays(obj, source_id=os.path.splitext(os.path.basename(path))[0])
    raise ValueError(f"Unsupported pickle structure: {type(obj)}")


def load_records(data_paths: List[str]) -> List[dict]:
    paths = []
    for data_path in data_paths:
        if os.path.isdir(data_path):
            for root, _, files in os.walk(data_path):
                for f in files:
                    if f.lower().endswith((".csv", ".npz", ".pkl", ".pickle")):
                        paths.append(os.path.join(root, f))
        else:
            paths.append(data_path)
    paths = sorted(set(paths))

    all_rows = []
    for p in sorted(paths):
        try:
            if p.lower().endswith(".csv"):
                rows = _read_csv(p)
            elif p.lower().endswith(".npz"):
                rows = _read_npz(p)
            elif p.lower().endswith((".pkl", ".pickle")):
                rows = _read_pkl(p)
            else:
                continue
            for r in rows:
                rr = dict(r)
                rr["_source_file"] = p
                all_rows.append(rr)
        except Exception as e:
            warnings.warn(f"Failed to load {p}: {e}")
    if not all_rows:
        warnings.warn(f"No rows loaded from paths: {data_paths}")
    return all_rows


def canonicalize_rows(rows: List[dict]) -> List[dict]:
    out = []
    if not rows:
        return out
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    alias_map = _build_alias_map(list(all_keys))

    missing = [k for k in ["sigma_min_Ag", "cost_total", "solve_time_ms"] if k not in alias_map]
    if missing:
        warnings.warn(f"Missing key mappings for required-ish fields: {missing}")

    for i, r in enumerate(rows):
        c = {}
        c["_source_file"] = r.get("_source_file", "")
        for std, real in alias_map.items():
            c[std] = r.get(real)

        src_file = str(r.get("_source_file", f"row{i}"))
        src_tag = os.path.splitext(os.path.basename(src_file))[0]
        raw_run_id = c.get("run_id")
        if raw_run_id is None or str(raw_run_id).strip() == "":
            c["run_id"] = src_tag
        else:
            c["run_id"] = f"{src_tag}::{str(raw_run_id)}"
        step_val = _parse_float(c.get("step", np.nan))
        c["step"] = int(step_val) if np.isfinite(step_val) else i

        for k in [
            "K",
            "K_opt",
            "rho",
            "lambda_g",
            "sigma_min_Ag",
            "loc",
            "slack_norm",
            "slack_max",
            "kkt_residual",
            "cost_total",
            "ee_rmse",
            "tracking_error",
            "success",
            "success_rate",
            "solve_time_ms",
            "noise_level",
        ]:
            c[k] = _parse_float(c.get(k, np.nan))

        c["method"] = _parse_method(c.get("method", None), c)
        out.append(c)
    return out


def aggregate_episode(rows: List[dict]) -> List[dict]:
    by_run = defaultdict(list)
    for r in rows:
        by_run[str(r["run_id"])].append(r)

    out = []
    for run_id, grp in by_run.items():
        grp = sorted(grp, key=lambda x: int(x.get("step", 0)))
        steps_est = int(max([int(g.get("step", 0)) for g in grp]) + 1) if grp else 0
        method = grp[0]["method"]
        K = np.nanmedian([g["K"] for g in grp if np.isfinite(g["K"])]) if any(np.isfinite(g["K"]) for g in grp) else np.nan
        rho = np.nanmedian([g["rho"] for g in grp if np.isfinite(g["rho"])]) if any(np.isfinite(g["rho"]) for g in grp) else np.nan
        lambda_g = np.nanmedian([g["lambda_g"] for g in grp if np.isfinite(g["lambda_g"])]) if any(np.isfinite(g["lambda_g"]) for g in grp) else np.nan

        sigma = np.array([g["sigma_min_Ag"] for g in grp], dtype=float)
        loc = np.array([g["loc"] for g in grp], dtype=float)
        slack = np.array(
            [
                g["slack_norm"] if np.isfinite(g["slack_norm"]) else g["slack_max"]
                for g in grp
            ],
            dtype=float,
        )
        kkt = np.array([g["kkt_residual"] for g in grp], dtype=float)
        kopt = np.array([g["K_opt"] for g in grp], dtype=float)
        cost = np.array([g["cost_total"] for g in grp], dtype=float)
        rmse = np.array([g["ee_rmse"] for g in grp], dtype=float)
        err_ts = np.array([g["tracking_error"] for g in grp], dtype=float)
        solv = np.array([g["solve_time_ms"] for g in grp], dtype=float)
        succ = np.array(
            [
                g["success"] if np.isfinite(g["success"]) else g["success_rate"]
                for g in grp
            ],
            dtype=float,
        )
        noise = np.array([g["noise_level"] for g in grp], dtype=float)

        sse = np.nan
        reach_time = np.nan
        reach_hit = 0.0
        if np.any(np.isfinite(err_ts)):
            e = err_ts[np.isfinite(err_ts)]
            w = int(max(1, SSE_WINDOW))
            sse = float(np.mean(e[-w:])) if e.size > 0 else np.nan
            hold = int(max(1, REACH_HOLD))
            if e.size >= hold:
                ok = (e <= float(REACH_EPS)).astype(int)
                run = np.convolve(ok, np.ones(hold, dtype=int), mode="valid")
                idx = np.where(run >= hold)[0]
                if idx.size > 0:
                    reach_time = float(idx[0])
                    reach_hit = 1.0
                else:
                    # Censored at episode horizon if not reached.
                    reach_time = float(e.size)
            else:
                reach_time = float(e.size)
        else:
            # Fallback if step-wise tracking error is unavailable.
            if np.any(np.isfinite(rmse)):
                sse = float(rmse[np.where(np.isfinite(rmse))[0][-1]])

        ep = {
            "run_id": run_id,
            "method": method,
            "K": float(K),
            "K_opt_mean": float(np.nanmean(kopt)) if np.any(np.isfinite(kopt)) else np.nan,
            "rho": float(rho),
            "lambda_g": float(lambda_g),
            "sigma_min_Ag": float(np.nanmin(sigma)) if np.any(np.isfinite(sigma)) else np.nan,
            "loc": float(np.nanmean(loc)) if np.any(np.isfinite(loc)) else np.nan,
            "slack_norm": float(np.nanmax(slack)) if np.any(np.isfinite(slack)) else np.nan,
            "kkt_residual": float(np.nanmax(kkt)) if np.any(np.isfinite(kkt)) else np.nan,
            "cost_total": float(cost[np.where(np.isfinite(cost))[0][0]]) if np.any(np.isfinite(cost)) else np.nan,
            "ee_rmse": float(rmse[np.where(np.isfinite(rmse))[0][0]]) if np.any(np.isfinite(rmse)) else np.nan,
            "sse": float(sse),
            "reach_time": float(reach_time),
            "reach_hit": float(reach_hit),
            "success": float(np.nanmax(succ)) if np.any(np.isfinite(succ)) else np.nan,
            "steps": float(steps_est),
            "solve_time_ms": float(np.nanmean(solv)) if np.any(np.isfinite(solv)) else np.nan,
            "noise_level": float(np.nanmedian(noise)) if np.any(np.isfinite(noise)) else np.nan,
        }
        out.append(ep)
    return out


def _bootstrap_ci(vals: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(v))
    if v.size == 1:
        return mean, mean, mean
    idx = np.random.randint(0, v.size, size=(int(n_boot), v.size))
    means = np.mean(v[idx], axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return mean, lo, hi


def _plot_metric_ci(ax, groups: Dict[str, np.ndarray], title: str, higher_is_better: bool = False):
    names = list(groups.keys())
    stats = [_bootstrap_ci(groups[n], BOOTSTRAP_N) for n in names]
    m = np.array([s[0] for s in stats], dtype=float)
    lo = np.array([s[1] for s in stats], dtype=float)
    hi = np.array([s[2] for s in stats], dtype=float)
    x = np.arange(len(names))
    yerr = np.vstack([m - lo, hi - m])
    ax.errorbar(x, m, yerr=yerr, fmt="o", capsize=3, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_title(title + (" (higher better)" if higher_is_better else " (lower better)"))
    ax.grid(True, alpha=0.25)


def _safe_log_axis(ax, arr: np.ndarray, axis: str):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0 or np.any(a <= 0):
        warnings.warn(f"Skip log-scale on {axis}-axis due to non-positive or empty values.")
        return
    if axis == "x":
        ax.set_xscale("log")
    else:
        ax.set_yscale("log")


def make_baseline_figure(ep_rows: List[dict], out_dir: str):
    groups = {
        "fixedK_small": [],
        "fixedK_large": [],
        "fixedK_lambda": [],
        "adaptiveK": [],
    }
    for r in ep_rows:
        m = r["method"]
        if m == "fixedK" and np.isfinite(r["K"]) and int(round(r["K"])) == int(K_SMALL):
            groups["fixedK_small"].append(r)
        elif m == "fixedK" and np.isfinite(r["K"]) and int(round(r["K"])) == int(K_LARGE):
            groups["fixedK_large"].append(r)
        elif m == "fixedK_lambda":
            if LAMBDA_G_LIST:
                if np.isfinite(r["lambda_g"]) and any(np.isclose(r["lambda_g"], v) for v in LAMBDA_G_LIST):
                    groups["fixedK_lambda"].append(r)
            else:
                groups["fixedK_lambda"].append(r)
        elif m == "adaptiveK":
            if RHO_LIST_DEFAULT:
                if np.isfinite(r["rho"]) and any(np.isclose(r["rho"], v) for v in RHO_LIST_DEFAULT):
                    groups["adaptiveK"].append(r)
            else:
                groups["adaptiveK"].append(r)

    for k, v in groups.items():
        if len(v) == 0:
            warnings.warn(f"Baseline group empty: {k}")

    metric_map = {
        "cost_total": ("cost_total", False),
        "ee_rmse": ("ee_rmse", False),
        "slack_norm": ("slack_norm", False),
        "sigma_min_Ag": ("sigma_min_Ag", True),
        "solve_time_ms": ("solve_time_ms", False),
        "success": ("success", True),
    }
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), dpi=300)
    axs = axs.reshape(-1)
    for ax, (label, (col, hib)) in zip(axs, metric_map.items()):
        gvals = {
            g: np.array([x[col] for x in rows], dtype=float)
            for g, rows in groups.items()
        }
        _plot_metric_ci(ax, gvals, label, higher_is_better=hib)
        if label == "sigma_min_Ag":
            all_vals = np.concatenate([v for v in gvals.values() if v.size > 0]) if any(v.size > 0 for v in gvals.values()) else np.array([])
            _safe_log_axis(ax, all_vals, "y")
    fig.suptitle("Baseline Comparison", y=0.995)
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_baseline_compare.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_robustness_figure(ep_rows: List[dict], out_dir: str):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=300)

    noise_vals = np.array([r["noise_level"] for r in ep_rows], dtype=float)
    has_noise = np.any(np.isfinite(noise_vals))
    if has_noise:
        methods = sorted(set(r["method"] for r in ep_rows))
        for m in methods:
            sub = [r for r in ep_rows if r["method"] == m and np.isfinite(r["noise_level"])]
            if not sub:
                continue
            x_unique = sorted(set(float(r["noise_level"]) for r in sub))
            succ = []
            cost = []
            for x in x_unique:
                grp = [r for r in sub if np.isclose(r["noise_level"], x)]
                succ.append(np.nanmean([r["success"] for r in grp]))
                cost.append(np.nanmean([r["cost_total"] for r in grp]))
            axs[0].plot(x_unique, succ, marker="o", label=m)
            axs[1].plot(x_unique, cost, marker="o", label=m)
        axs[0].set_xlabel("noise_level")
        axs[1].set_xlabel("noise_level")
        axs[0].set_title("Success vs Noise")
        axs[1].set_title("Cost vs Noise")
    else:
        adap = [r for r in ep_rows if r["method"] == "adaptiveK" and np.isfinite(r["rho"])]
        if adap:
            x = sorted(set(float(r["rho"]) for r in adap))
            y = []
            for rho in x:
                grp = [r for r in adap if np.isclose(r["rho"], rho)]
                y.append(np.nanmean([r["success"] for r in grp]))
            axs[0].plot(x, y, marker="o", label="adaptiveK")
            axs[0].set_xlabel("rho")
            _safe_log_axis(axs[0], np.array(x), "x")
            axs[0].set_title("Success Rate vs rho")
        by_m = sorted(set(r["method"] for r in ep_rows))
        y = [np.nanmean([r["success"] for r in ep_rows if r["method"] == m]) for m in by_m]
        axs[1].bar(by_m, y)
        axs[1].set_title("Success Rate vs Method")
        axs[1].set_ylim(0, 1.02)

    axs[0].set_ylabel("success_rate")
    axs[1].set_ylabel("cost_total / success_rate")
    for ax in axs:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    fig.suptitle("Robustness", y=0.995)
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_robustness.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_loc_vs_sigma(ep_rows: List[dict], out_dir: str):
    x = np.array([r["loc"] for r in ep_rows], dtype=float)
    y = np.array([r["sigma_min_Ag"] for r in ep_rows], dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.0), dpi=300)
    if not np.any(m):
        warnings.warn("loc vs sigma plot skipped: no valid loc/sigma data.")
        ax.text(0.5, 0.5, "no valid data", ha="center", va="center", transform=ax.transAxes)
    else:
        methods = sorted(set(r["method"] for r in ep_rows))
        for meth in methods:
            mm = m & np.array([r["method"] == meth for r in ep_rows])
            ax.scatter(x[mm], y[mm], s=20, alpha=0.75, label=meth)
        _safe_log_axis(ax, y[m], "y")
    ax.set_xlabel("Loc(K)")
    ax.set_ylabel("sigma_min_Ag")
    ax.set_title("Localness-Conditioning Tradeoff")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_loc_vs_sigma.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(a):
        order = np.argsort(a)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(a.size, dtype=float)
        return r
    rx = rank(x)
    ry = rank(y)
    rx = (rx - np.mean(rx)) / (np.std(rx) + 1e-12)
    ry = (ry - np.mean(ry)) / (np.std(ry) + 1e-12)
    return float(np.mean(rx * ry))


def make_sigma_vs_cost(ep_rows: List[dict], out_dir: str):
    x = np.array([r["sigma_min_Ag"] for r in ep_rows], dtype=float)
    y = np.array([r["cost_total"] if np.isfinite(r["cost_total"]) else r["ee_rmse"] for r in ep_rows], dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.0), dpi=300)
    if np.any(m):
        methods = sorted(set(r["method"] for r in ep_rows))
        for meth in methods:
            mm = m & np.array([r["method"] == meth for r in ep_rows])
            ax.scatter(x[mm], y[mm], s=20, alpha=0.75, label=meth)
        lx = np.log10(x[m])
        coef = np.polyfit(lx, y[m], deg=1)
        xx = np.linspace(np.min(lx), np.max(lx), 100)
        yy = coef[0] * xx + coef[1]
        ax.plot(10 ** xx, yy, "k--", lw=1.2, label="fit on log10(sigma)")
        corr = _spearman(x[m], y[m])
        ax.text(0.03, 0.96, f"Spearman={corr:.3f}", transform=ax.transAxes, va="top")
        _safe_log_axis(ax, x[m], "x")
    else:
        warnings.warn("sigma vs performance plot skipped: no positive sigma and target metric.")
        ax.text(0.5, 0.5, "no valid data", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("sigma_min_Ag")
    ax.set_ylabel("cost_total (or ee_rmse fallback)")
    ax.set_title("Conditioning-Performance Link")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_sigma_vs_cost.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_adaptive_distribution(step_rows: List[dict], ep_rows: List[dict], out_dir: str):
    adap_step = [r for r in step_rows if r["method"] == "adaptiveK" and np.isfinite(r["rho"]) and np.isfinite(r["K_opt"])]
    adap_ep = [r for r in ep_rows if r["method"] == "adaptiveK" and np.isfinite(r["rho"])]

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.8), dpi=300)
    if adap_step:
        rho_vals = sorted(set(float(r["rho"]) for r in adap_step))
        data = [np.array([r["K_opt"] for r in adap_step if np.isclose(r["rho"], rv)], dtype=float) for rv in rho_vals]
        ax.boxplot(data, positions=np.arange(len(rho_vals)), widths=0.6, showfliers=False)
        med = [np.nanmedian(d) if d.size > 0 else np.nan for d in data]
        ax.plot(np.arange(len(rho_vals)), med, "o-", label="median K_opt", color="tab:blue")
        ax.set_xticks(np.arange(len(rho_vals)))
        ax.set_xticklabels([f"{v:g}" for v in rho_vals])
        ax.set_xlabel("rho")
        ax.set_ylabel("K_opt distribution")
        _safe_log_axis(ax, np.array(rho_vals), "x")

        if adap_ep:
            ax2 = ax.twinx()
            solve = [np.nanmean([r["solve_time_ms"] for r in adap_ep if np.isclose(r["rho"], rv)]) for rv in rho_vals]
            ax2.plot(np.arange(len(rho_vals)), solve, "s--", color="tab:red", label="mean solve_time_ms")
            ax2.set_ylabel("solve_time_ms")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
    else:
        warnings.warn("Adaptive-K distribution figure skipped: no per-step adaptive K_opt data.")
        ax.text(0.5, 0.5, "no valid adaptive K_opt", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Adaptive-K Uses Minimum Necessary K")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_adaptiveK_distribution.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_sse_reach_split_figures(ep_rows: List[dict], out_dir: str):
    out_paths = []

    def _agg(rows, x_field, y_field):
        xs = sorted(set(float(r[x_field]) for r in rows if np.isfinite(r.get(x_field, np.nan))))
        med, p10, p90, succ = [], [], [], []
        for x in xs:
            grp = [r for r in rows if np.isfinite(r.get(x_field, np.nan)) and np.isclose(r[x_field], x)]
            vals = np.array([r[y_field] for r in grp], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                med.append(float(np.median(vals)))
                p10.append(float(np.quantile(vals, 0.1)))
                p90.append(float(np.quantile(vals, 0.9)))
            else:
                med.append(np.nan)
                p10.append(np.nan)
                p90.append(np.nan)
            svals = np.array([r["success"] for r in grp], dtype=float)
            svals = svals[np.isfinite(svals)]
            succ.append(float(np.mean(svals)) if svals.size > 0 else np.nan)
        return np.array(xs, dtype=float), np.array(med), np.array(p10), np.array(p90), np.array(succ)

    fixed_rows = [r for r in ep_rows if r["method"] in ("fixedK", "fixedK_lambda") and np.isfinite(r["K"])]
    adap_rows = [r for r in ep_rows if r["method"] == "adaptiveK" and np.isfinite(r["rho"])]

    # Fixed: SSE vs K
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8), dpi=300)
    if fixed_rows:
        x, m, lo, hi, _ = _agg(fixed_rows, "K", "sse")
        mask = np.isfinite(x) & np.isfinite(m)
        if np.any(mask):
            ax.plot(x[mask], m[mask], marker="o", lw=1.4, label="median SSE")
            ax.fill_between(x[mask], lo[mask], hi[mask], alpha=0.2, label="p10-p90")
            _safe_log_axis(ax, m[mask], "y")
        else:
            ax.text(0.5, 0.5, "no valid data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "no fixed-K data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("K")
    ax.set_ylabel("Steady-state error")
    ax.set_title("SSE vs K (fixed)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_sse_vs_k.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(p)

    # Fixed: Reach time vs K + success rate
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8), dpi=300)
    if fixed_rows:
        x, m, lo, hi, succ = _agg(fixed_rows, "K", "reach_time")
        mask = np.isfinite(x) & np.isfinite(m)
        if np.any(mask):
            ax.plot(x[mask], m[mask], marker="o", lw=1.4, color="tab:blue", label="median reach time")
            ax.fill_between(x[mask], lo[mask], hi[mask], alpha=0.2, color="tab:blue")
            ax2 = ax.twinx()
            sm = np.isfinite(x) & np.isfinite(succ)
            if np.any(sm):
                ax2.plot(x[sm], succ[sm], "s--", color="tab:red", label="success rate")
            ax2.set_ylabel("Success rate")
            ax2.set_ylim(-0.02, 1.02)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")
        else:
            ax.text(0.5, 0.5, "no valid data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "no fixed-K data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("K")
    ax.set_ylabel("Reach time [step]")
    ax.set_title("Reach Time vs K (fixed)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_reach_time_vs_k.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(p)

    # Adaptive: SSE vs rho
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8), dpi=300)
    if adap_rows:
        x, m, lo, hi, _ = _agg(adap_rows, "rho", "sse")
        mask = np.isfinite(x) & np.isfinite(m)
        if np.any(mask):
            ax.plot(x[mask], m[mask], marker="o", lw=1.4, label="median SSE")
            ax.fill_between(x[mask], lo[mask], hi[mask], alpha=0.2, label="p10-p90")
            _safe_log_axis(ax, x[mask], "x")
            _safe_log_axis(ax, m[mask], "y")
        else:
            ax.text(0.5, 0.5, "no valid data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "no adaptive-K data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("rho")
    ax.set_ylabel("Steady-state error")
    ax.set_title("SSE vs rho (adaptive)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_sse_vs_rho.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(p)

    # Adaptive: Reach time vs rho + success rate
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8), dpi=300)
    if adap_rows:
        x, m, lo, hi, succ = _agg(adap_rows, "rho", "reach_time")
        mask = np.isfinite(x) & np.isfinite(m)
        if np.any(mask):
            ax.plot(x[mask], m[mask], marker="o", lw=1.4, color="tab:blue", label="median reach time")
            ax.fill_between(x[mask], lo[mask], hi[mask], alpha=0.2, color="tab:blue")
            _safe_log_axis(ax, x[mask], "x")
            ax2 = ax.twinx()
            sm = np.isfinite(x) & np.isfinite(succ)
            if np.any(sm):
                ax2.plot(x[sm], succ[sm], "s--", color="tab:red", label="success rate")
                _safe_log_axis(ax2, x[sm], "x")
            ax2.set_ylabel("Success rate")
            ax2.set_ylim(-0.02, 1.02)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")
        else:
            ax.text(0.5, 0.5, "no valid data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "no adaptive-K data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("rho")
    ax.set_ylabel("Reach time [step]")
    ax.set_title("Reach Time vs rho (adaptive)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_reach_time_vs_rho.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(p)
    return out_paths


def make_sigma_threshold_compare(ep_rows: List[dict], out_dir: str, sigma_threshold: Optional[float]):
    rows = [r for r in ep_rows if np.isfinite(r["sigma_min_Ag"])]
    if not rows:
        warnings.warn("sigma-threshold figure skipped: no finite sigma_min_Ag.")
        return None

    sig = np.array([r["sigma_min_Ag"] for r in rows], dtype=float)
    thr = float(sigma_threshold) if sigma_threshold is not None else float(np.quantile(sig, 0.25))
    low = [r for r in rows if r["sigma_min_Ag"] < thr]
    high = [r for r in rows if r["sigma_min_Ag"] >= thr]
    if len(low) == 0 or len(high) == 0:
        warnings.warn("sigma-threshold split empty on one side; adjust --sigma_threshold.")

    metrics = [
        ("cost_total", "Cost Total"),
        ("ee_rmse", "EE RMSE"),
        ("slack_norm", "Slack"),
        ("solve_time_ms", "Solve Time [ms]"),
        ("sse", "Steady-state Error"),
        ("reach_time", "Reach Time [step]"),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(12, 7), dpi=300)
    axs = axs.reshape(-1)
    for ax, (m, title) in zip(axs, metrics):
        a = np.array([r[m] for r in low], dtype=float)
        b = np.array([r[m] for r in high], dtype=float)
        ma, la, ha = _bootstrap_ci(a, BOOTSTRAP_N)
        mb, lb, hb = _bootstrap_ci(b, BOOTSTRAP_N)
        x = np.array([0, 1], dtype=float)
        y = np.array([ma, mb], dtype=float)
        yerr = np.array([[ma - la, mb - lb], [ha - ma, hb - mb]], dtype=float)
        ax.bar(x, y, color=["tab:red", "tab:blue"], alpha=0.75)
        if np.all(np.isfinite(yerr)):
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=3, lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"< thr (n={len(low)})", f">= thr (n={len(high)})"])
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if m in ("cost_total", "ee_rmse", "slack_norm", "sse", "solve_time_ms"):
            _safe_log_axis(ax, y, "y")

    fig.suptitle(f"Before/After sigma_min threshold (thr={thr:.3e})", y=0.995)
    fig.tight_layout()
    out = os.path.join(out_dir, "fig_sigma_threshold_compare.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def print_summary_table(ep_rows: List[dict]):
    by_method = defaultdict(list)
    for r in ep_rows:
        by_method[r["method"]].append(r)

    print("\nMethod Summary")
    print("method\t n\t cost_mean\t cost_median\t success_rate")
    for m in sorted(by_method.keys()):
        grp = by_method[m]
        cost = np.array([g["cost_total"] for g in grp], dtype=float)
        succ = np.array([g["success"] for g in grp], dtype=float)
        n = len(grp)
        cmean = np.nanmean(cost) if np.any(np.isfinite(cost)) else np.nan
        cmed = np.nanmedian(cost) if np.any(np.isfinite(cost)) else np.nan
        srate = np.nanmean(succ) if np.any(np.isfinite(succ)) else np.nan
        print(f"{m}\t {n}\t {cmean:.4g}\t {cmed:.4g}\t {srate:.4g}")


def main():
    global K_SMALL, K_LARGE, RHO_LIST_DEFAULT, RHO_BEST, LAMBDA_G_LIST, BOOTSTRAP_N
    global SSE_WINDOW, REACH_EPS, REACH_HOLD
    global SIGMA_THRESHOLD

    parser = argparse.ArgumentParser(description="Generate CDC-ready figures from experiment logs.")
    parser.add_argument(
        "--data_path",
        type=str,
        nargs="+",
        default=[],
        help="One or more CSV/NPZ/PKL file paths or directories.",
    )
    parser.add_argument(
        "--fixed_dir",
        type=str,
        nargs="*",
        default=[],
        help="Directory(ies) containing fixed-K results.",
    )
    parser.add_argument(
        "--adaptive_dir",
        type=str,
        nargs="*",
        default=[],
        help="Directory(ies) containing adaptive-K/rho-sweep results.",
    )
    parser.add_argument(
        "--lambda_dir",
        type=str,
        nargs="*",
        default=[],
        help="Directory(ies) containing fixedK+lambda results.",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output figure directory.")
    parser.add_argument("--k_small", type=int, default=K_SMALL)
    parser.add_argument("--k_large", type=int, default=K_LARGE)
    parser.add_argument("--rho_list", type=float, nargs="*", default=RHO_LIST_DEFAULT)
    parser.add_argument("--rho_best", type=float, default=RHO_BEST)
    parser.add_argument("--lambda_g_list", type=float, nargs="*", default=LAMBDA_G_LIST)
    parser.add_argument("--bootstrap_n", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--sse_window", type=int, default=SSE_WINDOW)
    parser.add_argument("--reach_eps", type=float, default=REACH_EPS)
    parser.add_argument("--reach_hold", type=int, default=REACH_HOLD)
    parser.add_argument("--sigma_threshold", type=float, default=SIGMA_THRESHOLD)
    args = parser.parse_args()

    K_SMALL = int(args.k_small)
    K_LARGE = int(args.k_large)
    RHO_LIST_DEFAULT = list(args.rho_list) if args.rho_list else []
    RHO_BEST = args.rho_best
    LAMBDA_G_LIST = list(args.lambda_g_list) if args.lambda_g_list else []
    BOOTSTRAP_N = int(args.bootstrap_n)
    SSE_WINDOW = int(args.sse_window)
    REACH_EPS = float(args.reach_eps)
    REACH_HOLD = int(args.reach_hold)
    SIGMA_THRESHOLD = args.sigma_threshold

    os.makedirs(args.out_dir, exist_ok=True)

    input_paths = []
    input_paths.extend(args.data_path)
    input_paths.extend(args.fixed_dir)
    input_paths.extend(args.adaptive_dir)
    input_paths.extend(args.lambda_dir)
    input_paths = [p for p in input_paths if p]
    if not input_paths:
        raise ValueError(
            "No input path provided. Use --data_path (or --fixed_dir/--adaptive_dir/--lambda_dir)."
        )

    raw = load_records(input_paths)
    rows = canonicalize_rows(raw)
    ep_rows = aggregate_episode(rows)

    if RHO_BEST is not None:
        ep_rows = [
            r
            for r in ep_rows
            if (r["method"] != "adaptiveK")
            or (np.isfinite(r["rho"]) and np.isclose(r["rho"], float(RHO_BEST)))
        ]
        rows = [
            r
            for r in rows
            if (r["method"] != "adaptiveK")
            or (np.isfinite(r["rho"]) and np.isclose(r["rho"], float(RHO_BEST)))
        ]

    fig_paths = []
    fig_paths.append(make_baseline_figure(ep_rows, args.out_dir))
    fig_paths.append(make_robustness_figure(ep_rows, args.out_dir))
    fig_paths.append(make_loc_vs_sigma(ep_rows, args.out_dir))
    fig_paths.append(make_sigma_vs_cost(ep_rows, args.out_dir))
    fig_paths.append(make_adaptive_distribution(rows, ep_rows, args.out_dir))
    fig_paths.extend(make_sse_reach_split_figures(ep_rows, args.out_dir))
    fig_paths.append(make_sigma_threshold_compare(ep_rows, args.out_dir, SIGMA_THRESHOLD))

    print(f"Loaded records: step_rows={len(rows)} episode_rows={len(ep_rows)}")
    print(f"Input paths: {input_paths}")
    print_summary_table(ep_rows)

    print("\nSaved figures:")
    for p in fig_paths:
        if p is not None:
            print(p)


if __name__ == "__main__":
    main()
