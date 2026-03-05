import argparse
import csv
import glob
import os

import numpy as np


SIG_KEYS = ["sigma_min_Mk", "sig_actual", "sigma_min_Hu", "sigma_min_Ag"]
DU_KEYS = ["delta_u_norm"]
DU0_KEYS = ["delta_u0", "delta_u_first", "delta_u"]
STATUS_KEYS = ["solver_status", "status", "status_history"]
KKT_KEYS = ["kkt_residual"]


def first_existing_key(keys, bag):
    for k in keys:
        if k in bag:
            return k
    return None


def as_float_1d(x):
    arr = np.asarray(x)
    if arr.dtype == object:
        try:
            arr = arr.astype(float)
        except Exception:
            out = []
            for v in arr.reshape(-1):
                try:
                    out.append(float(v))
                except Exception:
                    out.append(np.nan)
            return np.asarray(out, dtype=float)
    return np.asarray(arr, dtype=float).reshape(-1)


def as_status_1d(x):
    arr = np.asarray(x, dtype=object).reshape(-1)
    out = []
    for v in arr:
        try:
            s = str(v).strip().lower()
        except Exception:
            s = ""
        out.append(s)
    return np.asarray(out, dtype=object)


def robust_zscore_mad(v):
    x = np.asarray(v, dtype=float).reshape(-1)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad <= 0:
        return np.full(x.shape, np.nan, dtype=float)
    return 0.6745 * (x - med) / mad


def r2_score(y, yhat):
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    if y.size == 0:
        return np.nan
    ss_res = np.nansum((y - yhat) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    if ss_tot <= 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def save_rows_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def extract_delta_u0(npz_obj):
    key = first_existing_key(DU0_KEYS, npz_obj.files)
    if key is None:
        return None
    arr = np.asarray(npz_obj[key])
    if key == "delta_u0":
        return as_float_1d(arr)
    if arr.dtype == object:
        try:
            arr = arr.astype(float)
        except Exception:
            flat = []
            for v in arr.reshape(-1):
                try:
                    flat.append(float(v))
                except Exception:
                    flat.append(np.nan)
            return np.asarray(flat, dtype=float)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1)
    if arr.ndim >= 2 and arr.shape[1] >= 1:
        return arr[:, 0].reshape(-1)
    return None


def build_upper_envelope(x, y, n_bins):
    xx = np.asarray(x, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(xx) & np.isfinite(yy) & (xx > 0)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 4:
        return np.array([]), np.array([]), np.array([]), np.array([])

    edges = np.geomspace(np.nanmin(xx), np.nanmax(xx), int(max(4, n_bins)) + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    y_max = np.full(centers.shape, np.nan, dtype=float)
    counts = np.zeros(centers.shape, dtype=int)
    for i in range(centers.size):
        sel = (xx >= edges[i]) & (xx < edges[i + 1])
        if np.any(sel):
            y_max[i] = float(np.nanmax(yy[sel]))
            counts[i] = int(np.count_nonzero(sel))
    keep = np.isfinite(y_max) & (counts > 0)
    return edges, centers[keep], y_max[keep], counts[keep]


def main():
    parser = argparse.ArgumentParser(
        description="Figure-B pipeline: outlier removal -> x transform -> regression"
    )
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.npz")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument(
        "--x_mode",
        type=str,
        default="regularized",
        choices=["regularized", "inv_sigma2"],
        help="regularized: 1/(lambda_g + lambda_min(R)*sigma^2), inv_sigma2: 1/sigma^2",
    )
    parser.add_argument("--lambda_g", type=float, default=np.nan)
    parser.add_argument("--lambda_min_r", type=float, default=np.nan)
    parser.add_argument("--sigma_floor", type=float, default=1e-10)
    parser.add_argument(
        "--ok_status",
        nargs="+",
        default=["optimal", "optimal_inaccurate"],
        help="Solver status whitelist (substring match, lowercase)",
    )
    parser.add_argument(
        "--kkt_quantile",
        type=float,
        default=0.99,
        help="Keep rows with kkt_residual <= quantile among status-filtered finite rows",
    )
    parser.add_argument(
        "--mad_z",
        type=float,
        default=3.5,
        help="MAD z-score threshold for sigma and delta_u_norm (in log10 space)",
    )
    parser.add_argument(
        "--env_bins",
        type=int,
        default=24,
        help="Number of log-space bins for upper-envelope plot (sigma_min vs delta_u0).",
    )
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(args.logdir, "figb_pipeline")
    os.makedirs(outdir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.logdir, args.pattern)))
    if len(paths) == 0:
        raise FileNotFoundError(f"No npz found: {os.path.join(args.logdir, args.pattern)}")

    rows = []
    for p in paths:
        d = np.load(p, allow_pickle=True)

        sig_key = first_existing_key(SIG_KEYS, d.files)
        du_key = first_existing_key(DU_KEYS, d.files)
        st_key = first_existing_key(STATUS_KEYS, d.files)
        kkt_key = first_existing_key(KKT_KEYS, d.files)
        if sig_key is None or du_key is None or st_key is None or kkt_key is None:
            continue

        sig = as_float_1d(d[sig_key])
        du = as_float_1d(d[du_key])
        du0 = extract_delta_u0(d)
        st = as_status_1d(d[st_key])
        kkt = as_float_1d(d[kkt_key])

        if du0 is None:
            du0 = np.full(sig.shape, np.nan, dtype=float)
        n = min(sig.size, du.size, du0.size, st.size, kkt.size)
        if n < 2:
            continue
        sig = sig[:n]
        du = du[:n]
        du0 = du0[:n]
        st = st[:n]
        kkt = kkt[:n]

        lg = float(args.lambda_g)
        lmr = float(args.lambda_min_r)
        if "lambda_g" in d.files and np.isfinite(as_float_1d(d["lambda_g"])).any():
            lg = float(as_float_1d(d["lambda_g"])[0])
        if "lambda_min_R" in d.files and np.isfinite(as_float_1d(d["lambda_min_R"])).any():
            lmr = float(as_float_1d(d["lambda_min_R"])[0])

        run_name = os.path.basename(p)
        for i in range(n):
            rows.append(
                {
                    "run": run_name,
                    "step": int(i),
                    "sig_actual": float(sig[i]),
                    "delta_u_norm": float(du[i]),
                    "delta_u0": float(du0[i]),
                    "solver_status": st[i],
                    "kkt_residual": float(kkt[i]),
                    "lambda_g": lg,
                    "lambda_min_r": lmr,
                }
            )

    if len(rows) == 0:
        raise RuntimeError("No usable rows found. Check keys in your npz logs.")

    sig = np.asarray([r["sig_actual"] for r in rows], dtype=float)
    du = np.asarray([r["delta_u_norm"] for r in rows], dtype=float)
    du0 = np.asarray([r["delta_u0"] for r in rows], dtype=float)
    st = np.asarray([str(r["solver_status"]).lower() for r in rows], dtype=object)
    kkt = np.asarray([r["kkt_residual"] for r in rows], dtype=float)

    finite_mask = np.isfinite(sig) & np.isfinite(du) & np.isfinite(kkt) & (sig > 0) & (du > 0)
    ok_set = [s.lower() for s in args.ok_status]
    status_mask = np.array([any(ok in x for ok in ok_set) for x in st], dtype=bool)
    base_mask = finite_mask & status_mask

    if np.count_nonzero(base_mask) < 5:
        raise RuntimeError("Too few points after finite/status filtering.")

    kkt_thr = float(np.nanquantile(kkt[base_mask], args.kkt_quantile))
    kkt_mask = kkt <= kkt_thr

    # MAD-based outlier rejection in log-space
    z_sig = robust_zscore_mad(np.log10(np.maximum(sig, args.sigma_floor)))
    z_du = robust_zscore_mad(np.log10(np.maximum(du, args.sigma_floor)))
    mad_mask = (np.abs(z_sig) <= args.mad_z) & (np.abs(z_du) <= args.mad_z)

    keep = base_mask & kkt_mask & mad_mask
    if np.count_nonzero(keep) < 5:
        raise RuntimeError("Too few points after outlier filtering.")

    sig_k = sig[keep]
    du_k = du[keep]
    du0_k = du0[keep]
    lg_k = np.asarray([r["lambda_g"] for r in rows], dtype=float)[keep]
    lmr_k = np.asarray([r["lambda_min_r"] for r in rows], dtype=float)[keep]

    if args.x_mode == "regularized":
        if not np.isfinite(lg_k).all() or not np.isfinite(lmr_k).all():
            raise RuntimeError(
                "regularized x_mode requires lambda_g and lambda_min_r (from npz or args)."
            )
        x = 1.0 / (lg_k + lmr_k * np.square(sig_k))
    else:
        x = 1.0 / np.square(np.maximum(sig_k, args.sigma_floor))

    y = du_k

    # Linear regression: y = a*x + b
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    r2_lin = r2_score(y, yhat)

    # Log-log regression: log10(y) = c*log10(x) + d
    lx = np.log10(np.maximum(x, args.sigma_floor))
    ly = np.log10(np.maximum(y, args.sigma_floor))
    c, d = np.polyfit(lx, ly, 1)
    lyhat = c * lx + d
    r2_log = r2_score(ly, lyhat)

    # Save raw/filtered tables
    raw_csv = os.path.join(outdir, "pipeline_raw_rows.csv")
    filt_csv = os.path.join(outdir, "pipeline_filtered_rows.csv")
    fit_csv = os.path.join(outdir, "pipeline_fit_summary.csv")

    raw_fields = list(rows[0].keys())
    save_rows_csv(raw_csv, rows, raw_fields)

    filt_rows = []
    idx_keep = np.where(keep)[0]
    for idx, xv, yv in zip(idx_keep, x, y):
        rr = dict(rows[int(idx)])
        rr["x_transformed"] = float(xv)
        rr["y_delta_u_norm"] = float(yv)
        filt_rows.append(rr)
    save_rows_csv(filt_csv, filt_rows, list(filt_rows[0].keys()))

    fit_rows = [
        {
            "n_raw": int(len(rows)),
            "n_after_finite_status": int(np.count_nonzero(base_mask)),
            "n_after_outlier": int(np.count_nonzero(keep)),
            "kkt_quantile": float(args.kkt_quantile),
            "kkt_threshold": float(kkt_thr),
            "mad_z": float(args.mad_z),
            "x_mode": args.x_mode,
            "linear_slope_a": float(a),
            "linear_intercept_b": float(b),
            "linear_r2": float(r2_lin),
            "loglog_slope_c": float(c),
            "loglog_intercept_d": float(d),
            "loglog_r2": float(r2_log),
            "corr_x_y": float(np.corrcoef(x, y)[0, 1]),
        }
    ]

    # Envelope pipeline: x=sigma_min(M_k) (sig_actual), y=delta_u0
    env_mask = np.isfinite(sig_k) & np.isfinite(du0_k) & (sig_k > 0)
    x_env = sig_k[env_mask]
    y_env = du0_k[env_mask]
    e_edges, e_centers, e_ymax, e_counts = build_upper_envelope(x_env, y_env, n_bins=int(args.env_bins))

    exceed_rate = np.nan
    exceed_count = 0
    total_count = int(x_env.size)
    if e_centers.size >= 2:
        interp = np.interp(x_env, e_centers, e_ymax, left=np.nan, right=np.nan)
        valid_interp = np.isfinite(interp)
        if np.any(valid_interp):
            exceed = y_env[valid_interp] > (interp[valid_interp] + 1e-12)
            exceed_count = int(np.count_nonzero(exceed))
            exceed_rate = float(exceed_count / np.count_nonzero(valid_interp))

    fit_rows[0]["envelope_bins_used"] = int(e_centers.size)
    fit_rows[0]["envelope_total_points"] = int(total_count)
    fit_rows[0]["envelope_exceed_count"] = int(exceed_count)
    fit_rows[0]["envelope_exceed_rate"] = float(exceed_rate) if np.isfinite(exceed_rate) else np.nan
    save_rows_csv(fit_csv, fit_rows, list(fit_rows[0].keys()))

    # Plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)

    # Left: raw vs filtered
    x_raw = (
        1.0 / np.square(np.maximum(sig, args.sigma_floor))
        if args.x_mode == "inv_sigma2"
        else 1.0
        / (
            np.asarray([r["lambda_g"] for r in rows], dtype=float)
            + np.asarray([r["lambda_min_r"] for r in rows], dtype=float)
            * np.square(np.maximum(sig, args.sigma_floor))
        )
    )
    axes[0].scatter(x_raw, du, s=8, alpha=0.2, color="#9aa0a6", label="raw")
    axes[0].scatter(x, y, s=10, alpha=0.6, color="#1f77b4", label="filtered")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    if args.x_mode == "regularized":
        axes[0].set_xlabel(r"$1/(\lambda_g+\lambda_{\min}(R)\sigma^2)$")
    else:
        axes[0].set_xlabel(r"$1/\sigma^2$")
    axes[0].set_ylabel(r"$\|\Delta u\|_2$")
    axes[0].set_title("Outlier Removal Result")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(loc="best")

    # Right: regression
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    yfit_s = (a * xs + b)
    yfit_s = np.maximum(yfit_s, args.sigma_floor)
    axes[1].scatter(x, y, s=10, alpha=0.5, color="#1f77b4", label="filtered data")
    axes[1].plot(xs, yfit_s, color="#d62728", lw=1.5, label=f"linear fit (R2={r2_lin:.3f})")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    if args.x_mode == "regularized":
        axes[1].set_xlabel(r"$1/(\lambda_g+\lambda_{\min}(R)\sigma^2)$")
    else:
        axes[1].set_xlabel(r"$1/\sigma^2$")
    axes[1].set_ylabel(r"$\|\Delta u\|_2$")
    axes[1].set_title(
        "Linear Regression on Filtered Data\n"
        f"log-log slope={c:.3f}, log-log R2={r2_log:.3f}"
    )
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(loc="best")

    plt.tight_layout()
    fig_path = os.path.join(outdir, "figb_pipeline_regression.png")
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Dedicated requested plot: scatter + upper envelope (bin max)
    fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=180)
    ax2.scatter(x_env, y_env, s=10, alpha=0.35, color="#1f77b4", label="scatter")
    if e_centers.size > 0:
        ax2.plot(e_centers, e_ymax, "-o", color="#d62728", lw=1.6, ms=4, label="upper envelope (bin max)")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\sigma_{\min}(M_k)$")
    ax2.set_ylabel(r"$\Delta u_0$")
    title = "Delta u0 vs sigma_min(Mk) with Upper Envelope"
    if np.isfinite(exceed_rate):
        title += f"\nBound check exceed rate={exceed_rate:.3%} ({exceed_count}/{total_count})"
    ax2.set_title(title)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="best")
    plt.tight_layout()
    fig_env = os.path.join(outdir, "fig_sigma_minMk_vs_delta_u0_upper_envelope.png")
    fig2.savefig(fig_env, dpi=220, bbox_inches="tight")
    plt.close(fig2)

    env_csv = os.path.join(outdir, "sigma_minMk_delta_u0_upper_envelope.csv")
    env_rows = []
    for i in range(e_centers.size):
        env_rows.append(
            {
                "bin_center_sigma_minMk": float(e_centers[i]),
                "bin_max_delta_u0": float(e_ymax[i]),
                "bin_count": int(e_counts[i]),
            }
        )
    if len(env_rows) > 0:
        save_rows_csv(env_csv, env_rows, list(env_rows[0].keys()))

    print(f"Saved: {raw_csv}")
    print(f"Saved: {filt_csv}")
    print(f"Saved: {fit_csv}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {fig_env}")
    if len(env_rows) > 0:
        print(f"Saved: {env_csv}")


if __name__ == "__main__":
    main()
