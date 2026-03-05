import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def to_1d_tracking_error(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        return np.abs(a)
    # If vector error per step, use L2 norm per step.
    return np.linalg.norm(a, axis=1)


def main():
    parser = argparse.ArgumentParser(description="3-way plot from single offline npz run.")
    parser.add_argument(
        "--npz",
        type=str,
        default=os.path.join(
            "logs",
            "reacher",
            "offline_ksweep_dir",
            "run_fixed_K080_rho0.3_seed000.npz",
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("logs", "reacher", "offline_ksweep_dir", "paper_figs"),
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    d = np.load(args.npz, allow_pickle=True)

    sigma = np.asarray(d["sigma_min_Hu"], dtype=float).reshape(-1)
    track = to_1d_tracking_error(d["tracking_error"])
    slack = np.asarray(d["slack_inf"], dtype=float).reshape(-1)
    solve = np.asarray(d["solve_time_ms"], dtype=float).reshape(-1)

    n = min(len(sigma), len(track), len(slack), len(solve))
    t = np.arange(n, dtype=int)
    sigma = sigma[:n]
    track = track[:n]
    slack = slack[:n]
    solve = solve[:n]

    # Figure A: sigma vs tracking vs slack
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 62))

    l1 = ax1.plot(t, sigma, color="#1f77b4", lw=2.0, label=r"$\sigma_{\min}(H_u)$")
    l2 = ax2.plot(t, track, color="#d62728", lw=1.8, label="tracking error")
    l3 = ax3.plot(t, slack, color="#2ca02c", lw=1.8, label="slack inf")

    ax1.set_xlabel("time step")
    ax1.set_ylabel(r"$\sigma_{\min}(H_u)$", color="#1f77b4")
    ax2.set_ylabel("tracking error", color="#d62728")
    ax3.set_ylabel("slack inf", color="#2ca02c")
    ax1.tick_params(axis="y", colors="#1f77b4")
    ax2.tick_params(axis="y", colors="#d62728")
    ax3.tick_params(axis="y", colors="#2ca02c")
    ax1.grid(alpha=0.3)
    ax1.set_title("Figure 1 (offline_ksweep_dir): sigma_min(Hu) vs tracking vs instability")
    lines = l1 + l2 + l3
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="best")
    plt.tight_layout()
    out_a = os.path.join(args.outdir, "fig1_threeway_time_sigma_tracking_slack.png")
    plt.savefig(out_a, dpi=220)
    plt.close(fig)

    # Figure B: sigma vs tracking vs solve time
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 62))

    l1 = ax1.plot(t, sigma, color="#1f77b4", lw=2.0, label=r"$\sigma_{\min}(H_u)$")
    l2 = ax2.plot(t, track, color="#d62728", lw=1.8, label="tracking error")
    l3 = ax3.plot(t, solve, color="#9467bd", lw=1.8, label="solve time (ms)")

    ax1.set_xlabel("time step")
    ax1.set_ylabel(r"$\sigma_{\min}(H_u)$", color="#1f77b4")
    ax2.set_ylabel("tracking error", color="#d62728")
    ax3.set_ylabel("solve time (ms)", color="#9467bd")
    ax1.tick_params(axis="y", colors="#1f77b4")
    ax2.tick_params(axis="y", colors="#d62728")
    ax3.tick_params(axis="y", colors="#9467bd")
    ax1.grid(alpha=0.3)
    ax1.set_title("Figure 1 (offline_ksweep_dir): sigma_min(Hu) vs tracking vs solve time")
    lines = l1 + l2 + l3
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="best")
    plt.tight_layout()
    out_b = os.path.join(args.outdir, "fig1_threeway_time_sigma_tracking_solvems.png")
    plt.savefig(out_b, dpi=220)
    plt.close(fig)

    # Quick correlations for sanity.
    corr = {
        "corr_sigma_tracking": float(np.corrcoef(sigma, track)[0, 1]),
        "corr_sigma_slack": float(np.corrcoef(sigma, slack)[0, 1]),
        "corr_sigma_solve_ms": float(np.corrcoef(sigma, solve)[0, 1]),
    }
    with open(os.path.join(args.outdir, "fig1_threeway_time_correlations.txt"), "w", encoding="utf-8") as f:
        for k, v in corr.items():
            f.write(f"{k}={v:.6f}\n")

    print(f"Saved: {out_a}")
    print(f"Saved: {out_b}")
    print(f"Saved: {os.path.join(args.outdir, 'fig1_threeway_time_correlations.txt')}")


if __name__ == "__main__":
    main()

