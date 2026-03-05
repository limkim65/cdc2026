import argparse
import csv
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Reuse the existing benchmark helpers to keep experiment settings identical.
sys.path.append(os.path.dirname(__file__))
import run_cdc_reacher_benchmark as base  # noqa: E402


class MeasurementNoiseWrapper:
    """Inject Gaussian noise into measured state before controller call."""

    def __init__(self, controller, noise_std: float, seed: int):
        self._controller = controller
        self._noise_std = float(noise_std)
        self._rng = np.random.default_rng(seed=int(seed))

    def compute_action(self, state, reference):
        s = np.asarray(state, dtype=float)
        if self._noise_std > 0:
            s = s + self._rng.normal(0.0, self._noise_std, size=s.shape)
        return self._controller.compute_action(s, reference)

    def __getattr__(self, name):
        return getattr(self._controller, name)


def run_once_with_noise(
    args,
    controller_args,
    seed: int,
    mode: str,
    K: int,
    rho: float,
    noise_std: float,
):
    env = base.get_reacher_simulator(
        num_steps=int(args.max_steps),
        video_folder=os.path.join(args.outdir, "video"),
        video_title=(
            f"{mode}_K{int(K)}_rho{float(rho):.3g}_noise{float(noise_std):.3g}_"
            f"seed{int(seed):03d}"
        ),
    )
    adaptive = bool(mode == "adaptive_rho")
    deepc = base.SelectDeePC(
        controller_args,
        selector_callback=base.LkSelector(),
        num_hankel_cols=int(K),
        n_iter=int(args.n_iter),
        adaptive_k=adaptive,
        K_min=int(args.Kmin),
        K_max=int(args.Kmax),
        K_step=int(args.Kstep),
        rho=float(rho),
        gamma_min=float(args.gamma_min),
    )
    deepc._eps_sigma = float(args.eps_sigma)

    noisy_controller = MeasurementNoiseWrapper(
        deepc,
        noise_std=float(noise_std),
        seed=int(seed + 1000 * round(1000.0 * float(noise_std))),
    )

    scheduler = base.ReacherSetpointScheduler(
        env, controller_args.deepc_dims, num_extra_targets=int(args.num_extra_targets)
    )
    x_traj, u_traj, _, _, cost_obj, _ = base.run_reacher_simulator(
        env,
        noisy_controller,
        scheduler,
        seed=int(seed),
        deepc_cost_accumulator=base.PerformanceAccumulatorWrapper(
            base.DeePCCostAccumulator(controller_args.controller_costs),
            base.IntegralSquareError(mask=np.array([0, 0, 0, 0, 1, 1, 0, 0])),
            base.IntegralAbsoluteError(mask=np.array([0, 0, 0, 0, 1, 1, 0, 0])),
        ),
    )

    cost = base.get_cost_dict(cost_obj)
    k_hist = np.asarray(deepc.get_K_opt_history(), dtype=float)
    sigma_ag_hist = np.asarray(deepc.get_sigma_min_A_history(), dtype=float)
    slack_hist = np.asarray(deepc.get_slack_norm_inf_history(), dtype=float)
    kkt_hist = np.asarray(deepc.get_kkt_residual_history(), dtype=float)
    eq_hist = np.asarray(deepc.get_eq_residual_history(), dtype=float)
    solve_hist = np.asarray(deepc.get_solve_time_ms_history(), dtype=float)
    status_hist = np.asarray(deepc.get_status_history(), dtype=object)

    loc_hist, kappa_hist = base.compute_localness_and_conditioning(deepc, x_traj)
    tgt_hist = base.replay_targets(x_traj, num_extra_targets=int(args.num_extra_targets))
    err = base.compute_tracking_error(x_traj, tgt_hist)
    act = base.compute_action_metrics(u_traj, u_limit=0.5)
    success = base.compute_success(
        err, eps=float(args.success_eps), window=int(args.success_window)
    )
    fail_solver = np.array([0 if ("optimal" in str(s)) else 1 for s in status_hist], dtype=float)

    tag = (
        f"{mode}_K{int(K):03d}_rho{float(rho):.3g}_noise{float(noise_std):.3g}_"
        f"Kmin{int(args.Kmin)}_Kmax{int(args.Kmax)}_Kstep{int(args.Kstep)}_"
        f"gmin{float(args.gamma_min):.3g}"
    )
    npz_path = os.path.join(args.outdir, f"run_{tag}_seed{int(seed):03d}.npz")
    deepc.save_history_npz(
        npz_path,
        extra={
            "mode": mode,
            "seed": int(seed),
            "K": int(K),
            "rho": float(rho),
            "measurement_noise_std": float(noise_std),
            "steps": int(np.asarray(x_traj).shape[0]),
            "sigma_min_Ag_episode_min": float(base.safe_stat(sigma_ag_hist, np.min)),
            "kappa_Ag_mean": float(base.safe_stat(kappa_hist, np.mean)),
            "localness_mean": float(base.safe_stat(loc_hist, np.mean)),
            "localness_max": float(base.safe_stat(loc_hist, np.max)),
            "kkt_res_max": float(base.safe_stat(kkt_hist, np.max)),
            "eq_res_max": float(base.safe_stat(eq_hist, np.max)),
            "slack_max": float(base.safe_stat(slack_hist, np.max)),
            "slack_mean": float(base.safe_stat(slack_hist, np.mean)),
            "solver_fail_rate": float(base.safe_stat(fail_solver, np.mean, default=0.0)),
            "rmse_ee": float(np.sqrt(base.safe_stat(err * err, np.mean, default=np.nan))),
            "err_p95": float(base.safe_stat(err, lambda x: np.quantile(x, 0.95))),
            "err_max": float(base.safe_stat(err, np.max)),
            "err_final": float(base.safe_stat(err[-1:], np.mean)),
            "success": int(success),
            "effort_sum_sq": float(act["effort_sum_sq"]),
            "jerk_sum_sq": float(act["jerk_sum_sq"]),
            "jerk_max": float(act["jerk_max"]),
            "u_violation_rate": float(act["u_violation_rate"]),
            "total_cost": float(cost.get("cost", np.nan)),
            "ISE": float(cost.get("ISE", np.nan)),
            "IAE": float(cost.get("IAE", np.nan)),
            "solve_ms_mean": float(base.safe_stat(solve_hist, np.mean)),
            "solve_ms_total": float(base.safe_stat(solve_hist, np.sum)),
            "k_opt_mean": float(base.safe_stat(k_hist, np.mean)),
            "k_opt_var": float(base.safe_stat(k_hist, np.var)),
            "executed_state_traj": np.asarray(x_traj, dtype=float),
            "executed_input_traj": np.asarray(u_traj, dtype=float),
            "tracking_error": np.asarray(err, dtype=float),
            "target_xy": np.asarray(tgt_hist, dtype=float),
            "localness_hist": np.asarray(loc_hist, dtype=float),
            "kappa_Ag_hist": np.asarray(kappa_hist, dtype=float),
        },
    )
    sel_png = os.path.join(args.outdir, f"selection_count_{tag}_seed{int(seed):03d}.png")
    save_selection_count_plot(k_hist=k_hist, out_png=sel_png)

    return {
        "mode": mode,
        "seed": int(seed),
        "K": int(K),
        "rho": float(rho),
        "measurement_noise_std": float(noise_std),
        "steps": int(np.asarray(x_traj).shape[0]),
        "k_opt_mean": float(base.safe_stat(k_hist, np.mean)),
        "k_opt_var": float(base.safe_stat(k_hist, np.var)),
        "sigma_min_Ag": float(base.safe_stat(sigma_ag_hist, np.min)),
        "kappa_Ag_mean": float(base.safe_stat(kappa_hist, np.mean)),
        "localness_mean": float(base.safe_stat(loc_hist, np.mean)),
        "slack_max": float(base.safe_stat(slack_hist, np.max)),
        "slack_mean": float(base.safe_stat(slack_hist, np.mean)),
        "kkt_res_max": float(base.safe_stat(kkt_hist, np.max)),
        "eq_res_max": float(base.safe_stat(eq_hist, np.max)),
        "solver_fail_rate": float(base.safe_stat(fail_solver, np.mean, default=0.0)),
        "rmse_ee": float(np.sqrt(base.safe_stat(err * err, np.mean, default=np.nan))),
        "err_p95": float(base.safe_stat(err, lambda x: np.quantile(x, 0.95))),
        "err_max": float(base.safe_stat(err, np.max)),
        "err_final": float(base.safe_stat(err[-1:], np.mean)),
        "success": int(success),
        "effort_sum_sq": float(act["effort_sum_sq"]),
        "jerk_sum_sq": float(act["jerk_sum_sq"]),
        "jerk_max": float(act["jerk_max"]),
        "u_violation_rate": float(act["u_violation_rate"]),
        "total_cost": float(cost.get("cost", np.nan)),
        "ISE": float(cost.get("ISE", np.nan)),
        "IAE": float(cost.get("IAE", np.nan)),
        "solve_ms_mean": float(base.safe_stat(solve_hist, np.mean)),
        "solve_ms_total": float(base.safe_stat(solve_hist, np.sum)),
        "npz_path": npz_path,
    }


def summarize_by_noise(rows: List[dict]) -> List[dict]:
    out = []
    levels = sorted(set(float(r["measurement_noise_std"]) for r in rows))
    for nstd in levels:
        subset = [r for r in rows if float(r["measurement_noise_std"]) == nstd]
        out.append(
            {
                "measurement_noise_std": float(nstd),
                "runs": int(len(subset)),
                "median_total_cost": base.safe_stat([r["total_cost"] for r in subset], np.median),
                "median_rmse_ee": base.safe_stat([r["rmse_ee"] for r in subset], np.median),
                "median_sigma_min_Ag": base.safe_stat([r["sigma_min_Ag"] for r in subset], np.median),
                "median_slack_max": base.safe_stat([r["slack_max"] for r in subset], np.median),
                "median_solve_ms": base.safe_stat([r["solve_ms_mean"] for r in subset], np.median),
                "success_rate": base.safe_stat([r["success"] for r in subset], np.mean, default=0.0),
            }
        )
    return out


def save_selection_count_plot(k_hist: np.ndarray, out_png: str) -> None:
    k = np.asarray(k_hist, dtype=float).reshape(-1)
    if k.size == 0:
        return
    steps = np.arange(1, k.size + 1, dtype=int)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.8), dpi=150)
    ax.plot(steps, k, color="tab:blue", lw=1.4)
    ax.set_xlabel("step")
    ax.set_ylabel("selected rows (K_opt)")
    ax.set_title("Selection Count Over Time")
    ax.grid(True, alpha=0.25)
    ymax = float(np.nanmax(k)) if np.any(np.isfinite(k)) else float(np.max(k))
    if np.isfinite(ymax) and ymax > 0:
        ax.set_ylim(0, ymax * 1.05)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="CDC Reacher benchmark with measurement-noise injection."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(REPO_ROOT, "logs", "reacher", "reacher_cdc_both_noise"),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fixed_k", "adaptive_rho", "both"],
        default="both",
    )
    parser.add_argument("--measurement_noise_std", type=float, nargs="+", default=[0.0, 0.005, 0.01])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--Kmin", type=int, default=40)
    parser.add_argument("--Kmax", type=int, default=200)
    parser.add_argument("--Kstep", type=int, default=10)
    parser.add_argument("--rho", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--gamma_min", type=float, default=1e-6)
    parser.add_argument("--eps_sigma", type=float, default=1e-3)
    parser.add_argument("--success_eps", type=float, default=0.01)
    parser.add_argument("--success_window", type=int, default=20)
    parser.add_argument("--num_extra_targets", type=int, default=0)
    parser.add_argument(
        "--dataset", type=str, choices=["iid", "random_walk", "both"], default="iid"
    )
    parser.add_argument("--enable_measurement_constraint", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    iid = base.load_data_from_folder(os.path.join(REPO_ROOT, "data", "reacher", "dataset_iid"), "iid")
    rw = base.load_data_from_folder(
        os.path.join(REPO_ROOT, "data", "reacher", "dataset_random_walk"), "random_walk"
    )
    for ds in [iid, rw]:
        base.fix_dataset(ds)

    if args.dataset == "iid":
        data = iid
    elif args.dataset == "random_walk":
        data = rw
    else:
        data = iid + rw

    controller_args = base.setup_deepc_reacher(
        data, enable_measurement_constraint=bool(args.enable_measurement_constraint)
    )

    rows: List[dict] = []
    run_fixed = args.mode in ["fixed_k", "both"]
    run_adapt = args.mode in ["adaptive_rho", "both"]

    for noise_std in args.measurement_noise_std:
        if run_fixed:
            for K in range(int(args.Kmin), int(args.Kmax) + 1, int(args.Kstep)):
                for seed in range(int(args.seeds)):
                    row = run_once_with_noise(
                        args=args,
                        controller_args=controller_args,
                        seed=seed,
                        mode="fixed_k",
                        K=K,
                        rho=float(args.rho[0]),
                        noise_std=float(noise_std),
                    )
                    rows.append(row)
                    print(
                        f"[fixed_k] noise={float(noise_std):.4g} K={int(K):3d} seed={int(seed):3d} "
                        f"cost={row['total_cost']:.3e} rmse={row['rmse_ee']:.3e} succ={int(row['success'])}"
                    )

        if run_adapt:
            for rho in args.rho:
                for seed in range(int(args.seeds)):
                    row = run_once_with_noise(
                        args=args,
                        controller_args=controller_args,
                        seed=seed,
                        mode="adaptive_rho",
                        K=int(args.Kmin),
                        rho=float(rho),
                        noise_std=float(noise_std),
                    )
                    rows.append(row)
                    print(
                        f"[adaptive_rho] noise={float(noise_std):.4g} rho={float(rho):.3g} seed={int(seed):3d} "
                        f"cost={row['total_cost']:.3e} rmse={row['rmse_ee']:.3e} succ={int(row['success'])}"
                    )

    summary_path = os.path.join(args.outdir, "summary.csv")
    if rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        by_noise = summarize_by_noise(rows)
        by_noise_path = os.path.join(args.outdir, "summary_by_noise.csv")
        with open(by_noise_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(by_noise[0].keys()))
            writer.writeheader()
            writer.writerows(by_noise)
    else:
        with open(summary_path, "w", newline="") as f:
            f.write("")
        by_noise_path = os.path.join(args.outdir, "summary_by_noise.csv")
        with open(by_noise_path, "w", newline="") as f:
            f.write("")

    print(f"\nSaved run summary: {summary_path}")
    print(f"Saved by-noise summary: {by_noise_path}")
    print(f"Saved logs under: {args.outdir}")


if __name__ == "__main__":
    main()
