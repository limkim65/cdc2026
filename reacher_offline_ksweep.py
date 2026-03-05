import argparse
import os
import sys

import numpy as np


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Reuse existing, validated setup helpers.
from experiments.reacher.run_cdc_reacher_benchmark import (  # noqa: E402
    ReacherSetpointScheduler,
    compute_tracking_error,
    compute_success,
    fix_dataset,
    get_cost_dict,
    setup_deepc_reacher,
)
from select_deepc.data_selectors import LkSelector  # noqa: E402
from select_deepc.data_selectors import AdaptiveLkSelector  # noqa: E402
from select_deepc.deepc_controller import SelectDeePC  # noqa: E402
from select_deepc.deepc_utils import (  # noqa: E402
    DeePCCostAccumulator,
    IntegralAbsoluteError,
    IntegralSquareError,
    PerformanceAccumulatorWrapper,
    get_reacher_simulator,
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

def sigma_min_matrix(M: np.ndarray) -> float:
    if M.size == 0 or M.shape[1] == 0:
        return np.nan
    try:
        s = np.linalg.svd(M, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.nan
    return float(s[-1])


from scipy.linalg import qr

def cpqr_pivot_indices(M: np.ndarray, normalize_cols: bool = True, eps: float = 1e-12) -> np.ndarray:
    if M.size == 0 or M.shape[1] == 0:
        return np.array([], dtype=np.int64)
    A = M
    if normalize_cols:
        norms = np.linalg.norm(M, axis=0)
        A = M / (norms + eps)
    _, _, p = qr(A, pivoting=True, mode="economic")
    return np.asarray(p, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Simple Reacher runner.")
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("logs", "reacher", "offline_ksweep_dir"),
    )
    parser.add_argument("--dataset", choices=["iid", "random_walk", "both"], default="iid")
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=80, help="Fixed K, or initial K for adaptive.")
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--adaptive_k", action="store_true")
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--Kmin", type=int, default=40)
    parser.add_argument("--Kmax", type=int, default=320)
    parser.add_argument("--Kstep", type=int, default=1)
    parser.add_argument("--gamma_min", type=float, default=1e-6)
    parser.add_argument("--eps_sigma", type=float, default=1e-3)
    parser.add_argument("--success_eps", type=float, default=0.01)
    parser.add_argument("--success_window", type=int, default=20)
    parser.add_argument("--num_extra_targets", type=int, default=0)
    parser.add_argument("--enable_measurement_constraint", action="store_true")
    parser.add_argument("--N_loc", type=int, default=1000)
    parser.add_argument("--d_max", type=float, default=10.0)
    parser.add_argument("--sigma_bar", type=float, default=1e-6)
    args = parser.parse_args()
 
    n_iter=1
    Kmax=10000
    Kstep=1
    N_loc=1000
    d_max=10.0
    sigma_bar=0.55

    os.makedirs(args.outdir, exist_ok=True)

    iid = load_data_from_folder(os.path.join(REPO_ROOT, "data", "reacher", "dataset_iid"), "iid")
    rw = load_data_from_folder(
        os.path.join(REPO_ROOT, "data", "reacher", "dataset_random_walk"), "random_walk"
    )
    for ds in [iid, rw]:
        fix_dataset(ds)

    if args.dataset == "iid":
        data = iid
    elif args.dataset == "random_walk":
        data = rw
    else:
        data = iid + rw

    controller_args = setup_deepc_reacher(
        data, enable_measurement_constraint=bool(args.enable_measurement_constraint)
    )
    env = get_reacher_simulator(
        num_steps=int(args.max_steps),
        video_folder=os.path.join(args.outdir, "video"),
        video_title="reacher_run",
    )

    controller_args.controller_costs.g_regularization_pi = 0

    deepc = SelectDeePC(
        controller_args,
        selector_callback=AdaptiveLkSelector(order=2),
        sigma_bar=float(args.sigma_bar),
        N_loc=int(args.N_loc),
        d_max=float(args.d_max),
        num_hankel_cols=int(args.K),
        n_iter=int(args.n_iter),
        K_min=int(args.Kmin),
        K_max=int(args.Kmax),
    )
    deepc._eps_sigma = 1e-6

    scheduler = ReacherSetpointScheduler(
        env, controller_args.deepc_dims, num_extra_targets=int(args.num_extra_targets)
    )
    x_traj, u_traj, _, _, cost_obj, _ = run_reacher_simulator(
        env,
        deepc,
        scheduler,
        seed=int(args.seed),
        deepc_cost_accumulator=PerformanceAccumulatorWrapper(
            DeePCCostAccumulator(controller_args.controller_costs),
            IntegralSquareError(mask=np.array([0, 0, 0, 0, 1, 1, 0, 0])),
            IntegralAbsoluteError(mask=np.array([0, 0, 0, 0, 1, 1, 0, 0])),
        ),
    )

    target_xy = np.asarray(
        scheduler._targets.targets[scheduler._target_idx], dtype=float
    ).reshape(1, 2)
    target_traj = np.repeat(target_xy, repeats=max(1, len(x_traj)), axis=0)
    err = compute_tracking_error(np.asarray(x_traj, dtype=float), target_traj)
    success = compute_success(err, eps=float(args.success_eps), window=int(args.success_window))

    mode_tag = "adaptive" if args.adaptive_k else "fixed"
    out_npz = os.path.join(
        args.outdir,
        f"run_{mode_tag}_K{int(args.K):03d}_rho{float(args.rho):.3g}_seed{int(args.seed):03d}.npz",
    )
    deepc.save_history_npz(
        out_npz,
        extra={
            "seed": int(args.seed),
            "K": int(args.K),
            "adaptive_k": int(bool(args.adaptive_k)),
            "rho": float(args.rho),
            "total_cost": float(get_cost_dict(cost_obj).get("cost", np.nan)),
            "rmse_ee": float(np.sqrt(safe_stat(err * err, np.mean, default=np.nan))),
            "success": int(success),
            "N_loc": int(args.N_loc),
            "d_max": float(args.d_max),
            "sigma_bar": float(args.sigma_bar),
            "local_gate_fail": int(deepc._local_gate_fail),
            "cond_gate_fail": int(deepc._cond_gate_fail),
            "d_k_history": np.asarray(deepc._d_k_history, dtype=float),
            "selected_idcs_history": np.array(deepc._selected_idcs_history, dtype=object),
            "executed_state_traj": np.asarray(x_traj, dtype=float),
            "executed_input_traj": np.asarray(u_traj, dtype=float),
            "tracking_error": np.asarray(err, dtype=float),
            "K_opt_history": np.asarray(deepc._K_opt_history, dtype=int),
        },
    )

    Hu = deepc.get_whole_input_hankel_matrix()
    Hy = deepc.get_whole_output_hankel_matrix()
    
    ranked_idcs_history = deepc.get_ranked_idcs_history()
    ranked_norms_history = deepc.get_ranked_norms_history()
    sigma_min_Mk_history_fixed = []
    step_list = [10, 50, 100, 150, 200, 250, 300, 350]
    k_list = list(range(10, 301, 10))

    #fixed k, selection based on only similarity
    for step in step_list:
        ranked_idcs = np.array(list(map(int, ranked_idcs_history[step])), dtype=np.int64)
        for k in k_list:
            ranked_idcs_k=ranked_idcs[:k]
            Hu_k=Hu[:, ranked_idcs_k]
            Mk=Hu_k
            step_min_sigma_Mk = sigma_min_matrix(Mk)
            sigma_min_Mk_history_fixed.append(step_min_sigma_Mk)

    #0) plot fixed k, selection based on only similarity
    import matplotlib.pyplot as plt
    arr = np.asarray(sigma_min_Mk_history_fixed, dtype=float).reshape(len(step_list), len(k_list))
    plt.figure(figsize=(10, 5))
    for i, step in enumerate(step_list):
        plt.plot(k_list, arr[i, :], marker='o', linewidth=1.5, label=f"step={step}")

    plt.xlabel("k")
    plt.ylabel("min sigma(Mk)")
    plt.title("min sigma(Mk) vs k (8 curves by step)")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "sigma_min_Mk_vs_k_by_step.png"), dpi=200)
    plt.show()
    
    
    #adaptive k, selection based on local gate and condition gate
    sigma_min_Mk_history_cpqr = []
    sigma_min_Mk_history_orig = []
    ratio_history = []
    for step in step_list:
    # 1) local gate / locality ranking 결과 (가까운 순)
        cand = np.array(list(map(int, ranked_idcs_history[step])), dtype=np.int64)

        # locality 유지: 가까운 애들 중에서만(=pool) conditioning 보정
        cand_pool = cand[:N_loc]   # 너가 쓰는 N_loc

        # 2) 후보 pool에서 CPQR로 재정렬 (conditioning 우선)
        Hu_sub = Hu[:, cand_pool]
        piv = cpqr_pivot_indices(Hu_sub, normalize_cols=True)
        ranked_cpqr = cand_pool[piv]  # 원본 column index로 복원

        # 3) k-sweep
        for k in k_list:
            idx_orig = cand_pool[:k]
            idx_cpqr = ranked_cpqr[:k]
            Mk_cpqr = Hu[:, idx_cpqr]
            Mk_orig = Hu[:, idx_orig]
            step_min_sigma_Mk_cpqr = sigma_min_matrix(Mk_cpqr)
            step_min_sigma_Mk_orig = sigma_min_matrix(Mk_orig)
            sigma_min_Mk_history_cpqr.append(step_min_sigma_Mk_cpqr)
            sigma_min_Mk_history_orig.append(step_min_sigma_Mk_orig)
            ratio_history.append(step_min_sigma_Mk_cpqr / (step_min_sigma_Mk_orig + 1e-12))

    sigma_orig_arr = np.asarray(sigma_min_Mk_history_orig, dtype=float).reshape(len(step_list), len(k_list))
    sigma_cpqr_arr = np.asarray(sigma_min_Mk_history_cpqr, dtype=float).reshape(len(step_list), len(k_list))
    ratio_arr      = np.asarray(ratio_history, dtype=float).reshape(len(step_list), len(k_list))

    #1) plot min sigma(Mk)_cpqr vs k
    plt.figure(figsize=(10, 5))
    for i, step in enumerate(step_list):
        plt.plot(k_list, sigma_cpqr_arr[i, :], marker='o', linewidth=1.5, label=f"step={step}")

    plt.xlabel("k")
    plt.ylabel("min sigma(Mk)_cpqr")
    plt.title("min sigma(Mk)_cpqr vs k (8 curves by step)")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "sigma_min_Mk_cpqr_vs_k_by_step.png"), dpi=200)
    plt.show()


    # 1) plot ratio plot
    plt.figure(figsize=(10, 5))
    for i, step in enumerate(step_list):
        plt.plot(k_list, ratio_arr[i, :], marker='o', linewidth=1.5, label=f"step={step}")
    plt.axhline(1.0, linestyle="--", linewidth=1.0)  # 기준선
    plt.xlabel("k")
    plt.ylabel("sigma_min(CPQR) / sigma_min(orig)")
    plt.title("conditioning gain ratio (CPQR vs orig)")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "ratio_cpqr_over_orig_vs_k_by_step.png"), dpi=200)
    plt.show()
        

if __name__ == "__main__":
    main()
