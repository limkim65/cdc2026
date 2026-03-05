import argparse
import glob
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def to_2d(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a


def extract_ee_xy(x_traj: np.ndarray) -> np.ndarray:
    x = to_2d(x_traj)
    # Original MuJoCo reacher observation in this repo is 10D:
    # ee_xy = obs[:,4:6] + obs[:,8:10]
    if x.shape[1] >= 10:
        return x[:, 4:6] + x[:, 8:10]
    # Some logs may already store transformed 8D observation.
    if x.shape[1] >= 6:
        return x[:, 4:6]
    raise ValueError(f"Unexpected state dimension for EE extraction: {x.shape}")


def align_target(target_xy: np.ndarray, T: int) -> np.ndarray:
    t = to_2d(target_xy)
    if t.shape[0] == T:
        return t
    if t.shape[0] == 1:
        return np.repeat(t, T, axis=0)
    if t.shape[0] > T:
        return t[:T]
    # t shorter than trajectory: hold last reference.
    pad = np.repeat(t[-1:, :], T - t.shape[0], axis=0)
    return np.vstack([t, pad])


def plot_one(npz_path: str, outdir: str) -> str:
    d = np.load(npz_path, allow_pickle=True)
    if "executed_state_traj" not in d.files:
        raise ValueError(f"`executed_state_traj` not found: {npz_path}")
    if "target_xy" not in d.files:
        raise ValueError(f"`target_xy` not found: {npz_path}")

    x_traj = np.asarray(d["executed_state_traj"], dtype=float)
    ee_xy = extract_ee_xy(x_traj)
    T = ee_xy.shape[0]
    ref_xy = align_target(np.asarray(d["target_xy"], dtype=float), T)
    steps = np.arange(T, dtype=int)

    fig, axs = plt.subplots(2, 1, figsize=(9.0, 5.6), dpi=170, sharex=True)
    axs[0].plot(steps, ee_xy[:, 0], color="tab:blue", lw=1.4, label="ee_x")
    axs[0].plot(steps, ref_xy[:, 0], color="tab:blue", lw=1.2, ls="--", label="ref_x")
    axs[0].set_ylabel("x")
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(frameon=False, ncol=2)

    axs[1].plot(steps, ee_xy[:, 1], color="tab:orange", lw=1.4, label="ee_y")
    axs[1].plot(steps, ref_xy[:, 1], color="tab:orange", lw=1.2, ls="--", label="ref_y")
    axs[1].set_ylabel("y")
    axs[1].set_xlabel("step")
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(frameon=False, ncol=2)

    title = os.path.basename(npz_path).replace(".npz", "")
    fig.suptitle(f"Reacher EE x/y vs Reference: {title}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, f"{title}_xy_vs_ref.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def gather_npz(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    return sorted(glob.glob(os.path.join(input_path, "*.npz")))


def main():
    parser = argparse.ArgumentParser(description="Plot Reacher EE x/y trajectory with reference.")
    parser.add_argument("--input", type=str, required=True, help="npz file or directory")
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("logs", "reacher", "figure", "trajectory_xy_ref"),
    )
    parser.add_argument("--max_runs", type=int, default=10)
    args = parser.parse_args()

    files = gather_npz(args.input)
    if not files:
        print(f"No npz found in: {args.input}")
        return

    saved = []
    limit = int(max(1, args.max_runs))
    for p in files[:limit]:
        try:
            out = plot_one(p, args.outdir)
            saved.append(out)
        except Exception as e:
            print(f"[skip] {os.path.basename(p)}: {e}")

    print(f"Plotted {len(saved)} run(s).")
    for s in saved:
        print(f"Saved: {s}")


if __name__ == "__main__":
    main()

