import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


class OnlineBufferDebugVisualizer:
    def __init__(
        self,
        max_steps: int,
        pause_s: float = 0.02,
        save_dir: str = "results",
        gamma: Optional[float] = None,
    ) -> None:
        self.max_steps = int(max_steps)
        self.pause_s = float(pause_s)
        self.save_dir = save_dir
        self.gamma = gamma
        os.makedirs(self.save_dir, exist_ok=True)

        self._sigma_hist = []
        self._step_hist = []
        self._save_requested = False

        plt.ion()
        self.fig, self.axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def _on_key_press(self, event):
        if event.key == "s":
            self._save_requested = True

    def update(self, snap: dict, selected_k: Optional[int] = None):
        step_idx = int(snap["step_idx"])
        u_hist = snap["u_history"]
        y_hist = snap["y_history"]
        sigma_min = snap["sigma_min"]
        buffer_size = int(snap["buffer_size"])
        num_segments = int(snap["num_segments"])

        self._step_hist.append(step_idx)
        self._sigma_hist.append(sigma_min)

        recent = 200
        st = max(0, u_hist.shape[0] - recent)
        u_recent = u_hist[st:, :]
        y_recent = y_hist[st:, :]
        x_axis = np.arange(st, st + u_recent.shape[0])

        ax_u, ax_y, ax_s, ax_b = self.axes
        for ax in self.axes:
            ax.cla()

        # [1] Input history
        ax_u.plot(x_axis, u_recent[:, 0], label="u")
        ax_u.axvline(step_idx, color="k", linestyle="--", linewidth=1)
        ax_u.set_title("Input history (recent 200)")
        ax_u.set_ylabel("u")
        ax_u.grid(True, alpha=0.3)
        ax_u.legend()

        # [2] Output history
        for j in range(y_recent.shape[1]):
            ax_y.plot(x_axis, y_recent[:, j], label=f"y[{j}]")
        ax_y.axvline(step_idx, color="k", linestyle="--", linewidth=1)
        ax_y.set_title("Output history y (measurement)")
        ax_y.set_ylabel("y")
        ax_y.grid(True, alpha=0.3)
        ax_y.legend(ncols=min(3, y_recent.shape[1]), fontsize=8)

        # [3] Sigma-min trajectory
        ax_s.plot(self._step_hist, self._sigma_hist, label="sigma_min(U_selected)")
        if self.gamma is not None:
            ax_s.axhline(self.gamma, color="r", linestyle="--", linewidth=1, label="gamma")
        ax_s.set_title("Sigma-min trajectory")
        ax_s.set_ylabel("sigma_min")
        ax_s.grid(True, alpha=0.3)
        ax_s.legend()

        # [4] Buffer utilization
        util = 100.0 * buffer_size / max(self.max_steps, 1)
        ax_b.bar(["size"], [buffer_size], color="tab:blue", alpha=0.8, label="buffer_size")
        ax_b.axhline(self.max_steps, color="k", linestyle="--", linewidth=1, label="max_steps")
        txt = f"util={util:.1f}%\nsegments={num_segments}\nK={selected_k if selected_k is not None else 'N/A'}"
        ax_b.text(0.05, 0.95, txt, transform=ax_b.transAxes, va="top")
        ax_b.set_title("Buffer utilization")
        ax_b.set_ylabel("count")
        ax_b.grid(True, alpha=0.3)
        ax_b.legend()

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        plt.pause(self.pause_s)

        if self._save_requested:
            self.save_snapshot(snap)
            self._save_requested = False

    def save_snapshot(self, snap: dict):
        step_idx = int(snap["step_idx"])
        png_path = os.path.join(self.save_dir, f"debug_snapshot_step_{step_idx}.png")
        npz_path = os.path.join(self.save_dir, f"debug_snapshot_step_{step_idx}.npz")
        self.fig.savefig(png_path, dpi=180, bbox_inches="tight")
        np.savez(
            npz_path,
            U_ini=np.array([]) if snap["U_ini"] is None else snap["U_ini"],
            H_u=np.array([]) if snap["H_u"] is None else snap["H_u"],
            selected_indices=np.array([]) if snap["selected_indices"] is None else snap["selected_indices"],
            singular_values=np.array([]) if snap["singular_values"] is None else snap["singular_values"],
        )
