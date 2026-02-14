import warnings
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class OnlineTrajectoryBuffer:
    def __init__(
        self,
        m: int,
        p: int,
        max_steps: int,
        segment_len: int,
        stride: int = 1,
    ) -> None:
        if m <= 0 or p <= 0:
            raise ValueError("m and p must be positive.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if segment_len <= 0:
            raise ValueError("segment_len must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")

        self.m = int(m)
        self.p = int(p)
        self.max_steps = int(max_steps)
        self.segment_len = int(segment_len)
        self.stride = int(stride)

        self._u_buf = np.zeros((self.max_steps, self.m), dtype=float)
        self._y_buf = np.zeros((self.max_steps, self.p), dtype=float)
        self._t_buf = np.full((self.max_steps,), np.nan, dtype=float)
        self._count = 0
        self._head = 0

    def reset(self) -> None:
        self._u_buf.fill(0.0)
        self._y_buf.fill(0.0)
        self._t_buf.fill(np.nan)
        self._count = 0
        self._head = 0

    def append(self, u_t: NDArray, y_t: NDArray, t_t: Optional[float] = None) -> None:
        u = np.asarray(u_t, dtype=float).reshape(-1)
        y = np.asarray(y_t, dtype=float).reshape(-1)
        if u.size != self.m:
            raise ValueError(f"u_t has dim {u.size}, expected {self.m}.")
        if y.size != self.p:
            raise ValueError(f"y_t has dim {y.size}, expected {self.p}.")

        if not np.isfinite(u).all() or not np.isfinite(y).all():
            warnings.warn("OnlineTrajectoryBuffer received NaN/Inf in append().")

        idx = self._head
        self._u_buf[idx, :] = u
        self._y_buf[idx, :] = y
        self._t_buf[idx] = float(t_t) if t_t is not None else float(self._count)

        self._head = (self._head + 1) % self.max_steps
        self._count = min(self._count + 1, self.max_steps)

    def size(self) -> int:
        return self._count

    def can_extract(self) -> bool:
        return self._count >= self.segment_len

    def _ordered_histories(self) -> Tuple[NDArray, NDArray, NDArray]:
        if self._count == 0:
            return (
                np.zeros((0, self.m), dtype=float),
                np.zeros((0, self.p), dtype=float),
                np.zeros((0,), dtype=float),
            )
        if self._count < self.max_steps:
            return (
                self._u_buf[: self._count, :].copy(),
                self._y_buf[: self._count, :].copy(),
                self._t_buf[: self._count].copy(),
            )

        idx = self._head
        return (
            np.vstack((self._u_buf[idx:, :], self._u_buf[:idx, :])).copy(),
            np.vstack((self._y_buf[idx:, :], self._y_buf[:idx, :])).copy(),
            np.hstack((self._t_buf[idx:], self._t_buf[:idx])).copy(),
        )

    def extract_segments(self, mode: str = "rolling") -> Tuple[NDArray, NDArray]:
        if mode != "rolling":
            raise ValueError(f"Unsupported mode: {mode}")

        u_hist, y_hist, _ = self._ordered_histories()
        T = u_hist.shape[0]
        if T < self.segment_len:
            return (
                np.zeros((0, self.segment_len, self.m), dtype=float),
                np.zeros((0, self.segment_len, self.p), dtype=float),
            )

        starts = range(0, T - self.segment_len + 1, self.stride)
        u_segs = [u_hist[s : s + self.segment_len, :] for s in starts]
        y_segs = [y_hist[s : s + self.segment_len, :] for s in starts]
        return np.stack(u_segs, axis=0), np.stack(y_segs, axis=0)

    @staticmethod
    def _block_hankel(w: NDArray, L: int) -> NDArray:
        T, d = w.shape
        if L > T:
            raise ValueError(f"L={L} must be <= T={T}")
        H = np.zeros((d * L, T - L + 1), dtype=float)
        w_vec = w.reshape(-1)
        for i in range(T - L + 1):
            H[:, i] = w_vec[d * i : d * (L + i)]
        return H

    def build_hankel(self, L: int) -> Tuple[NDArray, NDArray]:
        u_hist, y_hist, _ = self._ordered_histories()
        T = u_hist.shape[0]
        if T < L:
            return (
                np.zeros((self.m * L, 0), dtype=float),
                np.zeros((self.p * L, 0), dtype=float),
            )
        return self._block_hankel(u_hist, L), self._block_hankel(y_hist, L)

    def get_Uini_Yini(self, T_ini: int) -> Tuple[NDArray, NDArray]:
        if T_ini <= 0:
            raise ValueError("T_ini must be positive.")
        if self._count < T_ini:
            raise ValueError(
                f"Insufficient data for Uini/Yini: have {self._count}, need {T_ini}."
            )
        u_hist, y_hist, _ = self._ordered_histories()
        u_ini = u_hist[-T_ini:, :].reshape(-1)
        y_ini = y_hist[-T_ini:, :].reshape(-1)
        return u_ini, y_ini

    def debug_snapshot(
        self,
        step_idx: int,
        u_t: Optional[NDArray] = None,
        y_t: Optional[NDArray] = None,
        T_ini: Optional[int] = None,
        L: Optional[int] = None,
        selected_indices: Optional[NDArray] = None,
    ) -> dict:
        u_hist, y_hist, _ = self._ordered_histories()
        snap = {
            "step_idx": int(step_idx),
            "buffer_size": int(self.size()),
            "u_t": None if u_t is None else np.asarray(u_t, dtype=float).reshape(-1),
            "y_t": None if y_t is None else np.asarray(y_t, dtype=float).reshape(-1),
            "u_history": u_hist.copy(),
            "y_history": y_hist.copy(),
            "num_segments": 0,
            "selected_indices": None,
            "sigma_min": np.nan,
            "sigma_max": np.nan,
            "condition_number": np.inf,
            "H_u_shape": None,
            "H_y_shape": None,
            "U_ini": None,
            "Y_ini": None,
            "H_u": None,
            "H_y": None,
            "singular_values": None,
        }

        if self.can_extract():
            T = self.size()
            snap["num_segments"] = 1 + max(0, (T - self.segment_len) // self.stride)

        if T_ini is not None:
            try:
                u_ini, y_ini = self.get_Uini_Yini(int(T_ini))
                snap["U_ini"] = u_ini
                snap["Y_ini"] = y_ini
            except ValueError:
                pass

        if L is None:
            L = self.segment_len
        H_u, H_y = self.build_hankel(int(L))
        snap["H_u"] = H_u
        snap["H_y"] = H_y
        snap["H_u_shape"] = H_u.shape
        snap["H_y_shape"] = H_y.shape

        if selected_indices is not None:
            sel = np.asarray(selected_indices, dtype=int).reshape(-1)
            sel = sel[(sel >= 0) & (sel < H_u.shape[1])]
            snap["selected_indices"] = sel
        elif H_u.shape[1] > 0:
            snap["selected_indices"] = np.arange(H_u.shape[1], dtype=int)

        sel = snap["selected_indices"]
        if sel is not None and sel.size > 0 and H_u.shape[1] > 0:
            U_sel = H_u[:, sel]
            svals = np.linalg.svd(U_sel, compute_uv=False)
            snap["singular_values"] = svals
            smax = float(np.max(svals))
            smin = float(np.min(svals))
            snap["sigma_max"] = smax
            snap["sigma_min"] = smin
            if smin > 0:
                snap["condition_number"] = float(smax / smin)
            else:
                snap["condition_number"] = np.inf

        if (
            not np.isfinite(snap["sigma_min"])
            or not np.isfinite(snap["sigma_max"])
            or np.isinf(snap["condition_number"])
        ):
            warnings.warn(
                f"DEBUG SNAPSHOT step={step_idx}: invalid sigma/condition number values."
            )

        return snap
