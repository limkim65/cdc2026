import warnings

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from typing import Optional

from select_deepc.base_controllers import BaseController
from select_deepc.data_selectors import *
from select_deepc.deepc_utils import *
from select_deepc.hankel_generation import *

import numpy as np
from scipy.linalg import qr as qr_pivoted

def _to_scalar_float(value, default=np.nan) -> float:
    if hasattr(value, "value"):
        value = value.value
    if value is None:
        return float(default)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return float(default)
    if arr.size == 0:
        return float(default)
    return float(arr[0])


class DataDrivenPredictiveController(BaseController):
    """Data Driven Predictive Controller class which produces control actions based on
    the linearized system equations.
    """

    def __init__(
        self,
        H_u: NDArray,
        H_y: NDArray,
        deepc_dims: DeePCDims,
        controller_costs: DeePCCost,
        controller_constraints: DeePCConstraints,
        input_reference,
        verbose: bool = False,
        decompose_hankel_matrix: bool = False,
    ) -> None:
        """Init the mpc controller and set up all the relevant variables"""
        self._T_fut = deepc_dims.T_fut
        self._T_past = deepc_dims.T_past

        self._m = deepc_dims.m
        self._p = deepc_dims.p

        self._input_reference = input_reference

        self._initial_run = True

        self._verbose = verbose

        self._prev_input = None
        self._prev_u = None

        self._open_loop_index = 1

        self._setup_solver(
            H_u,
            H_y,
            controller_costs,
            controller_constraints,
            decompose_hankel_matrix,
        )

        self._initial_state_overridden = False
        self._last_solver_status = None
        self._last_eq_residual = np.nan
        self._last_eq_res_u_past = np.nan
        self._last_eq_res_y_past = np.nan
        self._last_eq_res_u_fut = np.nan
        self._last_kkt_residual = np.nan

    @classmethod
    def create_from_data(
        cls,
        deepc_args: DeePCControllerArgs,
        decompose_hankel_matrices=False,
    ):
        H_u, H_y = HankelMatrixGenerator(
            deepc_args.deepc_dims.T_past,
            deepc_args.deepc_dims.T_fut,
            deepc_args.T_hankel,
        ).generate_hankel_matrices(deepc_args.trajectory_data)
        print(f"deepc has hankel shapes h_u: {H_u.shape} and h_y: {H_y.shape}")

        return cls(
            H_u,
            H_y,
            deepc_args.deepc_dims,
            deepc_args.controller_costs,
            deepc_args.controller_constraints,
            deepc_args.input_reference,
            deepc_args.verbose,
            decompose_hankel_matrices,
        )

    def _setup_solver(
        self,
        H_u: NDArray,
        H_y: NDArray,
        controller_costs: DeePCCost,
        controller_constraints: DeePCConstraints,
        decompose_hankel_matrix: bool,
    ) -> None:
        """Initialize the mpc solver.

        We set up the convex problem once as it does not change from iteration to
        iteration to save some computational effort. The problem is initialized
        using the x_0 parameter which indicates the initial state of the rocket.
        We furthermore use a slack variable for the input constraints to ensure that
        the problem does not become infeasible during operation.
        We also do not directly penalize actuator action but changes in actuator
        commands.
        """

        if not decompose_hankel_matrix:
            U_p = H_u[: self._m * self._T_past, :]
            U_f = H_u[self._m * self._T_past :, :]
            Y_p = H_y[: self._p * self._T_past, :]
            Y_f = H_y[self._p * self._T_past :, :]
        else:
            H = np.vstack((H_u, H_y))
            jitter = 1e-10
            while True:
                try:
                    hankel_factor = np.linalg.cholesky(
                        H @ H.T + jitter * np.eye(H.shape[0])
                    )
                    # print(f"added jitter of {jitter} to hankel matrix for pd-ness")
                    break

                except np.linalg.LinAlgError:
                    jitter *= 10

            U_p = hankel_factor[: self._m * self._T_past, :]
            U_f = hankel_factor[
                self._m * self._T_past : self._m * (self._T_past + self._T_fut), :
            ]
            Y_p = hankel_factor[
                self._m
                * (self._T_past + self._T_fut) : self._m
                * (self._T_past + self._T_fut)
                + self._p * self._T_past,
                :,
            ]
            Y_f = hankel_factor[
                self._m * (self._T_past + self._T_fut) + self._p * self._T_past :,
                :,
            ]

        # Save decomposition blocks for residual logging after each solve.
        self._U_p = U_p
        self._U_f = U_f
        self._Y_p = Y_p
        self._Y_f = Y_f

        self._g = cp.Variable(U_f.shape[-1])
        self._u = cp.Variable(self._T_fut * self._m)
        self._y = cp.Variable(self._T_fut * self._p)

        self._u_past = cp.Parameter(self._T_past * self._m)
        self._y_past = cp.Parameter(self._T_past * self._p)
        self._reference_traj = cp.Parameter(self._T_fut * self._p)
        self._slack_cost = float(controller_costs.slack_cost)
        self._Qblk = np.kron(np.eye(self._T_fut), controller_costs.Q)
        self._Rblk = np.kron(np.eye(self._T_fut), controller_costs.R)

        cost = 0
        constr = []

        self._y_past_slack = cp.Variable(self._p * self._T_past)

        # Phi = generate_multistep_predictor_alt(
        #     H_u, H_y, DeePCDims(self._T_past, self._T_fut, self._p, self._m)
        # )
        # u_p_size = self._T_past * self._m
        # y_p_size = self._T_past * self._p
        # Phi_u_p = Phi[:, :u_p_size]
        # Phi_y_p = Phi[:, u_p_size : u_p_size + y_p_size]
        # Phi_u_f = Phi[:, u_p_size + y_p_size :]
        # constr += [
        #     self._y
        #     == Phi_u_p @ self._u_past
        #     + Phi_y_p @ (self._y_past + self._y_past_slack)
        #     + Phi_u_f @ self._u
        # ]

        # H_z = np.vstack((U_p, Y_p, U_f, np.ones(U_p.shape[-1])))
        # H_z_inv = np.linalg.pinv(H_z)
        # H_z_inv_u_p = H_z_inv[:, :u_p_size]
        # H_z_inv_y_p = H_z_inv[:, u_p_size : u_p_size + y_p_size]
        # H_z_inv_u_f = H_z_inv[:, u_p_size + y_p_size : -1]
        # H_z_inv_1 = H_z_inv[:, -1].reshape(-1, 1)
        # constr += [
        #     self._y
        #     == Y_f
        #     @ (
        #         H_z_inv_u_p @ self._u_past
        #         + H_z_inv_y_p @ (self._y_past + self._y_past_slack)
        #         + H_z_inv_u_f @ self._u
        #         + H_z_inv_1 @ np.ones((1,))
        #     )
        # ]

        constr += [
            U_p @ self._g == self._u_past,
            U_f @ self._g == self._u,
            Y_p @ self._g == self._y_past + self._y_past_slack,
            Y_f @ self._g == self._y,
            cp.sum(self._g) == 1,
        ]

        cost += controller_costs.slack_cost * (
            cp.norm1(self._y_past_slack)
        )  # + cp.norm2(self._y_past_slack))

        cost += cp.quad_form(
            self._y - self._reference_traj,
            self._Qblk,
        )

        cost += cp.quad_form(
            (self._u - np.tile(self._input_reference, self._T_fut)),
            self._Rblk,
        )

        if controller_costs.regularizer_cost_g_1 != 0:
            cost += controller_costs.regularizer_cost_g_1 * cp.norm(self._g, 1)

        u_mat = np.vstack((U_p, U_f, Y_p, np.ones((1, Y_p.shape[-1]))))
        pi = np.linalg.pinv(u_mat) @ u_mat

        if controller_costs.regularizer_cost_g_pi != 0:
            cost += controller_costs.regularizer_cost_g_pi * cp.sum_squares(
                (np.eye(pi.shape[0]) - pi) @ self._g
            )

        for k in range(self._T_fut):
            if controller_constraints.A_u is not None:
                constr += [
                    controller_constraints.A_u
                    @ self._u[k * self._m : (k + 1) * self._m]
                    <= controller_constraints.b_u
                ]

            if controller_constraints.A_y is not None:
                constr += [
                    controller_constraints.A_y
                    @ (self._y[k * self._p : (k + 1) * self._p])
                    <= controller_constraints.b_y
                ]

        self._problem = cp.Problem(cp.Minimize(cost), constr)
        ####-----####
        #custom
        self._problem_tracking_cost=cp.quad_form(
            self._y - self._reference_traj,
            self._Qblk,
        )
        self._problem_input_cost=cp.quad_form(
            (self._u - np.tile(self._input_reference, self._T_fut)),
            self._Rblk,
        )


    def compute_action(self, state: NDArray, reference: NDArray) -> NDArray:
        """Evaluate the deepc policy for a given state"""
        if isinstance(reference, list):
            reference = np.array(reference)

        if reference.ndim == 1:
            self._reference_traj.value = np.tile(reference, self._T_fut)
        elif reference.ndim == 2:
            self._reference_traj.value = reference.reshape(-1)

        # Append current state to past state trajectory and truncate first elements
        if not self._initial_state_overridden:
            if not self._initial_run:
                self._y_past.value = np.append(
                    self._y_past.value[self._p :].copy(), state
                )

            else:
                self._initial_run = False
                self._y_past.value = np.tile(state, self._T_past)
                self._u_past.value = np.tile(np.zeros(self._m), self._T_past)

        solver_status = None

        try:
            self._problem.solve(
                warm_start=False,
                verbose=self._verbose,
                solver="OSQP",
                max_iter=6000,
                eps_abs=1e-5,
                eps_rel=1e-4,
            )
            solver_status = self._problem.status

            if self._y_past_slack.value is not None:
                self._last_slack_inf = float(
                    np.linalg.norm(self._y_past_slack.value, ord=np.inf)
                )
            else:
                self._last_slack_inf = np.nan

        except cp.SolverError as e:
            print(f"SolverError:\n{e}")
            solver_status = "failed"

        if "optimal" in solver_status:
            computed_action = self._u[: self._m].value.copy()
            self._prev_u = self._u.value
            self._open_loop_index = 1

        else:
            warnings.warn(
                f"MPC solver status: {solver_status}, returning open loop input.",
                RuntimeWarning,
            )
            try:
                computed_action = self._prev_u[
                    (self._open_loop_index)
                    * self._m : (self._open_loop_index + 1)
                    * self._m
                ]
                self._open_loop_index += 1
            except:
                warnings.warn(
                    "Could not use open loop trajectory as fallback. Sending 0 input",
                    RuntimeWarning,
                )
                computed_action = np.zeros(self._m)

        # Guard against shape corruption in fallback mode:
        # slicing an exhausted open-loop plan can return an empty array
        # without raising, which later breaks _u_past dimension constraints.
        computed_action = np.asarray(computed_action, dtype=float).reshape(-1)
        if computed_action.size != self._m:
            warnings.warn(
                (
                    "Fallback action has invalid size "
                    f"{computed_action.size} (expected {self._m}); sending 0 input."
                ),
                RuntimeWarning,
            )
            computed_action = np.zeros(self._m, dtype=float)

        # Residual logging: primal equality residuals and a practical KKT proxy.
        if self._g.value is not None:
            g_val = np.asarray(self._g.value, dtype=float).reshape(-1)
            u_val = (
                np.asarray(self._u.value, dtype=float).reshape(-1)
                if self._u.value is not None
                else None
            )
            yslack_val = (
                np.asarray(self._y_past_slack.value, dtype=float).reshape(-1)
                if self._y_past_slack.value is not None
                else np.zeros(self._p * self._T_past, dtype=float)
            )

            r_u_p = self._U_p @ g_val - np.asarray(self._u_past.value, dtype=float).reshape(-1)
            r_y_p = self._Y_p @ g_val - (
                np.asarray(self._y_past.value, dtype=float).reshape(-1) + yslack_val
            )
            if u_val is None:
                r_u_f = np.array([], dtype=float)
            else:
                r_u_f = self._U_f @ g_val - u_val

            self._last_eq_res_u_past = float(np.linalg.norm(r_u_p, ord=2))
            self._last_eq_res_y_past = float(np.linalg.norm(r_y_p, ord=2))
            self._last_eq_res_u_fut = float(np.linalg.norm(r_u_f, ord=2))
            self._last_eq_residual = (
                self._last_eq_res_u_past
                + self._last_eq_res_y_past
                + self._last_eq_res_u_fut
            )

            # KKT proxy: stationarity over primal vars from available gradients.
            # This is solver-agnostic and stable for online logging.
            grad_g = np.zeros_like(g_val)
            grad_u = np.zeros(self._T_fut * self._m, dtype=float)
            grad_s = np.zeros(self._p * self._T_past, dtype=float)
            y_val = (
                np.asarray(self._y.value, dtype=float).reshape(-1)
                if self._y.value is not None
                else np.zeros(self._T_fut * self._p, dtype=float)
            )
            y_ref = (
                np.asarray(self._reference_traj.value, dtype=float).reshape(-1)
                if self._reference_traj.value is not None
                else np.zeros(self._T_fut * self._p, dtype=float)
            )
            grad_y = 2.0 * self._Qblk @ (y_val - y_ref)
            grad_u = 2.0 * self._Rblk @ (
                (u_val if u_val is not None else np.zeros(self._T_fut * self._m))
                - np.tile(self._input_reference, self._T_fut)
            )
            # l1 slack subgradient (minimum-norm choice at zero -> 0)
            grad_s = self._slack_cost * np.sign(yslack_val)
            self._last_kkt_residual = float(
                np.linalg.norm(grad_g, ord=2)
                + np.linalg.norm(grad_y, ord=2)
                + np.linalg.norm(grad_u, ord=2)
                + np.linalg.norm(grad_s, ord=2)
                + self._last_eq_residual
            )
        else:
            self._last_eq_res_u_past = np.nan
            self._last_eq_res_y_past = np.nan
            self._last_eq_res_u_fut = np.nan
            self._last_eq_residual = np.nan
            self._last_kkt_residual = np.nan

        self._last_solver_status = solver_status

        # Append computed input to past input trajectory.
        self._u_past.value = np.append(
            self._u_past.value[self._m :].copy(), computed_action
        )
        self._prev_input = computed_action.copy()
        self._initial_state_overridden = False
        return computed_action

    def get_last_slack_inf(self):
        return self._last_slack_inf

    def get_last_solver_status(self):
        return self._last_solver_status
    
    def get_last_eq_residual(self):
        return self._last_eq_residual

    def get_last_eq_residual_u_past(self):
        return self._last_eq_res_u_past

    def get_last_eq_residual_y_past(self):
        return self._last_eq_res_y_past

    def get_last_eq_residual_u_fut(self):
        return self._last_eq_res_u_fut

    def get_last_kkt_residual(self):
        return self._last_kkt_residual

    def set_initial_state(self, y_past, u_past):
        self._initial_state_overridden = True
        self._initial_run = False
        self._y_past.value = y_past
        self._u_past.value = u_past

    def get_planned_state_trajectory(self):
        if self._y.value is None:
            return np.zeros((self._T_past + self._T_fut, self._p))

        y_value = np.append(self._y_past.value, self._y.value)

        return y_value.reshape((-1, self._p))

    def get_planned_input_trajectory(self):
        if self._u.value is None:
            return np.zeros((self._T_past + self._T_fut, self._m))

        u_value = np.append(self._u_past.value, self._u.value)

        return u_value.reshape((-1, self._m))
    
    def get_tracking_cost(self):
        if self._problem_tracking_cost is None:
            return 0.0
        return _to_scalar_float(self._problem_tracking_cost, default=0.0)

    def get_input_cost(self):
        if self._problem_input_cost is None:
            return 0.0
        return _to_scalar_float(self._problem_input_cost, default=0.0)

class RocketControllerWrapper(BaseController):
    def __init__(self, base_controller: DataDrivenPredictiveController):
        self._base_controller = base_controller
        self._is_landed = False

    def _landing_detector(self, state: NDArray) -> bool:
        # both legs have contact with the ground
        if np.any(state[6:]):
            self._is_landed = True

        # if np.any(state[6:]) and abs(state[5]) < 0.4:
        #     self._is_landed = True

        return self._is_landed

    def compute_action(self, state: NDArray, reference: NDArray) -> NDArray:
        if self._landing_detector(state):
            # Stabilize the rocket if we came in a bit too hot
            if (np.sign(state[4]) == np.sign(state[5])) and np.abs(state[4]) > 0.1:
                return np.array([0, 5 * state[4], 0])
            else:
                return np.zeros(3)

        return self._base_controller.compute_action(state[:6], reference)

    def get_planned_state_trajectory(self):
        return self._base_controller.get_planned_state_trajectory()

    def get_planned_input_trajectory(self):
        return self._base_controller.get_planned_input_trajectory()


class StreamingDeePC(BaseController):
    """Collect the entire DeePC pipeline such that it fits into the simulator interface"""

    def __init__(
        self,
        H_u,
        H_y,
        deepc_args: DeePCControllerArgs,
    ):
        self._H_u = H_u
        self._H_y = H_y

        self._u_accumulator = np.zeros_like(H_u[:, 0])
        self._y_accumulator = np.zeros_like(H_y[:, 0])
        self._m = deepc_args.deepc_dims.m  # input size
        self._p = deepc_args.deepc_dims.p  # sensor size
        self._idx = 0
        self._L = deepc_args.deepc_dims.T_past + deepc_args.deepc_dims.T_fut

        self._T_past = deepc_args.deepc_dims.T_past

        self._deepc = None

        self._controller_args = [
            deepc_args.deepc_dims,
            deepc_args.controller_costs,
            deepc_args.controller_constraints,
            deepc_args.input_reference,
            deepc_args.verbose,
        ]

        self._deepc = None

    def compute_action(self, state, reference):
        """Compute action and update hankel matrices with new data"""
        if self._deepc is None:
            self._y_init = np.tile(state, self._T_past)
            self._u_init = np.tile(np.zeros(self._m), self._T_past)

        else:
            self._y_init = np.append(self._y_init[self._p :], state)

        self._deepc = DataDrivenPredictiveController(
            self._H_u,
            self._H_y,
            *self._controller_args,
            decompose_hankel_matrix=self._H_u.shape[-1] > 400,
        )
        self._deepc.set_initial_state(
            self._y_init,
            self._u_init,
        )
        action = self._deepc.compute_action(state, reference)
        self._u_init = np.append(self._u_init[self._m :], action)

        if self._idx < self._L:
            self._u_accumulator[self._m * self._idx : self._m * self._idx + self._m] = (
                action
            )
            self._y_accumulator[self._p * self._idx : self._p * self._idx + self._p] = (
                state
            )
            self._idx += 1
        else:
            self._u_accumulator = np.append(self._u_accumulator[self._m :], action)
            self._y_accumulator = np.append(self._y_accumulator[self._p :], state)
            self._H_u = np.append(
                self._H_u[:, 1:], self._u_accumulator.reshape(-1, 1), axis=1
            )
            self._H_y = np.append(
                self._H_y[:, 1:], self._y_accumulator.reshape(-1, 1), axis=1
            )

        return action

    def get_planned_state_trajectory(self):
        try:
            return self._deepc.get_planned_state_trajectory()
        except:
            return np.zeros((self._deepc._T_fut, self._p))

    def get_planned_input_trajectory(self):
        try:
            return self._deepc.get_planned_input_trajectory()
        except:
            return np.zeros((self._deepc._T_fut, self._m))


class SelectDeePC(BaseController):
    """Select-DeePC which selects the datapoints with minimal weights."""

    def __init__(
        self,
        deepc_args: DeePCControllerArgs,
        selector_callback=None,
        num_hankel_cols=180,
        n_iter: int = 1,
        debug: bool = False,
    ):
        self._debug = debug
        self._T_past = deepc_args.deepc_dims.T_past
        self._T_fut = deepc_args.deepc_dims.T_fut
        self._dims = deepc_args.deepc_dims
        hankel_gen = HankelMatrixGenerator(
            deepc_args.deepc_dims.T_past, deepc_args.deepc_dims.T_fut
        )
        self._H_u, self._H_y = hankel_gen.generate_hankel_matrices(
            deepc_args.trajectory_data
        )
        finite_u = np.all(np.isfinite(self._H_u), axis=0)
        finite_y = np.all(np.isfinite(self._H_y), axis=0)
        finite_cols = finite_u & finite_y
        dropped_cols = int(np.size(finite_cols) - np.count_nonzero(finite_cols))
        if dropped_cols > 0:
            warnings.warn(
                f"SelectDeePC: dropped {dropped_cols} non-finite Hankel columns.",
                RuntimeWarning,
            )
            self._H_u = self._H_u[:, finite_cols]
            self._H_y = self._H_y[:, finite_cols]
        if self._H_u.shape[1] == 0:
            raise ValueError("SelectDeePC: no finite Hankel columns available after filtering.")
        # print(self._H_u.shape)
        self._controller_args = [
            deepc_args.controller_costs,
            deepc_args.controller_constraints,
            deepc_args.input_reference,
            deepc_args.verbose,
        ]
        self._deepc = None

        self._num_hankel_cols = (
            num_hankel_cols if num_hankel_cols != -1 else self._H_u.shape[-1]
        )

        self._prev_traj_y = None
        self._prev_traj_u = None

        if selector_callback is None:
            self._selector_callback = LkSelector(deepc_args.deepc_dims)
        else:
            self._selector_callback = selector_callback

        self._n_iter = n_iter
        self._solve_time_avg = {
            "selection": RecursiveAverager(),
            "solve": RecursiveAverager(),
        }
        self._min_selected_sigma_history = []
        self._last_selected_idcs = np.array([], dtype=int)

    def compute_action(self, state, reference):
        step_min_sigma = np.nan
        for curr_iter in range(self._n_iter):
            if self._deepc is None:
                self._y_past = np.tile(state, self._T_past)
                self._u_past = np.tile(np.zeros(self._dims.m), self._T_past)
                state_traj = np.tile(state, self._T_past + self._T_fut).reshape(-1, 1)
                input_traj = np.zeros(
                    self._dims.m * (self._T_past + self._T_fut)
                ).reshape(-1, 1)

            else:
                if curr_iter == 0:
                    self._y_past = np.append(self._y_past[self._dims.p :], state)

                state_traj = self._deepc.get_planned_state_trajectory().reshape(-1, 1)
                input_traj = self._deepc.get_planned_input_trajectory().reshape(-1, 1)

            time_before_sel = perf_counter()
            idcs, norms = self._selector_callback(
                input_traj,
                state_traj,
                self._H_u,
                self._H_y,
                reference,
            )
            idcs_ranked = np.asarray(idcs, dtype=int).reshape(-1)
            time_after_sel = perf_counter()

            idcs = idcs[: self._num_hankel_cols]
            self._last_selected_idcs = idcs.copy()
            if idcs.size > 0:
                singular_values = np.linalg.svd(self._H_u[:, idcs], compute_uv=False)
                step_min_sigma = float(np.min(singular_values))

            prev_u = None
            if self._deepc is not None:
                prev_u = self._deepc._prev_u
                ol_idx = self._deepc._open_loop_index

            if self._debug:
                plt.plot(self._H_y[0::5, idcs], self._H_y[2::5, idcs])
                # plt.plot(reference[:, 0], reference[:, 1], "b--")
                plt.plot(
                    state_traj[0::5],
                    state_traj[2::5],
                    c="red",
                    linestyle="--",
                    marker="x",
                )

            self._deepc = DataDrivenPredictiveController(
                self._H_u[:, idcs],
                self._H_y[:, idcs],
                self._dims,
                *self._controller_args,
                decompose_hankel_matrix=idcs.size > 400,
            )
            if prev_u is not None:
                self._deepc._prev_u = prev_u
                self._deepc._open_loop_index = ol_idx

            self._deepc.set_initial_state(self._y_past, self._u_past)
            time_before_solve = perf_counter()
            action = self._deepc.compute_action(state, reference)
            time_after_solve = perf_counter()

            self._solve_time_avg["selection"].update(time_after_sel - time_before_sel)
            self._solve_time_avg["solve"].update(time_after_solve - time_before_solve)

        self._min_selected_sigma_history.append(step_min_sigma)
        self._u_past = np.append(self._u_past[self._dims.m :], action)
        return action

    

    def get_min_selected_singular_values(self):
        return np.array(self._min_selected_sigma_history, dtype=float)

    def get_last_selected_idcs(self):
        return np.array(self._last_selected_idcs, dtype=int)

    def get_planned_state_trajectory(self):
        try:
            return self._deepc.get_planned_state_trajectory()
        except:
            return np.zeros((self._deepc._T_fut, self._p))

    def get_planned_input_trajectory(self):
        try:
            return self._deepc.get_planned_input_trajectory()
        except:
            return np.zeros((self._deepc._T_fut, self._m))


class AdaptiveSelectDeePC(BaseController):
    """Select-DeePC which selects the datapoints with minimal weights."""

    def __init__(
        self,
        deepc_args: DeePCControllerArgs,
        selector_callback=None,
        num_hankel_cols=180,
        n_iter: int = 1,
        debug: bool = False,
        adaptive_k: bool = False,
        K_min: int = 20,
        K_max: int = 200,
        K_step: int = 10,
        sigma_bar: float = 1e-6,
        N_loc: int = 1000,
        d_max: float = 10.0,
        d_gate_enabled: bool = True,
        cond_gate_enabled: bool = True,
    ):
        self._debug = debug
        self._T_past = deepc_args.deepc_dims.T_past
        self._T_fut = deepc_args.deepc_dims.T_fut
        self._dims = deepc_args.deepc_dims
        hankel_gen = HankelMatrixGenerator(
            deepc_args.deepc_dims.T_past, deepc_args.deepc_dims.T_fut
        )
        self._H_u, self._H_y = hankel_gen.generate_hankel_matrices(
            deepc_args.trajectory_data
        )
        # Drop non-finite Hankel columns once so downstream SVD/solver never sees NaN/Inf.
        finite_u = np.all(np.isfinite(self._H_u), axis=0)
        finite_y = np.all(np.isfinite(self._H_y), axis=0)
        finite_cols = finite_u & finite_y
        dropped_cols = int(finite_cols.size - np.count_nonzero(finite_cols))
        if dropped_cols > 0:
            warnings.warn(
                f"SelectDeePC: dropped {dropped_cols} non-finite Hankel columns.",
                RuntimeWarning,
            )
            self._H_u = self._H_u[:, finite_cols]
            self._H_y = self._H_y[:, finite_cols]
        if self._H_u.shape[1] == 0:
            raise ValueError(
                "SelectDeePC: no finite Hankel columns available after filtering."
            )

        #nolmarization
        # print(self._H_u.shape)
        self._controller_args = [
            deepc_args.controller_costs,
            deepc_args.controller_constraints,
            deepc_args.input_reference,
            deepc_args.verbose,
        ]
        self._deepc = None

        self._num_hankel_cols = (
            num_hankel_cols if num_hankel_cols != -1 else self._H_u.shape[-1]
        )

        self._prev_traj_y = None
        self._prev_traj_u = None

        if selector_callback is None:
            self._selector_callback = LkSelector(deepc_args.deepc_dims)
        else:
            self._selector_callback = selector_callback

        self._n_iter = n_iter
        self._solve_time_avg = {
            "selection": RecursiveAverager(),
            "solve": RecursiveAverager(),
        }
        self._min_selected_sigma_history = []
        self._last_selected_idcs = np.array([], dtype=int)

        #custom
        self._eps_sigma=0.0
        self._K_opt_history = []
        self._selected_idcs_history = []
        self._sigma_min_A_history = []
        self._sigma_min_Mk_history = []
        self._slack_norm_inf_history = []
        self._slack_violation_history = []
        self._tracking_cost_history = []
        self._input_cost_history = []
        self._stage_cost_history = []
        self._solve_time_ms_history = []
        self._status_history = []
        self._eq_residual_history = []
        self._eq_res_u_past_history = []
        self._eq_res_y_past_history = []
        self._eq_res_u_fut_history = []
        self._kkt_residual_history = []
        self._input_history = []

        self._adaptive_k = adaptive_k
        self._K_min = K_min
        self._K_max = K_max
        self._K_step = K_step
        

        # Respect constructor arguments instead of hard-coded defaults.
        self._N_loc = int(N_loc)
        self._d_max = float(d_max)
        self._local_gate_fail=0
        self._cond_gate_fail=0
        self._sigma_bar = sigma_bar
        self._d_k_history = []
        self._sigma_min_all = None
        self._last_cand = None
        self._d_gate_enabled = d_gate_enabled
        self._cond_gate_enabled = cond_gate_enabled
        self._gate_enabled = d_gate_enabled or cond_gate_enabled
        # Fixed-K mode should honor num_hankel_cols directly.
    
        self._A_g = np.vstack([self._H_u, self._H_y[: self._dims.p * self._T_past, :]])


    def compute_action(self, state, reference):
        step_min_sigma = np.nan
        for curr_iter in range(self._n_iter):
            if self._deepc is None:
                self._y_past = np.tile(state, self._T_past)
                self._u_past = np.tile(np.zeros(self._dims.m), self._T_past)
                state_traj = np.tile(state, self._T_past + self._T_fut).reshape(-1, 1)
                input_traj = np.zeros(
                    self._dims.m * (self._T_past + self._T_fut)
                ).reshape(-1, 1)

            else:
                if curr_iter == 0:
                    self._y_past = np.append(self._y_past[self._dims.p :], state)

                state_traj = self._deepc.get_planned_state_trajectory().reshape(-1, 1)
                input_traj = self._deepc.get_planned_input_trajectory().reshape(-1, 1)

            time_before_sel = perf_counter()
            idcs, norms = self._selector_callback(
                input_traj,
                state_traj,
                self._H_u,
                self._H_y,
                reference,
            )
            idcs_ranked = np.asarray(idcs, dtype=int).reshape(-1)
            time_after_sel = perf_counter()
            
            #debug
            print(np.min(norms), np.median(norms), np.max(norms))
            print(np.percentile(norms, [1,5,10,50,90,99]))
            
            N_loc=self._N_loc
            d_max=self._d_max
            m = self._dims.m
            p = self._dims.p
            Tp = self._T_past
            Tf = self._T_fut
            u_ini_rows = m * Tp
            u_tot_rows = m * (Tp + Tf)
            Up_all = self._H_u[:u_ini_rows, :]               # (mTp, N)
            Uf_all = self._H_u[u_ini_rows:u_tot_rows, :]     # (mTf, N)

            if self._sigma_min_all is None:
                Mk=np.vstack([Up_all, Uf_all])
                sigma_min_all=self._sigma_min_matrix(Mk)
                self._sigma_min_all=sigma_min_all


            if self._d_gate_enabled and self._cond_gate_enabled:
                cand= idcs[norms[idcs] <= d_max]
                if cand.size <self._K_min:
                    cand = idcs[:self._K_min]
                    self._local_gate_fail += 1
            
                S,sig_actual,s_k,fail = self.cpqr_sigma_gate(
                    cand=cand,
                    Up_all=Up_all,
                    Uf_all=Uf_all,
                    sigma_bar=self._sigma_bar,
                    K_min=self._K_min,
                    K_max=min(self._K_max, cand.size),
                )
                if fail:
                    self._cond_gate_fail += 1
                    # 정책: bank update / sigma_bar 완화 / N_loc 확대
                    # 일단은 fallback으로 K_max 사용(혹은 bank update)
                idcs = S
                self._last_selected_idcs = idcs.copy()
            elif self._d_gate_enabled and not self._cond_gate_enabled:
                cand = idcs[norms[idcs] <= d_max]
                if cand.size <self._K_min:
                    cand = idcs[:self._K_min]
                    local_gate_fail += 1
                idcs = cand
                self._last_selected_idcs = idcs.copy()    
            elif not self._d_gate_enabled and self._cond_gate_enabled:
                cand = idcs[:self._K_max]
                S,sig_actual,s_k,fail = self.cpqr_sigma_gate(
                    cand=cand,
                    Up_all=Up_all,
                    Uf_all=Uf_all,
                    sigma_bar=self._sigma_bar,
                    K_min=self._K_min,
                    K_max=min(self._K_max, cand.size),
                )
                if fail:    
                    self._cond_gate_fail += 1
                    # 정책: bank update / sigma_bar 완화 / N_loc 확대
                    # 일단은 fallback으로 K_max 사용(혹은 bank update)
                idcs = S
                self._last_selected_idcs = idcs.copy()
            elif not self._d_gate_enabled and not self._cond_gate_enabled:
                idcs = idcs[:self._num_hankel_cols]
                self._last_selected_idcs = idcs.copy()
     
           
            Mk_sel = np.vstack([Up_all[:, idcs], Uf_all[:, idcs]])
            sigma_min_Mk_selected = self._sigma_min_matrix(Mk_sel)
            self._sigma_min_Mk_history.append(sigma_min_Mk_selected)
            self._selected_idcs_history.append(idcs.copy())
            self._last_selected_idcs = np.array(idcs, dtype=int)
            self._K_opt_history.append(idcs.size)
            #####----#####
            
            # #####----#####

            if idcs.size > 0:
                singular_values = np.linalg.svd(self._H_u[:, idcs], compute_uv=False)
                step_min_sigma = float(np.min(singular_values))

            prev_u = None
            if self._deepc is not None:
                prev_u = self._deepc._prev_u
                ol_idx = self._deepc._open_loop_index

            if self._debug:
                plt.plot(self._H_y[0::5, idcs], self._H_y[2::5, idcs])
                # plt.plot(reference[:, 0], reference[:, 1], "b--")
                plt.plot(
                    state_traj[0::5],
                    state_traj[2::5],
                    c="red",
                    linestyle="--",
                    marker="x",
                )

            self._deepc = DataDrivenPredictiveController(
                self._H_u[:, idcs],
                self._H_y[:, idcs],
                self._dims,
                *self._controller_args,
                decompose_hankel_matrix=idcs.size > 400,
            )
            if prev_u is not None:
                self._deepc._prev_u = prev_u
                self._deepc._open_loop_index = ol_idx

            self._deepc.set_initial_state(self._y_past, self._u_past)
            time_before_solve = perf_counter()
            action = self._deepc.compute_action(state, reference)
            time_after_solve = perf_counter()

            self._solve_time_avg["selection"].update(time_after_sel - time_before_sel)
            self._solve_time_avg["solve"].update(time_after_solve - time_before_solve)

        self._min_selected_sigma_history.append(step_min_sigma)
        self._u_past = np.append(self._u_past[self._dims.m :], action)


        ####-----####
        #custom
        slack_inf = float(self._deepc.get_last_slack_inf())
        self._slack_norm_inf_history.append(slack_inf)
        eps = self._eps_sigma# 미리 클래스에 저장해둔 threshold

        tracking_cost = float(self._deepc.get_tracking_cost())
        input_cost = float(self._deepc.get_input_cost())
        self._tracking_cost_history.append(tracking_cost)
        self._solve_time_ms_history.append(1000.0*(time_after_solve - time_before_solve))
        self._status_history.append(self._deepc.get_last_solver_status())
        self._input_history.append(action)

        ####-----####


        return action

    def _sigma_min_cols(self, M: np.ndarray, cols: np.ndarray) -> float:
        """Return sigma_min of M[:, cols]. cols는 1D index array."""
        if cols.size == 0:
            return np.nan
        try:
            s = np.linalg.svd(M[:, cols], compute_uv=False)
        except np.linalg.LinAlgError:
            return np.nan
        return float(s[-1])

    def _sigma_min_matrix(self, M):
        if M.size == 0 or M.shape[1] == 0: return np.nan
        try:
            s = np.linalg.svd(M, compute_uv=False)
        except np.linalg.LinAlgError:
            return np.nan
        return float(s[-1])

    def cpqr_sigma_gate(
        self,
        cand: np.ndarray,
        Up_all: np.ndarray,
        Uf_all: np.ndarray,
        sigma_bar: float,
        K_min: int = 40,
        K_max: int | None = None,
    ) -> tuple[np.ndarray | None, float, int, bool]:
        """
        cand: global column indices (1D)
        returns: (S, sigma_min_actual, s_k, fail)
        """
        if cand.size == 0:
            return None, float("nan"), 0, True
        if K_max is None:
            K_max = int(cand.size)
        K_max = min(K_max, int(cand.size))

        # candidate matrix for CPQR
        Mloc = np.vstack([Up_all[:, cand], Uf_all[:, cand]])  # (mL, |cand|)

        # CPQR: Mloc[:, piv] = Q R
        _, _, piv = qr_pivoted(Mloc, pivoting=True, mode="economic")

        # grow subset along pivot order, find minimal s_k satisfying sigma gate
        for s in range(max(1, K_min), K_max + 1):
            S = cand[piv[:s]]
            sig = self._sigma_min_matrix(np.vstack([Up_all[:, S], Uf_all[:, S]]))
            if np.isfinite(sig) and sig >= sigma_bar*self._sigma_min_all:
                return S, sig, s, False

        # fail
        sig_last = self._sigma_min_matrix(np.vstack([Up_all[:, cand[piv[:K_max]]], Uf_all[:, cand[piv[:K_max]]]]))
        S_last=cand[piv[:K_max]]
        return S_last, sig_last, K_max, True

    def _adaptive_k_selector(self, idcs: np.ndarray) -> Tuple[np.ndarray, int, float, float]:
        L = int(idcs.size)
        if L == 0:
            return idcs, 0, np.nan, np.nan

        A_g = self._A_g
        H_u=self._H_u

        # reference size should use K_max (not K_min)
        Kref = min(self._K_max, L)

        sigma_ref = self._sigma_ref_H_u
        
        Gamma = max(self._gamma_min, self._rho * sigma_ref)

        K = min(self._K_min, Kref)
        sigma_Ag = self._sigma_min_cols(A_g, idcs[:K])
        sigma_Hu = self._sigma_min_cols(H_u, idcs[:K])

        while (K < Kref) and (np.isnan(sigma_Hu) or (sigma_Hu < Gamma)):
            K = min(K + self._K_step, Kref)
            sigma_Hu = self._sigma_min_cols(H_u, idcs[:K])

        sigma_Ag = self._sigma_min_cols(A_g, idcs[:K])
        return idcs[:K], K, sigma_ref, sigma_Hu

    def get_sigma_min_all(self):
        return float(self._sigma_min_all)

    def get_whole_input_hankel_matrix(self):
        return self._H_u
    def get_whole_output_hankel_matrix(self):
        return self._H_y
        
    def get_ranked_idcs_history(self):
        return np.asarray(getattr(self, "_ranked_idcs_history", []), dtype=object)

    def get_ranked_norms_history(self):
        return np.asarray(getattr(self, "_ranked_norms_history", []), dtype=float)

    def get_min_selected_singular_values(self):
        return np.array(self._min_selected_sigma_history, dtype=float)

    def get_last_selected_idcs(self):
        return np.array(self._last_selected_idcs, dtype=int)

    def get_planned_state_trajectory(self):
        try:
            return self._deepc.get_planned_state_trajectory()
        except:
            return np.zeros((self._deepc._T_fut, self._p))

    def get_planned_input_trajectory(self):
        try:
            return self._deepc.get_planned_input_trajectory()
        except:
            return np.zeros((self._deepc._T_fut, self._m))

    def get_K_opt_history(self):
        return np.array(self._K_opt_history, dtype=int)
    def get_selected_idcs_history(self):
        # Adaptive-K can produce variable-length index sets per step.
        # Return an object array to preserve ragged history safely.
        return np.asarray(self._selected_idcs_history, dtype=object)
    def get_sigma_min_A_history(self):
        return np.array(self._sigma_min_A_history, dtype=float)
    def get_sigma_min_Mk_history(self):
        return np.asarray(self._sigma_min_Mk_history, dtype=float)
    def get_slack_norm_inf_history(self):
        return np.array(self._slack_norm_inf_history, dtype=float)
    def get_slack_violation_history(self):
        return np.array(self._slack_violation_history, dtype=int)
    def get_tracking_cost_history(self):
        return np.array(self._tracking_cost_history, dtype=float)
    def get_input_cost_history(self):
        return np.array(self._input_cost_history, dtype=float)
    def get_stage_cost_history(self):
        return np.array(self._stage_cost_history, dtype=float)
    def get_solve_time_ms_history(self):
        return np.array(self._solve_time_ms_history, dtype=float)
    def get_eps_sigma_history(self):
        return np.array([float(getattr(self, "_eps_sigma", np.nan))])
    def get_status_history(self):
        return np.asarray(self._status_history, dtype=object)
    def get_eq_residual_history(self):
        return np.asarray(self._eq_residual_history, dtype=float)
    def get_eq_residual_u_past_history(self):
        return np.asarray(self._eq_res_u_past_history, dtype=float)
    def get_eq_residual_y_past_history(self):
        return np.asarray(self._eq_res_y_past_history, dtype=float)
    def get_eq_residual_u_fut_history(self):
        return np.asarray(self._eq_res_u_fut_history, dtype=float)
    def get_kkt_residual_history(self):
        return np.asarray(self._kkt_residual_history, dtype=float)
    def get_input_history(self):
        return np.asarray(self._input_history, dtype=float)
    def save_history_npz(self, path: str, extra: Optional[dict] = None) -> None:
        """Save per-step histories to a .npz file for offline postprocessing."""
        payload = {
            "K_opt": np.asarray(self._K_opt_history, dtype=float),
            "selected_idcs": np.asarray(self._selected_idcs_history, dtype=object),
            "sigma_min_Hu": np.asarray(self._min_selected_sigma_history, dtype=float),
            "sigma_min_Ag": np.asarray(self._sigma_min_A_history, dtype=float),
            "sigma_min_Mk": np.asarray(self._sigma_min_Mk_history, dtype=float),
            "slack_inf": np.asarray(self._slack_norm_inf_history, dtype=float),
            "slack_fail": np.asarray(self._slack_violation_history, dtype=int),
            "tracking_cost": np.asarray(self._tracking_cost_history, dtype=float),
            "input_cost": np.asarray(self._input_cost_history, dtype=float),
            "stage_cost": np.asarray(self._stage_cost_history, dtype=float),
            "solve_time_ms": np.asarray(self._solve_time_ms_history, dtype=float),
            "solver_status": np.asarray(self._status_history, dtype=object),
            "eq_residual": np.asarray(self._eq_residual_history, dtype=float),
            "eq_res_u_past": np.asarray(self._eq_res_u_past_history, dtype=float),
            "eq_res_y_past": np.asarray(self._eq_res_y_past_history, dtype=float),
            "eq_res_u_fut": np.asarray(self._eq_res_u_fut_history, dtype=float),
            "kkt_residual": np.asarray(self._kkt_residual_history, dtype=float),
            "eps_sigma": np.asarray([float(getattr(self, "_eps_sigma", np.nan))]),
        }
        if extra is not None:
            for k, v in extra.items():
                payload[str(k)] = v
        np.savez_compressed(path, **payload)


class SelectDeePC_analyze(BaseController):
    """Select-DeePC which selects the datapoints with minimal weights."""

    def __init__(
        self,
        deepc_args: DeePCControllerArgs,
        selector_callback=None,
        num_hankel_cols=180,
        n_iter: int = 1,
        debug: bool = False,
        adaptive_k: bool = False,
        K_min: int = 20,
        K_max: int = 200,
        K_step: int = 1,
        sigma_bar: float = 1e-6,
        rho: float = 0.1,
        gamma_min: float = 1e-6,
        N_loc: int = 1000,
        d_max: float = 10.0,
    ):
        self._debug = debug
        self._T_past = deepc_args.deepc_dims.T_past
        self._T_fut = deepc_args.deepc_dims.T_fut
        self._dims = deepc_args.deepc_dims
        hankel_gen = HankelMatrixGenerator(
            deepc_args.deepc_dims.T_past, deepc_args.deepc_dims.T_fut
        )
        self._H_u, self._H_y = hankel_gen.generate_hankel_matrices(
            deepc_args.trajectory_data
        )
        # Drop non-finite Hankel columns once so downstream SVD/solver never sees NaN/Inf.
        finite_u = np.all(np.isfinite(self._H_u), axis=0)
        finite_y = np.all(np.isfinite(self._H_y), axis=0)
        finite_cols = finite_u & finite_y
        dropped_cols = int(finite_cols.size - np.count_nonzero(finite_cols))
        if dropped_cols > 0:
            warnings.warn(
                f"SelectDeePC: dropped {dropped_cols} non-finite Hankel columns.",
                RuntimeWarning,
            )
            self._H_u = self._H_u[:, finite_cols]
            self._H_y = self._H_y[:, finite_cols]
        if self._H_u.shape[1] == 0:
            raise ValueError(
                "SelectDeePC: no finite Hankel columns available after filtering."
            )

        #nolmarization
        # print(self._H_u.shape)
        self._controller_args = [
            deepc_args.controller_costs,
            deepc_args.controller_constraints,
            deepc_args.input_reference,
            deepc_args.verbose,
        ]
        self._deepc = None

        self._num_hankel_cols = (
            num_hankel_cols if num_hankel_cols != -1 else self._H_u.shape[-1]
        )

        self._prev_traj_y = None
        self._prev_traj_u = None

        if selector_callback is None:
            self._selector_callback = LkSelector(deepc_args.deepc_dims)
        else:
            self._selector_callback = selector_callback

        self._n_iter = n_iter
        self._solve_time_avg = {
            "selection": RecursiveAverager(),
            "solve": RecursiveAverager(),
        }
        self._min_selected_sigma_history = []
        self._last_selected_idcs = np.array([], dtype=int)

        #custom
        self._eps_sigma=0.0
        self._K_opt_history = []
        self._selected_idcs_history = []
        self._sigma_min_A_history = []
        self._sigma_min_Mk_history = []
        self._slack_norm_inf_history = []
        self._slack_violation_history = []
        self._tracking_cost_history = []
        self._input_cost_history = []
        self._stage_cost_history = []
        self._solve_time_ms_history = []
        self._status_history = []
        self._eq_residual_history = []
        self._eq_res_u_past_history = []
        self._eq_res_y_past_history = []
        self._eq_res_u_fut_history = []
        self._kkt_residual_history = []
        self._input_history = []
        self._y_past_history = []
        self._u_past_history = []

        self._adaptive_k = adaptive_k
        self._K_min = K_min
        self._K_max = K_max
        self._K_step = K_step
        self._rho = rho
        self._gamma_min = gamma_min
        

        # Respect constructor arguments instead of hard-coded defaults.
        self._N_loc = int(N_loc)
        self._d_max = float(d_max)
        self._local_gate_fail=0
        self._cond_gate_fail=0
        self._sigma_bar = sigma_bar
        self._d_k_history = []
        self._sigma_min_all = None

        # Fixed-K mode should honor num_hankel_cols directly.
        # The adaptive CPQR gate below uses (K_min, K_max), so lock both to requested K.
        if not self._adaptive_k:
            k_fixed = int(max(1, self._num_hankel_cols))
            self._K_min = k_fixed
            self._K_max = k_fixed
            self._N_loc = max(self._N_loc, k_fixed)



        self._A_g = np.vstack([self._H_u, self._H_y[: self._dims.p * self._T_past, :]])
        try:
            singular_values_A_g_offline = np.linalg.svd(self._A_g, compute_uv=False)
            self._sigma_ref_Ag = float(np.min(singular_values_A_g_offline))
        except np.linalg.LinAlgError:
            warnings.warn(
                "SelectDeePC: offline A_g SVD did not converge, falling back to gamma_min.",
                RuntimeWarning,
            )
            self._sigma_ref_Ag = float(self._gamma_min)
        try:
            singular_values_H_u_offline = np.linalg.svd(self._H_u, compute_uv=False)
            self._sigma_ref_H_u = float(np.min(singular_values_H_u_offline))
        except np.linalg.LinAlgError:
            warnings.warn(
                "SelectDeePC: offline A_g SVD did not converge, falling back to gamma_min.",
                RuntimeWarning,
            )
            self._sigma_ref_H_u = float(self._gamma_min)


        

    def compute_action(self, state, reference):
        step_min_sigma = np.nan

        once=True


        if self._deepc is None:
            self._y_past = np.tile(state, self._T_past)
            self._u_past = np.tile(np.zeros(self._dims.m), self._T_past)
            state_traj = np.tile(state, self._T_past + self._T_fut).reshape(-1, 1)
            input_traj = np.zeros(
                self._dims.m * (self._T_past + self._T_fut)
            ).reshape(-1, 1)

        else:
            state_traj = self._deepc.get_planned_state_trajectory().reshape(-1, 1)
            input_traj = self._deepc.get_planned_input_trajectory().reshape(-1, 1)

        #sensitivity analysis
        self._y_past_history.append(self._y_past.copy())
        self._u_past_history.append(self._u_past.copy())


        time_before_sel = perf_counter()
        idcs, norms = self._selector_callback(
            input_traj,
            state_traj,
            self._H_u,
            self._H_y,
            reference,
        )
        time_after_sel = perf_counter()

        # after: idcs, norms = self._selector_callback(...)
        if not hasattr(self, "_ranked_idcs_history"):
            self._ranked_idcs_history = []
            self._ranked_norms_history = []
        self._ranked_idcs_history.append(np.array(idcs, dtype=int))
        self._ranked_norms_history.append(np.array(norms, dtype=float))

        N_loc=self._N_loc
        d_max=self._d_max
        m = self._dims.m
        p = self._dims.p
        Tp = self._T_past
        Tf = self._T_fut
        u_ini_rows = m * Tp
        u_tot_rows = m * (Tp + Tf)
        Up_all = self._H_u[:u_ini_rows, :]               # (mTp, N)
        Uf_all = self._H_u[u_ini_rows:u_tot_rows, :]     # (mTf, N)

        # selectin gates
        # ---- d-gate ----
        cand = idcs[:N_loc]
        d_k = norms[cand].max()
        if d_k > d_max:
            self._local_gate_fail += 1
            # 최소 fallback: 후보 풀을 넓혀서라도 시도 
            cand = idcs[:min(idcs.size, 3*N_loc)]
        self._d_k_history.append(float(d_k))
        

        # ---- CPQR + sigma-gate: minimal subset ----
        S, sig_actual, s_k, fail = self.cpqr_sigma_gate(
            cand=cand,
            Up_all=Up_all,
            Uf_all=Uf_all,
            sigma_bar=self._sigma_bar,   # = underline_sigma
            K_min=self._K_min,
            K_max=min(self._K_max, cand.size),
        )

        if fail:
            self._cond_gate_fail += 1
            # 정책: bank update / sigma_bar 완화 / N_loc 확대
            # 일단은 fallback으로 top-K 사용(혹은 bank update)
            idcs = cand[:self._num_hankel_cols]
        else:
            idcs = S
            self._sigma_min_A_history.append(sig_actual)

        if self._sigma_min_all is None:
            Mk=np.vstack([Up_all, Uf_all])
            sigma_min_all=self._sigma_min_matrix(Mk)
            self._sigma_min_all=sigma_min_all
        
    
        # if once:
        #     print(f"step_min_sigma_Mk: {step_min_sigma_Mk}")
        #     once=False
        Mk_sel = np.vstack([Up_all[:, idcs], Uf_all[:, idcs]])
        sigma_min_Mk_selected = self._sigma_min_matrix(Mk_sel)
        self._sigma_min_Mk_history.append(sigma_min_Mk_selected)
        self._selected_idcs_history.append(idcs.copy())
        self._last_selected_idcs = np.array(idcs, dtype=int)
        self._K_opt_history.append(idcs.size)
        #####----#####
        


        prev_u = None
        if self._deepc is not None:
            prev_u = self._deepc._prev_u
            ol_idx = self._deepc._open_loop_index

        if self._debug:
            plt.plot(self._H_y[0::5, idcs], self._H_y[2::5, idcs])
            # plt.plot(reference[:, 0], reference[:, 1], "b--")
            plt.plot(
                state_traj[0::5],
                state_traj[2::5],
                c="red",
                linestyle="--",
                marker="x",
            )

        self._deepc = DataDrivenPredictiveController(
            self._H_u[:, idcs],
            self._H_y[:, idcs],
            self._dims,
            *self._controller_args,
            decompose_hankel_matrix=idcs.size > 400,
        )
        if prev_u is not None:
            self._deepc._prev_u = prev_u
            self._deepc._open_loop_index = ol_idx
        
        self._deepc.set_initial_state(self._y_past, self._u_past)
        time_before_solve = perf_counter()
        action = self._deepc.compute_action(state, reference)
        time_after_solve = perf_counter()



        self._solve_time_avg["selection"].update(time_after_sel - time_before_sel)
        self._solve_time_avg["solve"].update(time_after_solve - time_before_solve)


        self._u_past = np.append(self._u_past[self._dims.m :], action)


        ####-----####
        #custom
        slack_inf = float(self._deepc.get_last_slack_inf())
        self._slack_norm_inf_history.append(slack_inf)

        tracking_cost = float(self._deepc.get_tracking_cost())
        input_cost = float(self._deepc.get_input_cost())
        self._tracking_cost_history.append(tracking_cost)
        self._input_cost_history.append(input_cost)
        self._stage_cost_history.append(tracking_cost + input_cost)
        self._solve_time_ms_history.append(1000.0*(time_after_solve - time_before_solve))
        self._status_history.append(self._deepc.get_last_solver_status())
        self._input_history.append(action)

        ####-----####

        return action

    def _sigma_min_cols(self, M: np.ndarray, cols: np.ndarray) -> float:
        """Return sigma_min of M[:, cols]. cols는 1D index array."""
        if cols.size == 0:
            return np.nan
        try:
            s = np.linalg.svd(M[:, cols], compute_uv=False)
        except np.linalg.LinAlgError:
            return np.nan
        return float(s[-1])

    def _sigma_min_matrix(self, M):
        if M.size == 0 or M.shape[1] == 0: return np.nan
        try:
            s = np.linalg.svd(M, compute_uv=False)
        except np.linalg.LinAlgError:
            return np.nan
        return float(s[-1])

    def cpqr_sigma_gate(
        self,
        cand: np.ndarray,
        Up_all: np.ndarray,
        Uf_all: np.ndarray,
        sigma_bar: float,
        K_min: int = 20,
        K_max: int | None = None,
    ) -> tuple[np.ndarray | None, float, int, bool]:
        """
        cand: global column indices (1D)
        returns: (S, sigma_min_actual, s_k, fail)
        """
        if cand.size == 0:
            return None, float("nan"), 0, True
        if K_max is None:
            K_max = int(cand.size)
        K_max = min(K_max, int(cand.size))

        # candidate matrix for CPQR
        Mloc = np.vstack([Up_all[:, cand], Uf_all[:, cand]])  # (mL, |cand|)

        # CPQR: Mloc[:, piv] = Q R
        _, _, piv = qr_pivoted(Mloc, pivoting=True, mode="economic")

        # grow subset along pivot order, find minimal s_k satisfying sigma gate
        for s in range(max(1, K_min), K_max + 1):
            S = cand[piv[:s]]
            sig = self._sigma_min_matrix(np.vstack([Up_all[:, S], Uf_all[:, S]]))
            if np.isfinite(sig) and sig >= sigma_bar:
                return S, sig, s, False

        # fail
        sig_last = self._sigma_min_matrix(np.vstack([Up_all[:, cand[piv[:K_max]]], Uf_all[:, cand[piv[:K_max]]]]))
        return None, sig_last, K_max, True

    def _adaptive_k_selector(self, idcs: np.ndarray) -> Tuple[np.ndarray, int, float, float]:
        L = int(idcs.size)
        if L == 0:
            return idcs, 0, np.nan, np.nan

        A_g = self._A_g
        H_u=self._H_u

        # reference size should use K_max (not K_min)
        Kref = min(self._K_max, L)

        sigma_ref = self._sigma_ref_H_u
        
        Gamma = max(self._gamma_min, self._rho * sigma_ref)

        K = min(self._K_min, Kref)
        sigma_Ag = self._sigma_min_cols(A_g, idcs[:K])
        sigma_Hu = self._sigma_min_cols(H_u, idcs[:K])

        while (K < Kref) and (np.isnan(sigma_Hu) or (sigma_Hu < Gamma)):
            K = min(K + self._K_step, Kref)
            sigma_Hu = self._sigma_min_cols(H_u, idcs[:K])

        sigma_Ag = self._sigma_min_cols(A_g, idcs[:K])
        return idcs[:K], K, sigma_ref, sigma_Hu

    def get_sigma_min_all(self):
        return float(self._sigma_min_all)

    def get_whole_input_hankel_matrix(self):
        return self._H_u
    def get_whole_output_hankel_matrix(self):
        return self._H_y
        
    def get_ranked_idcs_history(self):
        return np.asarray(getattr(self, "_ranked_idcs_history", []), dtype=object)

    def get_ranked_norms_history(self):
        return np.asarray(getattr(self, "_ranked_norms_history", []), dtype=float)

    def get_min_selected_singular_values(self):
        return np.array(self._min_selected_sigma_history, dtype=float)

    def get_last_selected_idcs(self):
        return np.array(self._last_selected_idcs, dtype=int)

    def get_planned_state_trajectory(self):
        try:
            return self._deepc.get_planned_state_trajectory()
        except:
            return np.zeros((self._deepc._T_fut, self._p))

    def get_planned_input_trajectory(self):
        try:
            return self._deepc.get_planned_input_trajectory()
        except:
            return np.zeros((self._deepc._T_fut, self._m))

    def get_K_opt_history(self):
        return np.array(self._K_opt_history, dtype=int)
    def get_selected_idcs_history(self):
        # Adaptive-K can produce variable-length index sets per step.
        # Return an object array to preserve ragged history safely.
        return np.asarray(self._selected_idcs_history, dtype=object)
    def get_sigma_min_A_history(self):
        return np.array(self._sigma_min_A_history, dtype=float)
    def get_sigma_min_Mk_history(self):
        return np.asarray(self._sigma_min_Mk_history, dtype=float)
    def get_slack_norm_inf_history(self):
        return np.array(self._slack_norm_inf_history, dtype=float)
    def get_slack_violation_history(self):
        return np.array(self._slack_violation_history, dtype=int)
    def get_tracking_cost_history(self):
        return np.array(self._tracking_cost_history, dtype=float)
    def get_input_cost_history(self):
        return np.array(self._input_cost_history, dtype=float)
    def get_stage_cost_history(self):
        return np.array(self._stage_cost_history, dtype=float)
    def get_solve_time_ms_history(self):
        return np.array(self._solve_time_ms_history, dtype=float)
    def get_eps_sigma_history(self):
        return np.array([float(getattr(self, "_eps_sigma", np.nan))])
    def get_status_history(self):
        return np.asarray(self._status_history, dtype=object)
    def get_eq_residual_history(self):
        return np.asarray(self._eq_residual_history, dtype=float)
    def get_eq_residual_u_past_history(self):
        return np.asarray(self._eq_res_u_past_history, dtype=float)
    def get_eq_residual_y_past_history(self):
        return np.asarray(self._eq_res_y_past_history, dtype=float)
    def get_eq_residual_u_fut_history(self):
        return np.asarray(self._eq_res_u_fut_history, dtype=float)
    def get_kkt_residual_history(self):
        return np.asarray(self._kkt_residual_history, dtype=float)
    def get_input_history(self):
        return np.asarray(self._input_history, dtype=float)
    def save_history_npz(self, path: str, extra: Optional[dict] = None) -> None:
        """Save per-step histories to a .npz file for offline postprocessing."""
        payload = {
            "K_opt": np.asarray(self._K_opt_history, dtype=float),
            "selected_idcs": np.asarray(self._selected_idcs_history, dtype=object),
            "sigma_min_Hu": np.asarray(self._min_selected_sigma_history, dtype=float),
            "sigma_min_Ag": np.asarray(self._sigma_min_A_history, dtype=float),
            "sigma_min_Mk": np.asarray(self._sigma_min_Mk_history, dtype=float),
            "slack_inf": np.asarray(self._slack_norm_inf_history, dtype=float),
            "slack_fail": np.asarray(self._slack_violation_history, dtype=int),
            "tracking_cost": np.asarray(self._tracking_cost_history, dtype=float),
            "input_cost": np.asarray(self._input_cost_history, dtype=float),
            "stage_cost": np.asarray(self._stage_cost_history, dtype=float),
            "solve_time_ms": np.asarray(self._solve_time_ms_history, dtype=float),
            "solver_status": np.asarray(self._status_history, dtype=object),
            "eq_residual": np.asarray(self._eq_residual_history, dtype=float),
            "eq_res_u_past": np.asarray(self._eq_res_u_past_history, dtype=float),
            "eq_res_y_past": np.asarray(self._eq_res_y_past_history, dtype=float),
            "eq_res_u_fut": np.asarray(self._eq_res_u_fut_history, dtype=float),
            "kkt_residual": np.asarray(self._kkt_residual_history, dtype=float),
            "eps_sigma": np.asarray([float(getattr(self, "_eps_sigma", np.nan))]),
        }
        if extra is not None:
            for k, v in extra.items():
                payload[str(k)] = v
        np.savez_compressed(path, **payload)
