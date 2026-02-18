import warnings

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from typing import Optional

from .base_controllers import BaseController
from .data_selectors import *
from .deepc_utils import *
from .hankel_generation import *


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

        self._g = cp.Variable(U_f.shape[-1])
        self._u = cp.Variable(self._T_fut * self._m)
        self._y = cp.Variable(self._T_fut * self._p)

        self._u_past = cp.Parameter(self._T_past * self._m)
        self._y_past = cp.Parameter(self._T_past * self._p)
        self._reference_traj = cp.Parameter(self._T_fut * self._p)

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
            np.kron(np.eye(self._T_fut), controller_costs.Q),
        )

        cost += cp.quad_form(
            (self._u - np.tile(self._input_reference, self._T_fut)),
            np.kron(np.eye(self._T_fut), controller_costs.R),
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
            np.kron(np.eye(self._T_fut), controller_costs.Q),
        )
        self._problem_input_cost=cp.quad_form(
            (self._u - np.tile(self._input_reference, self._T_fut)),
            np.kron(np.eye(self._T_fut), controller_costs.R),
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

        # Append computed input to past input trajectory.
        self._u_past.value = np.append(
            self._u_past.value[self._m :].copy(), computed_action
        )
        self._prev_input = computed_action.copy()
        self._initial_state_overridden = False
        return computed_action

    def get_last_slack_inf(self):
        return self._last_slack_inf

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
            return 0
        tracking_cost = self._problem_tracking_cost

        return tracking_cost

    def get_input_cost(self):
        if self._problem_input_cost is None:
            return 0
        input_cost = self._problem_input_cost

        return input_cost

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


class SelectDeePC_original(BaseController):
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
        self._slack_norm_inf_history = []
        self._slack_violation_history = []
        self._tracking_cost_history = []
        self._input_cost_history = []
        self._stage_cost_history = []
        self._solve_time_ms_history = []
        self._status_history = []
        

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
            time_after_sel = perf_counter()

            idcs = idcs[: self._num_hankel_cols]
            self._last_selected_idcs = idcs.copy()
            if idcs.size > 0:
                singular_values = np.linalg.svd(self._H_u[:, idcs], compute_uv=False)
                step_min_sigma = float(np.min(singular_values))

            #####----#####
            #custom 
            k_opt = idcs.size
            self._K_opt_history.append(k_opt)
            self._selected_idcs_history.append(idcs)
             
            m = self._dims.m
            p = self._dims.p
            Tp = self._T_past
            Tf = self._T_fut
      
            u_ini_rows = m * Tp
            u_tot_rows = m * (Tp+ Tf)
            y_ini_rows = p * Tp
            y_tot_rows = p * (Tp+Tf)
            U_ini_selected = self._H_u[:u_ini_rows, idcs]          # U_p
            U_fut_selected = self._H_u[u_ini_rows:u_tot_rows, idcs] # U_f
            Y_ini_selected = self._H_y[:y_ini_rows, idcs]          # Y_p
            # debugging..
            # U_ini_selected, U_fut_selected, Y_ini_selected 슬라이싱 인덱싱이 잘 되었는지 확인
            # 세 부분을 다시 vstack (원래 H_u, H_y의 선택된 열과 비교)
            U_sel_reconstructed = np.vstack([U_ini_selected, U_fut_selected])
            Y_sel_reconstructed = Y_ini_selected  # Y는 Y_p 만 사용했으므로 비교는 아래와 같이
            H_u_sel = self._H_u[:, idcs]
            H_y_sel = self._H_y[:, idcs]
            
            check_u = np.allclose(U_sel_reconstructed, H_u_sel)
            check_y = np.allclose(Y_sel_reconstructed, H_y_sel[:y_ini_rows, :])

            if not (check_u and check_y):
                print(
                    f"[sanity check] U 재조합 일치 여부: {check_u}, Y 재조합 일치 여부: {check_y} "
                    f"U_sel_reconstructed.shape={U_sel_reconstructed.shape}, H_u_sel.shape={H_u_sel.shape}, "
                    f"Y_sel_reconstructed.shape={Y_sel_reconstructed.shape}, H_y_sel.shape={H_y_sel.shape}"
                )
            
            #A=np.block(
            #    [
            #        [U_ini_selected, np.zeros(U_ini_selected.shape[0],self._m * (self._T_past + self._T_fut)), np.zeros(U_ini_selected.shape[0],self._p * (self._T_past))]
            #        [Y_ini_selected, np.zeros(Y_ini_selected.shape[0], self._m * (self._T_past + self._T_fut)), np.eyes(self._p * (self._T_past))]
            #        [Y_fut_selected, -np.eye(self._m * (self._T_past + self._T_fut)), np.zeros(self._m * (self._T_past + self._T_fut),self._p*self._T_past)]
            #    ]
            #)
            A_g=np.vstack(
                [
                    U_ini_selected,
                    Y_ini_selected,
                    U_fut_selected,
                ]
            )
            sigular_values_A_g=np.linalg.svd(A_g, compute_uv=False)
            step_min_sigma_A_g = float(np.min(singular_values_A_g))
            self._sigma_min_A_history.append(step_min_sigma_A_g)

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

        self._min_selected_sigma_history.append(step_min_sigma)
        self._u_past = np.append(self._u_past[self._dims.m :], action)


        ####-----####
        #custom
        slack_inf = float(self._deepc.get_last_slack_inf())
        self._slack_norm_inf_history.append(slack_inf)
        eps = self._eps_sigma# 미리 클래스에 저장해둔 threshold
        self._slack_violation_history.append(int(slack_inf > eps))

        self._tracking_cost_history.append(self._deepc.get_tracking_cost())
        self._input_cost_history.append(self._deepc.get_input_cost())
        self._stage_cost_history.append(self._deepc.get_tracking_cost()+self._deepc.get_input_cost())
        self._solve_time_ms_history.append(1000.0*(time_after_solve - time_before_solve))
        self._status_history.append(None)

        ####-----####


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


    def save_history_npz(self, path: str, extra: Optional[dict] = None) -> None:
        """Save per-step histories to a .npz file for offline postprocessing."""
        payload = {
            "K_opt": np.asarray(self._K_opt_history, dtype=float),
            "sigma_min_Hu": np.asarray(self._min_selected_sigma_history, dtype=float),
            "sigma_min_Ag": np.asarray(self._sigma_min_A_history, dtype=float),
            "slack_inf": np.asarray(self._slack_norm_inf_history, dtype=float),
            "slack_fail": np.asarray(self._slack_violation_history, dtype=int),
            "tracking_cost": np.asarray(self._tracking_cost_history, dtype=float),
            "input_cost": np.asarray(self._input_cost_history, dtype=float),
            "stage_cost": np.asarray(self._stage_cost_history, dtype=float),
            "solve_time_ms": np.asarray(self._solve_time_ms_history, dtype=float),
            "eps_sigma": np.asarray([float(getattr(self, "_eps_sigma", np.nan))]),
        }
        if extra is not None:
            for k, v in extra.items():
                payload[str(k)] = v
        np.savez_compressed(path, **payload)