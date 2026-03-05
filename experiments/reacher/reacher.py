##1. 초기 설정
##2. offline data collection/ collected data load
##3. deepc controller 설정
##4. setpoint scheduler 설정
##5. run deepc simulation
##6. save results
##7. plot results

##1. 초기 설정
import numpy as np
np.set_printoptions(suppress=True, precision=3)


import matplotlib.pyplot as plt
from numpy.typing import NDArray

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from select_deepc.base_controllers import UniformInputGenerator, RandomWalkInputGenerator 
from select_deepc.deepc_controller import *
from select_deepc.deepc_utils import *
from select_deepc.deepc_rocket_utils import *

from select_deepc.hankel_generation import *

from itertools import product

import pandas as pd
import pickle

from tqdm import tqdm

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.sans-serif": "Times",
        "font.size": 10,
    }
)

PAPER_COL_WIDTH_IN = (6.75 - 0.25) / 2
PAPER_CONTENT_WIDTH_IN = 6.75
PAPER_COL_WIDTH_CM = PAPER_COL_WIDTH_IN * 2.54
PAPER_CONTENT_WIDTH_CM = PAPER_CONTENT_WIDTH_IN * 2.5

color_iso = "salmon"
color_l1 = "chocolate"
color_rand = "orchid"



##2. offline data collection/ collected data load
iid_trajectories = load_data_from_folder(
    os.path.join(REPO_ROOT, "data", "reacher", "dataset_iid"),
    "iid",
)
random_walk_trajectories = load_data_from_folder(
    os.path.join(REPO_ROOT, "data", "reacher", "dataset_random_walk"),
    "random_walk",
)


def fix_dataset(dataset: TrajectoryDataSet):
    for traj in dataset.dataset:
        traj.state_trajectory[:, [4, 5]] += traj.state_trajectory[:, [8, 9]]
        # traj.state_trajectory[:, [0]] = np.arctan2(traj.state_trajectory[:, 2], traj.state_trajectory[:, 0])
        # traj.state_trajectory[:, [1]] = np.arctan2(traj.state_trajectory[:, 3], traj.state_trajectory[:, 1])
        # traj.state_trajectory = traj.state_trajectory[:, [0, 1, 4, 5, 6, 7]]
        traj.state_trajectory = traj.state_trajectory[:, [0, 1, 2, 3, 4, 5, 6, 7]]

#dataset차원 맞추기. 전처리. fix
for dataset in [iid_trajectories, random_walk_trajectories]:
    fix_dataset(dataset)


##3. deepc controller 설정
def setup_DeePC(trajectory_data: TrajectoryDataSet, enable_measurement_constraint=False) -> DeePCControllerArgs:
    """"Set up the Data Driven Controller

    NOTE: Paramters in here can be adapted to tune the desired performance
    """
    p = 8 # sensor state size
    m = 2 # input size
    n = 75 # "estimated" system size
    T_past = 2
    T_fut = 15
    T_hankel= (m + 1) * (T_past + T_fut) + n - 1

    # x y w1 w2
    Q = np.diag([0, 0, 0, 0, 40000, 40000, 10, 10])
    R = np.diag([10, 10])
    g_regularization_1 = 10.000
    g_regularization_pi =  10000
    slack_cost = 5000000
    controller_costs = DeePCCost(Q, R, slack_cost, g_regularization_1, g_regularization_pi)

    A_u = None
    b_u = None
    A_y = None
    b_y = None

    A_u = np.array(
        [
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
        ]
    )
    b_u = np.array(
        [
            0.5,
            0.5,
            0.5,
            0.5,
        ]
    )
    if enable_measurement_constraint:
        A_y = np.array(
            [
                [0, 0, 0, 0, 0, 1, 0, 0],
            ]
        )
        b_y = np.array(
            [
                [0.1],
            ]
        )

    controller_constraints = DeePCConstraints(A_u, b_u, A_y, b_y)

    return DeePCControllerArgs(
        trajectory_data,
        DeePCDims(
            T_past,
            T_fut,
            p,
            m,
        ),
        T_hankel,
        controller_costs,
        controller_constraints,
        [0, 0],
        False,
    )

##4. setpoint scheduler 설정
class ReacherTargets:
    def __init__(self):
        num_steps = 4
        angles = np.linspace(3*np.pi/4, 2*np.pi, num_steps)
        np.random.seed(42)
        dists = np.random.uniform(0.04, 0.15, num_steps)

        self.targets = [[0.043, 0.092]]
        for angle, dist in zip(angles, dists):
            self.targets.append([dist*np.cos(angle), dist*np.sin(angle)])

        for target in self.targets:
            target[1] = np.minimum(target[1], 0.09)

class ReacherSetpointScheduler(SetPointScheduler):
    """Track a custom sequence of"""
    def __init__(self, env, deepc_dims: DeePCDims):
        super().__init__(env)
        self._dims = deepc_dims

        self._targets = ReacherTargets()
        self._sim_step = 1
        self._target_idx = 0

    def __call__(self, state, **kwargs):
        # target_pos = state[4:6]
        if (
            np.linalg.norm(
                state[4:6]
                + state[8:10]
                - np.array(self._targets.targets[self._target_idx])
            )
            < 0.005
        ):
            self._target_idx = (self._target_idx + 1) % len(self._targets.targets)
            print("updating setpoint")
            print(self._targets.targets[self._target_idx])

        self._sim_step += 1

        target_pos = self._targets.targets[self._target_idx]

        return [0, 0, 0, 0, target_pos[0], target_pos[1], 0, 0]

    def is_successful(self, state):
        return True


class DefaultReacherSetpoint(SetPointScheduler):
    """Track the setpoint provided by the env"""
    def __init__(self):
        pass

    def __call__(self, state, **kwargs):
        target_pos = state[4:6]
        return [0, 0, 0, 0, target_pos[0], target_pos[1], 0, 0]

    def is_successful(self, state):
        return True

##5. run deepc simulation
cost={}
solve_times={}

K_opts=[]
eps_sigmas=[]
selectec_idcs=[]
min_sigular_values_A=[]
slack_norm_infs=[]
slack_violations=[]
tracking_costs=[]
input_costs=[]
stage_costs=[]
selection_times=[]
total_times=[]
solve_time_ms=[]
status_counts=[]


controller_args = setup_DeePC(
    iid_trajectories,
    enable_measurement_constraint=True,
)

for num_cols in [200]: #[100, 150, 200, 300, 1000]:
    #num of steps=100, video folder, video title
    #TODO: custom video folder and title
    env=get_reacher_simulator(100, video_title=f"select_deepc")

    selector = LkSelector()

    deepc=SelectDeePC(controller_args, selector, num_cols, 5)

    

    (
        x_traj_deepc,
        u_traj_deepc,
        x_traj_deepc_predictions,
        u_traj_deepc_predictions,
        cost,
        solve_time
    ) = run_reacher_simulator(
        env,
        deepc,
        ReacherSetpointScheduler(env, controller_args.deepc_dims),
        deepc_cost_accumulator=PerformanceAccumulatorWrapper(
            DeePCCostAccumulator(controller_args.controller_costs),
            IntegralSquareError(mask = np.array([0,0,0,0, 1, 1, 0, 0])),
            IntegralAbsoluteError(mask = np.array([0,0,0,0, 1, 1, 0, 0])),
        ),
    )
    print(cost)

    solve_times[num_cols] = solve_time

    


##7. plot results
plt.plot(x_traj_deepc[:,4] + x_traj_deepc[:,8], x_traj_deepc[:, 5] + x_traj_deepc[:,9])

i = 0
for arr in x_traj_deepc_predictions:
    i+= 1
    if i in [-1,]:
        continue

    plt.plot(arr[:, 4], arr[:, 5], "--", alpha=0.75)

#목표점, constraint 궤적 표시
for sp in ReacherTargets().targets:
    plt.scatter(sp[0], sp[1], marker="x", c="black")

plt.fill_between(np.linspace(-.25, .25), 0.1 * np.ones(50), 0.25 * np.ones(50), color="red", alpha=0.5, zorder=0)
plt.title(f"Select-DeePC with {num_cols} columns")
plt.xlabel("x")
plt.ylabel("y")
out_path = fr"C:\Users\yelim\Desktop\paper\cdc2026\choose-wisely-paper-main\experiments\reacher\figures\reacher\select_deepc_num_cols{num_cols}.png"

plt.savefig(out_path)
plt.show()


##6. save results
K_opts=deepc.get_K_opt_history()
eps_sigmas=deepc.get_eps_sigma_history()
selected_idcs=deepc.get_selected_idcs_history()
min_sigular_values_A=deepc.get_sigma_min_A_history()
slack_norm_infs=deepc.get_slack_norm_inf_history()
slack_violations=deepc.get_slack_violation_history()
tracking_costs=deepc.get_tracking_cost_history()
input_costs=deepc.get_input_cost_history()
stage_costs=deepc.get_stage_cost_history()

deepc.save_history_npz(fr"C:\Users\yelim\Desktop\paper\cdc2026\choose-wisely-paper-main\experiments\reacher\results\select_deepc_num_cols{num_cols}.npz")

with open("data/reacher/costs_l1.pkl", "rb") as f:
    costs = pickle.load(f)
with open("data/reacher/costs_rand.pkl", "rb") as f:
    costs_rand = pickle.load(f)
with open("data/reacher/costs_clustered_iso.pkl", "rb") as f:
    costs_iso = pickle.load(f)


from matplotlib.lines import Line2D

blue = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
orange = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
green = plt.rcParams["axes.prop_cycle"].by_key()["color"][2]

fig, ax = plt.subplots(1,1, figsize=(PAPER_COL_WIDTH_IN,1.5))

ax.loglog(
    costs_rand.keys(),
    1e0*np.array([item["cost"] for item in costs_rand.values()]),
    linestyle="--",
    c=blue
)
ax.loglog(
    costs_rand.keys(),
    1e5*np.array([item["ISE"] for item in costs_rand.values()]),
    linestyle="--",
    c=green
)
ax.loglog(
    costs_rand.keys(),
    1e5 * np.array([item["IAE"] for item in costs_rand.values()]),
    linestyle="--",
    c=orange
)
ax.loglog(costs.keys(),  1e0* np.array([item["cost"] for item in costs.values()]),c=blue)
ax.loglog(costs.keys(), 1e5 * np.array([item["ISE"] for item in costs.values()]), c=green)
ax.loglog(costs.keys(), 1e5 * np.array([item["IAE"] for item in costs.values()]), c=orange)

ax.loglog(
    costs_iso.keys(), 1e0 * np.array([item["cost"] for item in costs_iso.values()]), ":", c=blue
)
ax.loglog(
    costs_iso.keys(), 1e5 * np.array([item["ISE"] for item in costs_iso.values()]), ":", c=green
)
ax.loglog(
    costs_iso.keys(), 1e5 * np.array([item["IAE"] for item in costs_iso.values()]), ":", c=orange
)

custom_lines = [
    Line2D([0],[0], color='black', linestyle="-"),
    Line2D([0],[0], color=blue, linestyle="-"),
    Line2D([0],[0], color='black',  linestyle="--"),
    Line2D([0],[0], color=orange, linestyle="-"),
    Line2D([0],[0], color='black',  linestyle=":"),
    Line2D([0],[0], color=green, linestyle="-"),
]
ax.set_xlabel("Number of Cols")
ax.set_ylabel("Cost")
ax.set_title("Cost in Reacher Simulation as\na Function of Hankel size")
ax.set_ylim(1e-1, None)
fig.legend(
    custom_lines,
    ["$L_1$", "Cost", "Random", "IAE", "Isomap", "ISE"],
    loc="lower left",
    bbox_to_anchor=(0.0, -.6),
    ncols=3,
)
# fig.legend(custom_lines_2, ["Cost", "IAE", "ISE"], loc="lower right")
fig.savefig(fr"C:\Users\yelim\Desktop\paper\cdc2026\choose-wisely-paper-main\experiments\reacher\figures\reacher\select_deepc_num_cols{num_cols}_costs_vs_rows.png", bbox_inches="tight")