import sys
import os

# 현재 파일의 부모의 부모 디렉토리(상위 디렉토리)를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from setup_deepc import *
from setup import *
from visualize import *
from offline_hankel_collection import *
from select_deepc.deepc_controller import *
from select_deepc.hankel_generation import HankelMatrixGenerator

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import casadi as ca
from IPython.display import clear_output, display
import time

import matplotlib.ticker as ticker
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



# parameters
corrupted = False
compare = False

#offline data collection
ref_off, reset_time, u_off, y_off = collect_closedloop_controller_data(animation=False)

trajectory_data = TrajectoryDataSet(
    [], "swingup_data"
)

trajectory_data=create_trajectory_dataset(reset_time, u_off, y_off, "swingup_data")

corrupted_trajectory_data = TrajectoryDataSet(
    [], "corrupted_swingup_data"
)

corrupted_trajectory_data=create_corrupted_trajectory_dataset(reset_time, u_off, y_off, "corrupted_swingup_data")


if corrupted is True:
    controller_args = setup_DeePC(corrupted_trajectory_data)
else:
    controller_args = setup_DeePC(trajectory_data)

H_u_offline, H_y_offline = HankelMatrixGenerator(
    controller_args.deepc_dims.T_past,
    controller_args.deepc_dims.T_fut,
).generate_hankel_matrices(controller_args.trajectory_data)

# SVD can fail if H_u contains NaN/Inf columns from unstable rollouts.
finite_col_mask = np.all(np.isfinite(H_u_offline), axis=0)
H_u_svd = H_u_offline[:, finite_col_mask]
if H_u_svd.shape[1] == 0:
    raise RuntimeError("No finite columns left in H_u for SVD.")
sv_offline = np.linalg.svd(H_u_svd, compute_uv=False)
k = 30  # 1-based rank
print(f"{k}th largest singular value = {sv_offline[k-1]}")
print("num trajectories:", len(controller_args.trajectory_data.dataset))
print("H_u shape:", H_u_offline.shape)



#Setup controller
select_deepc = SelectDeePC(
    controller_args,
    LkSelector(
        1,
        controller_args.deepc_dims,
        custom_callback=selector_cb,
        forgetting_factor=0.8
    ),
    num_hankel_cols=controller_args.deepc_dims.T_fut*controller_args.deepc_dims.m + 10,
    n_iter=1,
    debug=False,
)
#run select deepc
traj_select_deepc, preds_select_deepc, g_vals_select_deepc, cost_select_deepc, ref_traj_select = run_experiment(
    select_deepc, "select-deepc", DeePCCostAccumulator(controller_args.controller_costs)
)
sigma_min_hist = select_deepc.get_min_selected_singular_values()

# compare with deepc
if compare == True:
    deepc = DataDrivenPredictiveController.create_from_data(controller_args, True)
    traj_deepc, preds_deepc, _, _, ref_traj_deepc = run_experiment(deepc, "deepc", DeePCCostAccumulator(controller_args.controller_costs))
    plt.plot(traj_deepc[:, 2])
    for idx, pred in enumerate(preds_deepc):
        plt.plot(np.arange(10*idx-2, 10*idx+pred.shape[0]-2, 1), pred[:,2], "--")

#plot results
plt.plot(traj_select_deepc[:, 2])

for idx, pred in enumerate(preds_select_deepc):
    plt.plot(np.arange(10*idx-2, 10*idx+pred.shape[0]-2, 1), pred[:,2], "--")

plt.show()

# Position trajectory with reference line.
plt.figure()
plt.plot(traj_select_deepc[:, 0], label="Select-DeePC")
if compare == True:
    plt.plot(traj_deepc[:, 0], label="Full Data DeePC")
ref_pos = ref_traj_select[:, 0]
plt.plot(ref_pos, color="black", linestyle="--", linewidth=1, label="Reference")
plt.xlabel("Step")
plt.ylabel("Cart position")
plt.title("Position trajectory")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

angle = np.arctan2(traj_select_deepc[:,1], traj_select_deepc[:,2])
angle[angle < -1] += 2*np.pi
angle_plot = angle
plt.plot(angle_plot)
plt.plot([0]*angle_plot.size, "black", linestyle="--")
plt.show()



fig, axs = plt.subplots(1, 1, figsize=(3, 2.5))

blue = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
orange = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]


pole_begin, pole_end = generate_pole_end_pos(traj_select_deepc)
if compare == True:
    pole_b_d, pole_e_d = generate_pole_end_pos(traj_deepc)
axs.plot(
    np.linspace(-1,1),
    np.zeros(50),
    c="black", 
    linewidth=2,
)
vis_idcs = [0, 15, 22, 32, 120]


for idx, i in enumerate(vis_idcs):
    axs.plot(
    [pole_begin[i,0]-0.03, pole_begin[i,0]+0.03],
    [pole_begin[i,1], pole_begin[i,1]],
    color="black",
    linewidth=7,
    )
    axs.plot(
        [pole_begin[i,0], pole_end[i,0]],
        [pole_begin[i,1], pole_end[i,1]],
        linewidth=6,
        color="black",
    )
    axs.plot(
        [pole_begin[i,0], pole_end[i,0]],
        [pole_begin[i,1], pole_end[i,1]],
        linewidth=5,
        color="grey",
    )

    if idx == 0:
        continue

    _, pole_end_traj = generate_pole_end_pos(preds_select_deepc[idx])
    axs.plot(
        pole_end_traj[:,0],
        pole_end_traj[:,1],
        c=blue,
        linestyle="--",
        zorder=500,
        alpha=0.5
    )


axs.tick_params(
    axis='both',
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False, # labels along the bottom edge are off)
    labelleft=False,
)

axs.plot(pole_end[:,0], pole_end[:,1], c=blue, linestyle="-", alpha=1, zorder=50, label="Select-DeePC")
if compare == True:
    axs.plot(pole_e_d[:,0], pole_e_d[:,1], c=orange, linestyle="-", alpha=1, label="Full Data DeePC")

# axs.set_box_aspect(1)
axs.set_ylim(-0.75,0.75)
axs.set_aspect("equal")
axs.set_xlabel("$x$")
axs.set_ylabel("$y$")
axs.set_title("Closed Loop Trajectory")
fig.legend(ncols=2, bbox_to_anchor=(0.89, 0.07), fontsize="small", handlelength=1.25)
fig.tight_layout()
if compare == True:
    fig.savefig("figures/inverted_pendulum/inv-pendulum-cl.pdf", bbox_inches='tight')
else:
    fig.savefig("figures/inverted_pendulum/inv-pendulum-cl-select-deepc.pdf", bbox_inches='tight')


generate_g_val_plot(g_vals_select_deepc)
plt.show()

plt.figure()
plt.plot(sigma_min_hist)
plt.xlabel("Step")
plt.ylabel("Min singular value (selected Hankel)")
plt.title("Per-step minimum selected singular value")
plt.grid(True, alpha=0.3)
plt.show()

plt.figure()
plt.semilogy(sv_offline, marker="o", markersize=3, linewidth=1)
plt.xlabel("Index")
plt.ylabel("Singular value")
plt.title("Offline input Hankel (H_u) singular values")
plt.grid(True, alpha=0.3)
plt.show()




