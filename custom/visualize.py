import numpy as np
import matplotlib.pyplot as plt

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



def generate_pole_end_pos(traj):
    pos_xs = traj[:,0].reshape(-1,1)
    pole_begin = np.concatenate((pos_xs, np.zeros_like(pos_xs)), axis=1)
    # traj slice is (N,)
    # (N, 2, 2)
    sins = np.atleast_2d(traj[:,1]).T
    coss = np.atleast_2d(traj[:,2]).T
    rot_matrices = np.concatenate(
        (
            np.concatenate((coss, -sins), axis=1).reshape(-1, 1, 2),
            np.concatenate((sins, coss), axis= 1).reshape(-1, 1, 2),
        ),
        axis=1,
    )
    rot_matrices = rot_matrices.transpose([0,2,1])
    pole_end_loc = np.array([[0, -1],[1, 0]]) @ rot_matrices @ np.array([0.6, 0])

    pole_end = pole_begin + pole_end_loc
    return pole_begin, pole_end



def generate_g_val_animation(g_vals):
    import matplotlib.animation as animation

    g_vals_cleaned = []
    for g in g_vals:
        if g is not None:
            g_vals_cleaned.append(g)
    fig, ax = plt.subplots()

    (line2,) = ax.plot(g_vals_cleaned[0], label=f"g values")
    ax.set(ylim=[-15, 15])
    ax.legend()

    def update(frame):
        # for each frame, update the data stored on each artist.

        # update the line plot:
        line2.set_ydata(g_vals_cleaned[frame])
        return line2

    ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
    ani.save("g_vals.gif")
    plt.show()

def generate_g_val_plot(g_vals):
    num_g_nonzero = []
    for g_vec in g_vals:
        # due to solver failure
        if g_vec is None:
            continue
        num_g_nonzero.append(np.sum(np.abs(g_vec) > 1e-1))

    fig, ax = plt.subplots(1, 1, figsize=(PAPER_COL_WIDTH_IN, 1.5))
    ax.plot(num_g_nonzero)
    ax.set_title("Number of Linearly Combined Trajectories")
    ax.set_ylabel("\# Nonzero Elements")
    ax.set_xlabel("Simulation Timestep")
    fig.savefig("figures/inverted_pendulum/num_nonzero_elem.pdf", bbox_inches="tight")
