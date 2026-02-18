import numpy as np
import gymnasium as gym
from datetime import datetime
from select_deepc.deepc_controller import *
from select_deepc.deepc_dataclasses import *
from select_deepc.deepc_utils import *
from experiments.online_buffer import OnlineTrajectoryBuffer


def setup_DeePC(trajectory_data: TrajectoryDataSet) -> DeePCControllerArgs:
    """"Set up the Data Driven Controller

    NOTE: Paramters in here can be adapted to tune the desired performance
    """
    p = 5 # sensor state size
    m = 1 # input size
    n = 10 # "estimated" system size
    T_past = 2
    T_fut = 30
    T_hankel= (m + 1) * (T_past + T_fut) + n - 1

    # x y w1 w2
    Q = np.diag([5, 0, 10000, 0.1, 0.1])
    R = np.diag([1])
    g_regularization_1 = 50000.0
    g_regularization_pi =  0.0
    slack_cost = 5000000
    controller_costs = DeePCCost(Q, R, slack_cost, g_regularization_1, g_regularization_pi)

    A_u = None
    b_u = None
    A_y = None
    b_y = None

    A_y = np.array(
        [
            [1, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0],
        ]
    )

    b_y = np.array(
        [
            1,
            1,
        ]
    )

    A_u = np.array(
        [
            [1],
            [-1],
        ]
    )

    b_u = np.array(
        [
            3,
            3,
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
        [0],
        False,
    )



def run_experiment(
    controller,
    name,
    accumulator,
    use_online_buffer: bool = False,
    online_buffer: OnlineTrajectoryBuffer = None,
    online_buffer_kwargs: dict = None,
):
    env = gym.make("InvertedPendulum-v4-swingup", render_mode="rgb_array")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name_prefix = f"{name}_{timestamp}"
    env = gym.wrappers.RecordVideo(env=env, video_folder='video', name_prefix=video_name_prefix, episode_trigger=lambda ep: True)

    obs, _ = env.reset(seed=3)
    obs = np.array([obs[0], np.sin(obs[1]), np.cos(obs[1]), obs[2], obs[3]])

    traj = []
    refs = []
    preds = []
    g_vals = []

    if use_online_buffer and online_buffer is None:
        m = 1
        p = obs.size
        try:
            m = int(controller._dims.m)
            L = int(controller._dims.T_past + controller._dims.T_fut)
        except Exception:
            L = 32
        kwargs = online_buffer_kwargs or {}
        online_buffer = OnlineTrajectoryBuffer(
            m=m,
            p=p,
            max_steps=int(kwargs.get("max_steps", 4000)),
            segment_len=int(kwargs.get("segment_len", L)),
            stride=int(kwargs.get("stride", 1)),
        )

    T_sim = 300
    for i in range(T_sim):
        ref = [0, 0, 1, 0, 0]
        if i>100:
            ref=[0.5, 0, 1, 0, 0]
        refs.append(np.array(ref, dtype=float))
        action = controller.compute_action(obs, ref)
        if use_online_buffer and online_buffer is not None:
            online_buffer.append(
                np.asarray(action).reshape(-1),
                obs.reshape(-1),
                t_t=float(i),
            )
        accumulator.update_cost(obs, ref, action)
        try:
            g_vals.append(controller._deepc._g.value)
        except:
            pass
        obs, _, _, _, _ = env.step(action)
        obs = np.array([obs[0], np.sin(obs[1]), np.cos(obs[1]), obs[2], obs[3]])
        if i in [0, 15, 22, 32, 120]:
            preds.append(controller.get_planned_state_trajectory())
        traj.append(obs)

    env.close()

    controller.save_history_npz(
    f"logs/run_K{K}_seed{seed}.npz",
    extra={"K": K, "seed": seed}
    )

    traj = np.array(traj)
    refs = np.array(refs)
    return traj, preds, g_vals, accumulator.cost, refs


def selector_cb(adj_H_u, adj_H_y):
    return 1*adj_H_u, np.tile([1, 1, 1, 1, 1], int(adj_H_y.shape[0]/5)).reshape(-1,1) * adj_H_y
