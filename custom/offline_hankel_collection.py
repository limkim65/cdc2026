
import sys
from os import path
sys.path.append(path.join(path.dirname(path.abspath(""))))
sys.path.append(path.join(path.dirname(path.abspath("")), "select_deepc"))
from select_deepc.deepc_controller import *
import time
import numpy as np
import gymnasium as gym

def collect_closedloop_controller_data(animation=False):
        
    gym.envs.register(
        "InvertedPendulum-v4-swingup",
        entry_point="inverted_pendulum_v4_swingup:InvertedPendulumSwingupEnv",
        max_episode_steps=1000,
        reward_threshold=950.0,
    )

    # Perform the swingu up with a feedback controller, to gather data
    np.random.seed(4)

    # whether to show the animation, faster to run without
    show_animation = animation
    # whether to show the predictions, cleaner plots without
    show_predictions = 0

    # Parameters for the feedback controller
    ksu = 1.05
    kcw = 1.5
    kvw = 5
    kem = 1e-13
    eta = 1.0

    m = 4.78
    M = 10.47
    L = 0.3
    g = 9.81
    Vd = 2 * m * g * L

    xmax = 0.7
    vmax = 10

    # PID controller parameters
    Kp_p = -3.2
    Ki_p = -0.0
    Kd_p = 0.5

    Kp_c = 0.2
    Ki_c = -0.005
    Kd_c = 0.5

    integral_p = 0.0
    integral_c = 0.0

    stabilise = 0
    excitation = 0.02
    numSteps = 20000
    y = np.zeros(numSteps)
    x = np.zeros((4, numSteps))

    u = np.random.normal(0, 1, numSteps)

    reset_time = []
    failed = []
    seed = 3

    refpos = 0

    if show_animation:
        env = gym.make("InvertedPendulum-v4-swingup", render_mode="human")
    else:
        env = gym.make("InvertedPendulum-v4-swingup")
    obs, info = env.reset(seed=seed)

    # TODO: directly gather TrajectoryData
    for k in range(numSteps):
            obs += np.random.normal(0, 0.00001, (np.size(obs)))
            angle = obs[1]  # Angle of the pendulum
            dangle = obs[3]
            pos = obs[0]
            speed = obs[2]
            error_p = -angle 
            error_c = refpos - pos  
            integral_c += error_c
                    
            angle = obs[1] - np.pi
            V = 0.5 * m * L**2 * dangle**2 + m * g * L * (1 - np.cos(angle))
            angle = obs[1]
            
            # if the pendulum is close to the upright position, switch to the stabilising controller
            if np.abs(obs[1]) < 0.2:
                if stabilise == 0:
                    refpos = pos
                    integral_c = 0
                stabilise = 1

            if not stabilise:
                # the swing up controller
                u[k] = -ksu * np.sign(dangle * np.cos(angle)) + kcw * np.sign(pos) * np.log(1 - np.abs(pos)/(xmax*2)) + \
                        kem * (np.exp(np.abs(V - eta * Vd)) - 1) * np.sign(V - Vd) * np.sign(dangle * np.cos(angle)) + np.random.normal(0, excitation)
                if k > 200:
                    u[k] += np.random.normal(0, 0.1)
            else:
                # the stabilising controller
                u[k] = Kp_p * error_p + Ki_p * integral_p + Kd_p * dangle \
                    + Kp_c * error_c + Ki_c * integral_c + Kd_c * speed + np.random.normal(0, 0.05)
                
                if (k % 20) == 0:
                    u[k] += np.random.normal(0,1)
                    
            x[:,k] = obs
            

            # Clip the control input to the action space limits
            u[k] = np.clip(u[k], env.action_space.low[0], env.action_space.high[0])
            obs, reward, done, truncated, info = env.step([u[k]])
            
            if show_animation:
                time.sleep(0.05)
                
            if done or truncated or (k + 1)  % 100 == 0:
                seed += 1
                if np.abs(obs[1]) > 0.1:
                    failed += [k]
                obs, info = env.reset(seed = seed)
                reset_time += [k]
                stabilise = 0
                integral_c = 0


    env.close()

    y = x
    y_ref = y[:, 0:reset_time[0] + 1]
    u_ref = u[0:reset_time[0] + 1]

    print("# of failed trajectory:", len(failed))
    print("# of collected trajectory:", len(reset_time))

    ref = np.vstack((u_ref.T, y_ref))
    return ref, reset_time, u, y
    
def create_trajectory_dataset(reset_time, u, y, dataset_name="swingup_data"):

    swingup_dataset = TrajectoryDataSet(
        [], dataset_name
    )
    last_t=-1
    for t in reset_time:
        # if t in failed:
        #     last_t = t
        #     continue

        # Extract sections for y and u
        y_section = y[:, last_t+1:t+1].T
        y_section_featurized = np.array(
            [
                y_section[:,0],
                np.sin(y_section[:,1]),
                np.cos(y_section[:,1]),
                y_section[:,2],
                y_section[:,3],
            ]
        ).T
        u_section = u[last_t+1:t+1].reshape(-1,1)
        if y_section.size == 0:
            continue
        last_t = t
        numSteps_section = len(u_section)
        swingup_dataset.dataset.append(
            TrajectoryData(y_section_featurized.copy(), u_section.copy())
        )
            
        
    return swingup_dataset

def create_corrupted_trajectory_dataset(reset_time, u, y, dataset_name="corrupted_swingup_data"):
    corrupted_swingup_dataset = TrajectoryDataSet(
        [], dataset_name
    )
    last_t=-1
    for t in reset_time:
        # if t in failed:
        #     last_t = t
        #     continue

        # Extract sections for y and u
        y_section = y[:, last_t+1:t+1].T
        u_section = u[last_t+1:t+1].reshape(-1,1)

        # cos(theta) 원본 값 (각도는 y_section[:,1])
        cos_raw = np.cos(y_section[:, 1])         # (T,)

        # cos(theta) >= 0.5 가 되는 첫 시점 찾기
        upper_indices = np.where(cos_raw >= 0.5)[0]

        
        if len(upper_indices) > 0:
            cut_idx = upper_indices[0]           # 0~cut_idx-1 까지만 사용
            if cut_idx < 2:                      # 너무 짧으면 버리기 (옵션)
                last_t = t
                continue
            y_section = y_section[:cut_idx, :]
            u_section = u_section[:cut_idx, :]


        y_section_featurized = np.array(
            [
                y_section[:,0],
                np.sin(y_section[:,1]),
                np.cos(y_section[:,1]),
                y_section[:,2],
                y_section[:,3],
            ]
        ).T
        if y_section.size == 0:
            continue
        
        #debugging.. nan 제거용
        if not np.isfinite(y_section_featurized).all() or not np.isfinite(u_section).all():
            last_t = t
            continue

        last_t = t
        numSteps_section = len(u_section)



        corrupted_swingup_dataset.dataset.append(
            TrajectoryData(y_section_featurized.copy(), u_section.copy())
        )
    return corrupted_swingup_dataset