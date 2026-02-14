# PID Swingup controller and inverted pendulum mujoco environments have been provided by Xian Li


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


from select_deepc.deepc_controller import *

PAPER_COL_WIDTH_IN = (6.75 - 0.25) / 2
PAPER_CONTENT_WIDTH_IN = 6.75
PAPER_COL_WIDTH_CM = PAPER_COL_WIDTH_IN * 2.54
PAPER_CONTENT_WIDTH_CM = PAPER_CONTENT_WIDTH_IN * 2.5

color_iso = "salmon"
color_l1 = "chocolate"
color_rand = "orchid"

