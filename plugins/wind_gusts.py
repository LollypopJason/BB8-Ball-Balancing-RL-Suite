"""Example plugin: gusty wind on the head (Ornstein-Uhlenbeck process).

Unlike the built-in short pushes this is a *continuous*, slowly varying force - good for
training robustness against airflow / cable drag / a hand resting on the head.
"""
import math

import numpy as np

PLUGIN_INFO = {"name": "Wind gusts", "version": "1.0",
               "description": "Continuous random wind force on the head (mean + OU gusts)."}


def register(api):
    mean_n = api.setting("mean_force_n", 1.0, "steady wind force [N]")
    gust_n = api.setting("gust_force_n", 2.0, "gust standard deviation [N]")
    tau = api.setting("gust_time_s", 1.5, "gust correlation time [s]")
    direction = api.setting("direction_deg", 0.0, "steady wind direction [deg] (world frame)")
    state = {}

    def on_reset(env):
        state[id(env)] = np.zeros(2)

    def on_step(env):
        g = state.setdefault(id(env), np.zeros(2))
        dt = env.dt_ctrl
        g += -g * dt / tau + gust_n * math.sqrt(2 * dt / tau) * env._rng.normal(size=2)
        a = math.radians(direction)
        f = mean_n * np.array([math.cos(a), math.sin(a)]) + g
        env.add_external_force(env.head_bid, (f[0], f[1], 0.0))

    api.on_reset(on_reset)
    api.on_step(on_step)
