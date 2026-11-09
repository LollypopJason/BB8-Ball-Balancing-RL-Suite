"""Example plugin: open-loop wheel sine sweep (a "controller" for verification, not balancing).

Use it in the GUI (policy source -> Plugin: Sine sweep) with the robot on a stand / the
validation rig to compare encoder speed vs. command (latency, deadband, missed steps).
"""
import math

import numpy as np

PLUGIN_INFO = {"name": "Sine sweep controller", "version": "1.0",
               "description": "Open-loop chirp on all wheels for motor/encoder verification."}


def register(api):
    amp = api.setting("amplitude", 0.3, "normalised command amplitude (0..1)")
    f0 = api.setting("f_start_hz", 0.2, "start frequency [Hz]")
    f1 = api.setting("f_end_hz", 3.0, "end frequency [Hz]")
    dur = api.setting("duration_s", 20.0, "sweep duration [s]")

    def factory(cfg):
        dt = cfg.robot.frame_skip * cfg.robot.physics_timestep_s

        class Sweep:
            def __init__(self):
                self.t = 0.0

            def reset(self):
                self.t = 0.0

            def __call__(self, obs_full):
                t = self.t % dur
                k = (f1 - f0) / dur
                phase = 2 * math.pi * (f0 * t + 0.5 * k * t * t)
                self.t += dt
                return np.array([amp * math.sin(phase + i * 2 * math.pi / 3) for i in range(3)])
        return Sweep()

    api.controller("Sine sweep", factory)
