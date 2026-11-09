"""Example plugin: reward term that keeps the head (the "eye") pointing in its start direction.

Shows how to shape RL behaviour without touching the core code. The term is added to the
reward every control step as  weight * value  (weight editable in the Plugin Manager).
"""
import math

PLUGIN_INFO = {"name": "Heading hold reward", "version": "1.0",
               "description": "Penalises head yaw rate and heading error (keeps BB-8 looking forward)."}


def register(api):
    k_rate = api.setting("yaw_rate_weight", 0.05, "penalty per (rad/s)^2 of yaw rate")
    k_head = api.setting("heading_weight", 0.2, "penalty per rad of heading error")

    def heading_term(env):
        roll, pitch, yaw = env.last_raw["rpy"]
        gz = env.last_raw["gyro"][2]
        return -(k_rate * gz * gz + k_head * abs(math.atan2(math.sin(yaw), math.cos(yaw))))

    api.reward_term("heading_hold", heading_term, weight=1.0)
