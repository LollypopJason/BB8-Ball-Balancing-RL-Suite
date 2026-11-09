"""
classical_controller.py
=======================

Model-based classical balance controller for the head-balancing BB-8 ballbot.

It is a baseline / plant sanity check for both omniwheel models ("virtual" and
"roller") and uses only the sensors the real robot has: IMU roll / pitch / gyro and
the three wheel encoders (velocity).

Plant (per horizontal head-frame axis, linearised about upright)
---------------------------------------------------------------
    phi   : ball rotation angle            (ball travel  x = R * phi)
    theta : head tilt (absolute)
    u     : head-relative-to-ball rate     (what the omniwheels impose:  wheel = J u)

The steppers are stiff velocity sources, so ``u`` is the commanded quantity and the
free dynamics follow from the head + ball Lagrangian with the internal motor torque
eliminated::

    theta'' = a * theta + b * u'          a = m_h g l / M,  b = (M11 + M12) / M
    phi'    = theta' - u                  M = M11 + 2 M12 + M22

Taking the relative-rate *derivative* ``v = u'`` as input gives the 4-state model
``x = [phi, theta, theta', u]`` which is placed by Ackermann's formula with all poles
at ``-bandwidth`` (the ball-position pole at ``-pos_pole_frac * bandwidth``). The
command ``u`` is integrated at the control rate and mapped to the three wheel speeds
with the exact omniwheel Jacobian ``wheel = J u``.

Usage::

    from classical_controller import controller_from_config
    env = BB8BalanceEnv(cfg, training=False)
    ctrl = controller_from_config(cfg)
    obs, info = env.reset(seed=0); ctrl.reset()
    while True:
        obs, r, term, trunc, info = env.step(ctrl(info["obs_full"]))
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config_and_distributions import GRAVITY, AppConfig, default_config
from mjcf_generator import omniwheel_jacobian

# indices into the full (unscaled) observation vector, see config_and_distributions.OBS_GROUPS
_RPY, _GYRO, _ENC_V = slice(0, 3), slice(3, 6), slice(9, 12)


@dataclass
class ControllerParams:
    bandwidth: float = 4.0          # closed-loop pole magnitude [rad/s]
    pos_pole_frac: float = 0.35     # ball-position pole = -pos_pole_frac * bandwidth (0 -> velocity hold only)
    yaw_rate_gain: float = 0.5      # relative yaw rate command per rad/s of head yaw rate
    max_cmd_frac: float = 0.9       # wheel-speed command limit as a fraction of stepper_max_speed_rads
    pos_limit_m: float = 0.5        # clamp of the integrated ball-position error (anti wind-up)


def _place(A: np.ndarray, B: np.ndarray, poles) -> np.ndarray:
    """Ackermann's formula for single-input pole placement: returns K with eig(A - B K) = poles."""
    n = A.shape[0]
    C = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])
    coeffs = np.real(np.poly(poles))           # s^n + c1 s^(n-1) + ... + cn
    phi = sum(c * np.linalg.matrix_power(A, n - k) for k, c in enumerate(coeffs))
    e_n = np.zeros((1, n))
    e_n[0, -1] = 1.0
    return (e_n @ np.linalg.solve(C, phi)).ravel()


def plant_coefficients(cfg: AppConfig) -> tuple:
    """(a, b) of  theta'' = a theta + b u'  for the nominal robot."""
    rc = cfg.robot
    R = rc.ball_radius_m
    mb = cfg.randomization.ball.mass_kg.nominal()
    Ib = rc.ball_inertia_factor * mb * R ** 2
    mh = cfg.randomization.head.mass_kg.nominal()
    l = R + rc.head_cog_height_m
    rh, hh = 0.75 * rc.head_radius_m, 0.14            # same inertia model as mjcf_generator
    Ih = mh * (3 * rh ** 2 + hh ** 2) / 12
    M11 = (mb + mh) * R ** 2 + Ib
    M12 = mh * R * l
    M22 = mh * l ** 2 + Ih
    M = M11 + 2 * M12 + M22
    return mh * GRAVITY * l / M, (M11 + M12) / M


class ClassicalController:
    """Pole-placement balance controller producing normalised wheel-speed actions."""

    def __init__(self, cfg: Optional[AppConfig] = None, params: Optional[ControllerParams] = None):
        self.cfg = cfg or default_config()
        if self.cfg.network.action_mode == "torque":
            raise ValueError("ClassicalController needs the velocity action mode")
        self.params = params or ControllerParams()
        rc = self.cfg.robot
        self.dt = rc.control_dt_s
        self.max_speed = rc.stepper_max_speed_rads
        self.J = omniwheel_jacobian(rc.ball_radius_m, rc.zenith_angle_deg, rc.wheel_azimuths_deg,
                                    [rc.wheel_radius_m] * 3)
        self.J_inv = np.linalg.inv(self.J)
        a, b = plant_coefficients(self.cfg)
        self.a, self.b = a, b
        A = np.array([[0, 0, 1, -1], [0, 0, 1, 0], [0, a, 0, 0], [0, 0, 0, 0]], dtype=float)
        B = np.array([[0], [0], [b], [1]], dtype=float)
        w = self.params.bandwidth
        wp = self.params.pos_pole_frac * w
        if wp > 0:
            self.K = _place(A, B, [-w, -w, -w, -wp])
        else:   # velocity hold only: drop the position state
            self.K = np.concatenate([[0.0], _place(A[1:, 1:], B[1:], [-w, -w, -w])])
        self.reset()

    def reset(self) -> None:
        self.u_cmd = np.zeros(3)      # head-relative-to-ball rate command (head frame)
        self.phi = np.zeros(2)        # integrated ball rotation (odometry) about head x / y

    def __call__(self, obs_full: np.ndarray) -> np.ndarray:
        return self.act(obs_full)

    def act(self, obs_full: np.ndarray) -> np.ndarray:
        """``obs_full``: the full unscaled observation (``info["obs_full"]`` with ``training=False``)."""
        o = np.asarray(obs_full, dtype=np.float64)
        p, dt = self.params, self.dt
        roll, pitch = o[_RPY][0], o[_RPY][1]
        gyro = o[_GYRO]
        u_meas = self.J_inv @ o[_ENC_V]
        # head-frame tilt about x / y (small-angle ZYX: independent of yaw)
        theta = np.array([roll, pitch])
        theta_d = gyro[:2]
        phi_d = theta_d - u_meas[:2]                       # ball rate from IMU + encoder odometry
        R = self.cfg.robot.ball_radius_m
        lim = p.pos_limit_m / R
        self.phi = np.clip(self.phi + phi_d * dt, -lim, lim)
        for k in range(2):
            x = np.array([self.phi[k], theta[k], theta_d[k], self.u_cmd[k]])
            self.u_cmd[k] += -float(self.K @ x) * dt
        self.u_cmd[2] = -p.yaw_rate_gain * gyro[2]
        wheel = self.J @ self.u_cmd
        cap = p.max_cmd_frac * self.max_speed
        s = np.max(np.abs(wheel)) / cap
        if s > 1.0:                                         # keep the direction, saturate the magnitude
            wheel /= s
            self.u_cmd = self.J_inv @ wheel
        return np.clip(wheel / self.max_speed, -1.0, 1.0)


def controller_from_config(cfg: Optional[AppConfig] = None,
                           params: Optional[ControllerParams] = None) -> ClassicalController:
    return ClassicalController(cfg, params)
