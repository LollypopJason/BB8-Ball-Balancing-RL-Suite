"""
classical_controller.py
=======================

Classical (non-learned) balance controller for the head-balancing BB-8 ballbot.

Why it exists in an RL suite
----------------------------
* **Baseline**: a known-good reference to compare every RL policy against.
* **Teacher**: its demonstrations warm-start the tanh policy network by behaviour cloning
  (``rl_onnx_cpp_engine.behaviour_clone``) before PPO fine-tuning.
* **Safety fallback / first hardware test**: it exports to the same C++ API as the RL policy
  (``compute_stepper_commands``) so the Jetson runners can use it unchanged.

It only uses signals the real robot measures (IMU roll/pitch/yaw + gyro, wheel encoder
velocities and the odometry-based target error), i.e. exactly the policy observation vector.
The file depends on numpy only, so it can be copied to the Jetson.

Control law (per horizontal axis, head/heading frame, small angles)
------------------------------------------------------------------
    v_ball  = R * (w_ball_y, -w_ball_x),   w_ball = gyro - J^-1 * wheel_speeds      (odometry)
    a_cmd   = k_tilt * tilt + k_rate * tilt_rate + k_vel * v_ball + k_pos * pos_err (clipped)
              gains from pole placement (all poles at -bandwidth), see pole_placement_gains()
    v_cmd  += a_cmd * dt                                                             (clipped)
    wheels  = J * (w_head_ff - w_ball_cmd),   w_ball_cmd = (-v_cmd_y / R, v_cmd_x / R, 0)

The steppers are velocity-controlled, so integrating the desired ball acceleration into a
velocity command is the natural "inverted pendulum on a cart" structure. Pitch (+y rotation,
nose down) maps to +x, roll (+x rotation, right side down) maps to -y.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

# full observation layout (see config_and_distributions.ALL_OBS_CHANNELS)
I_RPY, I_GYRO, I_ENC_V, I_TGT = slice(0, 3), slice(3, 6), slice(9, 12), slice(15, 17)


def omniwheel_jacobian(R: float, wheel_r: Sequence[float], zenith_deg: float, azimuths_deg: Sequence[float]) -> np.ndarray:
    """wheel_speeds = J @ w_rel (head-relative-to-ball angular velocity, head frame)."""
    th = math.radians(zenith_deg)
    rows = []
    for az, r in zip(azimuths_deg, wheel_r):
        a = math.radians(az)
        rows.append((R / r) * np.array([-math.cos(th) * math.cos(a), -math.cos(th) * math.sin(a), math.sin(th)]))
    return np.stack(rows)


def pole_placement_gains(pendulum_length: float, bandwidth: float, g: float = 9.81) -> dict:
    """Gains for  a = k_tilt*theta + k_rate*theta_dot + k_pos*x + k_vel*v  on the linearised
    pendulum  theta_dd = (g/l) theta - a/l,  x_dd = a,  with all four closed-loop poles at -bandwidth.
    Characteristic polynomial: s^4 + (k2/l - k4) s^3 + (k1/l - k3 - g/l) s^2 + (g k4/l) s + g k3/l.
    Note: k_pos and k_vel are POSITIVE (same sign as the tilt gain) - to come back the ball must
    first roll a little further away so the head tips back (non-minimum-phase behaviour)."""
    l, p = pendulum_length, bandwidth
    k_pos = p ** 4 * l / g
    k_vel = 4 * p ** 3 * l / g
    k_rate = l * (4 * p + k_vel)
    k_tilt = l * (6 * p ** 2 + k_pos) + g
    return dict(k_tilt=k_tilt, k_rate=k_rate, k_pos=k_pos, k_vel=k_vel)


@dataclass
class ControllerParams:
    pendulum_length: float = 0.33   # [m] effective inverted-pendulum length (ball centre -> head CoG)
    bandwidth: float = 4.0          # [rad/s] closed-loop pole location (higher = stiffer, noisier)
    k_tilt: float = 0.0             # gains: 0 = computed from pendulum_length + bandwidth
    k_rate: float = 0.0
    k_pos: float = 0.0
    k_vel: float = 0.0
    k_int: float = 0.0              # optional integral of position (removes CoG-offset stand-off)
    int_max: float = 1.0            # [m*s] anti-windup clamp
    v_max: float = 1.0              # [m/s] commanded ball speed limit
    a_max: float = 8.0              # [m/s^2]
    pos_err_max: float = 0.5        # [m] clip of the position / target error
    k_yaw: float = 0.0              # [1/s] heading hold (0 = only keep head yaw locked to the ball)
    gyro_feedforward: bool = True   # add measured head rotation so the ball follows v_cmd exactly
    hold_position: bool = True      # integrate odometry to hold the start point when no target is set
    memoryless_lead_s: float = 0.0  # >0: v_cmd = v_measured + a*lead (stateless -> exactly learnable by an MLP)

    def resolved(self) -> "ControllerParams":
        out = ControllerParams(**asdict(self))
        pp = pole_placement_gains(self.pendulum_length, self.bandwidth)
        for k, v in pp.items():
            if getattr(out, k) == 0.0:
                setattr(out, k, v)
        return out


class ClassicalBalanceController:
    """Callable on the *full raw* observation vector (20 channels) -> 3 normalised wheel commands."""

    def __init__(self, ball_radius: float, wheel_radius: float, zenith_deg: float,
                 azimuths_deg: Sequence[float] = (0.0, 120.0, 240.0), max_wheel_speed: float = 45.0,
                 control_dt: float = 0.01, params: Optional[ControllerParams] = None):
        self.R = float(ball_radius)
        self.J = omniwheel_jacobian(self.R, [wheel_radius] * 3, zenith_deg, azimuths_deg)
        self.Jinv = np.linalg.inv(self.J)
        self.max_w = float(max_wheel_speed)
        self.dt = float(control_dt)
        self.params = params or ControllerParams()
        self.p = self.params.resolved()
        self.geometry = dict(ball_radius=ball_radius, wheel_radius=wheel_radius, zenith_deg=zenith_deg,
                             azimuths_deg=list(azimuths_deg), max_wheel_speed=max_wheel_speed, control_dt=control_dt)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.v_cmd = np.zeros(2)
        self.pos = np.zeros(2)       # odometry in the yaw-aligned start frame
        self.yaw0: Optional[float] = None
        self.integ = np.zeros(2)

    def ball_velocity(self, gyro: np.ndarray, enc_v: np.ndarray) -> np.ndarray:
        w_rel = self.Jinv @ enc_v
        w_ball = gyro - w_rel
        return self.R * np.array([w_ball[1], -w_ball[0]])

    def __call__(self, obs_full: np.ndarray) -> np.ndarray:
        p = self.p
        o = np.asarray(obs_full, dtype=np.float64)
        roll, pitch, yaw = o[I_RPY]
        gyro = o[I_GYRO]
        enc_v = o[I_ENC_V]
        tgt = o[I_TGT]
        v = self.ball_velocity(gyro, enc_v)
        # odometry (start frame) for position hold
        if self.yaw0 is None:
            self.yaw0 = yaw
        c, s = math.cos(yaw - self.yaw0), math.sin(yaw - self.yaw0)
        self.pos += self.dt * np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
        if np.any(tgt != 0.0):
            err = -tgt                                    # target error is already in the heading frame
        elif p.hold_position:
            err = np.array([c * self.pos[0] + s * self.pos[1], -s * self.pos[0] + c * self.pos[1]])
        else:
            err = np.zeros(2)
        n = np.linalg.norm(err)
        if n > p.pos_err_max:
            err *= p.pos_err_max / n
        tilt = np.array([pitch, -roll])                   # axis x <- pitch, axis y <- -roll
        rate = np.array([gyro[1], -gyro[0]])
        self.integ = np.clip(self.integ + err * self.dt, -p.int_max, p.int_max)
        a = p.k_tilt * tilt + p.k_rate * rate + p.k_vel * v + p.k_pos * err + p.k_int * self.integ
        a = np.clip(a, -p.a_max, p.a_max)
        if p.memoryless_lead_s > 0:
            self.v_cmd = np.clip(v + a * p.memoryless_lead_s, -p.v_max, p.v_max)
        else:
            self.v_cmd = np.clip(self.v_cmd + a * self.dt, -p.v_max, p.v_max)
        w_ball_cmd = np.array([-self.v_cmd[1] / self.R, self.v_cmd[0] / self.R, 0.0])
        w_head = np.array([gyro[0], gyro[1], gyro[2]]) if p.gyro_feedforward else np.zeros(3)
        w_head[2] = gyro[2] - p.k_yaw * yaw if p.k_yaw > 0 else 0.0   # damp/hold head yaw relative to the ball
        wheels = self.J @ (w_head - w_ball_cmd)
        return np.clip(wheels / self.max_w, -1.0, 1.0)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"type": "classical_balance_controller", "version": 2, "geometry": self.geometry,
                "params": asdict(self.params), "resolved_gains": asdict(self.p)}

    def save_json(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "ClassicalBalanceController":
        g = d["geometry"]
        return cls(g["ball_radius"], g["wheel_radius"], g["zenith_deg"], g["azimuths_deg"], g["max_wheel_speed"],
                   g["control_dt"], ControllerParams(**d.get("params", {})))

    @classmethod
    def load_json(cls, path: str) -> "ClassicalBalanceController":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def controller_from_config(cfg, params: Optional[ControllerParams] = None) -> ClassicalBalanceController:
    """Build from an AppConfig (simulator side)."""
    rc = cfg.robot
    return ClassicalBalanceController(rc.ball_radius_m, rc.wheel_radius_m, rc.zenith_angle_deg, rc.wheel_azimuths_deg,
                                      rc.stepper_max_speed_rads, rc.frame_skip * rc.physics_timestep_s, params)



# ============================================================================
# C++ export (same API as the RL policy export -> drop-in for the Jetson runners)
# ============================================================================
def generate_controller_cpp(ctrl: ClassicalBalanceController, out_dir: str,
                            basename: str = "bb8_policy_inference", steps_per_rad: float = 509.2958) -> dict:
    """Writes <basename>.h/.cpp exposing ``bb8::compute_stepper_commands(...)`` exactly like the
    RL export, so ``deployment_jetson`` builds it unchanged. The controller is stateful
    (velocity integrator + odometry): call ``bb8::reset_controller()`` before enabling motors."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    p, g = ctrl.p, ctrl.geometry
    chans = ["imu_roll_rad", "imu_pitch_rad", "imu_yaw_rad", "imu_gx_rads", "imu_gy_rads", "imu_gz_rads",
             "imu_ax_ms2", "imu_ay_ms2", "imu_az_ms2", "enc1_vel_rads", "enc2_vel_rads", "enc3_vel_rads",
             "enc1_pos_rad", "enc2_pos_rad", "enc3_pos_rad", "target_dx_m", "target_dy_m",
             "prev_motor1_cmd", "prev_motor2_cmd", "prev_motor3_cmd"]
    f = lambda x: f"{float(x):.9g}f" if any(ch in f"{float(x):.9g}" for ch in ".en") else f"{float(x):.9g}.0f"
    J = ctrl.J
    Ji = ctrl.Jinv
    mat = lambda M: "{" + ", ".join("{" + ", ".join(f(v) for v in row) + "}" for row in M) + "}"
    args = ",\n    ".join(f"float {c}" for c in chans)
    outs = ",\n    ".join([f"float* out_motor{i}_step_rate" for i in (1, 2, 3)] +
                          [f"float* out_motor{i}_rads = nullptr" for i in (1, 2, 3)])
    h = f"""// Auto-generated by BB-8 RL Suite - CLASSICAL BALANCE CONTROLLER (drop-in for the RL export)
#ifndef BB8_POLICY_INFERENCE_H
#define BB8_POLICY_INFERENCE_H
namespace bb8 {{
constexpr int   kNumPolicyInputs  = 20;
constexpr int   kNumPolicyOutputs = 3;
constexpr float kActionScale      = {f(ctrl.max_w)};
constexpr float kStepsPerRad      = {f(steps_per_rad)};
constexpr bool  kIsClassicalController = true;
void reset_controller();
/** obs_raw: all 20 channels in the standard order -> action_norm[3] in [-1, 1] */
void policy_forward(const float* obs_raw, float* action_norm);
void compute_stepper_commands(
    {args},
    {outs});
}}  // namespace bb8
extern "C" void bb8_policy_forward_c(const float* obs_raw, float* action_norm);
#endif
"""
    cpp = f"""// Auto-generated by BB-8 RL Suite - classical balance controller
// params: {json.dumps(asdict(p))}
// geometry: {json.dumps(g)}
#include "{basename}.h"
#include <cmath>
namespace bb8 {{
namespace {{
const float kJ[3][3] = {mat(J)};
const float kJinv[3][3] = {mat(Ji)};
const float kR = {f(ctrl.R)}, kDt = {f(ctrl.dt)}, kMaxW = {f(ctrl.max_w)};
const float kTilt = {f(p.k_tilt)}, kRate = {f(p.k_rate)}, kVel = {f(p.k_vel)}, kPos = {f(p.k_pos)};
const float kVmax = {f(p.v_max)}, kAmax = {f(p.a_max)}, kPosErrMax = {f(p.pos_err_max)}, kYaw = {f(p.k_yaw)};
const float kInt = {f(p.k_int)}, kIntMax = {f(p.int_max)};
const bool kGyroFF = {str(p.gyro_feedforward).lower()}, kHold = {str(p.hold_position).lower()};
float g_v[2] = {{0, 0}}, g_pos[2] = {{0, 0}}, g_int[2] = {{0, 0}}, g_yaw0 = 0.f; bool g_init = false;
inline float clampf(float x, float lo, float hi) {{ return x < lo ? lo : (x > hi ? hi : x); }}
}}  // namespace

void reset_controller() {{ g_v[0] = g_v[1] = g_pos[0] = g_pos[1] = g_int[0] = g_int[1] = 0.f; g_init = false; }}

void policy_forward(const float* o, float* out) {{
    const float roll = o[0], pitch = o[1], yaw = o[2], gx = o[3], gy = o[4], gz = o[5];
    float wr[3];
    for (int i = 0; i < 3; ++i) wr[i] = kJinv[i][0] * o[9] + kJinv[i][1] * o[10] + kJinv[i][2] * o[11];
    const float vx = kR * (gy - wr[1]), vy = -kR * (gx - wr[0]);
    if (!g_init) {{ g_yaw0 = yaw; g_init = true; }}
    const float c = std::cos(yaw - g_yaw0), s = std::sin(yaw - g_yaw0);
    g_pos[0] += kDt * (c * vx - s * vy); g_pos[1] += kDt * (s * vx + c * vy);
    float ex = 0.f, ey = 0.f;
    if (o[15] != 0.f || o[16] != 0.f) {{ ex = -o[15]; ey = -o[16]; }}
    else if (kHold) {{ ex = c * g_pos[0] + s * g_pos[1]; ey = -s * g_pos[0] + c * g_pos[1]; }}
    const float n = std::sqrt(ex * ex + ey * ey);
    if (n > kPosErrMax) {{ ex *= kPosErrMax / n; ey *= kPosErrMax / n; }}
    g_int[0] = clampf(g_int[0] + ex * kDt, -kIntMax, kIntMax);
    g_int[1] = clampf(g_int[1] + ey * kDt, -kIntMax, kIntMax);
    const float ax = clampf(kTilt * pitch + kRate * gy + kVel * vx + kPos * ex + kInt * g_int[0], -kAmax, kAmax);
    const float ay = clampf(-kTilt * roll - kRate * gx + kVel * vy + kPos * ey + kInt * g_int[1], -kAmax, kAmax);
    g_v[0] = clampf(g_v[0] + ax * kDt, -kVmax, kVmax);
    g_v[1] = clampf(g_v[1] + ay * kDt, -kVmax, kVmax);
    float w[3] = {{kGyroFF ? gx : 0.f, kGyroFF ? gy : 0.f, kYaw > 0.f ? gz - kYaw * yaw : 0.f}};
    w[0] -= -g_v[1] / kR; w[1] -= g_v[0] / kR;
    for (int i = 0; i < 3; ++i)
        out[i] = clampf((kJ[i][0] * w[0] + kJ[i][1] * w[1] + kJ[i][2] * w[2]) / kMaxW, -1.f, 1.f);
}}

void compute_stepper_commands({", ".join(f"float {c}" for c in chans)},
    float* out_motor1_step_rate, float* out_motor2_step_rate, float* out_motor3_step_rate,
    float* out_motor1_rads, float* out_motor2_rads, float* out_motor3_rads) {{
    const float o[20] = {{{", ".join(chans)}}};
    float a[3]; policy_forward(o, a);
    float* sr[3] = {{out_motor1_step_rate, out_motor2_step_rate, out_motor3_step_rate}};
    float* rd[3] = {{out_motor1_rads, out_motor2_rads, out_motor3_rads}};
    for (int i = 0; i < 3; ++i) {{ const float w = a[i] * kActionScale; *sr[i] = w * kStepsPerRad; if (rd[i]) *rd[i] = w; }}
}}
}}  // namespace bb8
extern "C" void bb8_policy_forward_c(const float* obs_raw, float* action_norm) {{ bb8::policy_forward(obs_raw, action_norm); }}
"""
    hp, cp = os.path.join(out_dir, basename + ".h"), os.path.join(out_dir, basename + ".cpp")
    open(hp, "w").write(h)
    open(cp, "w").write(cpp)
    return {"h": hp, "cpp": cp}
