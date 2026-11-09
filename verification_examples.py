"""
verification_examples.py - quick real-world experiments that verify the simulator
==================================================================================

Each example is a small, safe experiment you can do with the real robot (or parts of it) in a
few minutes. For every example the suite gives:

* the procedure + equipment + safety note,
* the **simulator's prediction** (computed from your current configuration),
* a comparison with what you measured (PASS / CHECK, with a hint what to fix),
* for several examples an **"apply"** step that writes your measurement into the configuration
  (robot dimensions/masses, IMU noise ranges, rolling resistance identified from a ramp roll-out).

Hardware measurements can be typed in or imported from the JSON written by
``deployment_jetson/jetson_control_loop.py --test <name>``.

CLI:
    python verification_examples.py list
    python verification_examples.py predict fall_time
    python verification_examples.py sheet > VERIFICATION.md       # printable checklist
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from config_and_distributions import DP, AppConfig, default_config


@dataclass
class Field:
    key: str
    label: str
    unit: str = ""
    tolerance: float = 0.1            # relative (if rel=True) or absolute tolerance for PASS
    rel: bool = True
    default: float = 0.0


@dataclass
class Example:
    key: str
    title: str
    category: str                     # "Measure", "Quick check", "Closed loop"
    minutes: int
    equipment: List[str]
    steps: List[str]
    fields: List[Field]
    predict: Callable[[AppConfig], Dict[str, float]]
    runner_test: str = ""             # jetson_control_loop.py --test <name>
    safety: str = ""
    apply: Optional[Callable[[AppConfig, Dict[str, float]], List[str]]] = None
    apply_label: str = ""
    why: str = ""


# ============================================================================
# Simulation helpers
# ============================================================================
def _ideal(cfg: AppConfig) -> AppConfig:
    c = cfg.copy()
    c.terrain = type(c.terrain)()                 # flat floor
    c.terrain.off_ideal_nominal = True
    c.disturbance.enabled = False
    c.plugins.enabled = []
    c.model_type = "bb8"
    return c


def simulate_coastdown(cfg: AppConfig, ramp_height: float, ramp_length: float = 1.0,
                       rolling_resistance: Optional[float] = None, max_s: float = 60.0) -> float:
    """Bare ball released at the top of a ramp; returns roll-out distance [m] from the ramp foot."""
    import mujoco
    b = cfg.randomization.ball
    R = cfg.robot.ball_radius_m
    m = b.mass_kg.nominal()
    I = cfg.robot.ball_inertia_factor * m * R * R
    roll = b.rolling_resistance.nominal() if rolling_resistance is None else rolling_resistance
    slide, tors = b.sliding_grip.nominal(), b.torsional_friction.nominal()
    a = math.asin(min(0.95, ramp_height / ramp_length))
    th = 0.01
    cx, cz = -ramp_length * math.cos(a) / 2, ramp_height / 2
    nx, nz = math.sin(a), math.cos(a)
    top = np.array([-ramp_length * math.cos(a) + 0.05 * math.cos(a), 0, ramp_height - 0.05 * math.sin(a)])
    start = top + np.array([nx, 0, nz]) * (R + 0.002)
    xml = f"""<mujoco><option timestep="0.002" integrator="implicitfast" cone="elliptic"/>
<worldbody>
  <geom name="floor" type="plane" size="30 5 0.1" friction="{slide} {tors} {roll}" condim="6" priority="2"/>
  <geom name="ramp" type="box" size="{ramp_length / 2 + 0.02} 0.5 {th}" pos="{cx - nx * th} 0 {cz - nz * th - 0.0015}"
        euler="0 {math.degrees(a)} 0" friction="{slide} {tors} {roll}" condim="6" priority="2"/>
  <body name="ball" pos="{start[0]} 0 {start[2]}"><freejoint/>
    <inertial pos="0 0 0" mass="{m}" diaginertia="{I} {I} {I}"/>
    <geom type="sphere" size="{R}" condim="6"/></body>
</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    moved = False
    while data.time < max_s:
        mujoco.mj_step(model, data)
        v = np.linalg.norm(data.qvel[:3])
        if v > 0.05:
            moved = True
        if moved and v < 0.005 and data.qpos[0] > 0:
            break
    return float(max(0.0, data.qpos[0]))


def simulate_fall_time(cfg: AppConfig, cog_height: Optional[float] = None, init_tilt_deg: float = 3.0) -> float:
    """Motors holding (zero speed command): time for the head to fall from 5 deg to 25 deg."""
    from bb8_gym_env import BB8BalanceEnv
    c = _ideal(cfg)
    if cog_height is not None:
        c.robot.head_cog_height_m = cog_height
    c.episode.init_tilt_deg = DP.const(init_tilt_deg)
    c.episode.auto_reset_on_fall = False
    c.episode.fall_tilt_deg = 89.0
    c.episode.max_episode_s = 10.0
    env = BB8BalanceEnv(c, training=False)
    env.reset(seed=0)
    t5 = t25 = None
    for _ in range(env.max_steps):
        _, _, _, trunc, info = env.step(np.zeros(env.action_space.shape))
        if t5 is None and info["tilt_deg"] > 5.0:
            t5 = info["time"]
        if info["tilt_deg"] > 25.0:
            t25 = info["time"]
            break
    env.close()
    return float(t25 - t5) if (t5 is not None and t25 is not None) else float("nan")


def _bisect(f: Callable[[float], float], target: float, lo: float, hi: float, increasing: bool, it: int = 14) -> float:
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if (f(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _classical_run(cfg: AppConfig, seconds: float, difficulty: float, push: Optional[Tuple[float, float]] = None,
                   seed: int = 0) -> Dict[str, float]:
    from bb8_gym_env import BB8BalanceEnv
    from classical_controller import controller_from_config
    c = cfg.copy()
    c.disturbance.enabled = False
    c.plugins.enabled = []
    c.episode.max_episode_s = seconds
    env = BB8BalanceEnv(c, training=False)
    env.set_difficulty(difficulty)
    env.reset(seed=seed)
    ctrl = controller_from_config(c)
    obs = env._observe()
    max_tilt, rec_t, pushed = 0.0, None, False
    for _ in range(env.max_steps):
        if push and not pushed and env.data.time >= 2.0:
            env.apply_push([push[0], 0, 0], push[1], "head")
            pushed = True
            max_tilt = 0.0
        _, _, term, trunc, info = env.step(ctrl(obs))
        obs = info["obs_full"]
        max_tilt = max(max_tilt, info["tilt_deg"])
        if pushed and rec_t is None and env.data.time > 2.0 + push[1] + 0.1 and info["tilt_deg"] < 2.0:
            rec_t = env.data.time - 2.0
        if term or trunc:
            break
    out = {"survived_s": float(env.data.time), "max_tilt_deg": max_tilt,
           "drift_m": float(np.linalg.norm(env.ball_pos()[:2])), "fell": bool(env.fallen)}
    if push:
        out["recovery_time_s"] = float(rec_t) if rec_t is not None else float("nan")
    env.close()
    return out


# ============================================================================
# Predictions / apply functions
# ============================================================================
def _wheel_jacobian(cfg):
    from mjcf_generator import omniwheel_jacobian
    rc = cfg.robot
    return omniwheel_jacobian(rc.ball_radius_m, rc.zenith_angle_deg, rc.wheel_azimuths_deg, [rc.wheel_radius_m] * 3)


def pred_measure(cfg):
    rc = cfg.robot
    return {"ball_diameter_m": 2 * rc.ball_radius_m, "ball_mass_kg": cfg.randomization.ball.mass_kg.nominal(),
            "head_mass_kg": rc.head_mass_kg, "head_cog_above_ball_m": rc.head_cog_height_m,
            "wheel_diameter_m": rc.wheel_diameter_m, "zenith_deg": rc.zenith_angle_deg}


def apply_measure(cfg, m):
    ch = []
    rc = cfg.robot
    if m.get("ball_diameter_m"):
        rc.ball_radius_m = m["ball_diameter_m"] / 2
        ch.append(f"ball radius -> {rc.ball_radius_m:.3f} m")
    if m.get("ball_mass_kg"):
        mm = m["ball_mass_kg"]
        cfg.randomization.ball.mass_kg = DP.uniform(0.95 * mm, 1.05 * mm, "kg", 0.05, 20)
        ch.append(f"ball mass -> U({0.95 * mm:.2f}, {1.05 * mm:.2f}) kg")
    if m.get("head_mass_kg"):
        hm = m["head_mass_kg"]
        rc.head_mass_kg = hm
        cfg.randomization.head.mass_kg = DP.uniform(0.95 * hm, 1.05 * hm, "kg", 0.2, 20)
        ch.append(f"head mass -> {hm:.2f} kg (randomised +-5 %)")
    if m.get("head_cog_above_ball_m"):
        rc.head_cog_height_m = m["head_cog_above_ball_m"]
        ch.append(f"head CoG height -> {rc.head_cog_height_m:.3f} m")
    if m.get("wheel_diameter_m"):
        rc.wheel_diameter_m = m["wheel_diameter_m"]
        ch.append(f"wheel diameter -> {rc.wheel_diameter_m * 1000:.1f} mm")
    if m.get("zenith_deg"):
        rc.zenith_angle_deg = m["zenith_deg"]
        ch.append(f"zenith angle -> {rc.zenith_angle_deg:.1f} deg")
    return ch


def pred_encoder_turns(cfg):
    return {f"wheel{i}_turns": 10.0 for i in (1, 2, 3)}


def pred_imu_noise(cfg):
    imu = cfg.randomization.imu
    return {"roll_std_deg": math.degrees(imu.orient_noise_std_rad.nominal()),
            "gyro_std_rads": imu.gyro_noise_std_rads.nominal(),
            "accel_std_ms2": imu.accel_noise_std_ms2.nominal(),
            "yaw_drift_deg_per_min": math.degrees(imu.yaw_drift_std_rad.nominal()) * math.sqrt(60.0),
            "accel_norm_ms2": 9.81}


def apply_imu_noise(cfg, m):
    imu = cfg.randomization.imu
    ch = []
    if m.get("roll_std_deg"):
        s = math.radians(m["roll_std_deg"])
        imu.orient_noise_std_rad = DP.uniform(0.5 * s, 2.0 * s, "rad", 0, 0.2)
        ch.append(f"orientation noise -> U({0.5 * s:.4f}, {2 * s:.4f}) rad")
    if m.get("gyro_std_rads"):
        s = m["gyro_std_rads"]
        imu.gyro_noise_std_rads = DP.uniform(0.5 * s, 2.0 * s, "rad/s", 0, 1)
        ch.append(f"gyro noise -> U({0.5 * s:.4f}, {2 * s:.4f}) rad/s")
    if m.get("accel_std_ms2"):
        s = m["accel_std_ms2"]
        imu.accel_noise_std_ms2 = DP.uniform(0.5 * s, 2.0 * s, "m/s^2", 0, 3)
        ch.append(f"accel noise -> U({0.5 * s:.3f}, {2 * s:.3f}) m/s^2")
    if m.get("yaw_drift_deg_per_min"):
        s = math.radians(abs(m["yaw_drift_deg_per_min"])) / math.sqrt(60.0)
        imu.yaw_drift_std_rad = DP.uniform(0.5 * s, 2.0 * s, "rad/sqrt(s)", 0, 0.2)
        ch.append(f"yaw drift -> U({0.5 * s:.5f}, {2 * s:.5f}) rad/sqrt(s)")
    return ch


COAST_H = 0.02    # ramp height used in the coast-down example [m] (low: the 50 cm ball rolls far)
COAST_L = 1.00


def pred_coastdown(cfg):
    return {"rollout_distance_m": simulate_coastdown(cfg, COAST_H, COAST_L)}


def apply_coastdown(cfg, m):
    d = m.get("rollout_distance_m")
    if not d:
        return []
    rr = _bisect(lambda x: simulate_coastdown(cfg, COAST_H, COAST_L, rolling_resistance=x), d, 1e-5, 0.03,
                 increasing=False, it=12)
    cfg.randomization.ball.rolling_resistance = DP.uniform(0.6 * rr, 1.4 * rr, "m", 0.0, 0.05)
    return [f"rolling resistance identified: {rr:.5f} m -> randomised U({0.6 * rr:.5f}, {1.4 * rr:.5f})"]


def pred_ratio(cfg):
    from bb8_gym_env import ValidationRigSim
    rc = cfg.robot
    rig = ValidationRigSim(cfg)
    speed, revs = 10.0, 10
    T = revs * 2 * math.pi / speed
    ball_angle, n = 0.0, int(T / 0.01)
    for _ in range(n):
        t = rig.step(speed, 0.01)
        ball_angle += t["ball_speed_meas_rads"] * 0.01
    return {"ball_revs": abs(ball_angle) / (2 * math.pi), "ball_revs_ideal": revs * rc.wheel_radius_m / rc.ball_radius_m}


def pred_motor_step(cfg):
    from bb8_gym_env import ValidationRigSim
    rig = ValidationRigSim(cfg)
    t63 = None
    vals = []
    for k in range(100):
        t = rig.step(10.0, 0.01)
        vals.append(t["enc_vel_rads"])
        if t63 is None and t["enc_vel_rads"] > 6.3:
            t63 = (k + 1) * 0.01
    return {"motor1_time_to_63pct_s": t63 or float("nan"), "motor1_speed_ratio": float(np.mean(vals[60:])) / 10.0}


def pred_spin(cfg):
    J = _wheel_jacobian(cfg)
    return {"head_yaw_rate_rads": float((np.linalg.inv(J) @ np.full(3, 10.0))[2])}


def pred_straight(cfg):
    return {"distance_m": 0.2 * 3.0, "heading_deg": 0.0}


def pred_fall_time(cfg):
    return {"time_5_to_25deg_s": simulate_fall_time(cfg)}


def pred_first_balance(cfg):
    a = _classical_run(cfg, 30.0, 0.0, seed=1)
    b = _classical_run(cfg, 30.0, 1.0, seed=2)
    return {"balanced_s": 30.0, "max_tilt_deg": max(a["max_tilt_deg"], b["max_tilt_deg"]),
            "drift_after_30s_m": max(a["drift_m"], b["drift_m"])}


def pred_push(cfg):
    r = _classical_run(cfg, 8.0, 0.0, push=(10.0, 0.2), seed=3)
    return {"max_tilt_deg": r["max_tilt_deg"], "recovery_time_s": r["recovery_time_s"]}


# ============================================================================
# Catalog
# ============================================================================
CATALOG: List[Example] = [
    Example("measure_robot", "Measure & enter your robot", "Measure", 10,
            ["kitchen / luggage scale", "tape measure or calliper", "ruler + knife edge (or a pencil)"],
            ["Weigh the bare ball and the complete head (with battery, Jetson, motors, wheels).",
             "Measure the ball diameter at 3 places and average.",
             "Head CoG: lay the fully assembled head on its side across a pencil and roll it until it balances; "
             "measure the balance point's height above the head plate, then add the gap between the ball top and "
             "the head plate (measured on the assembled robot) = CoG height above the ball top (+-1 cm is fine).",
             "Measure the wheel diameter and the contact (zenith) angle from your CAD / with a protractor."],
            [Field("ball_diameter_m", "Ball diameter", "m", 0.02), Field("ball_mass_kg", "Ball mass", "kg", 0.05),
             Field("head_mass_kg", "Head mass", "kg", 0.05), Field("head_cog_above_ball_m", "Head CoG above ball top", "m", 0.15),
             Field("wheel_diameter_m", "Wheel diameter", "m", 0.02), Field("zenith_deg", "Zenith contact angle", "deg", 0.03)],
            pred_measure, apply=apply_measure, apply_label="Apply measured values to the configuration",
            why="Every other prediction depends on these numbers."),
    Example("encoder_turns", "Encoder count check (by hand)", "Quick check", 2,
            ["robot on a stand (ball removed)", "a tape mark on each wheel"],
            ["Power the Jetson + encoders (motors may stay unpowered).",
             "Run the runner test, then turn each wheel exactly 10 turns by hand in its positive direction.",
             "Wrong count -> magnet gap / mux channel; negative -> flip encoder_direction_sign."],
            [Field(f"wheel{i}_turns", f"Wheel {i} turns", "rev", 0.02) for i in (1, 2, 3)],
            pred_encoder_turns, runner_test="encoder-turns",
            why="Encoder velocity is a policy input; a wrong sign makes every learned controller fail instantly."),
    Example("signs", "Wiring & sign check (interactive)", "Quick check", 3,
            ["robot on a stand (ball removed)", "the Jetson terminal"],
            ["Run the runner test and follow the prompts: tilt nose-down, tilt right side down, rotate the head "
             "counter-clockwise, turn each wheel forward by hand.",
             "Every value must be True. Fix IMU axes in the Arduino sketch (kAxisMap / kAxisSign) and encoder signs "
             "in robot_hw_config.json, then repeat."],
            [Field("pitch_positive_when_nose_down", "Pitch + when nose down"),
             Field("roll_positive_when_right_down", "Roll + when right side down"),
             Field("gz_positive_when_ccw", "Yaw rate + when turning CCW")] +
            [Field(f"encoder{i}_positive", f"Encoder {i} + when turned forward") for i in (1, 2, 3)],
            lambda cfg: {k: True for k in ("pitch_positive_when_nose_down", "roll_positive_when_right_down",
                                            "gz_positive_when_ccw", "encoder1_positive", "encoder2_positive",
                                            "encoder3_positive")},
            runner_test="signs", why="The simulator's sign conventions must match the hardware exactly."),
    Example("imu_noise", "IMU noise & drift (robot still)", "Quick check", 2,
            ["robot standing still on its stand or held still"],
            ["Run the runner test (60 s), do not touch the robot.",
             "Import the JSON. 'Apply' writes your measured noise into the IMU randomisation ranges."],
            [Field("roll_std_deg", "Roll noise std", "deg", 1.0), Field("gyro_std_rads", "Gyro noise std", "rad/s", 1.0),
             Field("accel_std_ms2", "Accel noise std", "m/s^2", 1.0),
             Field("yaw_drift_deg_per_min", "Yaw drift", "deg/min", 3.0, rel=False),
             Field("accel_norm_ms2", "Accel magnitude (gravity)", "m/s^2", 0.03)],
            pred_imu_noise, runner_test="imu-noise", apply=apply_imu_noise,
            apply_label="Use measured IMU noise for domain randomisation",
            why="Sim-to-real: train with the noise your sensor really has."),
    Example("ball_coastdown", "Ball roll-out on a ramp (no electronics)", "Quick check", 5,
            ["the bare ball", f"a straight board {COAST_L:.1f} m long, one end raised {COAST_H * 100:.0f} cm (books)",
             "tape measure", "a long flat hallway (the ball rolls several metres)"],
            [f"Raise one end of the board by exactly {COAST_H * 100:.0f} cm.",
             "Place the ball at the top end and release it without pushing.",
             "Measure from the foot of the ramp to where the ball stops. Repeat 3x, enter the average.",
             "'Apply' identifies the ball's rolling resistance on your floor."],
            [Field("rollout_distance_m", "Roll-out distance", "m", 0.25)],
            pred_coastdown, apply=apply_coastdown, apply_label="Identify rolling resistance from the roll-out",
            why="Rolling resistance of a hand-coated ball on your floor is hard to guess - and matters for drift."),
    Example("transmission_ratio", "Wheel -> ball transmission ratio (rig)", "Quick check", 5,
            ["validation rig: one motor + omniwheel pressing on the ball, ball free to turn (e.g. on 3 casters)",
             "tape mark on the ball"],
            ["Run the runner test (motor 1 turns the wheel 10 revolutions at 10 rad/s).",
             "Count the ball revolutions about the wheel axle (fractions by the mark position)."],
            [Field("ball_revs", "Ball revolutions", "rev", 0.08)],
            pred_ratio, runner_test="ratio",
            why="Checks wheel diameter, microstep setting and roller slip in one go."),
    Example("motor_step", "Motor step response", "Quick check", 3,
            ["robot on a stand, wheels spinning freely"],
            ["Run the runner test: each motor +10 rad/s, stop, -10 rad/s.",
             "Import the JSON (motor 1 is compared with the simulated stepper)."],
            [Field("motor1_time_to_63pct_s", "Motor 1 time to 63 %", "s", 0.05, rel=False),
             Field("motor1_speed_ratio", "Motor 1 speed ratio", "", 0.05)],
            pred_motor_step, runner_test="motor-step",
            why="Speed ratio != 1 -> microsteps / missed steps; slow response -> latency the policy must handle."),
    Example("spin_in_place", "Spin-in-place (Jacobian & signs)", "Quick check", 2,
            ["robot on the floor, someone holding the head upright"],
            ["Hold the head level with both hands (ball on the floor).",
             "Run the runner test: all wheels +10 rad/s for 5 s -> the head rotates about the vertical axis.",
             "Compare the yaw rate (sign AND magnitude)."],
            [Field("head_yaw_rate_rads", "Head yaw rate", "rad/s", 0.12)],
            pred_spin, runner_test="spin", safety="Keep fingers away from the omniwheels.",
            why="Validates zenith angle, wheel radius, motor order and direction signs - exactly the Jacobian."),
    Example("straight_line", "Drive straight (head held)", "Quick check", 3,
            ["robot on the floor, someone holding the head upright", "tape measure, floor tape at the start"],
            ["Point the eye along the floor tape.", "Run the runner test (0.2 m/s for 3 s, head held level).",
             "Measure distance and direction of travel relative to the eye direction."],
            [Field("distance_m", "Distance travelled", "m", 0.15), Field("heading_deg", "Direction vs. eye", "deg", 15, rel=False)],
            pred_straight, runner_test="straight",
            why="If the robot moves sideways or backwards, the policy's 'forward' differs from the hardware's."),
    Example("fall_time", "Passive fall time (motors holding)", "Measure", 5,
            ["the complete robot", "a helper to catch it", "soft mat"],
            ["Motors powered and holding (zero speed).",
             "Balance the robot by hand, tilt ~3 deg, release, and catch it after ~30 deg.",
             "The runner test records the IMU; or film at 120 fps and count frames from 5 to 25 deg.",
             "Compare: a big mismatch points at wrong masses / inertia / wheel holding torque (the fall time is only "
             "weakly sensitive to the CoG height, so use example 1 for the CoG)."],
            [Field("time_5_to_25deg_s", "Time from 5 to 25 deg", "s", 0.12)],
            pred_fall_time, runner_test="fall-time",
            safety="Always catch the robot; never let a 5+ kg robot fall freely.",
            why="Checks the passive (open-loop) dynamics: masses, inertias and that the steppers really hold."),
    Example("first_balance", "First balance with the classical controller", "Closed loop", 5,
            ["complete robot", "tether rope from the ceiling / a frame", "kill switch on the motor supply"],
            ["Run: jetson_control_loop.py --controller classical --seconds 30 (with tether).",
             "Read max tilt and odometry drift from the summary / log.",
             "Only after this works: try the RL policy."],
            [Field("balanced_s", "Balanced for", "s", 0.1), Field("max_tilt_deg", "Max tilt", "deg", 1.0),
             Field("drift_after_30s_m", "Drift after 30 s", "m", 1.0)],
            pred_first_balance, safety="Tether + kill switch. Stay clear of the head height.",
            why="A known-good baseline separates hardware problems from RL problems."),
    Example("push_recovery", "Push recovery (10 N, 0.2 s)", "Closed loop", 5,
            ["balancing robot (classical or RL)", "luggage / spring scale", "tether"],
            ["Hook the spring scale to the head, pull horizontally to ~10 N (1 kg on the scale) for ~0.2 s and release.",
             "Read the maximum tilt and the time until the tilt is back below 2 deg (from the log)."],
            [Field("max_tilt_deg", "Max tilt", "deg", 0.5), Field("recovery_time_s", "Recovery time", "s", 0.6)],
            pred_push, safety="Tether. Start with ~5 N.",
            why="Compares disturbance rejection of sim and robot - tells you if the randomisation ranges are wide enough."),
]
for _i, _e in enumerate(CATALOG):
    _e.title = f"{_i + 1}. {_e.title}"
BY_KEY = {e.key: e for e in CATALOG}


def compare(ex: Example, measured: Dict[str, float], predicted: Dict[str, float]) -> List[Dict[str, object]]:
    rows = []
    for f in ex.fields:
        m, p = measured.get(f.key), predicted.get(f.key)
        if m is None or p is None or (isinstance(p, float) and math.isnan(p)):
            rows.append({"field": f.label, "measured": m, "predicted": p, "error": None, "ok": None})
            continue
        if isinstance(m, bool) or isinstance(p, bool):
            ok = bool(m) == bool(p)
            rows.append({"field": f.label, "measured": m, "predicted": p, "error": None, "ok": ok})
            continue
        err = (m - p) / p if (f.rel and abs(p) > 1e-9) else (m - p)
        ok = abs(err) <= f.tolerance
        rows.append({"field": f.label, "unit": f.unit, "measured": m, "predicted": p, "error": err, "ok": ok,
                     "rel": f.rel})
    return rows


def import_runner_json(path: str) -> Tuple[str, Dict[str, float]]:
    """Map a jetson_control_loop.py --test result onto catalog fields."""
    with open(path) as f:
        d = json.load(f)
    test = d.get("test", "")
    meas = {k: v for k, v in d.get("measured", {}).items() if isinstance(v, (int, float, bool)) and v is not None}
    key = {"encoder-turns": "encoder_turns", "imu-noise": "imu_noise", "ratio": "transmission_ratio",
           "motor-step": "motor_step", "spin": "spin_in_place", "straight": "straight_line",
           "fall-time": "fall_time", "signs": "signs"}.get(test, test)
    if key == "straight_line":
        meas = {"distance_m": meas.get("odometry_distance_m"), "heading_deg": meas.get("odometry_heading_deg")}
    return key, {k: v for k, v in meas.items() if v is not None}


def checklist_markdown(cfg: Optional[AppConfig] = None, with_predictions: bool = False) -> str:
    cfg = cfg or default_config()
    out = ["# BB-8 real-world verification checklist", "",
           "Quick experiments that compare the real robot with the simulator. Do them in order: each one "
           "assumes the previous ones passed. Runner commands are run on the Jetson in the deployment package.", ""]
    for ex in CATALOG:
        out += [f"## {ex.title}", f"*{ex.category} - about {ex.minutes} min.* {ex.why}", "",
                "**Equipment:** " + "; ".join(ex.equipment), ""]
        out += [f"{i + 1}. {s}" for i, s in enumerate(ex.steps)]
        if ex.runner_test:
            out += ["", f"Runner: `python3 jetson_control_loop.py --test {ex.runner_test}`"]
        if ex.safety:
            out += ["", f"> Safety: {ex.safety}"]
        pred = ex.predict(cfg) if with_predictions else {}
        out += ["", "| Measure | Unit | Simulator prediction | Your value |", "|---|---|---|---|"]
        for f in ex.fields:
            p = pred.get(f.key)
            out.append(f"| {f.label} | {f.unit} | {'' if p is None else f'{p:.4g}'} | |")
        if ex.apply:
            out += ["", f"In the app: *{ex.apply_label}*."]
        out.append("")
    return "\n".join(out)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] == "list":
        for e in CATALOG:
            print(f"{e.key:20s} {e.category:12s} {e.minutes:3d} min  {e.title}")
        return 0
    if argv[0] == "predict":
        e = BY_KEY[argv[1]]
        for k, v in e.predict(default_config()).items():
            print(f"{k:28s} {v:.5g}")
        return 0
    if argv[0] == "sheet":
        print(checklist_markdown(default_config(), with_predictions="--predict" in argv))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
