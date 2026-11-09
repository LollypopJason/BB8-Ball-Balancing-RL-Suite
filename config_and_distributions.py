"""
config_and_distributions.py
===========================

Foundation layer of the BB-8 RL Suite & Deployment Studio.

Contents
--------
* ``DistributionParameter``  - Constant / Uniform / Normal sampler used by every
  randomisable quantity (re-sampled on every episode reset).
* Hierarchical dataclass configuration (robot geometry, domain randomisation,
  terrain, disturbances, episode/waypoints, network, training, resources, viewer).
* Observation / action channel specification shared by the Gym env, the ONNX/C++
  exporters and the Jetson deployment code (single source of truth for ordering,
  physical units and normalisation scales).
* JSON (de)serialisation and timestamped experiment-bundle ZIP packaging.
* Dark / Light Qt stylesheets and matching PyQtGraph palettes.

This module has **no** dependency on MuJoCo, Qt or PyTorch so it can be imported
by headless training workers and unit tests.
"""
from __future__ import annotations

import copy
import dataclasses
import datetime as _dt
import json
import math
import os
import typing
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

APP_NAME = "BB-8 RL Suite & Deployment Studio"
APP_VERSION = "1.0.0"
CONFIG_SCHEMA_VERSION = 1

# ----------------------------------------------------------------------------
# Hardware ground-truth constants (datasheet values, used as defaults)
# ----------------------------------------------------------------------------
STEPPER_MODEL = "JK42HS40-1704-83AF"
STEPPER_HOLDING_TORQUE_NM = 0.41
STEPPER_RATED_CURRENT_A = 1.7
STEPPER_FULL_STEPS_PER_REV = 200
STEPPER_ROTOR_INERTIA_KGM2 = 54e-7  # 54 g*cm^2 (typical 40 mm NEMA17)
DEFAULT_WHEEL_DIAMETER_M = 0.058
DEFAULT_ZENITH_DEG = 31.0
DEFAULT_WHEEL_AZIMUTHS_DEG = (0.0, 120.0, 240.0)
GRAVITY = 9.81


# ============================================================================
# 1. Statistical distribution engine
# ============================================================================
DIST_MODES = ("constant", "uniform", "normal")


@dataclass
class DistributionParameter:
    """A scalar that is either constant or drawn from U(low, high) / N(mean, std).

    Every randomised quantity in the suite is one of these. ``sample`` is called
    on *every* episode reset (manual or automatic after a fall).

    ``nominal()`` returns the deterministic "ideal physics" value (constant value,
    uniform mid-point or normal mean) used when the *Off (Ideal Nominal Physics)*
    preset is active.
    """

    mode: str = "constant"
    value: float = 0.0
    low: float = 0.0
    high: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    clamp: bool = True
    clamp_low: float = -1e9
    clamp_high: float = 1e9
    units: str = ""

    # ---- constructors -----------------------------------------------------
    @classmethod
    def const(cls, c: float, units: str = "", lo: float = -1e9, hi: float = 1e9):
        return cls(mode="constant", value=c, low=c, high=c, mean=c, std=0.0,
                   clamp_low=lo, clamp_high=hi, units=units)

    @classmethod
    def uniform(cls, a: float, b: float, units: str = "", lo: float = -1e9, hi: float = 1e9):
        a, b = min(a, b), max(a, b)
        return cls(mode="uniform", value=0.5 * (a + b), low=a, high=b,
                   mean=0.5 * (a + b), std=(b - a) / math.sqrt(12.0),
                   clamp_low=lo, clamp_high=hi, units=units)

    @classmethod
    def normal(cls, m: float, s: float, units: str = "", lo: float = -1e9, hi: float = 1e9):
        return cls(mode="normal", value=m, low=m - 2 * s, high=m + 2 * s, mean=m,
                   std=abs(s), clamp_low=lo, clamp_high=hi, units=units)

    # ---- sampling ---------------------------------------------------------
    def sample(self, rng: np.random.Generator) -> float:
        if self.mode == "uniform":
            lo, hi = min(self.low, self.high), max(self.low, self.high)
            x = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
        elif self.mode == "normal":
            x = float(rng.normal(self.mean, max(self.std, 0.0)))
        else:
            x = float(self.value)
        return self._clip(x)

    def nominal(self) -> float:
        if self.mode == "uniform":
            return self._clip(0.5 * (self.low + self.high))
        if self.mode == "normal":
            return self._clip(self.mean)
        return self._clip(self.value)

    def draw(self, rng: np.random.Generator, ideal: bool = False) -> float:
        """Sample, or return the nominal value when ``ideal`` physics is requested."""
        return self.nominal() if ideal else self.sample(rng)

    def is_random(self) -> bool:
        if self.mode == "uniform":
            return self.high > self.low
        if self.mode == "normal":
            return self.std > 0.0
        return False

    def _clip(self, x: float) -> float:
        if self.clamp:
            return float(min(max(x, self.clamp_low), self.clamp_high))
        return float(x)

    def describe(self) -> str:
        u = f" {self.units}" if self.units else ""
        if self.mode == "uniform":
            return f"U({self.low:g}, {self.high:g}){u}"
        if self.mode == "normal":
            return f"N({self.mean:g}, {self.std:g}){u}"
        return f"{self.value:g}{u}"


DP = DistributionParameter  # short alias used throughout the configs


def _c(v, units="", lo=-1e9, hi=1e9):
    return field(default_factory=lambda: DP.const(v, units, lo, hi))


def _u(a, b, units="", lo=-1e9, hi=1e9):
    return field(default_factory=lambda: DP.uniform(a, b, units, lo, hi))


def _n(m, s, units="", lo=-1e9, hi=1e9):
    return field(default_factory=lambda: DP.normal(m, s, units, lo, hi))


# ============================================================================
# 2. Configuration dataclasses
# ============================================================================
@dataclass
class RobotGeometryConfig:
    """Nominal (structural) robot geometry and actuator datasheet values."""

    ball_radius_m: float = 0.25             # 50 cm diameter ball
    ball_mesh_subdivisions: int = 3          # icosphere level (3 -> 642 verts)
    ball_inertia_factor: float = 0.62        # I = k*m*R^2 (0.4 solid, 0.667 thin shell)
    zenith_angle_deg: float = DEFAULT_ZENITH_DEG   # contact angle from top pole (31 -> 40)
    wheel_azimuths_deg: List[float] = field(default_factory=lambda: list(DEFAULT_WHEEL_AZIMUTHS_DEG))
    wheel_diameter_m: float = DEFAULT_WHEEL_DIAMETER_M
    wheel_width_m: float = 0.024
    rollers_per_row: int = 6                 # multi-roller approximation (2 staggered rows)
    roller_radius_frac: float = 0.8          # roller sphere radius / wheel radius (0.8 -> 0.85 % envelope ripple)
    omniwheel_model: str = "virtual"         # "virtual" (Jacobian transmission) | "roller" (contact rollers)
    head_mass_kg: float = 2.2
    head_radius_m: float = 0.15              # head dome radius (30 cm head, BB-8 proportions)
    head_cog_height_m: float = 0.085         # CoG height above the top of the ball
    imu_height_m: float = 0.06               # IMU site height above top of the ball
    stepper_holding_torque_nm: float = STEPPER_HOLDING_TORQUE_NM
    stepper_rotor_inertia: float = STEPPER_ROTOR_INERTIA_KGM2
    stepper_full_steps_per_rev: int = STEPPER_FULL_STEPS_PER_REV
    tmc2208_microsteps: int = 16
    stepper_kv: float = 0.6                  # velocity-servo stiffness [N*m/(rad/s)] emulating a stiff stepper
    stepper_corner_speed_rads: float = 15.0  # pull-out torque starts dropping above this speed
    stepper_max_speed_rads: float = 45.0     # command saturation (~430 rpm)
    stepper_torque_at_max_frac: float = 0.35 # remaining torque fraction at max speed
    physics_timestep_s: float = 0.002        # 500 Hz physics ("virtual" omniwheel model)
    roller_timestep_s: float = 0.001         # 1 kHz physics for the "roller" model / validation rig
    roller_contact_timeconst_s: float = 0.002  # roller-ball contact solref time constant (>= 2 x roller_timestep_s)
    control_hz: float = 100.0                # Jetson control loop rate

    @property
    def wheel_radius_m(self) -> float:
        return 0.5 * self.wheel_diameter_m

    @property
    def sim_timestep_s(self) -> float:
        """MuJoCo timestep actually simulated. The roller model's head rests on stiff, discrete
        roller contacts whose time constant must be resolved by several steps, so it runs finer."""
        return self.roller_timestep_s if self.omniwheel_model == "roller" else self.physics_timestep_s

    @property
    def frame_skip(self) -> int:
        """Physics sub-steps per control step."""
        return max(1, int(round(1.0 / (self.control_hz * self.sim_timestep_s))))

    @property
    def control_dt_s(self) -> float:
        return self.frame_skip * self.sim_timestep_s

    @property
    def steps_per_rad(self) -> float:
        return self.stepper_full_steps_per_rev * self.tmc2208_microsteps / (2.0 * math.pi)


@dataclass
class BallRandomization:
    mass_kg: DP = _u(2.0, 3.0, "kg", 0.05, 20)     # 50 cm hollow styrofoam + epoxy/sawdust shell (estimate - weigh yours)
    sliding_grip: DP = _u(0.7, 1.1, "mu", 0.05, 3)
    torsional_friction: DP = _u(0.002, 0.01, "m", 0.0, 0.1)
    rolling_resistance: DP = _u(0.0005, 0.004, "m", 0.0, 0.05)
    non_sphericity: DP = _c(0.0, "frac", 0.0, 0.1)    # ellipsoidal stretch fraction of R
    dent_depth_m: DP = _c(0.0, "m", 0.0, 0.02)        # max radial dent/bump depth
    dent_count: int = 12
    dent_width_rad: float = 0.25
    com_offset_m: DP = _c(0.0, "m", 0.0, 0.03)        # magnitude, random direction


@dataclass
class WheelRandomization:
    longitudinal_grip: DP = _u(0.7, 1.1, "mu", 0.05, 3)
    lateral_roller_resistance: DP = _u(0.0005, 0.003, "Nms/rad", 0.0, 0.1)
    rolling_resistance: DP = _u(0.0, 0.002, "Nms/rad", 0.0, 0.1)
    diameter_offset_frac: List[DP] = field(default_factory=lambda: [DP.const(0.0, "frac", -0.2, 0.2) for _ in range(3)])
    grip_offset_frac: List[DP] = field(default_factory=lambda: [DP.const(0.0, "frac", -0.8, 0.8) for _ in range(3)])


@dataclass
class HeadRandomization:
    mass_kg: DP = _u(2.0, 2.4, "kg", 0.2, 20)
    cog_dx_m: DP = _n(0.0, 0.004, "m", -0.05, 0.05)
    cog_dy_m: DP = _n(0.0, 0.004, "m", -0.05, 0.05)
    cog_dz_m: DP = _n(0.0, 0.006, "m", -0.08, 0.08)


@dataclass
class ActuatorRandomization:
    voltage_torque_scale: DP = _u(0.85, 1.0, "x", 0.1, 1.5)   # supply sag -> torque scale
    deadband_rads: DP = _u(0.0, 0.08, "rad/s", 0.0, 2.0)       # stepper low-speed deadband
    latency_steps: DP = _c(1.0, "ctrl steps", 0, 10)           # command latency (rounded)
    kv_scale: DP = _u(0.8, 1.2, "x", 0.1, 5)                   # servo stiffness variation


@dataclass
class IMURandomization:
    orient_noise_std_rad: DP = _u(0.002, 0.006, "rad", 0, 0.2)
    orient_drift_std_rad: DP = _u(0.0, 0.0005, "rad/sqrt(s)", 0, 0.05)
    yaw_drift_std_rad: DP = _u(0.0, 0.003, "rad/sqrt(s)", 0, 0.2)
    gyro_noise_std_rads: DP = _u(0.005, 0.02, "rad/s", 0, 1)
    gyro_drift_std_rads: DP = _u(0.0, 0.002, "rad/s/sqrt(s)", 0, 0.2)
    accel_noise_std_ms2: DP = _u(0.02, 0.12, "m/s^2", 0, 3)
    accel_drift_std_ms2: DP = _u(0.0, 0.01, "m/s^2/sqrt(s)", 0, 1)
    encoder_noise_std_rads: DP = _u(0.0, 0.05, "rad/s", 0, 2)


@dataclass
class DomainRandomizationConfig:
    ball: BallRandomization = field(default_factory=BallRandomization)
    wheels: WheelRandomization = field(default_factory=WheelRandomization)
    head: HeadRandomization = field(default_factory=HeadRandomization)
    actuators: ActuatorRandomization = field(default_factory=ActuatorRandomization)
    imu: IMURandomization = field(default_factory=IMURandomization)


@dataclass
class TerrainConfig:
    map_name: str = "flat"                  # "flat" | "obstacle_arena"
    off_ideal_nominal: bool = False         # [ ] Off (Ideal Nominal Physics)
    variable_grip: bool = False             # [ ] Variable Ground Texture / Grip
    rough_heightfield: bool = False         # [ ] Procedural Rough Ground (Heightfield)
    walls: bool = False                     # [ ] Walls / Arena Boundaries
    ramps: bool = False                     # [ ] Minor Ramps & Slopes
    bumps_stairs: bool = False              # [ ] Speed Bumps & Stairs
    arena_half_size_m: float = 3.0
    grip_tiles: int = 8
    tile_grip_factor: DP = _u(0.6, 1.25, "x", 0.05, 3)
    rough_amplitude_m: float = 0.012
    rough_resolution: int = 64
    ramp_slope_deg: float = 6.0
    bump_height_m: float = 0.012
    stair_step_m: float = 0.012
    layout_seed: int = 7


@dataclass
class DisturbanceConfig:
    enabled: bool = False
    target_body: str = "head"               # "head" | "ball"
    interval_s: DP = _u(2.0, 5.0, "s", 0.05, 600)
    magnitude_n: DP = _u(2.0, 8.0, "N", 0.0, 500)
    azimuth_deg: DP = _u(0.0, 360.0, "deg")
    elevation_deg: DP = _c(0.0, "deg", -90, 90)
    duration_s: DP = _c(0.10, "s", 0.002, 5)


@dataclass
class RewardWeights:
    alive: float = 1.0
    upright: float = 1.0
    ang_vel: float = 0.02
    action: float = 0.01
    action_rate: float = 0.05
    drift: float = 0.5
    ball_velocity: float = 1.0              # penalise |v_ball|^2 (prevents run-away drift to wheel-speed saturation)
    target_distance: float = 0.5
    target_progress: float = 5.0
    target_reached_bonus: float = 2.0
    fall_penalty: float = 50.0


@dataclass
class EpisodeConfig:
    auto_reset_on_fall: bool = True
    fall_tilt_deg: float = 30.0
    detect_contact_loss: bool = True
    max_episode_s: float = 20.0
    spawn_x_m: float = 0.0                  # dedicated centre start point (0, 0)
    spawn_y_m: float = 0.0
    init_tilt_deg: DP = _u(0.0, 3.0, "deg", 0, 20)
    init_yaw_deg: DP = _c(0.0, "deg")
    target_enabled: bool = False
    target_x_m: DP = _c(1.0, "m", -20, 20)
    target_y_m: DP = _c(0.0, "m", -20, 20)
    target_reach_radius_m: float = 0.10
    target_obs_clip_m: float = 2.0
    reward: RewardWeights = field(default_factory=RewardWeights)


@dataclass
class NetworkConfig:
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"                # fixed by design (export/C++ use tanh)
    init: str = "orthogonal"                # "orthogonal" | "xavier"
    obs_mask: Dict[str, bool] = field(default_factory=lambda: {g.key: g.default_on for g in OBS_GROUPS})
    action_mask: List[bool] = field(default_factory=lambda: [True, True, True])
    action_mode: str = "velocity"           # "velocity" (step-rate) | "torque"


@dataclass
class TrainingConfig:
    algorithm: str = "PPO"                  # "PPO" | "SAC"
    n_envs: int = 8
    vec_env: str = "subproc"                # "subproc" | "dummy"
    total_timesteps: int = 3_000_000
    learning_rate: float = 3e-4
    n_steps: int = 1024                     # PPO rollout per env
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    clip_range: float = 0.2
    log_std_init: float = -1.0
    sac_buffer_size: int = 500_000
    autosave_every_steps: int = 100_000
    autosave_every_minutes: float = 5.0
    best_window_episodes: int = 20
    seed: int = 0
    device: str = "cpu"


@dataclass
class ResourceConfig:
    cpu_percent: int = 75                   # 25 / 50 / 75 / 100
    ram_ceiling_percent: float = 85.0       # system RAM usage that triggers checkpoint
    pause_on_ram_ceiling: bool = False


@dataclass
class ViewerConfig:
    mode: str = "single"                    # "single" | "grid" | "headless"
    grid_count: int = 8
    theme: str = "dark"
    width: int = 960
    height: int = 640
    record_seconds: float = 60.0


@dataclass
class CustomModelBinding:
    path: str = ""
    import_mode: str = "standalone"         # "standalone" | "head_shell" (STL/OBJ on parametric robot)
    mesh_scale: float = 1.0                 # 0.001 for CAD meshes in millimetres
    imu_body: str = ""
    ball_body: str = ""
    wheel_joints: List[str] = field(default_factory=lambda: ["", "", ""])


@dataclass
class AppConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    model_type: str = "bb8"                 # "bb8" | "validation_rig" | "custom"
    robot: RobotGeometryConfig = field(default_factory=RobotGeometryConfig)
    randomization: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    custom: CustomModelBinding = field(default_factory=CustomModelBinding)
    workspace_dir: str = "workspace"

    # convenience ----------------------------------------------------------
    def copy(self) -> "AppConfig":
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AppConfig":
        return _from_dict(cls, d)

    @classmethod
    def from_json(cls, s: str) -> "AppConfig":
        return cls.from_dict(json.loads(s))

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return path

    @classmethod
    def load(cls, path: str) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())


def _from_dict(cls, d):
    """Recursive, forgiving dataclass reconstruction (unknown keys ignored,
    missing keys keep their defaults -> old session files stay loadable)."""
    if not dataclasses.is_dataclass(cls) or not isinstance(d, dict):
        return d
    hints = typing.get_type_hints(cls)
    obj = cls()
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]
        tp = hints.get(f.name, f.type)
        origin = typing.get_origin(tp)
        args = typing.get_args(tp)
        if dataclasses.is_dataclass(tp):
            val = _from_dict(tp, val)
        elif origin in (list, List) and args and dataclasses.is_dataclass(args[0]):
            val = [_from_dict(args[0], v) for v in val]
        elif origin in (dict, Dict) and isinstance(val, dict):
            merged = dict(getattr(obj, f.name))
            merged.update(val)
            val = merged
        setattr(obj, f.name, val)
    return obj


def default_config() -> AppConfig:
    return AppConfig()


# ============================================================================
# 3. Observation / action channel specification (single source of truth)
# ============================================================================
@dataclass(frozen=True)
class ObsGroup:
    key: str
    label: str
    channels: tuple            # channel names (also C++ argument names)
    scale: tuple               # raw -> normalised multiplier
    units: str
    default_on: bool = True


OBS_GROUPS: List[ObsGroup] = [
    ObsGroup("imu_rpy", "IMU Roll / Pitch / Yaw", ("imu_roll_rad", "imu_pitch_rad", "imu_yaw_rad"),
             (1 / 0.3, 1 / 0.3, 1 / math.pi), "rad"),
    ObsGroup("imu_gyro", "IMU Angular Velocities", ("imu_gx_rads", "imu_gy_rads", "imu_gz_rads"),
             (1 / 2.0, 1 / 2.0, 1 / 2.0), "rad/s"),
    ObsGroup("imu_accel", "IMU Linear Accelerations", ("imu_ax_ms2", "imu_ay_ms2", "imu_az_ms2"),
             (1 / GRAVITY, 1 / GRAVITY, 1 / GRAVITY), "m/s^2"),
    ObsGroup("enc_vel", "3x Magnetic Encoder Velocities", ("enc1_vel_rads", "enc2_vel_rads", "enc3_vel_rads"),
             (1 / 20.0, 1 / 20.0, 1 / 20.0), "rad/s"),
    ObsGroup("enc_pos", "3x Encoder Positions (sin/cos-free, wrapped)", ("enc1_pos_rad", "enc2_pos_rad", "enc3_pos_rad"),
             (1 / math.pi, 1 / math.pi, 1 / math.pi), "rad", False),
    ObsGroup("target_err", "Target (dx, dy) Error [heading frame]", ("target_dx_m", "target_dy_m"),
             (1 / 1.0, 1 / 1.0), "m"),
    ObsGroup("prev_cmd", "Previous Motor Commands", ("prev_motor1_cmd", "prev_motor2_cmd", "prev_motor3_cmd"),
             (1.0, 1.0, 1.0), "norm [-1,1]"),
]
OBS_GROUP_BY_KEY = {g.key: g for g in OBS_GROUPS}
ALL_OBS_CHANNELS: List[str] = [c for g in OBS_GROUPS for c in g.channels]
ALL_OBS_SCALES: List[float] = [s for g in OBS_GROUPS for s in g.scale]

ACTION_CHANNELS = ("motor1", "motor2", "motor3")


def active_obs_indices(obs_mask: Dict[str, bool]) -> List[int]:
    """Indices into the *full* observation vector that are fed to the network."""
    idx, k = [], 0
    for g in OBS_GROUPS:
        n = len(g.channels)
        if obs_mask.get(g.key, g.default_on):
            idx.extend(range(k, k + n))
        k += n
    return idx


def active_obs_names(obs_mask: Dict[str, bool]) -> List[str]:
    return [ALL_OBS_CHANNELS[i] for i in active_obs_indices(obs_mask)]


def active_action_indices(action_mask: Sequence[bool]) -> List[int]:
    return [i for i, on in enumerate(action_mask) if on]


# ============================================================================
# 4. Presets & helpers
# ============================================================================
def iter_distribution_params(cfg: AppConfig):
    """Yield (dotted_path, DistributionParameter) for every DP in the config."""
    def walk(obj, prefix):
        if isinstance(obj, DistributionParameter):
            yield prefix, obj
        elif dataclasses.is_dataclass(obj):
            for f in dataclasses.fields(obj):
                yield from walk(getattr(obj, f.name), f"{prefix}.{f.name}" if prefix else f.name)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                yield from walk(v, f"{prefix}[{i}]")
    yield from walk(cfg, "")


def apply_ideal_preset(cfg: AppConfig) -> AppConfig:
    """Return a copy where every DistributionParameter is collapsed to its nominal."""
    out = cfg.copy()
    for _, dp in iter_distribution_params(out):
        v = dp.nominal()
        dp.mode, dp.value = "constant", v
    out.terrain.off_ideal_nominal = True
    return out


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_workspace(cfg: AppConfig) -> str:
    ws = os.path.abspath(cfg.workspace_dir)
    for sub in ("", "checkpoints", "sessions", "exports", "telemetry"):
        os.makedirs(os.path.join(ws, sub), exist_ok=True)
    return ws


def create_experiment_bundle(out_dir: str, cfg: AppConfig, files: Dict[str, Optional[str]],
                             extra_text: Optional[Dict[str, str]] = None) -> str:
    """Package an experiment into ``bb8_experiment_<timestamp>.zip``.

    ``files`` maps archive-name -> local path (missing/None paths are skipped and
    listed in MANIFEST.txt so the bundle documents what was unavailable).
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = timestamp()
    zpath = os.path.join(out_dir, f"bb8_experiment_{stamp}.zip")
    manifest = [f"{APP_NAME} v{APP_VERSION}", f"Created: {stamp}", ""]
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("config_snapshot.json", cfg.to_json())
        manifest.append("config_snapshot.json  (all domain-randomisation & hyper-parameters)")
        for arc, p in files.items():
            if p and os.path.isfile(p):
                z.write(p, arc)
                manifest.append(f"{arc}  <- {p}")
            else:
                manifest.append(f"{arc}  (not available at export time)")
        for arc, text in (extra_text or {}).items():
            z.writestr(arc, text)
            manifest.append(arc)
        z.writestr("MANIFEST.txt", "\n".join(manifest) + "\n")
    return zpath


# ============================================================================
# 5. Themes (Qt stylesheets + PyQtGraph palette)
# ============================================================================
THEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        "bg": "#14171c", "panel": "#1c2028", "panel2": "#232833", "border": "#323946",
        "text": "#d9dee7", "muted": "#8b95a7", "accent": "#ff8a1f", "accent2": "#2fa4ff",
        "danger": "#e5484d", "ok": "#3ccf7e", "plot_bg": "#101318", "plot_fg": "#aab3c2",
    },
    "light": {
        "bg": "#f3f4f6", "panel": "#ffffff", "panel2": "#eceef2", "border": "#c9ced8",
        "text": "#1d2330", "muted": "#5d6778", "accent": "#d96800", "accent2": "#0b6bcb",
        "danger": "#c62828", "ok": "#1b8a4b", "plot_bg": "#ffffff", "plot_fg": "#333a46",
    },
}
PLOT_PENS = ["#ff8a1f", "#2fa4ff", "#3ccf7e", "#e5484d", "#b07cff", "#f5d142", "#38d9d9", "#ff5fa2"]


def make_stylesheet(theme: str = "dark") -> str:
    t = THEMES.get(theme, THEMES["dark"])
    return f"""
    QWidget {{ background-color: {t['bg']}; color: {t['text']}; font-size: 12px; }}
    QMainWindow::separator {{ background: {t['border']}; width: 3px; height: 3px; }}
    QDockWidget {{ titlebar-close-icon: none; }}
    QDockWidget::title {{ background: {t['panel2']}; padding: 4px; font-weight: 600; }}
    QToolBar {{ background: {t['panel']}; border-bottom: 1px solid {t['border']}; spacing: 6px; padding: 3px; }}
    QToolButton {{ padding: 3px 8px; border-radius: 4px; }}
    QToolButton:checked {{ background: {t['accent2']}; color: white; font-weight: 600; }}
    QToolButton:hover {{ border: 1px solid {t['accent2']}; }}
    QStatusBar {{ background: {t['panel']}; color: {t['muted']}; }}
    QGroupBox {{ border: 1px solid {t['border']}; border-radius: 6px; margin-top: 10px; padding: 6px;
                 background: {t['panel']}; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {t['accent']};
                        font-weight: 600; }}
    QPushButton {{ background: {t['panel2']}; border: 1px solid {t['border']}; border-radius: 5px;
                   padding: 4px 10px; }}
    QPushButton:hover {{ border-color: {t['accent2']}; }}
    QPushButton:pressed, QPushButton:checked {{ background: {t['accent2']}; color: white; }}
    QPushButton#danger {{ background: {t['danger']}; color: white; font-weight: 700; border: none; }}
    QPushButton#accent {{ background: {t['accent']}; color: black; font-weight: 700; border: none; }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
        background: {t['panel2']}; border: 1px solid {t['border']}; border-radius: 4px; padding: 2px 4px;
        selection-background-color: {t['accent2']}; }}
    QComboBox QAbstractItemView {{ background: {t['panel2']}; selection-background-color: {t['accent2']}; }}
    QTabWidget::pane {{ border: 1px solid {t['border']}; }}
    QTabBar::tab {{ background: {t['panel2']}; padding: 5px 9px; border: 1px solid {t['border']};
                    border-bottom: none; }}
    QTabBar::tab:selected {{ background: {t['panel']}; color: {t['accent']}; }}
    QTreeWidget, QTableWidget, QListWidget {{ background: {t['panel']}; border: 1px solid {t['border']};
                    alternate-background-color: {t['panel2']}; }}
    QHeaderView::section {{ background: {t['panel2']}; border: none; padding: 3px; }}
    QSlider::groove:horizontal {{ height: 6px; background: {t['panel2']}; border-radius: 3px; }}
    QSlider::handle:horizontal {{ background: {t['accent']}; width: 14px; margin: -5px 0; border-radius: 7px; }}
    QProgressBar {{ border: 1px solid {t['border']}; border-radius: 4px; text-align: center; }}
    QProgressBar::chunk {{ background: {t['accent2']}; }}
    QLabel#muted {{ color: {t['muted']}; }}
    QScrollArea {{ border: none; }}
    QCheckBox::indicator {{ width: 14px; height: 14px; }}
    """


__all__ = [n for n in dir() if not n.startswith("_")]
