"""
bb8_gym_env.py
==============

Gymnasium MuJoCo environment for the head-balancing BB-8 ballbot.

Features
--------
* Domain randomisation of every ``DistributionParameter`` on **every reset**
  (ball mass / grip / torsional & rolling friction / non-sphericity / dents / CoM,
  per-wheel wear (diameter + grip), head mass + 3D CoG, supply-voltage torque sag,
  stepper deadband, command latency, servo stiffness, IMU noise & random-walk drift).
* Stepper model (JK42HS40-1704-83AF + TMC2208): velocity servo with speed-dependent
  pull-out torque, voltage-scaled holding torque, deadband, micro-step rate
  quantisation, missed-step accounting, traction-limited slip.
* Simulated Arduino Nano 33 IoT IMU (roll/pitch/yaw, gyro, accelerometer) with
  white noise + bias random walk (no magnetometer -> yaw drifts).
* Magnetic encoders (absolute single-turn position + velocity) per wheel.
* Intermittent 3-D push disturbances (``xfrc_applied``) with distribution-driven
  interval / magnitude / azimuth / elevation / duration.
* Fall detection (tilt threshold and head-ball contact loss), optional auto-reset.
* Live waypoint target (``set_target``) and spawn at arena centre (0, 0).
* ``get_render_state``/``set_render_state`` for the GUI's grid viewer & timeline scrubber.

Also contains ``ValidationRigSim`` - a non-RL driver for the single-motor test bench.
"""
from __future__ import annotations

import collections
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from config_and_distributions import (ALL_OBS_CHANNELS, ALL_OBS_SCALES, GRAVITY, AppConfig,
                                      active_action_indices, active_obs_indices, default_config)
from mjcf_generator import (StructuralSample, build_xml, generate_validation_rig_xml, omniwheel_frames)
from plugins import load_plugins


def quat_to_rpy(q: np.ndarray) -> Tuple[float, float, float]:
    """MuJoCo quaternion (w, x, y, z) -> ZYX roll, pitch, yaw."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class BB8BalanceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, config: Optional[AppConfig | dict] = None, render_mode: Optional[str] = None,
                 seed: Optional[int] = None, render_size: Tuple[int, int] = (640, 480),
                 training: bool = True):
        super().__init__()
        if isinstance(config, dict):
            config = AppConfig.from_dict(config)
        self.cfg: AppConfig = (config or default_config()).copy()
        if self.cfg.model_type == "validation_rig":
            raise ValueError("Use ValidationRigSim for the validation rig model")
        self.render_mode = render_mode
        self.render_size = render_size
        self.training = training
        self._rng = np.random.default_rng(seed)
        self.model_version = 0
        self.difficulty = 1.0     # curriculum: 0 = nominal physics / no pushes, 1 = full randomisation
        self.plugins = load_plugins(self.cfg)
        self._structural_key = None
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.xml: str = ""
        self._renderer = None
        self.hfield_dirty = False
        self.xml_override: Optional[str] = None      # set by the GUI XML editor / importer
        self.model_overrides: List[Tuple[str, str, str, Any]] = []  # live inspector edits

        rc = self.cfg.robot
        self.dt_ctrl = rc.frame_skip * rc.physics_timestep_s
        self.max_steps = int(round(self.cfg.episode.max_episode_s / self.dt_ctrl))
        self.obs_idx = np.array(active_obs_indices(self.cfg.network.obs_mask), dtype=np.int64)
        self.act_idx = active_action_indices(self.cfg.network.action_mask)
        self.obs_scale = np.array(ALL_OBS_SCALES, dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(len(self.obs_idx),), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(max(1, len(self.act_idx)),), dtype=np.float32)

        self._build(self._sample_structural(ideal=True))
        self.target = np.array([self.cfg.episode.target_x_m.nominal(), self.cfg.episode.target_y_m.nominal()])
        self.target_enabled = self.cfg.episode.target_enabled
        self.p: Dict[str, Any] = {}
        self._init_episode_state()
        self._sample_params(ideal=True)
        self._apply_params()

    # ------------------------------------------------------------------ build
    @property
    def ideal(self) -> bool:
        return bool(self.cfg.terrain.off_ideal_nominal)

    def _sample_structural(self, ideal: bool) -> StructuralSample:
        rc, br, wr = self.cfg.robot, self.cfg.randomization.ball, self.cfg.randomization.wheels
        radii = [rc.wheel_radius_m * (1.0 + self._draw(d, ideal)) for d in wr.diameter_offset_frac]
        return StructuralSample(ball_radius=rc.ball_radius_m,
                                non_sphericity=self._draw(br.non_sphericity, ideal),
                                dent_depth=self._draw(br.dent_depth_m, ideal),
                                wheel_radii=radii, mesh_seed=int(self._rng.integers(1 << 30)))

    def _build(self, s: StructuralSample) -> None:
        key = s.key(self.cfg.robot.omniwheel_model)
        if self.xml_override:
            key = ("override", hash(self.xml_override))
        if key == self._structural_key and self.model is not None:
            self.structural = s
            return
        self.xml = self.xml_override or build_xml(self.cfg, s)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self._structural_key = key
        self.structural = s
        self.model_version += 1
        self._cache_ids()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _cache_ids(self) -> None:
        m = self.model
        nid = lambda t, n: mujoco.mj_name2id(m, t, n)
        self.ball_bid = nid(mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.head_bid = nid(mujoco.mjtObj.mjOBJ_BODY, "head")
        self.ball_gid = nid(mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.target_mocap = m.body_mocapid[nid(mujoco.mjtObj.mjOBJ_BODY, "target")]
        self.roller_mode = self.cfg.robot.omniwheel_model == "roller"
        self.torque_mode = self.cfg.network.action_mode == "torque"
        self.ground_gids = [g for g in range(m.ngeom)
                            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("ground_")]
        self.roller_gids = set(g for g in range(m.ngeom)
                               if "_roller_" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""))
        self.wheel_jids = [nid(mujoco.mjtObj.mjOBJ_JOINT, f"wheel{i}_joint") for i in (1, 2, 3)]
        self.wheel_qadr = [m.jnt_qposadr[j] for j in self.wheel_jids]
        self.wheel_dadr = [m.jnt_dofadr[j] for j in self.wheel_jids]
        self._wqadr = np.array(self.wheel_qadr, dtype=np.int64)
        self._wdadr = np.array(self.wheel_dadr, dtype=np.int64)
        self.roller_dofs = [m.jnt_dofadr[j] for j in range(m.njnt)
                            if "_roller_" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "")]
        sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)
        self.s_quat = m.sensor_adr[sid("imu_quat")]
        self.s_gyro = m.sensor_adr[sid("imu_gyro")]
        self.s_acc = m.sensor_adr[sid("imu_accel")]
        self.ball_dadr = m.jnt_dofadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]
        if self.roller_mode:
            self.ball_jadr = m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]
            self.head_jadr = m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "head_free")]
            self.head_dadr = m.jnt_dofadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "head_free")]
        else:
            self.ball_jadr = m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "ball_free")]
            jb = nid(mujoco.mjtObj.mjOBJ_JOINT, "head_ball_joint")
            self.hb_qadr, self.hb_dadr = m.jnt_qposadr[jb], m.jnt_dofadr[jb]
        self.hfield_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_HFIELD, "rough_hf")
        self.ext_force = np.zeros((m.nbody, 6))
        self._ext_active = False
        self._ext_cleared = True
        self._manual_push_bid = None
        # nominal values that randomisation scales
        self.nom_head_ipos = m.body_ipos[self.head_bid].copy()
        self.nom_head_inertia = m.body_inertia[self.head_bid].copy()
        self.nom_head_mass = float(m.body_mass[self.head_bid])
        self.wheel_mass_total = self.cfg.robot.head_mass_kg - self.nom_head_mass
        self.nom_gear = m.actuator_gear[:, :3].copy()
        self.frames = omniwheel_frames(self.cfg.robot.ball_radius_m, self.cfg.robot.zenith_angle_deg,
                                       self.cfg.robot.wheel_azimuths_deg, self.structural.wheel_radii)
        self.nominal_dof_damping = m.dof_damping.copy()

    # --------------------------------------------------------- randomisation
    def _sample_params(self, ideal: bool) -> None:
        r, rng = self.cfg.randomization, self._rng
        d = lambda dp: self._draw(dp, ideal)
        p = {
            "ball_mass": d(r.ball.mass_kg), "ball_slide": d(r.ball.sliding_grip),
            "ball_torsion": d(r.ball.torsional_friction), "ball_roll": d(r.ball.rolling_resistance),
            "ball_com_off": d(r.ball.com_offset_m),
            "wheel_grip": d(r.wheels.longitudinal_grip),
            "roller_res": d(r.wheels.lateral_roller_resistance),
            "wheel_roll_res": d(r.wheels.rolling_resistance),
            "wheel_grip_off": [d(x) for x in r.wheels.grip_offset_frac],
            "head_mass": d(r.head.mass_kg),
            "cog": np.array([d(r.head.cog_dx_m), d(r.head.cog_dy_m), d(r.head.cog_dz_m)]),
            "v_scale": d(r.actuators.voltage_torque_scale), "deadband": d(r.actuators.deadband_rads),
            "latency": int(round(d(r.actuators.latency_steps))), "kv_scale": d(r.actuators.kv_scale),
            "imu": {k: d(getattr(r.imu, k)) for k in r.imu.__dataclass_fields__},
            "tile_grip": {},
        }
        com_dir = rng.normal(size=3)
        p["ball_com_vec"] = p["ball_com_off"] * com_dir / (np.linalg.norm(com_dir) + 1e-9)
        for g in self.ground_gids:
            p["tile_grip"][g] = (self._draw(self.cfg.terrain.tile_grip_factor, ideal)
                                 if (self.cfg.terrain.variable_grip or self.cfg.terrain.rough_heightfield) else 1.0)
        self.p = p

    def _apply_params(self) -> None:
        m, p, rc = self.model, self.p, self.cfg.robot
        R = self.structural.ball_radius
        # ball
        m.body_mass[self.ball_bid] = p["ball_mass"]
        I = rc.ball_inertia_factor * p["ball_mass"] * R ** 2
        m.body_inertia[self.ball_bid] = [I, I, I]
        m.body_ipos[self.ball_bid] = p["ball_com_vec"]
        # ground friction = ball surface properties x local ground grip (ground geoms have priority)
        for g in self.ground_gids:
            f = p["tile_grip"].get(g, 1.0)
            m.geom_friction[g] = [p["ball_slide"] * f, p["ball_torsion"], p["ball_roll"] * f]
        m.geom_friction[self.ball_gid] = [p["ball_slide"], p["ball_torsion"], p["ball_roll"]]
        # head
        hm = max(0.1, p["head_mass"] - self.wheel_mass_total)
        m.body_mass[self.head_bid] = hm
        m.body_inertia[self.head_bid] = self.nom_head_inertia * hm / max(self.nom_head_mass, 1e-6)
        m.body_ipos[self.head_bid] = self.nom_head_ipos + p["cog"]
        # wheels / actuators
        self.wheel_radii = np.array(self.structural.wheel_radii)
        self.grip_i = np.array([p["wheel_grip"] * (1.0 + o) for o in p["wheel_grip_off"]])
        th = math.radians(rc.zenith_angle_deg)
        self.normal_load = p["head_mass"] * GRAVITY / (3.0 * max(math.cos(th), 0.2))
        self.traction_torque = self.grip_i * self.normal_load * self.wheel_radii
        kv = rc.stepper_kv * p["kv_scale"]
        if not self.torque_mode:
            m.actuator_gainprm[:, 0] = kv
            m.actuator_biasprm[:, 2] = -kv
        if self.roller_mode:
            for g in self.roller_gids:
                wi = int(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)[1]) - 1
                m.geom_friction[g, 0] = self.grip_i[wi]
            for dof in self.roller_dofs:
                m.dof_damping[dof] = p["roller_res"]
            for dof in self.wheel_dadr:
                m.dof_damping[dof] = p["wheel_roll_res"]
        else:
            # per-wheel wear changes the transmission ratio R / r_i
            for i, f in enumerate(self.frames):
                m.actuator_gear[i, :3] = (R / self.wheel_radii[i]) * f.m
            # reflected rotor inertia + roller/rolling resistance on the ball-joint dofs
            J = np.stack([(R / self.wheel_radii[i]) * f.m for i, f in enumerate(self.frames)])
            arm = rc.stepper_rotor_inertia * np.diag(J.T @ J)
            lat = sum(np.outer(np.cross(f.n, f.m), np.cross(f.n, f.m)) for f in self.frames)  # roller directions
            damp = p["roller_res"] * (R / rc.wheel_radius_m) ** 2 * np.diag(lat) + p["wheel_roll_res"] * np.diag(J.T @ J)
            m.dof_armature[self.hb_dadr:self.hb_dadr + 3] = arm
            m.dof_damping[self.hb_dadr:self.hb_dadr + 3] = 2e-4 + damp
        self._apply_overrides()
        mujoco.mj_setConst(m, self.data)

    def _apply_overrides(self) -> None:
        """Re-apply live inspector edits (e.g. a body mass) after every randomisation."""
        m = self.model
        for objtype, name, fld, value in self.model_overrides:
            t = mujoco.mjtObj.mjOBJ_BODY if objtype == "body" else mujoco.mjtObj.mjOBJ_GEOM
            i = mujoco.mj_name2id(m, t, name)
            if i >= 0:
                getattr(m, f"{objtype}_{fld}")[i] = value

    def set_override(self, objtype: str, name: str, fld: str, value) -> None:
        self.model_overrides = [o for o in self.model_overrides if o[:3] != (objtype, name, fld)]
        self.model_overrides.append((objtype, name, fld, np.array(value, dtype=float)))
        self._apply_overrides()
        mujoco.mj_setConst(self.model, self.data)

    def load_xml(self, xml: Optional[str]) -> None:
        """Recompile from an edited/imported MJCF (None = back to procedural model)."""
        if xml:
            mujoco.MjModel.from_xml_string(xml)  # validate first; raises on error
        self.xml_override = xml
        self._structural_key = None
        self.model_overrides = []

    def _randomize_terrain(self) -> None:
        if self.hfield_id < 0:
            return
        m = self.model
        nr, nc = m.hfield_nrow[self.hfield_id], m.hfield_ncol[self.hfield_id]
        adr = m.hfield_adr[self.hfield_id]
        if self.ideal:
            z = np.full((nr, nc), 0.5)
        else:
            z = np.zeros((nr, nc))
            for octave, amp in ((4, 0.55), (8, 0.3), (16, 0.15)):
                coarse = self._rng.random((octave + 1, octave + 1))
                xi = np.linspace(0, octave, nc)
                yi = np.linspace(0, octave, nr)
                x0, y0 = np.floor(xi).astype(int).clip(0, octave - 1), np.floor(yi).astype(int).clip(0, octave - 1)
                fx, fy = xi - x0, yi - y0
                fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
                c00 = coarse[np.ix_(y0, x0)]
                c10 = coarse[np.ix_(y0, x0 + 1)]
                c01 = coarse[np.ix_(y0 + 1, x0)]
                c11 = coarse[np.ix_(y0 + 1, x0 + 1)]
                top = c00 * (1 - fx) + c10 * fx
                bot = c01 * (1 - fx) + c11 * fx
                z += amp * (top * (1 - fy[:, None]) + bot * fy[:, None])
            # flatten the spawn area so the robot starts level
            yy, xx = np.meshgrid(np.linspace(-1, 1, nr), np.linspace(-1, 1, nc), indexing="ij")
            rr = np.sqrt(xx ** 2 + yy ** 2) * self.cfg.terrain.arena_half_size_m
            w = np.clip((rr - 0.3) / 0.4, 0, 1)
            z = 0.5 + (z - z.mean()) * w
            z = np.clip(z, 0, 1)
        m.hfield_data[adr:adr + nr * nc] = z.ravel()
        self.hfield_dirty = True

    # ------------------------------------------------------------- episode
    def _init_episode_state(self) -> None:
        self.step_count = 0
        self.sim_time = 0.0
        self.prev_action = np.zeros(3)
        self.cmd_queue: collections.deque = collections.deque()
        self.enc_pos = np.zeros(3)
        self.enc_vel = np.zeros(3)
        self.slip = np.zeros(3, dtype=bool)
        self.missed_steps = np.zeros(3)
        self.motor_torque = np.zeros(3)
        self.motor_cmd = np.zeros(3)
        self.imu_bias = np.zeros(9)
        self.fallen = False
        self.contact_lost_t = 0.0
        self.episode_return = 0.0
        self.tilt_hist: List[float] = []
        self.push_next_t = math.inf
        self.push_end_t = -1.0
        self.push_force = np.zeros(3)
        self.prev_dist = None
        self.target_reached = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        ideal = self.ideal
        self._build(self._sample_structural(ideal))
        self._sample_params(ideal)
        mujoco.mj_resetData(self.model, self.data)
        self._apply_params()
        self._randomize_terrain()
        self._init_episode_state()
        ep = self.cfg.episode
        d, m = self.data, self.model
        # spawn: centre (0,0) by default with a small random initial tilt / yaw
        tilt = math.radians(ep.init_tilt_deg.draw(self._rng, ideal))
        az = self._rng.uniform(0, 2 * math.pi)
        yaw = math.radians(ep.init_yaw_deg.draw(self._rng, ideal))
        axis = np.array([math.cos(az), math.sin(az), 0.0])
        q_tilt, q_yaw, q = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(q_tilt, axis, tilt)
        mujoco.mju_axisAngle2Quat(q_yaw, np.array([0, 0, 1.0]), yaw)
        mujoco.mju_mulQuat(q, q_tilt, q_yaw)
        a = self.ball_jadr
        d.qpos[a:a + 2] = [ep.spawn_x_m, ep.spawn_y_m]
        if self.roller_mode:
            h = self.head_jadr
            d.qpos[h:h + 3] = d.qpos[a:a + 3]
            # rotate head about the ball centre
            d.qpos[h + 3:h + 7] = q
        else:
            d.qpos[self.hb_qadr:self.hb_qadr + 4] = q
        if "target" in options:
            self.target = np.asarray(options["target"], float)[:2]
        elif ep.target_enabled:
            self.target = np.array([ep.target_x_m.draw(self._rng, ideal), ep.target_y_m.draw(self._rng, ideal)])
        self.target_enabled = bool(options.get("target_enabled", ep.target_enabled))
        d.mocap_pos[self.target_mocap] = [self.target[0], self.target[1], 0.002 if self.target_enabled else -5.0]
        mujoco.mj_forward(m, d)
        if self.roller_mode:  # settle the head onto the rollers
            for _ in range(50):
                mujoco.mj_step(m, d)
            d.qvel[:] = 0
            d.time = 0.0
            mujoco.mj_forward(m, d)
        self._schedule_push(first=True)
        self.plugins.run_reset(self)
        self.prev_dist = self._target_dist()
        obs_full = self._observe()
        return self._mask_obs(obs_full), self._info(obs_full, 0.0)

    # ---------------------------------------------------------------- API
    def _draw(self, dp, ideal: bool) -> float:
        """Sample a DistributionParameter, shrunk toward its nominal value by the curriculum difficulty."""
        x = dp.draw(self._rng, ideal)
        if ideal or self.difficulty >= 1.0:
            return x
        nom = dp.nominal()
        return nom + max(0.0, self.difficulty) * (x - nom)

    def set_difficulty(self, d: float) -> None:
        """Curriculum hook (applied from the next reset; pushes scale immediately)."""
        self.difficulty = float(min(1.0, max(0.0, d)))

    def set_target(self, x: float, y: float, enabled: bool = True) -> None:
        """Change the waypoint live (mid-episode)."""
        self.target = np.array([float(x), float(y)])
        self.target_enabled = bool(enabled)
        self.data.mocap_pos[self.target_mocap] = [x, y, 0.002 if enabled else -5.0]
        self.prev_dist = self._target_dist()
        self.target_reached = False

    def update_config(self, cfg: AppConfig) -> None:
        """Apply a new configuration; takes effect (incl. recompilation) on next reset."""
        self.cfg = cfg.copy()
        self.plugins = load_plugins(self.cfg)
        self._structural_key = None
        self.dt_ctrl = self.cfg.robot.frame_skip * self.cfg.robot.physics_timestep_s
        self.max_steps = int(round(self.cfg.episode.max_episode_s / self.dt_ctrl))

    # ------------------------------------------------------------ helpers
    def ball_pos(self) -> np.ndarray:
        return self.data.xpos[self.ball_bid].copy()

    def head_tilt(self) -> float:
        zz = self.data.xmat[self.head_bid].reshape(3, 3)[2, 2]
        return math.acos(max(-1.0, min(1.0, zz)))

    def _target_dist(self) -> float:
        return float(np.linalg.norm(self.ball_pos()[:2] - self.target)) if self.target_enabled else 0.0

    def _schedule_push(self, first: bool = False) -> None:
        dc = self.cfg.disturbance
        if not dc.enabled:
            self.push_next_t = math.inf
            return
        self.push_next_t = self.data.time + max(0.05, dc.interval_s.sample(self._rng))

    def _update_push(self) -> None:
        dc, d = self.cfg.disturbance, self.data
        bid = self.head_bid if dc.target_body == "head" else self.ball_bid
        if self._manual_push_bid is not None:
            bid = self._manual_push_bid
        if d.time >= self.push_next_t:
            self._manual_push_bid = None
            bid = self.head_bid if dc.target_body == "head" else self.ball_bid
            mag = dc.magnitude_n.sample(self._rng) * self.difficulty
            az = math.radians(dc.azimuth_deg.sample(self._rng))
            el = math.radians(dc.elevation_deg.sample(self._rng))
            self.push_force = mag * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            self.push_end_t = d.time + dc.duration_s.sample(self._rng)
            self._schedule_push()
        d.xfrc_applied[:] = self.ext_force          # plugin / external forces (persist per control step)
        self._ext_cleared = False
        if d.time < self.push_end_t:
            d.xfrc_applied[bid, :3] += self.push_force
        elif self.push_end_t > 0 and d.time > self.push_end_t + 0.5:
            self.push_force = np.zeros(3)
            self.push_end_t = -1.0

    def add_external_force(self, body, force=(0, 0, 0), torque=(0, 0, 0)) -> None:
        """For plugins: add a world-frame force [N] / torque [N*m] on a body (name or id) for the
        current control step. Cleared automatically before the next ``on_step`` call."""
        bid = body if isinstance(body, (int, np.integer)) else mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        self.ext_force[bid, :3] += np.asarray(force, float)
        self.ext_force[bid, 3:] += np.asarray(torque, float)
        self._ext_active = True

    def apply_push(self, force_xyz, duration: float = 0.1, body: str = "head") -> None:
        """Manual push from the GUI."""
        self.push_force = np.asarray(force_xyz, float)
        self.push_end_t = self.data.time + duration
        self._manual_push_bid = self.head_bid if body == "head" else self.ball_bid

    # -------------------------------------------------------------- step
    def step(self, action):
        rc, m, d = self.cfg.robot, self.model, self.data
        a = np.zeros(3)
        act = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        for k, i in enumerate(self.act_idx):
            if k < len(act):
                a[i] = act[k]
        # command latency (control steps)
        self.cmd_queue.append(a.copy())
        while len(self.cmd_queue) > self.p["latency"] + 1:
            self.cmd_queue.popleft()
        a_eff = self.cmd_queue[0]

        if self.torque_mode:
            cmd = a_eff * rc.stepper_holding_torque_nm
        else:
            cmd = a_eff * rc.stepper_max_speed_rads
            cmd = np.where(np.abs(cmd) < self.p["deadband"], 0.0, cmd)
            spr = rc.steps_per_rad
            cmd = np.round(cmd * spr * self.dt_ctrl) / (spr * self.dt_ctrl)  # integer micro-steps per tick
        self.motor_cmd = cmd
        if self._ext_active:
            self.ext_force[:] = 0.0
            self._ext_active = False
        if self.plugins.steps:
            self.plugins.run_step(self)
        self._integrate(cmd)
        self.step_count += 1
        self.sim_time = d.time
        obs_full = self._observe()
        tilt = self.head_tilt()
        self.tilt_hist.append(tilt)
        reward, terms = self._reward(obs_full, a, tilt)
        fell = tilt > math.radians(self.cfg.episode.fall_tilt_deg) or self._contact_lost()
        terminated = False
        if fell:
            self.fallen = True
            reward -= self.cfg.episode.reward.fall_penalty
            terminated = bool(self.cfg.episode.auto_reset_on_fall or self.training)
        if not np.all(np.isfinite(d.qpos)):
            terminated, reward = True, -self.cfg.episode.reward.fall_penalty
            mujoco.mj_resetData(m, d)
        truncated = self.step_count >= self.max_steps and not terminated
        self.prev_action = a
        self.episode_return += reward
        info = self._info(obs_full, reward, terms)
        if terminated or truncated:
            info["episode_stats"] = self.episode_stats()
        return self._mask_obs(obs_full), float(reward), terminated, bool(truncated), info

    def _wheel_speeds(self) -> np.ndarray:
        if self.roller_mode:
            return self.data.qvel[self._wdadr]
        return self.data.actuator_velocity.copy()

    def _integrate(self, cmd: np.ndarray) -> None:
        """Run ``frame_skip`` physics sub-steps with the stepper/TMC2208 model:
        speed-dependent pull-out torque x supply-voltage scale, traction-limited
        slip (virtual model), missed-step accounting and encoder integration."""
        rc, m, d = self.cfg.robot, self.model, self.data
        dt, spr = rc.physics_timestep_s, rc.steps_per_rad
        tq = rc.stepper_holding_torque_nm * self.p["v_scale"]
        corner = rc.stepper_corner_speed_rads
        span = max(rc.stepper_max_speed_rads - corner, 1e-6)
        drop = 1.0 - rc.stepper_torque_at_max_frac
        trac = self.traction_torque
        roller, torque_mode = self.roller_mode, self.torque_mode
        fr = m.actuator_forcerange
        ws = self._wheel_speeds()
        for _ in range(rc.frame_skip):
            x = np.minimum(np.maximum((np.abs(ws) - corner) / span, 0.0), 1.0)
            lim_step = tq * (1.0 - drop * x)
            limit = lim_step if roller else np.minimum(lim_step, trac)
            fr[:, 0] = -limit
            fr[:, 1] = limit
            d.ctrl[:] = np.minimum(np.maximum(cmd, -limit), limit) if torque_mode else cmd
            if self.push_next_t != math.inf or self.push_end_t > 0 or self._ext_active:
                self._update_push()
            elif self._ext_cleared is False:
                d.xfrc_applied[:] = 0.0
                self._ext_cleared = True
            mujoco.mj_step(m, d)
            ws = self._wheel_speeds()
            force = d.actuator_force
            sat = np.abs(force) >= 0.98 * limit
            if roller:
                slip = np.zeros(3, dtype=bool)
                enc_v = ws
            else:
                slip = sat & (trac <= lim_step)
                # a slipping wheel spins up toward the commanded rate (stepper unloaded)
                enc_v = np.where(slip, cmd if not torque_mode else 1.3 * ws, ws)
                self.enc_pos = self.enc_pos + enc_v * dt
                d.qpos[self._wqadr] = self.enc_pos
                d.qvel[self._wdadr] = enc_v
            if not torque_mode:
                lost = sat & ~slip
                if lost.any():
                    self.missed_steps += np.where(lost, np.abs(cmd - ws) * dt * spr, 0.0)
        if roller:
            self.enc_pos = d.qpos[self._wqadr].copy()
        self.slip = slip
        self.enc_vel = np.asarray(enc_v, dtype=np.float64).copy()
        self.motor_torque = d.actuator_force.copy()

    def stepper_torque_limit(self, speed: np.ndarray) -> np.ndarray:
        rc = self.cfg.robot
        x = np.clip((np.abs(speed) - rc.stepper_corner_speed_rads) /
                    max(rc.stepper_max_speed_rads - rc.stepper_corner_speed_rads, 1e-6), 0, 1)
        return rc.stepper_holding_torque_nm * self.p["v_scale"] * (1.0 - (1.0 - rc.stepper_torque_at_max_frac) * x)

    def _contact_lost(self) -> bool:
        if not self.cfg.episode.detect_contact_loss:
            return False
        d, m = self.data, self.model
        if self.roller_mode:
            touching = 0
            for i in range(d.ncon):
                c = d.contact[i]
                if (c.geom1 == self.ball_gid and c.geom2 in self.roller_gids) or \
                        (c.geom2 == self.ball_gid and c.geom1 in self.roller_gids):
                    touching += 1
            lost = touching == 0
        else:
            # head would lift off if the ball-joint reaction pulls the head *down*
            mujoco.mj_rnePostConstraint(m, d)
            f = d.cfrc_int[self.head_bid, 3:]
            zh = d.xmat[self.head_bid].reshape(3, 3)[:, 2]
            lost = float(f @ zh) < 0.0
        self.contact_lost_t = self.contact_lost_t + self.dt_ctrl if lost else 0.0
        return self.contact_lost_t > 0.08

    def _observe(self) -> np.ndarray:
        d = self.data
        ip = self.p["imu"]
        dt = self.dt_ctrl
        rng = self._rng
        q = d.sensordata[self.s_quat:self.s_quat + 4]
        rpy = np.array(quat_to_rpy(q))
        gyro = d.sensordata[self.s_gyro:self.s_gyro + 3].copy()
        acc = d.sensordata[self.s_acc:self.s_acc + 3].copy()
        # bias random walks
        sq = math.sqrt(dt)
        self.imu_bias[0:2] += rng.normal(0, ip["orient_drift_std_rad"] * sq, 2)
        self.imu_bias[2] += rng.normal(0, ip["yaw_drift_std_rad"] * sq)
        self.imu_bias[3:6] += rng.normal(0, ip["gyro_drift_std_rads"] * sq, 3)
        self.imu_bias[6:9] += rng.normal(0, ip["accel_drift_std_ms2"] * sq, 3)
        rpy = rpy + self.imu_bias[0:3] + rng.normal(0, ip["orient_noise_std_rad"], 3)
        rpy[2] = float(wrap_pi(rpy[2]))
        gyro = gyro + self.imu_bias[3:6] + rng.normal(0, ip["gyro_noise_std_rads"], 3)
        acc = acc + self.imu_bias[6:9] + rng.normal(0, ip["accel_noise_std_ms2"], 3)
        enc_v = self.enc_vel + rng.normal(0, ip["encoder_noise_std_rads"], 3)
        enc_p = wrap_pi(self.enc_pos)
        # target error in the (IMU-yaw) heading frame; without a waypoint it points back to the spawn
        # point (= odometry origin on the real robot) so the policy can observe and hold its position
        home = self.target if self.target_enabled else (
            np.array([self.cfg.episode.spawn_x_m, self.cfg.episode.spawn_y_m]) if self.cfg.episode.hold_spawn_position
            else None)
        if home is not None:
            e = home - self.ball_pos()[:2]
            n = np.linalg.norm(e)
            clip = self.cfg.episode.target_obs_clip_m
            if n > clip:
                e = e * clip / n
            cy, sy = math.cos(rpy[2]), math.sin(rpy[2])
            tgt = np.array([cy * e[0] + sy * e[1], -sy * e[0] + cy * e[1]])
        else:
            tgt = np.zeros(2)
        self.last_raw = {"rpy": rpy, "gyro": gyro, "acc": acc, "enc_v": enc_v, "enc_p": enc_p, "tgt": tgt}
        return np.concatenate([rpy, gyro, acc, enc_v, enc_p, tgt, self.prev_action]).astype(np.float64)

    def _mask_obs(self, full: np.ndarray) -> np.ndarray:
        o = (full * self.obs_scale)[self.obs_idx]
        return np.clip(o, -10.0, 10.0).astype(np.float32)

    def _reward(self, obs: np.ndarray, a: np.ndarray, tilt: float) -> Tuple[float, Dict[str, float]]:
        w = self.cfg.episode.reward
        fall = math.radians(self.cfg.episode.fall_tilt_deg)
        gyro = self.data.sensordata[self.s_gyro:self.s_gyro + 3]
        t = {
            "alive": w.alive,
            "upright": w.upright * (1.0 - min(1.0, (tilt / fall) ** 2)),
            "ang_vel": -w.ang_vel * float(gyro[:2] @ gyro[:2] + 0.25 * gyro[2] ** 2),
            "action": -w.action * float(a @ a),
            "action_rate": -w.action_rate * float((a - self.prev_action) @ (a - self.prev_action)),
        }
        if self.plugins.rewards:
            t.update(self.plugins.reward_terms(self))
        pos = self.ball_pos()[:2]
        v = self.data.qvel[self.ball_dadr:self.ball_dadr + 2]
        # linear in speed: cruising slowly must not be "almost free" (|v| instead of v^2)
        t["ball_velocity"] = -w.ball_velocity * float(np.linalg.norm(v)) * (0.2 if self.target_enabled else 1.0)
        if self.target_enabled:
            dist = self._target_dist()
            t["target_distance"] = -w.target_distance * min(dist, 3.0)
            t["target_progress"] = w.target_progress * ((self.prev_dist or dist) - dist)
            if dist < self.cfg.episode.target_reach_radius_m:
                t["target_reached"] = w.target_reached_bonus * self.dt_ctrl * 10
                self.target_reached = True
            self.prev_dist = dist
        else:
            spawn = np.array([self.cfg.episode.spawn_x_m, self.cfg.episode.spawn_y_m])
            t["drift"] = -w.drift * min(float(np.linalg.norm(pos - spawn)), 5.0)
        return float(sum(t.values())), t

    def _info(self, obs_full: np.ndarray, reward: float, terms: Optional[dict] = None) -> Dict[str, Any]:
        if self.training:  # keep IPC payload small for SubprocVecEnv
            return {"tilt_deg": math.degrees(self.head_tilt())}
        return {
            "time": float(self.data.time), "tilt_deg": math.degrees(self.head_tilt()),
            "ball_xy": self.ball_pos()[:2].copy(), "target_xy": self.target.copy(),
            "target_dist": self._target_dist(), "reward": reward, "reward_terms": terms or {},
            "obs_full": obs_full, "motor_cmd": self.motor_cmd.copy(), "motor_torque": self.motor_torque.copy(),
            "enc_vel": self.enc_vel.copy(), "slip": self.slip.copy(), "missed_steps": self.missed_steps.copy(),
            "push_force": self.push_force.copy(), "fallen": self.fallen,
        }

    def episode_stats(self) -> Dict[str, float]:
        tilt = np.degrees(np.array(self.tilt_hist)) if self.tilt_hist else np.zeros(1)
        return {"return": float(self.episode_return), "survival_s": float(self.step_count * self.dt_ctrl),
                "mean_tilt_deg": float(tilt.mean()), "max_tilt_deg": float(tilt.max()),
                "final_target_dist": self._target_dist(), "fallen": bool(self.fallen),
                "missed_steps": float(self.missed_steps.sum())}

    # ------------------------------------------------------- rendering/state
    def get_render_state(self) -> Dict[str, Any]:
        d = self.data
        st = {"qpos": d.qpos.copy(), "mocap_pos": d.mocap_pos.copy(), "time": float(d.time),
              "model_version": self.model_version, "tilt_deg": math.degrees(self.head_tilt()),
              "return": float(self.episode_return)}
        if self.hfield_dirty and self.hfield_id >= 0:
            st["hfield"] = self.model.hfield_data.copy()
        return st

    def set_render_state(self, st: Dict[str, Any]) -> None:
        self.data.qpos[:] = st["qpos"]
        self.data.mocap_pos[:] = st["mocap_pos"]
        mujoco.mj_kinematics(self.model, self.data)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.render_size[1], self.render_size[0])
        self._renderer.update_scene(self.data, camera="chase")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ============================================================================
# Validation rig driver (single motor + omniwheel + bearing-mounted sphere)
# ============================================================================
class ValidationRigSim:
    """Drives the test bench with a commanded wheel speed and reports
    torque / encoder / sphere-speed / slip so friction & torque physics can be
    validated independently of the RL agent."""

    def __init__(self, cfg: Optional[AppConfig] = None):
        self.cfg = (cfg or default_config()).copy()
        self.xml = generate_validation_rig_xml(self.cfg)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.model_version = 1
        m = self.model
        self.wheel_dof = m.jnt_dofadr[m.joint("wheel1_joint").id]
        self.wheel_q = m.jnt_qposadr[m.joint("wheel1_joint").id]
        self.ball_bid = m.body("test_ball").id
        rc = self.cfg.robot
        self.frame = omniwheel_frames(rc.ball_radius_m, rc.zenith_angle_deg, [0.0], [rc.wheel_radius_m])[0]
        self.v_scale = 1.0
        mujoco.mj_forward(m, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, cmd_rads: float, duration_s: float = 0.01) -> Dict[str, float]:
        m, d, rc = self.model, self.data, self.cfg.robot
        n = max(1, int(round(duration_s / rc.physics_timestep_s)))
        for _ in range(n):
            w = d.qvel[self.wheel_dof]
            s = abs(w)
            frac = 1.0 - (1.0 - rc.stepper_torque_at_max_frac) * min(1.0, max(0.0, (s - rc.stepper_corner_speed_rads) /
                                                                             (rc.stepper_max_speed_rads - rc.stepper_corner_speed_rads)))
            lim = rc.stepper_holding_torque_nm * self.v_scale * frac
            m.actuator_forcerange[0] = [-lim, lim]
            d.ctrl[0] = cmd_rads
            mujoco.mj_step(m, d)
        v = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.ball_bid, v, 0)
        w_ball = v[:3]
        wheel_v = float(d.qvel[self.wheel_dof])
        # no-slip prediction: w_ball . m = -(r/R) * wheel speed
        pred = -(self.frame.radius / rc.ball_radius_m) * wheel_v
        meas = float(w_ball @ self.frame.m)
        slip = 0.0 if abs(pred) < 1e-3 else 1.0 - meas / pred
        return {"time": float(d.time), "cmd_rads": cmd_rads, "enc_vel_rads": wheel_v,
                "enc_pos_rad": float(d.qpos[self.wheel_q]), "motor_torque_nm": float(d.actuator_force[0]),
                "torque_limit_nm": float(m.actuator_forcerange[0, 1]),
                "ball_speed_meas_rads": meas, "ball_speed_pred_rads": pred, "slip_ratio": slip,
                "surface_speed_ms": abs(meas) * rc.ball_radius_m, "ncon": int(d.ncon)}

    def get_render_state(self):
        return {"qpos": self.data.qpos.copy(), "mocap_pos": self.data.mocap_pos.copy(), "time": float(self.data.time),
                "model_version": self.model_version}


def make_env(config_dict: dict, rank: int = 0, seed: int = 0, training: bool = True):
    """SB3 VecEnv factory (picklable for SubprocVecEnv)."""
    def _init():
        env = BB8BalanceEnv(AppConfig.from_dict(config_dict), seed=seed + 1000 * rank, training=training)
        return env
    return _init


if __name__ == "__main__":
    env = BB8BalanceEnv()
    o, _ = env.reset(seed=0)
    t0 = time.time()
    n = 0
    for _ in range(2000):
        o, r, term, trunc, info = env.step(env.action_space.sample() * 0)
        n += 1
        if term or trunc:
            print("episode", info["episode_stats"])
            env.reset()
    print(f"{n / (time.time() - t0):.0f} control steps/s")
