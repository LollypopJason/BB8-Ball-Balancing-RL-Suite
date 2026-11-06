"""
mjcf_generator.py
=================

Procedural MJCF (MuJoCo XML) generation for the BB-8 RL Suite.

* ``generate_bb8_xml``            - parametric head-balancing ballbot (3 omniwheels at
                                     120 deg azimuth, adjustable 31-40+ deg zenith contact angle).
* ``generate_validation_rig_xml`` - single JK42HS40 stepper + 58 mm omniwheel test bench
                                     pressing on a bearing-mounted test sphere.
* ``dented_sphere_mesh``          - high-tessellation icosphere with non-sphericity,
                                     dents/bumps (vertex perturbation) for the hand-coated ball.
* ``terrain_xml``                 - flat / obstacle arena + modular layers (grip tiles,
                                     heightfield, walls, ramps, bumps & stairs).
* ``import_custom_model``         - MJCF / URDF / STL / OBJ importer that binds the RL
                                     IMU site, wheel actuators, floor and target marker.
* ``omniwheel_frames`` / ``omniwheel_jacobian`` - exact contact geometry shared with
                                     the gym env and the Jetson odometry code.

Omniwheel modelling (two selectable fidelity levels)
---------------------------------------------------
``virtual`` (default, fast, RL-friendly)
    The head is attached to the ball with a 3-DoF ball joint at the ball centre.
    Each omniwheel is an ideal transmission: wheel speed  phi_i = (R/r_i) * m_i . w_rel
    where ``w_rel`` is the head-relative-to-ball angular velocity (= ball-joint qvel)
    and ``m_i = n_i x t_i`` is the wheel axle. This is implemented natively with a
    MuJoCo ``velocity`` actuator on the ball joint whose 3-D ``gear`` is (R/r_i) m_i,
    so the stepper servo stiffness is integrated implicitly (``implicitfast``).
    Roller (lateral) friction, traction limits and slip are applied by the env.

``roller`` (high fidelity, slower)
    Head is a free body resting on the ball. Each omniwheel is a hinge-driven hub
    carrying 2 staggered rows of passive rollers (hinge joints about the rim tangent)
    whose sphere geoms make real frictional contact with the ball. This is the
    multi-roller approximation (stock MuJoCo cannot orient anisotropic friction).
"""
from __future__ import annotations

import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config_and_distributions import AppConfig, RobotGeometryConfig, TerrainConfig

# Contact bit-masks -----------------------------------------------------------
#   ground/obstacles : contype 1, conaffinity 5 (collide with ball + head shell)
#   ball             : contype 1, conaffinity 3 (collide with ground + rollers)
#   rollers          : contype 2, conaffinity 0 (collide only with the ball)
#   head shell       : contype 4, conaffinity 4 (collide only with ground when fallen)
GROUND_CT = 'contype="1" conaffinity="5"'
BALL_CT = 'contype="1" conaffinity="3"'
ROLLER_CT = 'contype="2" conaffinity="0" solref="0.004 1" solimp="0.95 0.99 0.001"'
HEAD_CT = 'contype="4" conaffinity="4"'
VIS = 'contype="0" conaffinity="0" group="1"'


def _v(x: Sequence[float], p: int = 6) -> str:
    return " ".join(f"{float(a):.{p}g}" for a in x)


# ============================================================================
# Geometry helpers
# ============================================================================
@dataclass
class WheelFrame:
    index: int
    azimuth: float          # rad
    zenith: float           # rad
    n: np.ndarray           # unit contact normal (ball centre -> contact), head frame
    t: np.ndarray           # unit drive direction (rim tangent at contact)
    m: np.ndarray           # unit wheel axle = n x t
    contact: np.ndarray     # contact point relative to ball centre
    center: np.ndarray      # wheel centre relative to ball centre
    radius: float           # wheel outer radius


def omniwheel_frames(ball_radius: float, zenith_deg: float, azimuths_deg: Sequence[float],
                     wheel_radii: Sequence[float]) -> List[WheelFrame]:
    th = math.radians(zenith_deg)
    out = []
    for i, (az_d, rw) in enumerate(zip(azimuths_deg, wheel_radii)):
        a = math.radians(az_d)
        n = np.array([math.sin(th) * math.cos(a), math.sin(th) * math.sin(a), math.cos(th)])
        t = np.array([-math.sin(a), math.cos(a), 0.0])
        m = np.cross(n, t)
        out.append(WheelFrame(i, a, th, n, t, m, ball_radius * n, (ball_radius + rw) * n, rw))
    return out


def omniwheel_jacobian(ball_radius: float, zenith_deg: float, azimuths_deg: Sequence[float],
                       wheel_radii: Sequence[float]) -> np.ndarray:
    """J (3x3):  wheel_speeds = J @ w_rel ,  w_rel = head-relative-to-ball angular
    velocity expressed in the head frame (the MuJoCo ball-joint qvel)."""
    fr = omniwheel_frames(ball_radius, zenith_deg, azimuths_deg, wheel_radii)
    return np.stack([(ball_radius / f.radius) * f.m for f in fr])


def icosphere(subdivisions: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
             (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
             (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in verts]
    f = list(faces)
    for _ in range(max(0, int(subdivisions))):
        cache: Dict[Tuple[int, int], int] = {}

        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                p = v[i] + v[j]
                v.append(p / np.linalg.norm(p))
                cache[key] = len(v) - 1
            return cache[key]

        nf = []
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), np.array(f, dtype=np.int64)


def dented_sphere_mesh(radius: float, subdivisions: int, non_sphericity: float = 0.0,
                       dent_depth: float = 0.0, dent_count: int = 0, dent_width: float = 0.25,
                       rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Procedurally imperfect ball.

    * ``non_sphericity``: random ellipsoidal stretch (fraction of R) along a random axis set.
    * ``dent_depth`` / ``dent_count``: Gaussian radial perturbations; negative = dent,
      positive = bump. NOTE: MuJoCo collides meshes through their convex hull, so an
      inward dent becomes a *flat spot* on the hull - exactly how a hard dented ball
      rolls on flat ground - while outward bumps act fully.
    """
    rng = rng or np.random.default_rng()
    v, f = icosphere(subdivisions)
    r = np.ones(len(v))
    if dent_depth > 0 and dent_count > 0:
        centres = rng.normal(size=(dent_count, 3))
        centres /= np.linalg.norm(centres, axis=1, keepdims=True)
        depths = rng.uniform(-1.0, 0.6, size=dent_count) * dent_depth / radius
        for c, d in zip(centres, depths):
            ang = np.arccos(np.clip(v @ c, -1, 1))
            r += d * np.exp(-(ang / max(dent_width, 1e-3)) ** 2)
    pts = v * r[:, None]
    if non_sphericity > 0:
        stretch = 1.0 + rng.uniform(-non_sphericity, non_sphericity, size=3)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        pts = ((pts @ q) * stretch) @ q.T
    return pts * radius, f


def dome_mesh(radius: float, height: float, segments: int = 24, rings: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Hemispherical (elliptic) head dome with flat bottom, origin at base centre."""
    verts = [(0.0, 0.0, 0.0)]
    for k in range(rings + 1):
        phi = (math.pi / 2) * k / rings
        for s in range(segments):
            a = 2 * math.pi * s / segments
            verts.append((radius * math.cos(phi) * math.cos(a), radius * math.cos(phi) * math.sin(a),
                          height * math.sin(phi)))
    faces = []
    for s in range(segments):
        faces.append((0, 1 + (s + 1) % segments, 1 + s))
    for k in range(rings):
        r0, r1 = 1 + k * segments, 1 + (k + 1) * segments
        for s in range(segments):
            s1 = (s + 1) % segments
            faces += [(r0 + s, r0 + s1, r1 + s1), (r0 + s, r1 + s1, r1 + s)]
    return np.array(verts), np.array(faces)


def _mesh_asset(name: str, verts: np.ndarray, faces: np.ndarray) -> str:
    return (f'<mesh name="{name}" vertex="{_v(verts.ravel(), 7)}" '
            f'face="{" ".join(str(int(i)) for i in faces.ravel())}"/>')


# ============================================================================
# Structural sample (things that require recompilation when they change)
# ============================================================================
@dataclass
class StructuralSample:
    ball_radius: float
    non_sphericity: float = 0.0
    dent_depth: float = 0.0
    wheel_radii: List[float] = field(default_factory=lambda: [0.029] * 3)
    mesh_seed: int = 0

    def key(self, omni_model: str) -> tuple:
        wr = tuple(round(r, 6) for r in self.wheel_radii) if omni_model == "roller" else ()
        dent = (round(self.non_sphericity, 6), round(self.dent_depth, 6),
                self.mesh_seed if (self.non_sphericity > 0 or self.dent_depth > 0) else 0)
        return (round(self.ball_radius, 6),) + dent + wr


# ============================================================================
# Terrain
# ============================================================================
def terrain_xml(tc: TerrainConfig) -> Tuple[str, str, Dict[str, object]]:
    """Return (asset_xml, worldbody_xml, info). Ground geoms are named ``ground_*``
    (their friction is set by the env from ball x ground-grip factors), obstacles ``obst_*``."""
    rng = np.random.default_rng(tc.layout_seed)
    H = tc.arena_half_size_m
    assets, body = [], []
    info: Dict[str, object] = {"hfield": None, "spawn_z_offset": 0.0, "tiles": []}
    floor_z = 0.0
    if tc.rough_heightfield:
        A = tc.rough_amplitude_m
        n = int(tc.rough_resolution)
        assets.append(f'<hfield name="rough_hf" nrow="{n}" ncol="{n}" size="{H} {H} {A:.5f} 0.05"/>')
        body.append(f'<geom name="ground_hfield" type="hfield" hfield="rough_hf" pos="0 0 {-A / 2:.5f}" '
                    f'material="ground_mat" {GROUND_CT} priority="3"/>')
        floor_z = -A / 2 - 0.002
        info["hfield"] = "rough_hf"
        info["spawn_z_offset"] = A / 2 + 0.002
    elif tc.variable_grip:
        n = int(tc.grip_tiles)
        s = 2 * H / n
        for i in range(n):
            for j in range(n):
                x, y = -H + (i + 0.5) * s, -H + (j + 0.5) * s
                shade = 0.32 + 0.12 * ((i + j) % 2)
                name = f"ground_tile_{i}_{j}"
                info["tiles"].append(name)
                body.append(f'<geom name="{name}" type="box" size="{s / 2:.4f} {s / 2:.4f} 0.01" '
                            f'pos="{x:.4f} {y:.4f} -0.01" rgba="{shade:.2f} {shade + 0.03:.2f} {shade + 0.07:.2f} 1" '
                            f'{GROUND_CT} priority="3"/>')
        floor_z = -0.021
    body.insert(0, f'<geom name="ground_floor" type="plane" size="{H + 4} {H + 4} 0.1" pos="0 0 {floor_z:.4f}" '
                   f'material="grid_mat" {GROUND_CT} priority="3"/>')

    if tc.walls or tc.map_name == "obstacle_arena":
        th, h = 0.05, 0.25
        for nm, pos, size in (("n", (0, H + th, h / 2), (H + 2 * th, th, h / 2)),
                              ("s", (0, -H - th, h / 2), (H + 2 * th, th, h / 2)),
                              ("e", (H + th, 0, h / 2), (th, H, h / 2)),
                              ("w", (-H - th, 0, h / 2), (th, H, h / 2))):
            body.append(f'<geom name="obst_wall_{nm}" type="box" size="{_v(size)}" pos="{_v(pos)}" '
                        f'material="wall_mat" {GROUND_CT}/>')
    if tc.map_name == "obstacle_arena":
        placed = 0
        for _ in range(200):
            if placed >= 9:
                break
            r = rng.uniform(1.0, H - 0.4)
            a = rng.uniform(0, 2 * math.pi)
            x, y = r * math.cos(a), r * math.sin(a)
            if rng.random() < 0.5:
                sx, sy = rng.uniform(0.08, 0.25, size=2)
                yaw = rng.uniform(0, 180)
                body.append(f'<geom name="obst_box_{placed}" type="box" size="{sx:.3f} {sy:.3f} 0.15" '
                            f'pos="{x:.3f} {y:.3f} 0.15" euler="0 0 {yaw:.1f}" material="obst_mat" {GROUND_CT}/>')
            else:
                rr = rng.uniform(0.06, 0.18)
                body.append(f'<geom name="obst_cyl_{placed}" type="cylinder" size="{rr:.3f} 0.15" '
                            f'pos="{x:.3f} {y:.3f} 0.15" material="obst_mat" {GROUND_CT}/>')
            placed += 1
    if tc.ramps:
        s = math.radians(tc.ramp_slope_deg)
        L, W, h = 0.7, 0.8, 0.02
        for k, (cx, cy) in enumerate(((1.7, 0.4), (-1.3, 1.5), (0.3, -1.9))):
            yaw = rng.uniform(0, 180)
            cyaw, syaw = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            for side in (-1, 1):  # two mirrored slabs form a gentle ridge
                cz = L / 2 * math.sin(s) - h * math.cos(s)
                off = side * (L / 2) * math.cos(s)
                px, py = cx + off * cyaw, cy + off * syaw
                pitch = side * math.degrees(s)
                body.append(f'<geom name="ground_ramp_{k}_{"ab"[side > 0]}" type="box" size="{L / 2} {W / 2} {h}" '
                            f'pos="{px:.4f} {py:.4f} {cz:.4f}" euler="0 {pitch:.3f} {yaw:.2f}" '
                            f'material="ramp_mat" {GROUND_CT} priority="3"/>')
    if tc.bumps_stairs:
        rb = tc.bump_height_m
        for k, (x, y, yaw) in enumerate(((0.9, -0.6, 90), (-0.8, -0.9, 30), (-1.0, 0.6, 0), (2.0, -1.4, 60))):
            body.append(f'<geom name="ground_bump_{k}" type="capsule" size="{rb:.4f} 0.45" pos="{x} {y} 0" '
                        f'euler="90 0 {yaw}" material="ramp_mat" {GROUND_CT} priority="3"/>')
        st = tc.stair_step_m
        for k in range(3):
            top = (k + 1) * st
            body.append(f'<geom name="ground_stair_{k}" type="box" size="{0.5 - 0.15 * k:.3f} 0.6 {top / 2:.4f}" '
                        f'pos="{-2.1 + 0.15 * k:.3f} -1.9 {top / 2:.4f}" material="ramp_mat" {GROUND_CT} priority="3"/>')
    return "\n    ".join(assets), "\n    ".join(body), info


def _common_assets() -> str:
    return """
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.32 0.38 0.48" rgb2="0.05 0.06 0.08" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.22 0.24 0.27" rgb2="0.17 0.19 0.22" width="512" height="512" mark="edge" markrgb="0.35 0.4 0.45"/>
    <material name="grid_mat" texture="grid" texrepeat="12 12" texuniform="true" reflectance="0.08"/>
    <material name="ground_mat" rgba="0.36 0.33 0.29 1"/>
    <material name="wall_mat" rgba="0.45 0.47 0.52 1"/>
    <material name="obst_mat" rgba="0.62 0.42 0.25 1"/>
    <material name="ramp_mat" rgba="0.40 0.44 0.50 1"/>
    <material name="ball_mat" rgba="0.94 0.93 0.90 1" specular="0.4" shininess="0.5"/>
    <material name="bb8_orange" rgba="0.95 0.50 0.10 1"/>
    <material name="head_mat" rgba="0.93 0.92 0.89 1" specular="0.5" shininess="0.6"/>
    <material name="metal_mat" rgba="0.55 0.57 0.60 1" specular="0.7"/>
    <material name="motor_mat" rgba="0.12 0.12 0.13 1"/>
    <material name="roller_mat" rgba="0.15 0.16 0.18 1"/>
    <material name="hub_mat" rgba="0.75 0.30 0.10 1"/>"""


def _header(model_name: str, timestep: float) -> str:
    return f"""<mujoco model="{model_name}">
  <compiler angle="degree" autolimits="true" inertiafromgeom="auto"/>
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="0">
    <flag multiccd="enable"/>
  </option>
  <size memory="64M"/>
  <visual>
    <global offwidth="1920" offheight="1200" azimuth="135" elevation="-22"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.35 0.35 0.38" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
    <map znear="0.01" zfar="60"/>
  </visual>"""


# ============================================================================
# Parametric BB-8 ballbot
# ============================================================================
def generate_bb8_xml(cfg: AppConfig, structural: Optional[StructuralSample] = None,
                     head_shell_mesh: Optional[str] = None, head_shell_scale: float = 1.0) -> str:
    rc: RobotGeometryConfig = cfg.robot
    R = structural.ball_radius if structural else rc.ball_radius_m
    wheel_r = list(structural.wheel_radii) if structural else [rc.wheel_radius_m] * 3
    roller = rc.omniwheel_model == "roller"
    torque_mode = cfg.network.action_mode == "torque"
    frames = omniwheel_frames(R, rc.zenith_angle_deg, rc.wheel_azimuths_deg, wheel_r)

    # --- ball mesh ---------------------------------------------------------
    rng = np.random.default_rng(structural.mesh_seed if structural else 0)
    ns = structural.non_sphericity if structural else 0.0
    dd = structural.dent_depth if structural else 0.0
    use_mesh = ns > 0 or dd > 0
    assets = [_common_assets()]
    if use_mesh:
        bv, bf = dented_sphere_mesh(R, rc.ball_mesh_subdivisions, ns, dd, cfg.randomization.ball.dent_count,
                                    cfg.randomization.ball.dent_width_rad, rng)
        assets.append(_mesh_asset("ball_mesh", bv, bf))
        ball_geom = f'<geom name="ball_geom" type="mesh" mesh="ball_mesh" material="ball_mat" {BALL_CT} priority="1" condim="6"/>'
    else:
        ball_geom = f'<geom name="ball_geom" type="sphere" size="{R:.5f}" material="ball_mat" {BALL_CT} priority="1" condim="6"/>'

    # head geometry heights (frame origin = ball centre)
    top_wheel = max(f.center[2] + f.radius * math.cos(f.zenith) for f in frames)
    plate_z = max(R + 0.015, top_wheel + 0.012)
    dome_h = max(0.06, rc.head_radius_m * 0.72)
    dv, df = dome_mesh(rc.head_radius_m, dome_h)
    assets.append(_mesh_asset("dome_mesh", dv, df))
    if head_shell_mesh:
        assets.append(f'<mesh name="head_shell" file="{os.path.abspath(head_shell_mesh)}" '
                      f'scale="{head_shell_scale} {head_shell_scale} {head_shell_scale}"/>')
    t_asset, t_body, t_info = terrain_xml(cfg.terrain)
    if t_asset:
        assets.append(t_asset)

    m_head = rc.head_mass_kg
    hub_mass, roller_mass = 0.035, 0.002
    n_rollers = 2 * rc.rollers_per_row
    wheel_total = 3 * (hub_mass + n_rollers * roller_mass) if roller else 3 * 1e-3
    head_body_mass = max(0.1, m_head - wheel_total)
    rh, hh = 0.11, 0.14
    ih = (head_body_mass * (3 * rh ** 2 + hh ** 2) / 12, head_body_mass * (3 * rh ** 2 + hh ** 2) / 12,
          head_body_mass * rh ** 2 / 2)
    cog_z = R + rc.head_cog_height_m

    # --- wheel bodies --------------------------------------------------------
    wheel_xml = []
    for f in frames:
        i = f.index + 1
        rw = f.radius
        rr = 0.4 * rw
        rcen = rw - rr
        motor_len = 0.04
        motor_c = f.center - (0.5 * rc.wheel_width_m + motor_len / 2 + 0.004) * f.m
        brk_top = np.array([motor_c[0], motor_c[1], plate_z])
        wheel_xml.append(
            f'<geom name="motor{i}_case" type="box" size="0.021 0.021 {motor_len / 2}" pos="{_v(motor_c)}" '
            f'zaxis="{_v(f.m)}" material="motor_mat" {VIS}/>')
        wheel_xml.append(
            f'<geom name="motor{i}_bracket" type="capsule" fromto="{_v(motor_c)} {_v(brk_top)}" size="0.006" '
            f'material="metal_mat" {VIS}/>')
        wheel_xml.append(
            f'<geom name="encoder{i}" type="cylinder" size="0.012 0.006" pos="{_v(motor_c - (motor_len / 2 + 0.006) * f.m)}" '
            f'zaxis="{_v(f.m)}" rgba="0.1 0.45 0.9 1" {VIS}/>')
        if roller:
            jw = (f'<joint name="wheel{i}_joint" type="hinge" axis="0 0 1" armature="{rc.stepper_rotor_inertia:.3g}" '
                  f'damping="0.0005"/>')
            winert = f'<inertial pos="0 0 0" mass="{hub_mass}" diaginertia="{hub_mass * rw ** 2 / 4:.3g} {hub_mass * rw ** 2 / 4:.3g} {hub_mass * rw ** 2 / 2:.3g}"/>'
        else:
            jw = f'<joint name="wheel{i}_joint" type="hinge" axis="0 0 1" damping="0"/>'
            winert = '<inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>'
        rollers = []
        offs = rc.wheel_width_m / 4
        for row in range(2):
            for k in range(rc.rollers_per_row):
                b = 2 * math.pi * (k + 0.5 * row) / rc.rollers_per_row
                pos = (rcen * math.cos(b), rcen * math.sin(b), (offs if row else -offs))
                ax = (-math.sin(b), math.cos(b), 0)
                if roller:
                    rollers.append(
                        f'<body name="w{i}_roller_{row}_{k}" pos="{_v(pos)}">'
                        f'<joint name="w{i}_roller_{row}_{k}_j" type="hinge" axis="{_v(ax)}" damping="0.001"/>'
                        f'<geom name="w{i}_roller_{row}_{k}_g" type="sphere" size="{rr:.5f}" mass="{roller_mass}" '
                        f'material="roller_mat" {ROLLER_CT} priority="4" condim="3" friction="1.0 0.005 0.0001"/>'
                        f'</body>')
                else:
                    rollers.append(f'<geom type="sphere" size="{rr:.5f}" pos="{_v(pos)}" material="roller_mat" {VIS}/>')
        wheel_xml.append(
            f'<body name="wheel{i}" pos="{_v(f.center)}" zaxis="{_v(f.m)}">\n          {jw}\n          {winert}\n'
            f'          <geom name="wheel{i}_hub" type="cylinder" size="{rcen * 0.95:.5f} {rc.wheel_width_m * 0.42:.5f}" '
            f'material="hub_mat" {VIS} mass="0"/>\n'
            f'          <geom name="wheel{i}_marker" type="box" size="{rcen * 0.9:.5f} 0.002 {rc.wheel_width_m * 0.45:.5f}" '
            f'rgba="1 1 1 1" {VIS} mass="0"/>\n          ' + "\n          ".join(rollers) + "\n        </body>")

    head_shell = (f'<geom name="head_shell_import" type="mesh" mesh="head_shell" pos="0 0 {plate_z:.4f}" '
                  f'rgba="0.9 0.9 0.88 1" {VIS}/>' if head_shell_mesh else "")
    head_joint = ('<freejoint name="head_free"/>' if roller
                  else '<joint name="head_ball_joint" type="ball" damping="0.0002"/>')
    head_inner = f"""
        {head_joint}
        <inertial pos="0 0 {cog_z:.5f}" mass="{head_body_mass:.4f}" diaginertia="{_v(ih)}"/>
        <site name="imu" pos="0 0 {R + rc.imu_height_m:.5f}" size="0.012" rgba="0.1 0.8 0.2 1" group="2"/>
        <site name="head_cog" pos="0 0 {cog_z:.5f}" size="0.01" rgba="1 0 1 1" group="3"/>
        <geom name="head_plate" type="cylinder" size="{rc.head_radius_m * 0.98:.4f} 0.004" pos="0 0 {plate_z:.4f}" material="metal_mat" {HEAD_CT} mass="0"/>
        <geom name="head_dome" type="mesh" mesh="dome_mesh" pos="0 0 {plate_z + 0.004:.4f}" material="head_mat" {HEAD_CT} mass="0"/>
        <geom name="head_band" type="cylinder" size="{rc.head_radius_m * 1.005:.4f} 0.012" pos="0 0 {plate_z + 0.018:.4f}" material="bb8_orange" {VIS} mass="0"/>
        <geom name="head_eye" type="sphere" size="0.028" pos="{rc.head_radius_m * 0.72:.4f} 0 {plate_z + dome_h * 0.62:.4f}" rgba="0.03 0.03 0.04 1" {VIS} mass="0"/>
        <geom name="head_eye_small" type="sphere" size="0.011" pos="{rc.head_radius_m * 0.58:.4f} 0.05 {plate_z + dome_h * 0.86:.4f}" rgba="0.03 0.03 0.04 1" {VIS} mass="0"/>
        <geom name="head_antenna" type="capsule" fromto="0 -0.03 {plate_z + dome_h:.4f} 0 -0.03 {plate_z + dome_h + 0.07:.4f}" size="0.003" material="metal_mat" {VIS} mass="0"/>
        <geom name="jetson_board" type="box" size="0.05 0.04 0.012" pos="0 0 {plate_z + 0.02:.4f}" rgba="0.1 0.5 0.2 1" group="3" contype="0" conaffinity="0" mass="0"/>
        {head_shell}
        """ + "\n        ".join(wheel_xml)

    # decals on the ball (visual only) to make rolling visible
    decals = []
    for k, d in enumerate([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                           (0.577, 0.577, 0.577), (-0.577, -0.577, 0.577), (0.577, -0.577, -0.577),
                           (-0.577, 0.577, -0.577)]):
        rad = 0.36 * R if k < 6 else 0.22 * R
        decals.append(f'<geom name="ball_decal_{k}" type="cylinder" size="{rad:.4f} 0.0015" pos="{_v(np.array(d) * (R + 0.0003))}" '
                      f'zaxis="{_v(d)}" material="bb8_orange" {VIS} mass="0"/>')

    bb = cfg.randomization.ball
    ball_mass = bb.mass_kg.nominal()
    I = rc.ball_inertia_factor * ball_mass * R ** 2
    spawn_z = R + float(t_info["spawn_z_offset"]) + 0.0005
    sx, sy = cfg.episode.spawn_x_m, cfg.episode.spawn_y_m
    ball_body_open = f"""<body name="ball" pos="{sx} {sy} {spawn_z:.5f}">
      <freejoint name="ball_free"/>
      <inertial pos="0 0 0" mass="{ball_mass:.4f}" diaginertia="{I:.6g} {I:.6g} {I:.6g}"/>
      {ball_geom}
      """ + "\n      ".join(decals)

    if roller:
        bodies = ball_body_open + "\n    </body>\n" + \
                 f'    <body name="head" pos="{sx} {sy} {spawn_z:.5f}">{head_inner}\n    </body>'
    else:
        bodies = ball_body_open + f'\n      <body name="head" pos="0 0 0">{head_inner}\n      </body>\n    </body>'

    # --- actuators ---------------------------------------------------------
    tq = rc.stepper_holding_torque_nm
    acts = []
    for f in frames:
        i = f.index + 1
        if roller:
            tr = f'joint="wheel{i}_joint"'
        else:
            g = (R / f.radius) * f.m
            tr = f'joint="head_ball_joint" gear="{_v(g)} 0 0 0"'
        if torque_mode:
            acts.append(f'<motor name="motor{i}" {tr} ctrlrange="{-tq} {tq}" forcerange="{-tq} {tq}"/>')
        else:
            acts.append(f'<velocity name="motor{i}" {tr} kv="{rc.stepper_kv}" ctrlrange="{-rc.stepper_max_speed_rads} '
                        f'{rc.stepper_max_speed_rads}" forcerange="{-tq} {tq}"/>')

    xml = f"""{_header("bb8_head_balancing_ballbot", rc.physics_timestep_s)}
  <asset>{"".join(assets)}
  </asset>
  <worldbody>
    <light name="key" pos="2 -2 5" dir="-0.4 0.4 -1" directional="true" castshadow="true" diffuse="0.7 0.7 0.68"/>
    <light name="fill" pos="-3 2 4" dir="0.5 -0.3 -1" directional="true" castshadow="false" diffuse="0.25 0.27 0.3"/>
    {t_body}
    <body name="target" mocap="true" pos="{cfg.episode.target_x_m.nominal()} {cfg.episode.target_y_m.nominal()} 0.002">
      <geom name="target_disc" type="cylinder" size="0.09 0.002" rgba="0.2 0.95 0.4 0.55" {VIS}/>
      <geom name="target_pole" type="capsule" fromto="0 0 0 0 0 0.45" size="0.006" rgba="0.2 0.95 0.4 0.8" {VIS}/>
      <geom name="target_flag" type="box" size="0.045 0.002 0.025" pos="0.045 0 0.42" rgba="0.2 0.95 0.4 0.9" {VIS}/>
    </body>
    <camera name="chase" mode="trackcom" pos="1.1 -1.1 0.75" xyaxes="0.707 0.707 0 -0.3 0.3 0.9"/>
    {bodies}
  </worldbody>
  <actuator>
    {chr(10).join("    " + a for a in acts).strip()}
  </actuator>
  <sensor>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
    <gyro name="imu_gyro" site="imu"/>
    <accelerometer name="imu_accel" site="imu"/>
    <framepos name="ball_pos" objtype="xbody" objname="ball"/>
    <framelinvel name="ball_linvel" objtype="xbody" objname="ball"/>
  </sensor>
</mujoco>
"""
    return xml


# ============================================================================
# Validation rig: single stepper + 58 mm omniwheel on a bearing-mounted sphere
# ============================================================================
def generate_validation_rig_xml(cfg: AppConfig, preload_n: Optional[float] = None,
                                bearing_damping: float = 0.0005) -> str:
    rc = cfg.robot
    R, rw = rc.ball_radius_m, rc.wheel_radius_m
    f = omniwheel_frames(R, rc.zenith_angle_deg, [0.0], [rw])[0]
    if preload_n is None:
        preload_n = rc.head_mass_kg * 9.81 / (3 * math.cos(f.zenith))
    k_spring = 3000.0
    cz = R + 0.35
    ball_mass = cfg.randomization.ball.mass_kg.nominal()
    I = rc.ball_inertia_factor * ball_mass * R ** 2
    rr = 0.4 * rw
    rcen = rw - rr
    offs = rc.wheel_width_m / 4
    rollers = []
    for row in range(2):
        for k in range(rc.rollers_per_row):
            b = 2 * math.pi * (k + 0.5 * row) / rc.rollers_per_row
            pos = (rcen * math.cos(b), rcen * math.sin(b), (offs if row else -offs))
            ax = (-math.sin(b), math.cos(b), 0)
            rollers.append(
                f'<body name="roller_{row}_{k}" pos="{_v(pos)}"><joint name="roller_{row}_{k}_j" type="hinge" '
                f'axis="{_v(ax)}" damping="0.001"/><geom name="roller_{row}_{k}_g" type="sphere" size="{rr:.5f}" '
                f'mass="0.002" material="roller_mat" {ROLLER_CT} priority="4" condim="3" friction="1.0 0.005 0.0001"/></body>')
    motor_c = -(0.5 * rc.wheel_width_m + 0.024) * np.array([0, 0, 1.0])
    tq = rc.stepper_holding_torque_nm
    return f"""{_header("validation_rig_stepper_omniwheel", rc.physics_timestep_s)}
  <asset>{_common_assets()}
  </asset>
  <worldbody>
    <light pos="1 -1 3" dir="-0.3 0.3 -1" directional="true" castshadow="true"/>
    <geom name="ground_floor" type="plane" size="3 3 0.1" material="grid_mat" {GROUND_CT}/>
    <geom name="stand_post" type="cylinder" fromto="0 0 0 0 0 {cz - R - 0.01:.4f}" size="0.02" material="metal_mat" {VIS}/>
    <geom name="stand_base" type="cylinder" size="0.18 0.01" pos="0 0 0.01" material="metal_mat" {VIS}/>
    <body name="test_ball" pos="0 0 {cz:.4f}">
      <joint name="test_ball_bearing" type="ball" damping="{bearing_damping}"/>
      <inertial pos="0 0 0" mass="{ball_mass:.4f}" diaginertia="{I:.6g} {I:.6g} {I:.6g}"/>
      <geom name="ball_geom" type="sphere" size="{R:.5f}" material="ball_mat" {BALL_CT} priority="1" condim="6"/>
      <geom name="ball_decal" type="cylinder" size="{0.3 * R:.4f} 0.0015" pos="{R:.4f} 0 0" zaxis="1 0 0" material="bb8_orange" {VIS} mass="0"/>
      <geom name="ball_decal2" type="cylinder" size="{0.3 * R:.4f} 0.0015" pos="0 {R:.4f} 0" zaxis="0 1 0" material="bb8_orange" {VIS} mass="0"/>
    </body>
    <body name="carriage" pos="{_v(f.center + np.array([0, 0, cz]))}">
      <joint name="preload_slide" type="slide" axis="{_v(f.n)}" stiffness="{k_spring}" springref="{-preload_n / k_spring:.6f}" damping="40"/>
      <inertial pos="0 0 0" mass="0.35" diaginertia="2e-4 2e-4 2e-4"/>
      <geom name="carriage_rail" type="capsule" fromto="0 0 0 {_v(0.12 * f.n)}" size="0.006" material="metal_mat" {VIS}/>
      <body name="wheel1" pos="0 0 0" zaxis="{_v(f.m)}">
        <joint name="wheel1_joint" type="hinge" axis="0 0 1" armature="{rc.stepper_rotor_inertia:.3g}" damping="0.0005"/>
        <inertial pos="0 0 0" mass="0.035" diaginertia="8e-6 8e-6 1.5e-5"/>
        <geom name="wheel1_hub" type="cylinder" size="{rcen * 0.95:.5f} {rc.wheel_width_m * 0.42:.5f}" material="hub_mat" {VIS} mass="0"/>
        <geom name="wheel1_marker" type="box" size="{rcen * 0.9:.5f} 0.002 {rc.wheel_width_m * 0.45:.5f}" rgba="1 1 1 1" {VIS} mass="0"/>
        <geom name="motor1_case" type="box" size="0.021 0.021 0.02" pos="{_v(motor_c)}" material="motor_mat" {VIS} mass="0"/>
        <geom name="encoder1" type="cylinder" size="0.012 0.006" pos="{_v(motor_c * 1.9)}" rgba="0.1 0.45 0.9 1" {VIS} mass="0"/>
        {"".join(rollers)}
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="motor1" joint="wheel1_joint" kv="{rc.stepper_kv}" ctrlrange="{-rc.stepper_max_speed_rads} {rc.stepper_max_speed_rads}" forcerange="{-tq} {tq}"/>
  </actuator>
  <sensor>
    <jointvel name="enc1_vel" joint="wheel1_joint"/>
    <jointpos name="enc1_pos" joint="wheel1_joint"/>
    <actuatorfrc name="motor1_torque" actuator="motor1"/>
  </sensor>
</mujoco>
"""


# ============================================================================
# Custom model importer (MJCF / URDF / STL / OBJ)
# ============================================================================
def _urdf_to_mjcf(path: str) -> str:
    import mujoco
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    base = os.path.dirname(os.path.abspath(path))
    import re
    txt = re.sub(r"package://[^/]+/", "", txt)
    if "<mujoco>" not in txt:
        txt = txt.replace("<robot", "<robot", 1)
        txt = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler meshdir="' + base.replace("\\", "/") +
                     r'" balanceinertia="true" discardvisual="false" fusestatic="false"/></mujoco>', txt, count=1)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False, dir=base if os.access(base, os.W_OK) else None)
    tmp.write(txt)
    tmp.close()
    try:
        model = mujoco.MjModel.from_xml_path(tmp.name)
        out = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
        out.close()
        mujoco.mj_saveLastXML(out.name, model)
        with open(out.name, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        os.unlink(tmp.name)


def ensure_binary_stl(path: str) -> str:
    """MuJoCo only reads binary STL; convert ASCII STL (common CAD export) to a temp binary copy."""
    if not path.lower().endswith(".stl"):
        return path
    with open(path, "rb") as f:
        head = f.read(512)
    if not head.lstrip().lower().startswith(b"solid") or b"facet" not in head:
        return path
    import struct
    verts = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            t = line.split()
            if t and t[0] == "vertex":
                verts.append(tuple(float(x) for x in t[1:4]))
    tris = [verts[i:i + 3] for i in range(0, len(verts) - len(verts) % 3, 3)]
    out = os.path.join(tempfile.gettempdir(), f"bb8_{abs(hash(os.path.abspath(path)))}_{len(tris)}.stl")
    with open(out, "wb") as f:
        f.write(b"converted by bb8-rl-suite".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for tri in tris:
            a, b, c = (np.array(v) for v in tri)
            n = np.cross(b - a, c - a)
            n = n / (np.linalg.norm(n) + 1e-12)
            f.write(struct.pack("<12fH", *n, *a, *b, *c, 0))
    return out


def import_custom_model(cfg: AppConfig) -> str:
    """Return an MJCF string for the imported model with RL bindings injected:
    floor (if missing), ``imu`` site + sensors on the bound IMU body, ``motor1..3``
    velocity actuators on the bound wheel joints, ``ball_pos`` sensor and a mocap
    ``target`` marker. Bindings left empty are skipped (viewer/inspector still works)."""
    b = cfg.custom
    path = os.path.abspath(b.path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".stl", ".obj"):
        path = ensure_binary_stl(path)
        if b.import_mode == "head_shell":
            return generate_bb8_xml(cfg, head_shell_mesh=path, head_shell_scale=b.mesh_scale)
        s = b.mesh_scale
        return f"""{_header("imported_mesh", cfg.robot.physics_timestep_s)}
  <asset>{_common_assets()}
    <mesh name="imported" file="{path}" scale="{s} {s} {s}"/>
  </asset>
  <worldbody>
    <light pos="1 -1 3" dir="-0.3 0.3 -1" directional="true"/>
    <geom name="ground_floor" type="plane" size="5 5 0.1" material="grid_mat" {GROUND_CT}/>
    <body name="imported_body" pos="0 0 0.6"><freejoint/><site name="imu" pos="0 0 0"/>
      <geom name="imported_geom" type="mesh" mesh="imported" rgba="0.9 0.9 0.88 1" {BALL_CT}/></body>
  </worldbody>
  <sensor><framequat name="imu_quat" objtype="site" objname="imu"/><gyro name="imu_gyro" site="imu"/>
    <accelerometer name="imu_accel" site="imu"/></sensor>
</mujoco>"""
    if ext == ".urdf":
        text = _urdf_to_mjcf(path)
        base = os.path.dirname(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        base = os.path.dirname(path)
    root = ET.fromstring(text)
    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler")
        root.insert(0, comp)
    for attr in ("meshdir", "texturedir", "assetdir"):
        v = comp.get(attr)
        if v and not os.path.isabs(v):
            comp.set(attr, os.path.normpath(os.path.join(base, v)))
    if comp.get("meshdir") is None and comp.get("assetdir") is None:
        comp.set("meshdir", base)
    wb = root.find("worldbody")
    if wb is None:
        wb = ET.SubElement(root, "worldbody")
    has_plane = any(g.get("type") == "plane" for g in wb.iter("geom"))
    if not has_plane:
        ET.SubElement(wb, "geom", name="ground_floor", type="plane", size="6 6 0.1", rgba="0.25 0.27 0.3 1",
                      contype="1", conaffinity="7")
    for top in list(wb.findall("body")):
        if top.find("joint") is None and top.find("freejoint") is None and b.import_mode == "standalone" and ext == ".urdf":
            ET.SubElement(top, "freejoint")
    if wb.find("./body[@name='target']") is None:
        tb = ET.SubElement(wb, "body", name="target", mocap="true", pos="1 0 0.002")
        ET.SubElement(tb, "geom", type="cylinder", size="0.09 0.002", rgba="0.2 0.95 0.4 0.55", contype="0", conaffinity="0")

    def find_body(name):
        for bd in root.iter("body"):
            if bd.get("name") == name:
                return bd
        return None

    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    if b.imu_body:
        bd = find_body(b.imu_body)
        if bd is None:
            raise ValueError(f"IMU body '{b.imu_body}' not found in imported model")
        ET.SubElement(bd, "site", name="imu", pos="0 0 0", size="0.01")
        ET.SubElement(sensor, "framequat", name="imu_quat", objtype="site", objname="imu")
        ET.SubElement(sensor, "gyro", name="imu_gyro", site="imu")
        ET.SubElement(sensor, "accelerometer", name="imu_accel", site="imu")
    if b.ball_body:
        ET.SubElement(sensor, "framepos", name="ball_pos", objtype="xbody", objname=b.ball_body)
    joints = [j for j in b.wheel_joints if j]
    if joints:
        act = root.find("actuator")
        if act is None:
            act = ET.SubElement(root, "actuator")
        tq = cfg.robot.stepper_holding_torque_nm
        for i, jn in enumerate(joints):
            ET.SubElement(act, "velocity", name=f"motor{i + 1}", joint=jn, kv=str(cfg.robot.stepper_kv),
                          ctrlrange=f"{-cfg.robot.stepper_max_speed_rads} {cfg.robot.stepper_max_speed_rads}",
                          forcerange=f"{-tq} {tq}")
    return ET.tostring(root, encoding="unicode")


def build_xml(cfg: AppConfig, structural: Optional[StructuralSample] = None) -> str:
    """Dispatch on ``cfg.model_type``."""
    if cfg.model_type == "validation_rig":
        return generate_validation_rig_xml(cfg)
    if cfg.model_type == "custom" and cfg.custom.path:
        return import_custom_model(cfg)
    return generate_bb8_xml(cfg, structural)


def list_model_tree(model) -> List[dict]:
    """Body/geom hierarchy for the GUI inspector tree."""
    import mujoco
    out = []
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}"
        geoms = []
        for g in range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b]):
            geoms.append((g, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"))
        out.append({"id": b, "name": name, "parent": int(model.body_parentid[b]), "geoms": geoms})
    return out


if __name__ == "__main__":  # quick CLI: write the default model to disk
    from config_and_distributions import default_config
    c = default_config()
    with open("bb8_default.xml", "w") as fh:
        fh.write(generate_bb8_xml(c))
    print("wrote bb8_default.xml")
