"""
gui_addons.py - user-friendliness add-ons for main_gui.py
=========================================================

* ``TOOLTIPS`` / ``tooltip_for``    - help text for every configuration field
* ``Overlays``                      - 3D overlays drawn into the MuJoCo scene (push arrows, ball trail,
                                      CoG markers, velocity / target lines) + contact visualisation flags
* ``CAMERA_PRESETS`` / ``apply_camera_preset``
* ``render_offscreen`` / ``export_timeline_video`` - high-res screenshots and MP4 / GIF episode videos
* ``VerificationDialog``            - real-world verification examples (predict / import / compare / apply)
* ``PluginManagerDialog``           - enable / configure plugins
* ``QUICK_START`` / ``SHORTCUTS``   - Help menu texts
"""
from __future__ import annotations

import collections
import json
import math
import os
from typing import Callable, Dict, List, Optional

import mujoco
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

# ============================================================================
# Tooltips  (key = "<DataclassName>.<field>"  or a DistributionWidget row label)
# ============================================================================
TOOLTIPS: Dict[str, str] = {
    # robot geometry
    "RobotGeometryConfig.wheel_diameter_m": "Outer diameter of the omniwheels (58 mm stock). Changes the transmission ratio R/r.",
    "RobotGeometryConfig.wheel_width_m": "Axial width of the omniwheel (two roller rows).",
    "RobotGeometryConfig.rollers_per_row": "Rollers per row for the multi-roller contact model (visual in the virtual model).",
    "RobotGeometryConfig.omniwheel_model": "Virtual = exact omniwheel kinematics, fast (use for training).\n"
                                           "Multi-roller = real roller contacts, slower, for fidelity checks.",
    "RobotGeometryConfig.ball_radius_m": "Ball radius (your ball: 0.25 m = 50 cm diameter).",
    "RobotGeometryConfig.ball_inertia_factor": "I = k*m*R^2. 0.4 = solid sphere, 0.667 = thin shell. A styrofoam core "
                                               "with a heavy epoxy shell is ~0.55-0.65.",
    "RobotGeometryConfig.ball_mesh_subdivisions": "Tessellation of the dented ball mesh (3 -> 642 vertices).",
    "RobotGeometryConfig.head_mass_kg": "Total head mass incl. motors, wheels, Jetson, battery (geometry/inertia).",
    "RobotGeometryConfig.head_cog_height_m": "Height of the head's centre of gravity above the TOP of the ball. "
                                             "The most important balancing parameter - measure it (Tools -> verification).",
    "RobotGeometryConfig.head_radius_m": "Visual head dome radius.",
    "RobotGeometryConfig.imu_height_m": "Height of the Arduino IMU above the ball top (affects measured acceleration).",
    "RobotGeometryConfig.stepper_holding_torque_nm": "Datasheet holding torque (JK42HS40-1704-83AF: 0.41 N*m).",
    "RobotGeometryConfig.tmc2208_microsteps": "Microsteps per full step (TMC2208 MRES). Used for step-rate output and quantisation.",
    "RobotGeometryConfig.stepper_kv": "Stiffness of the stepper velocity servo [N*m per rad/s]. Higher = stiffer, "
                                      "like a well-tuned stepper that does not lose steps.",
    "RobotGeometryConfig.stepper_corner_speed_rads": "Speed above which the stepper's pull-out torque starts to drop.",
    "RobotGeometryConfig.stepper_max_speed_rads": "Maximum commanded wheel speed (= action 1.0).",
    "RobotGeometryConfig.stepper_torque_at_max_frac": "Fraction of holding torque left at max speed.",
    "RobotGeometryConfig.physics_timestep_s": "MuJoCo physics step (0.002 s = 500 Hz).",
    "RobotGeometryConfig.control_hz": "Policy / Jetson control loop rate. Must match the real robot.",
    # terrain
    "TerrainConfig.map_name": "Flat ground or an arena with walls and obstacles.",
    "TerrainConfig.off_ideal_nominal": "Ignore all randomisation: every parameter uses its nominal value.",
    "TerrainConfig.variable_grip": "Floor split into tiles with different friction (carpet / tiles / wood).",
    "TerrainConfig.rough_heightfield": "Procedural bumpy floor, regenerated every episode.",
    "TerrainConfig.walls": "Arena boundary walls.",
    "TerrainConfig.ramps": "Gentle ridges to drive over.",
    "TerrainConfig.bumps_stairs": "Cable-like bumps and small steps.",
    "TerrainConfig.arena_half_size_m": "Half the side length of the arena.",
    "TerrainConfig.rough_amplitude_m": "Peak-to-peak height of the rough floor.",
    "TerrainConfig.ramp_slope_deg": "Slope of the ramps.",
    "TerrainConfig.bump_height_m": "Height of the speed bumps.",
    "TerrainConfig.stair_step_m": "Height of one stair step.",
    "TerrainConfig.layout_seed": "Random seed for obstacle / ramp placement.",
    "TerrainConfig.dent_count": "Number of dents / bumps on the ball.",
    "BallRandomization.dent_count": "Number of dents / bumps on the ball surface.",
    # disturbance
    "DisturbanceConfig.enabled": "Random pushes during episodes (trains disturbance rejection).",
    "DisturbanceConfig.target_body": "Push the head or the ball.",
    # episode
    "EpisodeConfig.auto_reset_on_fall": "Start a new episode automatically when the robot falls.",
    "EpisodeConfig.fall_tilt_deg": "Head tilt that counts as a fall (episode ends with a penalty).",
    "EpisodeConfig.detect_contact_loss": "Also end the episode if the head would lift off the ball.",
    "EpisodeConfig.max_episode_s": "Episode length limit.",
    "EpisodeConfig.spawn_x_m": "Start position X (0 = arena centre, useful to see directional drift).",
    "EpisodeConfig.spawn_y_m": "Start position Y.",
    "EpisodeConfig.target_reach_radius_m": "Distance at which a waypoint counts as reached.",
    # network / training
    "NetworkConfig.init": "Weight initialisation of the cold-start network.",
    "NetworkConfig.action_mode": "Velocity = stepper step-rate commands (matches the hardware). Torque = research only.",
    "TrainingConfig.algorithm": "PPO is robust and the default. SAC is more sample-efficient but needs tuning.",
    "TrainingConfig.n_envs": "Parallel simulations. ~1-2 per CPU core.",
    "TrainingConfig.vec_env": "Subproc = one process per env (fast on multi-core).",
    "TrainingConfig.total_timesteps": "Training length in control steps (100 steps = 1 simulated second).",
    "TrainingConfig.learning_rate": "Adam learning rate.",
    "TrainingConfig.n_steps": "PPO rollout length per env before each update.",
    "TrainingConfig.batch_size": "Minibatch size of the PPO update.",
    "TrainingConfig.gamma": "Discount factor. 0.995 ~ 2 s planning horizon at 100 Hz.",
    "TrainingConfig.autosave_every_steps": "Checkpoint interval in steps.",
    "TrainingConfig.autosave_every_minutes": "Checkpoint interval in minutes (whichever comes first).",
    "TrainingConfig.seed": "Random seed (reproducibility).",
    "TrainingConfig.warm_start": "Behaviour cloning of the classical controller before RL: the network starts out "
                                 "already balancing and RL improves it. Strongly recommended.",
    "TrainingConfig.curriculum": "Start with nominal physics and widen the randomisation automatically as the "
                                 "policy gets good (difficulty 0 -> 1).",
    "TrainingConfig.curriculum_threshold": "Mean survival (fraction of the episode) needed to raise the difficulty.",
    "TrainingConfig.curriculum_step": "Difficulty increment per curriculum stage.",
    "TrainingConfig.bc_samples": "Teacher samples collected for behaviour cloning.",
    # reward
    "RewardWeights.alive": "Reward per step for not having fallen.",
    "RewardWeights.upright": "Reward for small head tilt.",
    "RewardWeights.ang_vel": "Penalty on head angular velocity (smoothness).",
    "RewardWeights.action": "Penalty on command magnitude (energy).",
    "RewardWeights.action_rate": "Penalty on command changes (avoids jitter / missed steps).",
    "RewardWeights.drift": "Penalty on distance from the spawn point (no target).",
    "RewardWeights.ball_velocity": "Penalty on ball speed (stops slow cruising away).",
    "RewardWeights.target_distance": "Penalty on distance to the waypoint.",
    "RewardWeights.target_progress": "Reward for getting closer to the waypoint.",
    "RewardWeights.target_reached_bonus": "Bonus while inside the reach radius.",
    "RewardWeights.fall_penalty": "One-off penalty when the robot falls.",
}
ROW_TOOLTIPS: Dict[str, str] = {
    "Mass": "Ball mass. Weigh your ball.",
    "Sliding grip μ": "Ball-floor sliding friction coefficient.",
    "Torsional friction": "Resistance to spinning about the vertical axis.",
    "Rolling resistance": "Rolling resistance (MuJoCo length unit). Identify it with the ramp roll-out experiment.",
    "Non-sphericity": "Random ellipsoidal deformation (fraction of the radius).",
    "Dent / bump depth": "Maximum depth of dents / height of bumps of the hand-coated shell.",
    "CoM offset |d|": "Ball centre-of-mass offset (uneven coating).",
    "Longitudinal grip μ": "Omniwheel grip in the driving direction (limits transmissible torque -> slip).",
    "Lateral roller resistance": "Friction of the free-spinning rollers.",
    "Total head mass": "Randomised head mass incl. everything mounted.",
    "CoG offset dx": "Head CoG offset forward/back (unbalanced battery etc.).",
    "CoG offset dy": "Head CoG offset sideways.",
    "CoG offset dz": "Head CoG height error.",
    "Voltage sag torque scale": "Battery sag -> less motor torque.",
    "Stepper deadband": "Commands below this speed produce no motion.",
    "Command latency": "Delay of the command in control ticks (I2C/UART/compute).",
    "Servo stiffness scale": "Variation of the stepper servo stiffness.",
    "Ground grip factor": "Multiplies the ball-floor grip (per tile with 'Variable grip').",
    "Interval between pushes": "Time between automatic pushes.",
    "Force magnitude": "Push force.",
    "Direction azimuth": "Push direction around the vertical axis (0 = +x = robot forward at spawn).",
    "Direction elevation": "Push direction above the horizontal.",
    "Push duration": "How long each push lasts.",
    "Initial tilt": "Random head tilt at the start of each episode.",
    "Initial yaw": "Heading at the start of each episode.",
}


def tooltip_for(obj, attr: str) -> str:
    return TOOLTIPS.get(f"{type(obj).__name__}.{attr}", "")


# ============================================================================
# 3D overlays
# ============================================================================
def _add_geom(scn, gtype, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    scn.ngeom += 1
    return g


def _add_connector(scn, gtype, width, a, b, rgba):
    g = _add_geom(scn, gtype, np.zeros(3), np.zeros(3), np.eye(3), rgba)
    if g is not None:
        mujoco.mjv_connector(g, gtype, width, np.asarray(a, float), np.asarray(b, float))
    return g


class Overlays:
    """Extra scene geometry drawn on top of the MuJoCo scene (visualisation only)."""

    def __init__(self):
        self.flags = {"push": True, "trail": True, "cog": False, "velocity": False, "target_line": True,
                      "contacts": False}
        self.trail: collections.deque = collections.deque(maxlen=400)

    def clear(self):
        self.trail.clear()

    def record(self, ball_pos: np.ndarray):
        if not self.trail or np.linalg.norm(self.trail[-1][:2] - ball_pos[:2]) > 0.01:
            self.trail.append(np.array([ball_pos[0], ball_pos[1], 0.004]))

    def apply_option_flags(self, opt: mujoco.MjvOption):
        on = self.flags.get("contacts", False)
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = on
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = on

    def draw(self, scn, model, data, env=None):
        f = self.flags
        if f.get("trail") and len(self.trail) > 1:
            pts = list(self.trail)
            n = len(pts)
            for i in range(1, n):
                a = 0.25 + 0.75 * i / n
                _add_connector(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, pts[i - 1], pts[i], (0.18, 0.64, 1.0, a))
        if env is None or not hasattr(env, "head_bid"):
            return
        head_c = data.xipos[env.head_bid]
        ball_c = data.xpos[env.ball_bid]
        if f.get("push"):  # xfrc_applied holds built-in pushes + plugin forces (wind, ...)
            for bid, centre in ((env.head_bid, head_c), (env.ball_bid, ball_c)):
                force = data.xfrc_applied[bid, :3]
                nrm = float(np.linalg.norm(force))
                if nrm > 0.05:
                    L = min(0.6, 0.03 * nrm)
                    _add_connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, 0.02, centre - force / nrm * (L + 0.05),
                                   centre - force / nrm * 0.05, (1.0, 0.2, 0.15, 1.0))
        if f.get("cog"):
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.022, 0, 0], head_c, np.eye(3), (1, 0.1, 0.9, 1))
            com = data.subtree_com[env.ball_bid]
            _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.018, 0, 0], com, np.eye(3), (1, 1, 0.1, 1))
            _add_connector(scn, mujoco.mjtGeom.mjGEOM_LINE, 2, com, [com[0], com[1], 0.002], (1, 1, 0.1, 0.9))
            _add_connector(scn, mujoco.mjtGeom.mjGEOM_LINE, 2, ball_c, head_c, (1, 0.1, 0.9, 0.8))
        if f.get("velocity"):
            v = data.qvel[env.ball_dadr:env.ball_dadr + 3].copy()
            v[2] = 0
            sp = float(np.linalg.norm(v))
            if sp > 0.02:
                base = np.array([ball_c[0], ball_c[1], 0.01])
                _add_connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, 0.015, base, base + v * 0.8, (0.2, 1, 0.4, 1))
        if f.get("target_line") and getattr(env, "target_enabled", False):
            t = np.array([env.target[0], env.target[1], 0.006])
            _add_connector(scn, mujoco.mjtGeom.mjGEOM_LINE, 2, [ball_c[0], ball_c[1], 0.006], t, (0.2, 0.95, 0.4, 0.9))


# ============================================================================
# Camera presets
# ============================================================================
CAMERA_PRESETS = {
    "Follow (default)": dict(distance=2.4, azimuth=150, elevation=-14, follow=True),
    "Front (eye view)": dict(distance=1.6, azimuth=180, elevation=-6, follow=True),
    "Side": dict(distance=1.8, azimuth=90, elevation=-6, follow=True),
    "Top-down": dict(distance=3.0, azimuth=90, elevation=-89, follow=True),
    "Wheel close-up": dict(distance=0.45, azimuth=200, elevation=-25, follow=False, look="wheel1"),
    "Arena overview": dict(distance=7.0, azimuth=135, elevation=-40, follow=False, look=None),
}


def apply_camera_preset(view, name: str) -> None:
    p = CAMERA_PRESETS[name]
    cam, model, data = view.cam, view.model, view.data
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = p["distance"], p["azimuth"], p["elevation"]
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball") if model is not None else -1
    if ball < 0 and model is not None:
        ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "test_ball")
    if p.get("follow") and ball >= 0:
        view.follow_body = ball
        cam.lookat[:] = data.xpos[ball] + np.array([0, 0, 0.12])
    else:
        view.follow_body = -1
        look = p.get("look")
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, look) if (look and model is not None) else -1
        cam.lookat[:] = data.xpos[bid] if bid >= 0 else np.array([0, 0, 0.3])
    view.render_now()


# ============================================================================
# Screenshots & videos
# ============================================================================
def render_offscreen(model, data, cam, width=1920, height=1080, opt=None, overlay_fn=None) -> np.ndarray:
    g = model.vis.global_
    g.offwidth, g.offheight = max(g.offwidth, width), max(g.offheight, height)
    r = mujoco.Renderer(model, height, width)
    try:
        r.update_scene(data, camera=cam, scene_option=opt)
        if overlay_fn:
            overlay_fn(r.scene)
        return r.render().copy()
    finally:
        r.close()


def save_png(rgb: np.ndarray, path: str) -> str:
    h, w, _ = rgb.shape
    QtGui.QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy().save(path)
    return path


def export_timeline_video(model, frames: List[dict], cam, path: str, fps: int = 30, size=(1280, 720),
                          opt=None, follow_body: int = -1, progress: Optional[Callable[[int, int], bool]] = None,
                          overlay_flags: Optional[dict] = None) -> str:
    """Render recorded timeline frames (qpos/mocap) into an MP4 (imageio-ffmpeg) or GIF (Pillow).
    Falls back to a numbered PNG sequence if no encoder is installed."""
    data = mujoco.MjData(model)
    w, h = size
    g = model.vis.global_
    g.offwidth, g.offheight = max(g.offwidth, w), max(g.offheight, h)
    r = mujoco.Renderer(model, h, w)
    t0, t1 = frames[0]["t"], frames[-1]["t"]
    times = np.arange(t0, t1, 1.0 / fps)
    ts = np.array([f["t"] for f in frames])
    trail = collections.deque(maxlen=400)
    cam = _copy_cam(cam)
    ext = os.path.splitext(path)[1].lower()
    writer = None
    images = []
    try:
        if ext == ".mp4":
            import imageio.v2 as iio
            writer = iio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    except Exception:
        path = os.path.splitext(path)[0] + ".gif"
        ext = ".gif"
    try:
        for k, t in enumerate(times):
            fr = frames[int(min(len(frames) - 1, np.searchsorted(ts, t)))]
            if len(fr["qpos"]) != model.nq:
                continue
            data.qpos[:] = fr["qpos"]
            if len(fr["mocap"]) == len(data.mocap_pos):
                data.mocap_pos[:] = fr["mocap"]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_camlight(model, data)
            if follow_body >= 0:
                p = data.xpos[follow_body]
                cam.lookat[:] = 0.8 * cam.lookat + 0.2 * np.array([p[0], p[1], p[2] + 0.12])
                trail.append(np.array([p[0], p[1], 0.004]))
            r.update_scene(data, camera=cam, scene_option=opt)
            if (overlay_flags or {}).get("trail", True) and len(trail) > 1:
                pts = list(trail)
                for i in range(1, len(pts)):
                    _add_connector(r.scene, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, pts[i - 1], pts[i],
                                   (0.18, 0.64, 1.0, 0.25 + 0.75 * i / len(pts)))
            img = r.render().copy()
            if writer is not None:
                writer.append_data(img)
            else:
                images.append(img)
            if progress and not progress(k + 1, len(times)):
                break
    finally:
        r.close()
        if writer is not None:
            writer.close()
    if writer is None:
        try:
            from PIL import Image
            pil = [Image.fromarray(im).resize((min(w, 720), int(h * min(w, 720) / w))) for im in images]
            if not pil:
                raise ValueError("no frames")
            pil[0].save(path, save_all=True, append_images=pil[1:], duration=int(1000 / fps), loop=0)
        except ImportError:
            d = os.path.splitext(path)[0] + "_frames"
            os.makedirs(d, exist_ok=True)
            for i, im in enumerate(images):
                save_png(im, os.path.join(d, f"frame_{i:05d}.png"))
            path = d
    return path


def _copy_cam(cam):
    c = mujoco.MjvCamera()
    c.type, c.distance, c.azimuth, c.elevation = cam.type, cam.distance, cam.azimuth, cam.elevation
    c.lookat[:] = cam.lookat
    return c


# ============================================================================
# Verification dialog
# ============================================================================
class VerificationDialog(QtWidgets.QDialog):
    """Real-world verification examples: procedure, simulator prediction, measurement, verdict, apply."""

    def __init__(self, main_window):
        super().__init__(main_window)
        import verification_examples as V
        self.V = V
        self.mw = main_window
        self.setWindowTitle("Real-world verification examples")
        self.resize(1100, 720)
        self.pred: Dict[str, float] = {}
        lay = QtWidgets.QHBoxLayout(self)
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumWidth(330)
        for ex in V.CATALOG:
            it = QtWidgets.QListWidgetItem(f"{ex.title}\n   {ex.category} · {ex.minutes} min")
            it.setData(Qt.UserRole, ex.key)
            self.list.addItem(it)
        lay.addWidget(self.list)
        right = QtWidgets.QVBoxLayout()
        self.text = QtWidgets.QTextBrowser()
        right.addWidget(self.text, 3)
        g = QtWidgets.QGroupBox("Your measurement")
        self.form = QtWidgets.QFormLayout(g)
        right.addWidget(g)
        hb = QtWidgets.QHBoxLayout()
        self.btn_pred = QtWidgets.QPushButton("Compute simulator prediction")
        self.btn_import = QtWidgets.QPushButton("Import runner result (.json)…")
        self.btn_cmp = QtWidgets.QPushButton("Compare")
        self.btn_cmp.setObjectName("accent")
        self.btn_apply = QtWidgets.QPushButton("Apply to configuration")
        for b in (self.btn_pred, self.btn_import, self.btn_cmp, self.btn_apply):
            hb.addWidget(b)
        right.addLayout(hb)
        self.result = QtWidgets.QTableWidget(0, 5)
        self.result.setHorizontalHeaderLabels(["Quantity", "Simulator", "Measured", "Error", "Verdict"])
        self.result.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self.result, 2)
        hb2 = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Export printable checklist (.md)…")
        b.clicked.connect(self.export_sheet)
        hb2.addWidget(b)
        hb2.addStretch(1)
        right.addLayout(hb2)
        lay.addLayout(right, 1)
        self.list.currentRowChanged.connect(self.show_example)
        self.btn_pred.clicked.connect(self.compute_prediction)
        self.btn_import.clicked.connect(self.import_json)
        self.btn_cmp.clicked.connect(self.compare)
        self.btn_apply.clicked.connect(self.apply)
        self.list.setCurrentRow(0)

    @property
    def ex(self):
        return self.V.BY_KEY[self.list.currentItem().data(Qt.UserRole)]

    def show_example(self, _row):
        ex = self.ex
        steps = "".join(f"<li>{s}</li>" for s in ex.steps)
        runner = (f"<p><b>Jetson runner:</b> <code>python3 jetson_control_loop.py --test {ex.runner_test}</code> "
                  f"→ import the JSON it writes.</p>") if ex.runner_test else ""
        safety = f"<p style='color:#e5484d'><b>Safety:</b> {ex.safety}</p>" if ex.safety else ""
        self.text.setHtml(f"<h2>{ex.title}</h2><p><i>{ex.category} · ~{ex.minutes} min</i> — {ex.why}</p>"
                          f"<p><b>Equipment:</b> {', '.join(ex.equipment)}</p><ol>{steps}</ol>{runner}{safety}")
        while self.form.rowCount():
            self.form.removeRow(0)
        self.inputs = {}
        for f in ex.fields:
            if f.key.endswith(("_positive", "_ccw", "_down")):
                w = QtWidgets.QCheckBox("yes")
            else:
                w = QtWidgets.QDoubleSpinBox()
                w.setRange(-1e6, 1e6)
                w.setDecimals(4)
                w.setSpecialValueText("—")
                w.setMinimum(-1e6)
                w.setValue(-1e6)  # = empty
                w.setSuffix(f" {f.unit}" if f.unit else "")
            self.form.addRow(f.label, w)
            self.inputs[f.key] = w
        self.pred = {}
        self.result.setRowCount(0)
        self.btn_apply.setEnabled(ex.apply is not None)
        self.btn_apply.setText(ex.apply_label or "Apply to configuration")
        self.btn_import.setEnabled(bool(ex.runner_test))

    def _measured(self) -> Dict[str, float]:
        out = {}
        for k, w in self.inputs.items():
            if isinstance(w, QtWidgets.QCheckBox):
                out[k] = w.isChecked()
            elif w.value() > -1e5:
                out[k] = w.value()
        return out

    def compute_prediction(self):
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.pred = self.ex.predict(self.mw.cfg)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.compare(prediction_only=not self._measured())

    def import_json(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Runner result", "", "JSON (*.json)")
        if not p:
            return
        key, meas = self.V.import_runner_json(p)
        if key in self.V.BY_KEY and key != self.ex.key:
            for i in range(self.list.count()):
                if self.list.item(i).data(Qt.UserRole) == key:
                    self.list.setCurrentRow(i)
        for k, v in meas.items():
            w = self.inputs.get(k)
            if isinstance(w, QtWidgets.QCheckBox):
                w.setChecked(bool(v))
            elif w is not None:
                w.setValue(float(v))
        self.compare()

    def compare(self, prediction_only: bool = False):
        if not self.pred:
            self.compute_prediction()
            return
        rows = self.V.compare(self.ex, {} if prediction_only else self._measured(), self.pred)
        self.result.setRowCount(len(rows))
        for i, r in enumerate(rows):
            fmt = lambda v: "—" if v is None else (str(v) if isinstance(v, bool) else f"{v:.4g}")
            err = r.get("error")
            err_s = "—" if err is None else (f"{err * 100:+.1f} %" if r.get("rel") else f"{err:+.3g}")
            verdict = "—" if r["ok"] is None else ("PASS" if r["ok"] else "CHECK")
            for j, v in enumerate((r["field"], fmt(r["predicted"]), fmt(r["measured"]), err_s, verdict)):
                it = QtWidgets.QTableWidgetItem(v)
                if j == 4 and r["ok"] is not None:
                    it.setForeground(QtGui.QColor("#3ccf7e" if r["ok"] else "#e5484d"))
                self.result.setItem(i, j, it)
        self.result.resizeColumnsToContents()

    def apply(self):
        ex = self.ex
        meas = self._measured()
        if not meas:
            QtWidgets.QMessageBox.information(self, "Apply", "Enter (or import) your measurement first.")
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            changes = ex.apply(self.mw.cfg, meas)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if changes:
            self.mw.config_applied_externally("\n".join(changes))
            self.pred = {}
            QtWidgets.QMessageBox.information(self, "Configuration updated", "\n".join(changes))

    def export_sheet(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Checklist", "bb8_verification_checklist.md", "Markdown (*.md)")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.V.checklist_markdown(self.mw.cfg, with_predictions=True))


# ============================================================================
# Plugin manager
# ============================================================================
class PluginManagerDialog(QtWidgets.QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)
        import plugins as P
        self.P = P
        self.mw = main_window
        self.setWindowTitle("Plugin Manager")
        self.resize(900, 560)
        lay = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(f"Plugin folders: {'; '.join(P.plugin_dirs(main_window.cfg))}<br>"
                                "Enabled plugins are loaded in the viewport <b>and</b> in every training process.")
        info.setWordWrap(True)
        lay.addWidget(info)
        sp = QtWidgets.QSplitter()
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Enabled", "Plugin", "Description"])
        self.table.horizontalHeader().setStretchLastSection(True)
        sp.addWidget(self.table)
        self.settings = QtWidgets.QTableWidget(0, 3)
        self.settings.setHorizontalHeaderLabels(["Setting", "Value", "Description"])
        self.settings.horizontalHeader().setStretchLastSection(True)
        sp.addWidget(self.settings)
        sp.setSizes([480, 420])
        lay.addWidget(sp, 1)
        hb = QtWidgets.QHBoxLayout()
        for txt, fn in (("Open plugins folder", self.open_folder), ("Reload", self.reload)):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            hb.addWidget(b)
        hb.addStretch(1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept_changes)
        bb.rejected.connect(self.reject)
        hb.addWidget(bb)
        lay.addLayout(hb)
        self.table.currentCellChanged.connect(lambda r, *_: self.show_settings(r))
        self.settings_values: Dict[str, Dict[str, object]] = {k: dict(v) for k, v in main_window.cfg.plugins.settings.items()}
        self.reload()

    def reload(self):
        self.found = self.P.discover_plugins(self.mw.cfg)
        enabled = set(self.mw.cfg.plugins.enabled)
        self.table.setRowCount(len(self.found))
        for i, p in enumerate(self.found):
            cb = QtWidgets.QTableWidgetItem()
            cb.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            cb.setCheckState(Qt.Checked if p["file"] in enabled else Qt.Unchecked)
            self.table.setItem(i, 0, cb)
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{p.get('name', p['file'])} {p.get('version', '')}"))
            d = QtWidgets.QTableWidgetItem(p.get("description", ""))
            d.setToolTip(p.get("error", p["path"]))
            self.table.setItem(i, 2, d)
        self.table.resizeColumnsToContents()
        if self.found:
            self.table.setCurrentCell(0, 1)

    def _store_settings(self):
        name = getattr(self, "_settings_for", None)
        if not name:
            return
        vals = self.settings_values.setdefault(name, {})
        for r in range(self.settings.rowCount()):
            key = self.settings.item(r, 0).text()
            txt = self.settings.item(r, 1).text()
            try:
                vals[key] = json.loads(txt)
            except ValueError:
                vals[key] = txt

    def show_settings(self, row):
        self._store_settings()
        if row < 0 or row >= len(self.found):
            return
        p = self.found[row]
        name = os.path.splitext(p["file"])[0]
        self._settings_for = name
        # discover declared settings by registering into a scratch registry
        reg = self.P.PluginRegistry()
        try:
            mod = self.P._import_file(p["path"])
            mod.register(self.P.PluginAPI(reg, name, None))
        except Exception as e:
            reg.settings = {}
            self.settings.setRowCount(1)
            self.settings.setItem(0, 0, QtWidgets.QTableWidgetItem("error"))
            self.settings.setItem(0, 1, QtWidgets.QTableWidgetItem(str(e)))
            return
        decl = reg.settings.get(name, {})
        cur = self.settings_values.get(name, {})
        self.settings.setRowCount(len(decl))
        for i, (k, (default, doc)) in enumerate(decl.items()):
            it = QtWidgets.QTableWidgetItem(k)
            it.setFlags(Qt.ItemIsEnabled)
            self.settings.setItem(i, 0, it)
            self.settings.setItem(i, 1, QtWidgets.QTableWidgetItem(json.dumps(cur.get(k, default))))
            d = QtWidgets.QTableWidgetItem(doc)
            d.setFlags(Qt.ItemIsEnabled)
            self.settings.setItem(i, 2, d)
        self.settings.resizeColumnsToContents()

    def open_folder(self):
        d = self.P.plugin_dirs(self.mw.cfg)[0]
        os.makedirs(d, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(d))

    def accept_changes(self):
        self._store_settings()
        enabled = [self.found[i]["file"] for i in range(len(self.found))
                   if self.table.item(i, 0).checkState() == Qt.Checked]
        self.mw.cfg.plugins.enabled = enabled
        self.mw.cfg.plugins.settings = {k: v for k, v in self.settings_values.items() if v}
        self.accept()


# ============================================================================
# Help texts
# ============================================================================
QUICK_START = """
<h2>Quick start</h2>
<ol>
<li><b>Describe your robot</b>: <i>Tools → Real-world verification examples → 1. Measure & enter your robot</i>
    (ball diameter, masses, head CoG, wheel size, zenith angle) → <i>Apply</i>.</li>
<li><b>Check the physics</b> in the viewport: policy <i>Classical controller (baseline)</i> must balance.
    Press <b>Push</b> to disturb it.</li>
<li><b>Train</b>: <i>Network &amp; Training → ▶ Start</i>. The default pipeline first clones the classical
    controller (warm start, ~1–2 min), then PPO improves it while the curriculum widens the randomisation.
    Use <b>Headless Turbo</b> for speed; watch <i>Training Progress</i>.</li>
<li><b>Inspect</b> the best policy: policy source <i>Best policy</i>, pause and scrub the timeline,
    export a video (<i>File → Export → Video</i>).</li>
<li><b>Deploy</b>: <i>File → Export → Jetson deployment package</i> and follow its DEPLOY.md.
    Do the verification examples on the real robot before the first closed-loop run.</li>
</ol>
<p>Every field has a tooltip. Your configuration, window layout and theme are remembered between sessions;
<i>File → New</i> restores factory defaults.</p>
"""

SHORTCUTS = [
    ("Esc", "Emergency stop: kill training workers, stop the simulation, optional factory reset"),
    ("Space", "Pause / run the simulation"),
    ("Ctrl+R", "Reset episode"),
    ("Ctrl+P", "Push the robot"),
    ("Ctrl+S", "Save configuration preset"),
    ("Ctrl+O", "Open configuration preset"),
    ("Ctrl+N", "New (factory defaults)"),
    ("F12", "Screenshot"),
    ("Alt+1 … Alt+6", "Camera presets (follow, front, side, top, wheel, arena)"),
    ("Mouse: left drag", "Rotate camera (Shift: horizontal)"),
    ("Mouse: right drag", "Pan camera"),
    ("Mouse: wheel", "Zoom"),
    ("Double-click", "Select part (inspector)"),
    ("Ctrl+click / drag", "Set the waypoint target"),
]
