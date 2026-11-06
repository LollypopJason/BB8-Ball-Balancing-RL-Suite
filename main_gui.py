"""
main_gui.py
===========

BB-8 RL Suite & Deployment Studio - native PySide6 (Qt6) desktop application.

* Embedded MuJoCo OpenGL viewport (MuJoCo's own GL context renders off-screen and the
  frame is blitted into a native Qt widget -> no FBO/context sharing issues across
  Windows / Linux / macOS, no browser, no web server).
* Three execution modes: Single Interactive (60 FPS, mouse camera, live target
  dragging, 3D part selection), Parallel Visual Grid (up to 8 agents), Headless Turbo.
* Hardware resource governor (CPU-core %, RAM ceiling -> checkpoint).
* Model switcher: Validation Rig / Parametric BB-8 / Custom import (MJCF, URDF, STL, OBJ).
* Interactive part inspector with emissive highlight + live property & XML editor.
* Domain-randomisation editor (Constant / Uniform / Normal for every parameter).
* Training session controls (Start / Pause / Resume / Stop / Save / Load), live plots.
* Timeline scrubber synchronised with the 3D view and telemetry cursors.
* One-click exports: best .onnx, C++ policy (.cpp/.h), telemetry .csv, experiment .zip.
* ESC / "Reset & Kill Workers": immediate halt of all workers + optional factory reset.

Mouse (viewport): left-drag rotate, right-drag pan, wheel / middle-drag zoom,
double-click select part, Ctrl+click / Ctrl+drag set the live waypoint target.
"""
from __future__ import annotations

import multiprocessing

if __name__ == "__main__":
    # Frozen (PyInstaller .exe) child processes must be intercepted before Qt / torch are imported.
    multiprocessing.freeze_support()

import collections
import csv
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer, Signal

import mujoco
import pyqtgraph as pg

from config_and_distributions import (APP_NAME, APP_VERSION, DIST_MODES, OBS_GROUPS, PLOT_PENS, THEMES, AppConfig,
                                      DistributionParameter, apply_ideal_preset, create_experiment_bundle,
                                      default_config, ensure_workspace, make_stylesheet, timestamp)
from bb8_gym_env import BB8BalanceEnv, ValidationRigSim, quat_to_rpy, wrap_pi
from mjcf_generator import import_custom_model, list_model_tree
from rl_onnx_cpp_engine import (PolicyRunner, PolicySnapshot, TrainingController, allowed_cores,
                                ensure_initial_policy, export_onnx, generate_cpp, load_onnx_snapshot,
                                load_session, process_tree_rss_mb, save_session)

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


# ============================================================================
# MuJoCo viewport widget
# ============================================================================
class MjViewport(QtWidgets.QWidget):
    bodySelected = Signal(int, int)        # body id, geom id
    targetPicked = Signal(float, float)

    def __init__(self, parent=None, interactive: bool = True, label: str = "", max_size=(1920, 1200)):
        super().__init__(parent)
        self.setMinimumSize(160, 120)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.renderer: Optional[mujoco.Renderer] = None
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.interactive = interactive
        self.label = label
        self.overlay = ""
        self.highlight_geoms: set = set()
        self.follow_body = -1
        self.max_size = max_size
        self.image: Optional[QtGui.QImage] = None
        self.error: Optional[str] = None
        self.enabled = True
        self.hfield_upload: Optional[int] = None
        self._press = None
        self._btn = None

    # ---- model / renderer ---------------------------------------------
    def set_model(self, model, data, keep_camera: bool = False, follow_body: int = -1) -> None:
        self.release()
        self.model, self.data = model, data
        self.follow_body = follow_body
        self.highlight_geoms = set()
        if not keep_camera and model is not None:
            mujoco.mjv_defaultFreeCamera(model, self.cam)
            self.cam.distance = 2.4
            self.cam.elevation = -14
            self.cam.azimuth = 150
            self.cam.lookat[:] = [0, 0, 0.36]
        self.error = None
        self.enabled = True

    def release(self) -> None:
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
        self.renderer = None

    def _ensure_renderer(self) -> bool:
        if self.model is None:
            return False
        dpr = self.devicePixelRatioF()
        w = max(64, min(int(self.width() * dpr), self.max_size[0]))
        h = max(48, min(int(self.height() * dpr), self.max_size[1]))
        w, h = w - w % 4, h - h % 2
        if self.renderer is not None and (self.renderer.width, self.renderer.height) == (w, h):
            return True
        self.release()
        g = self.model.vis.global_
        g.offwidth, g.offheight = max(g.offwidth, w), max(g.offheight, h)
        self.renderer = mujoco.Renderer(self.model, h, w)
        return True

    def render_now(self) -> None:
        if not self.enabled or self.model is None or not self.isVisible():
            self.update()
            return
        try:
            if not self._ensure_renderer():
                return
            if self.follow_body >= 0:
                p = self.data.xpos[self.follow_body]
                self.cam.lookat[:] = 0.85 * self.cam.lookat + 0.15 * np.array([p[0], p[1], p[2] + 0.12])
            r = self.renderer
            r.update_scene(self.data, camera=self.cam, scene_option=self.opt)
            if self.highlight_geoms:
                scn = r.scene
                for i in range(scn.ngeom):
                    g = scn.geoms[i]
                    if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in self.highlight_geoms:
                        g.rgba[:] = (1.0, 0.15, 0.85, 1.0)
                        g.emission = 0.9
            if self.hfield_upload is not None:
                try:
                    r._gl_context.make_current()
                    mujoco.mjr_uploadHField(self.model, r._mjr_context, self.hfield_upload)
                except Exception:
                    pass
                self.hfield_upload = None
            px = r.render()
            h, w, _ = px.shape
            self.image = QtGui.QImage(np.ascontiguousarray(px).data, w, h, 3 * w,
                                      QtGui.QImage.Format_RGB888).copy()
        except Exception as e:  # GL unavailable etc. -> keep app usable
            self.error = f"Renderer unavailable: {e}"
            self.enabled = False
            self.release()
        self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#0b0d10"))
        if self.image is not None:
            p.drawImage(self.rect(), self.image)
        p.setPen(QtGui.QColor("#ffb066"))
        f = p.font()
        f.setPointSize(9)
        p.setFont(f)
        text = "\n".join(t for t in (self.label, self.overlay) if t)
        if text:
            p.drawText(self.rect().adjusted(8, 6, -8, -6), Qt.AlignLeft | Qt.AlignTop, text)
        if self.error:
            p.setPen(QtGui.QColor("#ff6b6b"))
            p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.error)
        p.end()

    # ---- mouse --------------------------------------------------------
    def _select(self, x: float, y: float):
        if self.renderer is None or self.model is None:
            return -1, -1, None
        aspect = self.width() / max(1, self.height())
        relx, rely = x / max(1, self.width()), (self.height() - y) / max(1, self.height())
        selpnt = np.zeros(3)
        geomid = np.zeros(1, dtype=np.int32)
        flexid = np.zeros(1, dtype=np.int32)
        skinid = np.zeros(1, dtype=np.int32)
        body = mujoco.mjv_select(self.model, self.data, self.opt, aspect, relx, rely, self.renderer.scene,
                                 selpnt, geomid, flexid, skinid)
        return int(body), int(geomid[0]), selpnt

    def mousePressEvent(self, ev):
        self._press = ev.position()
        self._btn = ev.button()
        if self.interactive and ev.button() == Qt.LeftButton and ev.modifiers() & Qt.ControlModifier:
            self._pick_target(ev.position())

    def _pick_target(self, pos):
        b, g, pnt = self._select(pos.x(), pos.y())
        if g >= 0 and pnt is not None:
            self.targetPicked.emit(float(pnt[0]), float(pnt[1]))

    def mouseDoubleClickEvent(self, ev):
        if not self.interactive:
            return
        b, g, _ = self._select(ev.position().x(), ev.position().y())
        self.bodySelected.emit(b, g)

    def mouseMoveEvent(self, ev):
        if self._press is None or self.model is None or self.renderer is None:
            return
        pos = ev.position()
        if self.interactive and self._btn == Qt.LeftButton and ev.modifiers() & Qt.ControlModifier:
            self._pick_target(pos)
            return
        dx = (pos.x() - self._press.x()) / max(1, self.height())
        dy = (pos.y() - self._press.y()) / max(1, self.height())
        self._press = pos
        shift = bool(ev.modifiers() & Qt.ShiftModifier)
        if self._btn == Qt.LeftButton:
            act = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif self._btn == Qt.RightButton:
            act = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
            self.follow_body = -1 if self.interactive else self.follow_body
        else:
            act = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self.model, act, dx, dy, self.renderer.scene, self.cam)
        if not self.parent_is_running():
            self.render_now()

    def mouseReleaseEvent(self, ev):
        self._press = None
        self._btn = None

    def wheelEvent(self, ev):
        if self.model is None or self.renderer is None:
            return
        steps = ev.angleDelta().y() / 120.0
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * steps, self.renderer.scene, self.cam)
        if not self.parent_is_running():
            self.render_now()

    def parent_is_running(self) -> bool:
        w = self.window()
        return bool(getattr(w, "sim_running", False))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.model is not None and self.enabled:
            QTimer.singleShot(0, self.render_now)


# ============================================================================
# Grid of viewports (up to 8 agents)
# ============================================================================
class GridViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout_ = QtWidgets.QGridLayout(self)
        self.layout_.setSpacing(3)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.slots: List[Dict[str, Any]] = []

    def set_count(self, n: int) -> None:
        self.release()
        for s in self.slots:
            s["view"].setParent(None)
            s["view"].deleteLater()
        self.slots = []
        cols = 1 if n == 1 else 2 if n <= 4 else 4
        for i in range(n):
            v = MjViewport(self, interactive=False, label=f"Agent {i + 1}", max_size=(800, 600))
            self.layout_.addWidget(v, i // cols, i % cols)
            self.slots.append({"view": v, "model": None, "data": None, "version": None})

    def update_slot(self, i: int, state: dict, xml: Optional[str] = None, model=None, data=None) -> None:
        if i >= len(self.slots):
            return
        s = self.slots[i]
        try:
            if model is not None:
                if s["model"] is not model:
                    s["model"], s["data"] = model, data
                    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
                    s["view"].set_model(model, data, follow_body=bid)
            else:
                if xml is not None:
                    m = mujoco.MjModel.from_xml_string(xml)
                    d = mujoco.MjData(m)
                    s["model"], s["data"] = m, d
                    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
                    s["view"].set_model(m, d, follow_body=bid)
                if s["model"] is None or state is None:
                    return
                m, d = s["model"], s["data"]
                if len(state["qpos"]) != m.nq:
                    return
                d.qpos[:] = state["qpos"]
                d.mocap_pos[:] = state["mocap_pos"]
                if "hfield" in state and m.nhfield > 0 and len(state["hfield"]) == len(m.hfield_data):
                    m.hfield_data[:] = state["hfield"]
                    s["view"].hfield_upload = 0
                mujoco.mj_kinematics(m, d)
                mujoco.mj_camlight(m, d)
            if state is not None:
                s["view"].overlay = f"t={state.get('time', 0):.1f}s  tilt={state.get('tilt_deg', 0):.1f}°  " \
                                    f"R={state.get('return', 0):.0f}"
            s["view"].render_now()
        except Exception as e:
            s["view"].error = str(e)

    def release(self) -> None:
        for s in self.slots:
            s["view"].release()
            s["model"] = s["data"] = None


# ============================================================================
# Timeline recorder
# ============================================================================
class TimelineRecorder:
    def __init__(self, max_frames: int = 6000):
        self.frames: collections.deque = collections.deque(maxlen=max_frames)
        self._cache = None

    def clear(self):
        self.frames.clear()
        self._cache = None

    def add(self, t: float, qpos, mocap, vals: Dict[str, float]):
        self.frames.append({"t": t, "qpos": np.array(qpos), "mocap": np.array(mocap), "vals": vals})
        self._cache = None

    def __len__(self):
        return len(self.frames)

    def arrays(self) -> Dict[str, np.ndarray]:
        if self._cache is None:
            if not self.frames:
                self._cache = {"t": np.zeros(0)}
            else:
                keys = list(self.frames[-1]["vals"].keys())
                out = {"t": np.array([f["t"] for f in self.frames])}
                for k in keys:
                    out[k] = np.array([f["vals"].get(k, np.nan) for f in self.frames], dtype=float)
                self._cache = out
        return self._cache

    def to_csv(self, path: str) -> str:
        arr = self.arrays()
        keys = list(arr.keys())
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for i in range(len(arr["t"])):
                w.writerow([f"{arr[k][i]:.6g}" for k in keys])
        return path


# ============================================================================
# Telemetry + training plots
# ============================================================================
PLOT_ROWS = [
    ("Attitude [deg]", ["tilt_deg", "roll_deg", "pitch_deg", "yaw_deg"]),
    ("Wheel speed [rad/s]", ["cmd1", "cmd2", "cmd3", "enc1", "enc2", "enc3", "cmd_rads", "enc_vel_rads"]),
    ("Torque [N·m]", ["tq1", "tq2", "tq3", "motor_torque_nm", "torque_limit_nm"]),
    ("IMU gyro [rad/s] / ball", ["gx", "gy", "gz", "ball_speed_meas_rads", "ball_speed_pred_rads"]),
    ("Reward / slip", ["reward", "target_dist", "slip_ratio"]),
]


class TelemetryPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self.glw)
        self.plots, self.curves, self.cursors = [], {}, []
        prev = None
        for r, (title, keys) in enumerate(PLOT_ROWS):
            p = self.glw.addPlot(row=r, col=0)
            p.setTitle(title, size="9pt")
            p.showGrid(x=True, y=True, alpha=0.25)
            p.addLegend(offset=(-5, 5), labelTextSize="7pt")
            if prev is not None:
                p.setXLink(prev)
            prev = p
            cur = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ff3fd2", width=1.5))
            cur.setVisible(False)
            p.addItem(cur)
            self.plots.append((p, keys))
            self.cursors.append(cur)

    def refresh(self, arr: Dict[str, np.ndarray], live: bool, window_s: float = 10.0):
        t = arr.get("t", np.zeros(0))
        if len(t) == 0:
            for c in self.curves.values():
                c.setData([], [])
            return
        for p, keys in self.plots:
            ci = 0
            for k in keys:
                if k not in arr:
                    continue
                if (id(p), k) not in self.curves:
                    style = Qt.DashLine if k.startswith("cmd") else Qt.SolidLine
                    pen = pg.mkPen(PLOT_PENS[(ci + (3 if k.startswith("enc") else 0)) % len(PLOT_PENS)], width=1.3,
                                   style=style)
                    self.curves[(id(p), k)] = p.plot([], [], pen=pen, name=k)
                self.curves[(id(p), k)].setData(t, arr[k])
                ci += 1
        if live:
            self.plots[0][0].setXRange(max(0.0, t[-1] - window_s), t[-1] + 0.05, padding=0)
        for c in self.cursors:
            c.setVisible(not live)

    def set_cursor(self, t: float):
        for c in self.cursors:
            c.setValue(t)
            c.setVisible(True)

    def clear(self):
        for (pid, k), c in list(self.curves.items()):
            c.setData([], [])

    def apply_theme(self, theme: str):
        t = THEMES[theme]
        self.glw.setBackground(t["plot_bg"])
        for p, _ in self.plots:
            for ax in ("left", "bottom"):
                p.getAxis(ax).setPen(t["plot_fg"])
                p.getAxis(ax).setTextPen(t["plot_fg"])


class TrainingPlots(QtWidgets.QWidget):
    KEYS = [("return", "Episode return"), ("survival_s", "Survival time [s]"),
            ("mean_tilt_deg", "Mean tilt [deg]"), ("final_target_dist", "Waypoint distance [m]")]

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self.glw)
        self.data: Dict[str, List[float]] = {k: [] for k, _ in self.KEYS}
        self.steps: List[int] = []
        self.items = {}
        prev = None
        for r, (k, title) in enumerate(self.KEYS):
            p = self.glw.addPlot(row=r, col=0)
            p.setTitle(title, size="9pt")
            p.showGrid(x=True, y=True, alpha=0.25)
            if prev is not None:
                p.setXLink(prev)
            prev = p
            sc = p.plot([], [], pen=None, symbol="o", symbolSize=3, symbolBrush=(47, 164, 255, 90), symbolPen=None)
            ma = p.plot([], [], pen=pg.mkPen("#ff8a1f", width=2))
            self.items[k] = (sc, ma)
        self._dirty = False

    def add(self, msg: dict):
        self.steps.append(msg.get("timesteps", 0))
        for k, _ in self.KEYS:
            self.data[k].append(float(msg.get(k, np.nan)))
        self._dirty = True

    def refresh(self):
        if not self._dirty or not self.steps:
            return
        x = np.array(self.steps, float)
        for k, _ in self.KEYS:
            y = np.array(self.data[k])
            sc, ma = self.items[k]
            sc.setData(x[-4000:], y[-4000:])
            n = min(50, len(y))
            if n >= 2:
                kern = np.ones(n) / n
                m = np.convolve(np.nan_to_num(y), kern, mode="valid")
                ma.setData(x[n - 1:], m)
        self._dirty = False

    def clear(self):
        self.steps = []
        self.data = {k: [] for k, _ in self.KEYS}
        for sc, ma in self.items.values():
            sc.setData([], [])
            ma.setData([], [])

    def to_csv(self, path: str):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timesteps"] + [k for k, _ in self.KEYS])
            for i, s in enumerate(self.steps):
                w.writerow([s] + [self.data[k][i] for k, _ in self.KEYS])

    def apply_theme(self, theme: str):
        self.glw.setBackground(THEMES[theme]["plot_bg"])


# ============================================================================
# DistributionParameter editor widget
# ============================================================================
class DistributionWidget(QtWidgets.QWidget):
    changed = Signal()

    def __init__(self, dp: DistributionParameter, decimals: int = 4, parent=None):
        super().__init__(parent)
        self.dp = dp
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Constant", "Uniform", "Normal"])
        self.mode.setCurrentIndex(DIST_MODES.index(dp.mode) if dp.mode in DIST_MODES else 0)
        self.mode.setFixedWidth(84)
        lay.addWidget(self.mode)
        self.a = self._spin(decimals)
        self.b = self._spin(decimals)
        lay.addWidget(self.a)
        lay.addWidget(self.b)
        self.units = QtWidgets.QLabel(dp.units)
        self.units.setObjectName("muted")
        self.units.setMinimumWidth(34)
        lay.addWidget(self.units)
        self._load()
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.a.valueChanged.connect(self._store)
        self.b.valueChanged.connect(self._store)

    @staticmethod
    def _spin(decimals):
        s = QtWidgets.QDoubleSpinBox()
        s.setDecimals(decimals)
        s.setRange(-1e6, 1e6)
        s.setSingleStep(10 ** (-max(1, decimals - 2)))
        s.setKeyboardTracking(False)
        s.setMinimumWidth(70)
        return s

    def _load(self):
        self.a.blockSignals(True)
        self.b.blockSignals(True)
        m = self.dp.mode
        if m == "uniform":
            self.a.setPrefix("min ")
            self.b.setPrefix("max ")
            self.a.setValue(self.dp.low)
            self.b.setValue(self.dp.high)
            self.b.setVisible(True)
        elif m == "normal":
            self.a.setPrefix("μ ")
            self.b.setPrefix("σ ")
            self.a.setValue(self.dp.mean)
            self.b.setValue(self.dp.std)
            self.b.setVisible(True)
        else:
            self.a.setPrefix("")
            self.a.setValue(self.dp.value)
            self.b.setVisible(False)
        tip = f"{self.dp.describe()}  (clamped to [{self.dp.clamp_low:g}, {self.dp.clamp_high:g}])"
        self.setToolTip(tip)
        self.a.blockSignals(False)
        self.b.blockSignals(False)

    def _mode_changed(self, i):
        nom = self.dp.nominal()
        self.dp.mode = DIST_MODES[i]
        if self.dp.mode == "constant":
            self.dp.value = nom
        elif self.dp.mode == "uniform" and self.dp.high <= self.dp.low:
            span = abs(nom) * 0.1 or 0.1
            self.dp.low, self.dp.high = nom - span, nom + span
        elif self.dp.mode == "normal":
            self.dp.mean = nom
            if self.dp.std <= 0:
                self.dp.std = abs(nom) * 0.05 or 0.01
        self._load()
        self.changed.emit()

    def _store(self):
        m = self.dp.mode
        if m == "uniform":
            self.dp.low, self.dp.high = self.a.value(), self.b.value()
        elif m == "normal":
            self.dp.mean, self.dp.std = self.a.value(), max(0.0, self.b.value())
        else:
            self.dp.value = self.a.value()
        self.setToolTip(self.dp.describe())
        self.changed.emit()


# ============================================================================
# Simple physics wrapper for imported custom models
# ============================================================================
class GenericSim:
    """Free physics for custom imports. When IMU + wheel joints are bound, it also builds
    the standard full observation vector so exported policies can be test-driven."""

    def __init__(self, xml: str, cfg: AppConfig):
        self.cfg = cfg
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.model_version = 1
        self.dt_ctrl = cfg.robot.frame_skip * self.model.opt.timestep
        m = self.model
        self.has_imu = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat") >= 0
        self.n_motor = sum(1 for i in range(3) if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i + 1}") >= 0)
        self.prev = np.zeros(3)
        mujoco.mj_forward(m, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def observe(self) -> np.ndarray:
        m, d = self.model, self.data
        if not self.has_imu:
            return np.zeros(20)
        a = lambda n: m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)]
        q = d.sensordata[a("imu_quat"):a("imu_quat") + 4]
        rpy = np.array(quat_to_rpy(q))
        gyro = d.sensordata[a("imu_gyro"):a("imu_gyro") + 3]
        acc = d.sensordata[a("imu_accel"):a("imu_accel") + 3]
        ev, ep = np.zeros(3), np.zeros(3)
        for i in range(self.n_motor):
            j = m.actuator_trnid[i, 0]
            ev[i] = d.qvel[m.jnt_dofadr[j]]
            ep[i] = d.qpos[m.jnt_qposadr[j]]
        return np.concatenate([rpy, gyro, acc, ev, wrap_pi(ep), np.zeros(2), self.prev])

    def step(self, action: np.ndarray):
        a = np.zeros(3)
        a[:min(3, len(action))] = np.clip(action[:3], -1, 1)
        self.prev = a
        if self.n_motor:
            self.data.ctrl[:self.n_motor] = a[:self.n_motor] * self.cfg.robot.stepper_max_speed_rads
        for _ in range(self.cfg.robot.frame_skip):
            mujoco.mj_step(self.model, self.data)

    def get_render_state(self):
        return {"qpos": self.data.qpos.copy(), "mocap_pos": self.data.mocap_pos.copy(), "time": self.data.time,
                "model_version": self.model_version}


# ============================================================================
# Form helpers
# ============================================================================
def fspin(value, lo, hi, step=0.01, decimals=3, suffix=""):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    s.setKeyboardTracking(False)
    if suffix:
        s.setSuffix(" " + suffix)
    return s


def ispin(value, lo, hi, step=1, suffix=""):
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(value)
    s.setKeyboardTracking(False)
    if suffix:
        s.setSuffix(" " + suffix)
    return s


def scroll(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    sa = QtWidgets.QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(widget)
    return sa


# ============================================================================
# Main window
# ============================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: Optional[AppConfig] = None):
        super().__init__()
        self.cfg = cfg or default_config()
        self.setWindowTitle(f"{APP_NAME}  v{APP_VERSION}")
        self.resize(1600, 980)
        ensure_workspace(self.cfg)
        self.trainer = TrainingController()
        self.recorder = TimelineRecorder(int(self.cfg.viewer.record_seconds * self.cfg.robot.control_hz))
        self.sim = None                    # BB8BalanceEnv | ValidationRigSim | GenericSim
        self.sim_running = True
        self.scrubbing = False
        self._live_state = None
        self.policy: Optional[PolicyRunner] = None
        self.policy_source = "zero"
        self.rig_cmd = 0.0
        self.speed = 1.0
        self.grid_envs: List[BB8BalanceEnv] = []
        self.selected = (-1, -1)
        self.resume_path: Optional[str] = None
        self.loaded_session_dir: Optional[str] = None
        self.last_obs_full = np.zeros(20)
        self._acc = 0.0
        self._last_tick = time.perf_counter()
        self._last_plot = 0.0
        self._fps_frames, self._fps_t, self.fps = 0, time.perf_counter(), 0.0
        self._rebuild_timer = QTimer(self, singleShot=True, interval=250, timeout=self.rebuild_sim)

        ensure_initial_policy(self.cfg)    # cold-start ONNX on startup
        self._build_ui()
        self.apply_theme(self.cfg.viewer.theme)
        self.rebuild_sim()
        self.set_mode(self.cfg.viewer.mode)

        self.timer = QTimer(self, interval=16, timeout=self._tick, timerType=Qt.PreciseTimer)
        self.timer.start()
        self.poll_timer = QTimer(self, interval=100, timeout=self._poll_trainer)
        self.poll_timer.start()
        self.res_timer = QTimer(self, interval=2000, timeout=self._resource_monitor)
        self.res_timer.start()
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_Escape), self, activated=self.emergency_reset,
                        context=Qt.ApplicationShortcut)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_Space), self, activated=self.toggle_sim)
        self.log(f"Workspace: {os.path.abspath(self.cfg.workspace_dir)}  (cold-start policy ready)")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self._build_toolbar()
        # central: stacked single / grid / headless
        self.stack = QtWidgets.QStackedWidget()
        self.viewport = MjViewport(self, interactive=True)
        self.viewport.bodySelected.connect(self.on_body_selected)
        self.viewport.targetPicked.connect(self.on_target_picked)
        self.grid = GridViewer(self)
        self.headless_lbl = QtWidgets.QLabel("HEADLESS TURBO MODE\n\nOpenGL renderer disabled - physics & training "
                                             "run as fast as the CPU allows.\nPlots and logs keep updating.")
        self.headless_lbl.setAlignment(Qt.AlignCenter)
        self.headless_lbl.setStyleSheet("font-size: 18px; font-weight: 600;")
        for w in (self.viewport, self.grid, self.headless_lbl):
            self.stack.addWidget(w)
        central = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(central)
        cl.setContentsMargins(2, 2, 2, 2)
        cl.addWidget(self.stack, 1)
        cl.addWidget(self._build_timeline())
        self.setCentralWidget(central)

        # left dock: configuration
        self.left_dock = QtWidgets.QDockWidget("Configuration", self)
        self.left_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.left_dock.setMinimumWidth(430)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)
        self._build_config_tabs()

        # right dock: telemetry, training curves, log
        self.right_dock = QtWidgets.QDockWidget("Telemetry & Training", self)
        self.right_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.right_dock.setMinimumWidth(430)
        tabs = QtWidgets.QTabWidget()
        self.telemetry = TelemetryPanel()
        self.train_plots = TrainingPlots()
        self.log_box = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log_box.setMaximumBlockCount(5000)
        tabs.addTab(self.telemetry, "Live Telemetry")
        tabs.addTab(self.train_plots, "Training Progress")
        tabs.addTab(self.log_box, "Log")
        self.right_dock.setWidget(tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_dock)
        self.statusBar().showMessage("Ready")

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.addWidget(QtWidgets.QLabel(" Mode: "))
        self.mode_group = QtGui.QActionGroup(self)
        self.mode_actions = {}
        for key, text, tip in (("single", "Single Interactive", "1 env @ 60 FPS, mouse camera, picking"),
                               ("grid", "Parallel Grid (8)", "Watch up to 8 agents learning live"),
                               ("headless", "Headless Turbo", "Renderer off, maximum training speed")):
            a = QtGui.QAction(text, self, checkable=True, toolTip=tip)
            a.triggered.connect(lambda _=False, k=key: self.set_mode(k))
            self.mode_group.addAction(a)
            tb.addAction(a)
            self.mode_actions[key] = a
        self.grid_count = ispin(self.cfg.viewer.grid_count, 1, 8, suffix="agents")
        self.grid_count.valueChanged.connect(self._grid_count_changed)
        tb.addWidget(self.grid_count)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Model: "))
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["Validation Rig (1 motor)", "Parametric BB-8 Ballbot", "Custom Import…"])
        self.model_combo.setCurrentIndex({"validation_rig": 0, "bb8": 1, "custom": 2}[self.cfg.model_type])
        self.model_combo.activated.connect(self._model_changed)
        tb.addWidget(self.model_combo)
        tb.addAction(QtGui.QAction("Import…", self, triggered=self.import_model))
        tb.addSeparator()
        self.play_act = QtGui.QAction("⏸ Pause Sim", self, triggered=self.toggle_sim)
        tb.addAction(self.play_act)
        tb.addAction(QtGui.QAction("⟲ Reset Episode", self, triggered=self.reset_episode))
        tb.addAction(QtGui.QAction("➜ Push", self, triggered=self.manual_push, toolTip="Apply a random push to the head"))
        self.addToolBarBreak()
        tb = self.addToolBar("Resources")
        tb.setMovable(False)
        tb.addWidget(QtWidgets.QLabel(" Resource governor  CPU: "))
        self.cpu_combo = QtWidgets.QComboBox()
        self.cpu_combo.addItems(["25%", "50%", "75%", "100%"])
        self.cpu_combo.setCurrentText(f"{self.cfg.resources.cpu_percent}%")
        self.cpu_combo.currentTextChanged.connect(self._cpu_changed)
        tb.addWidget(self.cpu_combo)
        tb.addWidget(QtWidgets.QLabel(" RAM ceiling: "))
        self.ram_spin = fspin(self.cfg.resources.ram_ceiling_percent, 10, 99, 1, 0, "%")
        self.ram_spin.valueChanged.connect(lambda v: setattr(self.cfg.resources, "ram_ceiling_percent", v))
        tb.addWidget(self.ram_spin)
        self.res_label = QtWidgets.QLabel("  RAM -- | CPU --")
        self.res_label.setObjectName("muted")
        self.res_label.setMinimumWidth(420)
        tb.addWidget(self.res_label)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.theme_btn = QtWidgets.QPushButton("☀ Light")
        self.theme_btn.setCheckable(True)
        self.theme_btn.clicked.connect(lambda: self.apply_theme("light" if self.cfg.viewer.theme == "dark" else "dark"))
        tb.addWidget(self.theme_btn)
        kill = QtWidgets.QPushButton("⛔ Reset && Kill Workers (ESC)")
        kill.setObjectName("danger")
        kill.clicked.connect(self.emergency_reset)
        tb.addWidget(kill)

    def _build_timeline(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(4, 2, 4, 2)
        self.tl_play = QtWidgets.QPushButton("⏸")
        self.tl_play.setFixedWidth(36)
        self.tl_play.clicked.connect(self.toggle_sim)
        lay.addWidget(self.tl_play)
        lay.addWidget(QtWidgets.QLabel("Timeline"))
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(self._scrub_begin)
        self.slider.valueChanged.connect(self._scrub)
        lay.addWidget(self.slider, 1)
        self.tl_label = QtWidgets.QLabel("t = 0.00 s")
        self.tl_label.setMinimumWidth(170)
        lay.addWidget(self.tl_label)
        live = QtWidgets.QPushButton("● Live")
        live.clicked.connect(self._go_live)
        lay.addWidget(live)
        lay.addWidget(QtWidgets.QLabel("Speed"))
        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(lambda s: setattr(self, "speed", float(s[:-1])))
        lay.addWidget(self.speed_combo)
        return w

    # ---------------------------------------------------- configuration tabs
    def _build_config_tabs(self):
        self.cfg_tabs = QtWidgets.QTabWidget()
        self.cfg_tabs.addTab(scroll(self._tab_robot()), "Robot")
        self.cfg_tabs.addTab(scroll(self._tab_randomization()), "Randomization")
        self.cfg_tabs.addTab(scroll(self._tab_terrain()), "Terrain && Pushes")
        self.cfg_tabs.addTab(scroll(self._tab_episode()), "Episode && Target")
        self.cfg_tabs.addTab(scroll(self._tab_training()), "Network && Training")
        self.cfg_tabs.addTab(self._tab_inspector(), "Inspector")
        self.left_dock.setWidget(self.cfg_tabs)

    def _bind(self, obj, attr, widget, kind="reset"):
        """Bind a widget's value to a config attribute. kind: live | reset | rebuild."""
        def setter(v):
            setattr(obj, attr, v)
            self.config_changed(kind)
        if isinstance(widget, (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox)):
            widget.valueChanged.connect(setter)
        elif isinstance(widget, QtWidgets.QCheckBox):
            widget.toggled.connect(setter)
        elif isinstance(widget, QtWidgets.QComboBox):
            widget.currentTextChanged.connect(lambda t: setter(widget.currentData() or t))
        return widget

    def _dp_row(self, form, label, dp, kind="reset", decimals=4):
        w = DistributionWidget(dp, decimals)
        w.changed.connect(lambda: self.config_changed(kind))
        form.addRow(label, w)
        return w

    def _tab_robot(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        rc = self.cfg.robot
        g = QtWidgets.QGroupBox("Omniwheel contact geometry")
        f = QtWidgets.QFormLayout(g)
        zl = QtWidgets.QHBoxLayout()
        self.zenith_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.zenith_slider.setRange(150, 600)
        self.zenith_slider.setValue(int(round(rc.zenith_angle_deg * 10)))
        self.zenith_lbl = QtWidgets.QLabel(f"{rc.zenith_angle_deg:.1f}°")
        self.zenith_lbl.setMinimumWidth(48)

        def zen(val):
            rc.zenith_angle_deg = val / 10.0
            self.zenith_lbl.setText(f"{rc.zenith_angle_deg:.1f}°")
            self.config_changed("rebuild")
        self.zenith_slider.valueChanged.connect(zen)
        zl.addWidget(self.zenith_slider)
        zl.addWidget(self.zenith_lbl)
        f.addRow("Zenith (contact) angle", zl)
        az = QtWidgets.QLineEdit(", ".join(f"{a:g}" for a in rc.wheel_azimuths_deg))

        def set_az():
            try:
                vals = [float(x) for x in az.text().replace(";", ",").split(",") if x.strip()]
                if len(vals) == 3:
                    rc.wheel_azimuths_deg = vals
                    self.config_changed("rebuild")
            except ValueError:
                pass
        az.editingFinished.connect(set_az)
        f.addRow("Wheel azimuths [deg]", az)
        f.addRow("Wheel diameter", self._bind(rc, "wheel_diameter_m", fspin(rc.wheel_diameter_m, 0.02, 0.2, 0.001, 4, "m"), "rebuild"))
        f.addRow("Wheel width", self._bind(rc, "wheel_width_m", fspin(rc.wheel_width_m, 0.005, 0.08, 0.001, 4, "m"), "rebuild"))
        f.addRow("Rollers per row", self._bind(rc, "rollers_per_row", ispin(rc.rollers_per_row, 3, 16), "rebuild"))
        om = QtWidgets.QComboBox()
        om.addItem("Virtual (Jacobian, fast RL)", "virtual")
        om.addItem("Multi-roller contact (high fidelity)", "roller")
        om.setCurrentIndex(0 if rc.omniwheel_model == "virtual" else 1)
        f.addRow("Omniwheel model", self._bind(rc, "omniwheel_model", om, "rebuild"))
        v.addWidget(g)
        g = QtWidgets.QGroupBox("Ball && head (nominal)")
        f = QtWidgets.QFormLayout(g)
        f.addRow("Ball radius", self._bind(rc, "ball_radius_m", fspin(rc.ball_radius_m, 0.05, 0.5, 0.005, 4, "m"), "rebuild"))
        f.addRow("Ball inertia factor k (I=kmR²)", self._bind(rc, "ball_inertia_factor", fspin(rc.ball_inertia_factor, 0.3, 0.7, 0.01, 3), "rebuild"))
        f.addRow("Ball mesh subdivisions", self._bind(rc, "ball_mesh_subdivisions", ispin(rc.ball_mesh_subdivisions, 1, 5), "rebuild"))
        f.addRow("Head mass (geometry)", self._bind(rc, "head_mass_kg", fspin(rc.head_mass_kg, 0.2, 20, 0.05, 3, "kg"), "rebuild"))
        f.addRow("Head CoG above ball top", self._bind(rc, "head_cog_height_m", fspin(rc.head_cog_height_m, 0.0, 0.5, 0.005, 4, "m"), "rebuild"))
        f.addRow("Head dome radius", self._bind(rc, "head_radius_m", fspin(rc.head_radius_m, 0.05, 0.4, 0.005, 3, "m"), "rebuild"))
        f.addRow("IMU height above ball top", self._bind(rc, "imu_height_m", fspin(rc.imu_height_m, 0, 0.4, 0.005, 3, "m"), "rebuild"))
        v.addWidget(g)
        g = QtWidgets.QGroupBox("JK42HS40-1704-83AF + TMC2208")
        f = QtWidgets.QFormLayout(g)
        f.addRow("Holding torque", self._bind(rc, "stepper_holding_torque_nm", fspin(rc.stepper_holding_torque_nm, 0.05, 3, 0.01, 3, "N·m"), "rebuild"))
        f.addRow("Microsteps", self._bind(rc, "tmc2208_microsteps", ispin(rc.tmc2208_microsteps, 1, 256), "reset"))
        f.addRow("Servo stiffness kv", self._bind(rc, "stepper_kv", fspin(rc.stepper_kv, 0.01, 10, 0.05, 3, "N·m·s"), "rebuild"))
        f.addRow("Corner speed", self._bind(rc, "stepper_corner_speed_rads", fspin(rc.stepper_corner_speed_rads, 1, 200, 1, 1, "rad/s"), "reset"))
        f.addRow("Max speed", self._bind(rc, "stepper_max_speed_rads", fspin(rc.stepper_max_speed_rads, 1, 300, 1, 1, "rad/s"), "rebuild"))
        f.addRow("Torque left at max speed", self._bind(rc, "stepper_torque_at_max_frac", fspin(rc.stepper_torque_at_max_frac, 0.05, 1, 0.05, 2), "reset"))
        f.addRow("Physics timestep", self._bind(rc, "physics_timestep_s", fspin(rc.physics_timestep_s, 0.0002, 0.01, 0.0005, 4, "s"), "rebuild"))
        f.addRow("Control rate", self._bind(rc, "control_hz", fspin(rc.control_hz, 10, 1000, 10, 0, "Hz"), "rebuild"))
        v.addWidget(g)
        g = QtWidgets.QGroupBox("Validation rig")
        f = QtWidgets.QFormLayout(g)
        self.rig_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.rig_slider.setRange(-450, 450)
        self.rig_lbl = QtWidgets.QLabel("0.0 rad/s")

        def rig(vv):
            self.rig_cmd = vv / 10.0
            self.rig_lbl.setText(f"{self.rig_cmd:.1f} rad/s")
        self.rig_slider.valueChanged.connect(rig)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.rig_slider)
        hl.addWidget(self.rig_lbl)
        f.addRow("Motor command", hl)
        self.rig_info = QtWidgets.QLabel("Select 'Validation Rig' in the model switcher.")
        self.rig_info.setWordWrap(True)
        self.rig_info.setObjectName("muted")
        f.addRow(self.rig_info)
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _tab_randomization(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        r = self.cfg.randomization
        note = QtWidgets.QLabel("Every parameter is re-sampled on every reset (manual or after a fall). "
                                "Constant / Uniform(min,max) / Normal(μ,σ) with safety clamping.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        v.addWidget(note)
        g = QtWidgets.QGroupBox("Ball")
        f = QtWidgets.QFormLayout(g)
        self._dp_row(f, "Mass", r.ball.mass_kg)
        self._dp_row(f, "Sliding grip μ", r.ball.sliding_grip)
        self._dp_row(f, "Torsional friction", r.ball.torsional_friction, decimals=5)
        self._dp_row(f, "Rolling resistance", r.ball.rolling_resistance, decimals=5)
        self._dp_row(f, "Non-sphericity", r.ball.non_sphericity)
        self._dp_row(f, "Dent / bump depth", r.ball.dent_depth_m, decimals=5)
        f.addRow("Dent count", self._bind(r.ball, "dent_count", ispin(r.ball.dent_count, 0, 200), "reset"))
        self._dp_row(f, "CoM offset |d|", r.ball.com_offset_m, decimals=5)
        v.addWidget(g)
        g = QtWidgets.QGroupBox("Omniwheels (asymmetric wear)")
        f = QtWidgets.QFormLayout(g)
        self._dp_row(f, "Longitudinal grip μ", r.wheels.longitudinal_grip)
        self._dp_row(f, "Lateral roller resistance", r.wheels.lateral_roller_resistance, decimals=5)
        self._dp_row(f, "Rolling resistance", r.wheels.rolling_resistance, decimals=5)
        for i in range(3):
            self._dp_row(f, f"Wheel {i + 1} diameter offset", r.wheels.diameter_offset_frac[i])
            self._dp_row(f, f"Wheel {i + 1} grip offset", r.wheels.grip_offset_frac[i])
        v.addWidget(g)
        g = QtWidgets.QGroupBox("Head")
        f = QtWidgets.QFormLayout(g)
        self._dp_row(f, "Total head mass", r.head.mass_kg)
        self._dp_row(f, "CoG offset dx", r.head.cog_dx_m)
        self._dp_row(f, "CoG offset dy", r.head.cog_dy_m)
        self._dp_row(f, "CoG offset dz", r.head.cog_dz_m)
        v.addWidget(g)
        g = QtWidgets.QGroupBox("Actuator imperfections")
        f = QtWidgets.QFormLayout(g)
        self._dp_row(f, "Voltage sag torque scale", r.actuators.voltage_torque_scale)
        self._dp_row(f, "Stepper deadband", r.actuators.deadband_rads)
        self._dp_row(f, "Command latency", r.actuators.latency_steps)
        self._dp_row(f, "Servo stiffness scale", r.actuators.kv_scale)
        v.addWidget(g)
        g = QtWidgets.QGroupBox("IMU (Nano 33 IoT LSM6DS3) && encoders")
        f = QtWidgets.QFormLayout(g)
        for k, lbl in (("orient_noise_std_rad", "Roll/pitch/yaw noise σ"), ("orient_drift_std_rad", "Roll/pitch drift"),
                       ("yaw_drift_std_rad", "Yaw drift (no magnetometer)"), ("gyro_noise_std_rads", "Gyro noise σ"),
                       ("gyro_drift_std_rads", "Gyro bias random walk"), ("accel_noise_std_ms2", "Accel noise σ"),
                       ("accel_drift_std_ms2", "Accel bias random walk"), ("encoder_noise_std_rads", "Encoder vel noise σ")):
            self._dp_row(f, lbl, getattr(r.imu, k), decimals=5)
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _tab_terrain(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        t = self.cfg.terrain
        g = QtWidgets.QGroupBox("Map && domain-randomisation layers")
        f = QtWidgets.QFormLayout(g)
        mp = QtWidgets.QComboBox()
        mp.addItem("Standard Flat Ground", "flat")
        mp.addItem("Obstacle Arena", "obstacle_arena")
        mp.setCurrentIndex(0 if t.map_name == "flat" else 1)
        f.addRow("Map", self._bind(t, "map_name", mp, "rebuild"))
        for attr, lbl in (("off_ideal_nominal", "Off (Ideal Nominal Physics)"),
                          ("variable_grip", "Variable Ground Texture / Grip"),
                          ("rough_heightfield", "Procedural Rough Ground (Heightfield)"),
                          ("walls", "Walls / Arena Boundaries"), ("ramps", "Minor Ramps && Slopes"),
                          ("bumps_stairs", "Speed Bumps && Stairs")):
            cb = QtWidgets.QCheckBox(lbl)
            cb.setChecked(getattr(t, attr))
            f.addRow(self._bind(t, attr, cb, "reset" if attr == "off_ideal_nominal" else "rebuild"))
        f.addRow("Arena half-size", self._bind(t, "arena_half_size_m", fspin(t.arena_half_size_m, 1, 20, 0.5, 1, "m"), "rebuild"))
        self._dp_row(f, "Ground grip factor", t.tile_grip_factor)
        f.addRow("Rough amplitude", self._bind(t, "rough_amplitude_m", fspin(t.rough_amplitude_m, 0.001, 0.1, 0.002, 4, "m"), "rebuild"))
        f.addRow("Ramp slope", self._bind(t, "ramp_slope_deg", fspin(t.ramp_slope_deg, 1, 25, 0.5, 1, "deg"), "rebuild"))
        f.addRow("Bump height", self._bind(t, "bump_height_m", fspin(t.bump_height_m, 0.002, 0.08, 0.002, 4, "m"), "rebuild"))
        f.addRow("Stair step", self._bind(t, "stair_step_m", fspin(t.stair_step_m, 0.002, 0.08, 0.002, 4, "m"), "rebuild"))
        f.addRow("Layout seed", self._bind(t, "layout_seed", ispin(t.layout_seed, 0, 99999), "rebuild"))
        v.addWidget(g)
        d = self.cfg.disturbance
        g = QtWidgets.QGroupBox("Intermittent external pushes (xfrc_applied)")
        f = QtWidgets.QFormLayout(g)
        cb = QtWidgets.QCheckBox("Enable automatic pushes")
        cb.setChecked(d.enabled)
        f.addRow(self._bind(d, "enabled", cb, "live"))
        tb = QtWidgets.QComboBox()
        tb.addItem("Head", "head")
        tb.addItem("Ball", "ball")
        tb.setCurrentIndex(0 if d.target_body == "head" else 1)
        f.addRow("Apply to", self._bind(d, "target_body", tb, "live"))
        self._dp_row(f, "Interval between pushes", d.interval_s, "live")
        self._dp_row(f, "Force magnitude", d.magnitude_n, "live")
        self._dp_row(f, "Direction azimuth", d.azimuth_deg, "live")
        self._dp_row(f, "Direction elevation", d.elevation_deg, "live")
        self._dp_row(f, "Push duration", d.duration_s, "live")
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _tab_episode(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        e = self.cfg.episode
        g = QtWidgets.QGroupBox("Episode control")
        f = QtWidgets.QFormLayout(g)
        cb = QtWidgets.QCheckBox("Auto-reset on fall")
        cb.setChecked(e.auto_reset_on_fall)
        f.addRow(self._bind(e, "auto_reset_on_fall", cb, "live"))
        f.addRow("Fall tilt threshold", self._bind(e, "fall_tilt_deg", fspin(e.fall_tilt_deg, 2, 90, 1, 1, "deg"), "live"))
        cb2 = QtWidgets.QCheckBox("Detect head-ball contact loss")
        cb2.setChecked(e.detect_contact_loss)
        f.addRow(self._bind(e, "detect_contact_loss", cb2, "live"))
        f.addRow("Max episode length", self._bind(e, "max_episode_s", fspin(e.max_episode_s, 1, 3600, 1, 1, "s"), "reset"))
        f.addRow("Spawn X (centre = 0)", self._bind(e, "spawn_x_m", fspin(e.spawn_x_m, -20, 20, 0.1, 3, "m"), "rebuild"))
        f.addRow("Spawn Y (centre = 0)", self._bind(e, "spawn_y_m", fspin(e.spawn_y_m, -20, 20, 0.1, 3, "m"), "rebuild"))
        self._dp_row(f, "Initial tilt", e.init_tilt_deg)
        self._dp_row(f, "Initial yaw", e.init_yaw_deg)
        v.addWidget(g)
        g = QtWidgets.QGroupBox("Live waypoint target (Ctrl+click in viewport)")
        f = QtWidgets.QFormLayout(g)
        self.target_cb = QtWidgets.QCheckBox("Enable target endpoint")
        self.target_cb.setChecked(e.target_enabled)
        self.tx = fspin(e.target_x_m.nominal(), -20, 20, 0.05, 3, "m")
        self.ty = fspin(e.target_y_m.nominal(), -20, 20, 0.05, 3, "m")
        self.target_cb.toggled.connect(self._target_changed)
        self.tx.valueChanged.connect(self._target_changed)
        self.ty.valueChanged.connect(self._target_changed)
        f.addRow(self.target_cb)
        f.addRow("Target X", self.tx)
        f.addRow("Target Y", self.ty)
        self._dp_row(f, "Training target X distribution", e.target_x_m)
        self._dp_row(f, "Training target Y distribution", e.target_y_m)
        f.addRow("Reach radius", self._bind(e, "target_reach_radius_m", fspin(e.target_reach_radius_m, 0.01, 2, 0.01, 3, "m"), "live"))
        v.addWidget(g)
        g = QtWidgets.QGroupBox("Reward weights")
        f = QtWidgets.QFormLayout(g)
        for k in e.reward.__dataclass_fields__:
            f.addRow(k, self._bind(e.reward, k, fspin(getattr(e.reward, k), 0, 1000, 0.01, 3), "live"))
        v.addWidget(g)
        v.addStretch(1)
        return w

    def _tab_training(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        n = self.cfg.network
        g = QtWidgets.QGroupBox("Policy network (tanh MLP)")
        f = QtWidgets.QFormLayout(g)
        self.hidden_edit = QtWidgets.QLineEdit(", ".join(str(x) for x in n.hidden_layers))
        self.hidden_edit.setToolTip("e.g. 64, 64  or  128, 128, 64")

        def set_hidden():
            try:
                vals = [int(x) for x in self.hidden_edit.text().replace(";", ",").split(",") if x.strip()]
                if vals and all(1 <= x <= 4096 for x in vals):
                    n.hidden_layers = vals
                    self.log(f"Hidden layers -> {vals} (new cold-start ONNX on next training start)")
            except ValueError:
                pass
        self.hidden_edit.editingFinished.connect(set_hidden)
        f.addRow("Hidden layers", self.hidden_edit)
        ini = QtWidgets.QComboBox()
        ini.addItem("Orthogonal", "orthogonal")
        ini.addItem("Xavier uniform", "xavier")
        ini.setCurrentIndex(0 if n.init == "orthogonal" else 1)
        f.addRow("Weight init", self._bind(n, "init", ini, "live"))
        am = QtWidgets.QComboBox()
        am.addItem("Stepper target velocity / step-rate", "velocity")
        am.addItem("Motor torque", "torque")
        am.setCurrentIndex(0 if n.action_mode == "velocity" else 1)
        f.addRow("Action type", self._bind(n, "action_mode", am, "rebuild"))
        v.addWidget(g)
        g = QtWidgets.QGroupBox("Active inputs (observation mask)")
        gl = QtWidgets.QVBoxLayout(g)
        for grp in OBS_GROUPS:
            cb = QtWidgets.QCheckBox(f"{grp.label}  ({len(grp.channels)})")
            cb.setChecked(n.obs_mask.get(grp.key, grp.default_on))
            cb.toggled.connect(lambda on, k=grp.key: (n.obs_mask.__setitem__(k, on), self.config_changed("rebuild")))
            gl.addWidget(cb)
        v.addWidget(g)
        g = QtWidgets.QGroupBox("Active outputs (action mask)")
        gl = QtWidgets.QHBoxLayout(g)
        for i in range(3):
            cb = QtWidgets.QCheckBox(f"Motor {i + 1}")
            cb.setChecked(n.action_mask[i])
            cb.toggled.connect(lambda on, i=i: (n.action_mask.__setitem__(i, on), self.config_changed("rebuild")))
            gl.addWidget(cb)
        v.addWidget(g)
        t = self.cfg.training
        g = QtWidgets.QGroupBox("Training (Stable-Baselines3)")
        f = QtWidgets.QFormLayout(g)
        alg = QtWidgets.QComboBox()
        alg.addItems(["PPO", "SAC"])
        alg.setCurrentText(t.algorithm)
        f.addRow("Algorithm", self._bind(t, "algorithm", alg, "live"))
        f.addRow("Parallel envs", self._bind(t, "n_envs", ispin(t.n_envs, 1, 64), "live"))
        ve = QtWidgets.QComboBox()
        ve.addItem("SubprocVecEnv (multi-core)", "subproc")
        ve.addItem("DummyVecEnv (single process)", "dummy")
        ve.setCurrentIndex(0 if t.vec_env == "subproc" else 1)
        f.addRow("Vectorisation", self._bind(t, "vec_env", ve, "live"))
        f.addRow("Total timesteps", self._bind(t, "total_timesteps", ispin(t.total_timesteps, 1000, 2_000_000_000, 100000), "live"))
        f.addRow("Learning rate", self._bind(t, "learning_rate", fspin(t.learning_rate, 1e-6, 1e-1, 1e-4, 6), "live"))
        f.addRow("Rollout steps / env", self._bind(t, "n_steps", ispin(t.n_steps, 16, 65536, 128), "live"))
        f.addRow("Batch size", self._bind(t, "batch_size", ispin(t.batch_size, 16, 65536, 64), "live"))
        f.addRow("γ", self._bind(t, "gamma", fspin(t.gamma, 0.8, 0.9999, 0.001, 4), "live"))
        f.addRow("Auto-save every", self._bind(t, "autosave_every_steps", ispin(t.autosave_every_steps, 1000, 100_000_000, 10000, "steps"), "live"))
        f.addRow("…or every", self._bind(t, "autosave_every_minutes", fspin(t.autosave_every_minutes, 0.5, 600, 0.5, 1, "min"), "live"))
        f.addRow("Seed", self._bind(t, "seed", ispin(t.seed, 0, 2 ** 31 - 1), "live"))
        v.addWidget(g)
        g = QtWidgets.QGroupBox("Session")
        gl = QtWidgets.QGridLayout(g)
        self.btn_start = QtWidgets.QPushButton("▶ Start")
        self.btn_start.setObjectName("accent")
        self.btn_pause = QtWidgets.QPushButton("⏸ Pause")
        self.btn_resume = QtWidgets.QPushButton("⏵ Resume")
        self.btn_stop = QtWidgets.QPushButton("⏹ Stop")
        self.btn_save = QtWidgets.QPushButton("💾 Save Session")
        self.btn_load = QtWidgets.QPushButton("📂 Load Session")
        for i, (b, fn) in enumerate(((self.btn_start, self.start_training), (self.btn_pause, self.pause_training),
                                     (self.btn_resume, self.resume_training), (self.btn_stop, self.stop_training),
                                     (self.btn_save, self.save_session_dialog), (self.btn_load, self.load_session_dialog))):
            b.clicked.connect(fn)
            gl.addWidget(b, i // 3, i % 3)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFormat("%v / %m steps")
        gl.addWidget(self.progress, 2, 0, 1, 3)
        self.train_lbl = QtWidgets.QLabel("Idle")
        self.train_lbl.setObjectName("muted")
        gl.addWidget(self.train_lbl, 3, 0, 1, 3)
        v.addWidget(g)
        g = QtWidgets.QGroupBox("Interactive policy")
        f = QtWidgets.QFormLayout(g)
        self.policy_combo = QtWidgets.QComboBox()
        self.policy_combo.addItem("Zero action (passive)", "zero")
        self.policy_combo.addItem("Initial policy (cold start)", "initial")
        self.policy_combo.addItem("Best policy (auto-reload)", "best")
        self.policy_combo.addItem("Load .onnx file…", "file")
        self.policy_combo.activated.connect(lambda i: self.set_policy(self.policy_combo.itemData(i)))
        f.addRow("Drive viewport with", self.policy_combo)
        v.addWidget(g)
        g = QtWidgets.QGroupBox("Export suite")
        gl = QtWidgets.QGridLayout(g)
        for i, (txt, fn) in enumerate((("Export Best .onnx", self.export_best_onnx),
                                       ("Export C++ Policy (.cpp / .h)", self.export_cpp),
                                       ("Export Telemetry (.csv)", self.export_csv),
                                       ("Export Complete Experiment Bundle (.zip)", self.export_bundle))):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            gl.addWidget(b, i // 2, i % 2)
        v.addWidget(g)
        v.addStretch(1)
        self._update_train_buttons()
        return w

    def _tab_inspector(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Body / Geom", "id"])
        self.tree.setColumnWidth(0, 260)
        self.tree.itemSelectionChanged.connect(self._tree_selected)
        v.addWidget(self.tree, 2)
        g = QtWidgets.QGroupBox("Properties (live)")
        self.prop_form = QtWidgets.QFormLayout(g)
        self.prop_edits: Dict[str, QtWidgets.QLineEdit] = {}
        v.addWidget(g)
        hb = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Apply properties")
        b.clicked.connect(self.apply_properties)
        hb.addWidget(b)
        b = QtWidgets.QPushButton("Clear highlight")
        b.clicked.connect(lambda: self.on_body_selected(-1, -1))
        hb.addWidget(b)
        v.addLayout(hb)
        g = QtWidgets.QGroupBox("XML editor (selected element)")
        gl = QtWidgets.QVBoxLayout(g)
        self.xml_edit = QtWidgets.QPlainTextEdit()
        self.xml_edit.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.xml_edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        gl.addWidget(self.xml_edit)
        hb = QtWidgets.QHBoxLayout()
        for txt, fn in (("Apply XML && Recompile", self.apply_xml), ("Edit full model XML", self.show_full_xml),
                        ("Revert to procedural", self.revert_xml)):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            hb.addWidget(b)
        gl.addLayout(hb)
        v.addWidget(g, 3)
        g = QtWidgets.QGroupBox("Custom model RL bindings")
        f = QtWidgets.QFormLayout(g)
        self.bind_imu = QtWidgets.QComboBox()
        self.bind_ball = QtWidgets.QComboBox()
        self.bind_wheels = [QtWidgets.QComboBox() for _ in range(3)]
        f.addRow("IMU body", self.bind_imu)
        f.addRow("Ball body", self.bind_ball)
        for i, cbx in enumerate(self.bind_wheels):
            f.addRow(f"Wheel {i + 1} joint", cbx)
        b = QtWidgets.QPushButton("Bind sensors/actuators && recompile")
        b.clicked.connect(self.apply_bindings)
        f.addRow(b)
        v.addWidget(g)
        return w

    # ---------------------------------------------------------- simulation
    def config_changed(self, kind: str):
        if kind == "rebuild":
            self._rebuild_timer.start()
        elif isinstance(self.sim, BB8BalanceEnv):
            self.sim.cfg = self.cfg.copy()
            if kind == "live":
                self.sim.max_steps = int(round(self.cfg.episode.max_episode_s / self.sim.dt_ctrl))
                if self.cfg.disturbance.enabled and self.sim.push_next_t == math.inf:
                    self.sim._schedule_push()
            for env in self.grid_envs:
                env.cfg = self.cfg.copy()

    def rebuild_sim(self):
        """(Re)create the interactive simulation from the current configuration."""
        old_cam = (self.viewport.cam.distance, self.viewport.cam.azimuth, self.viewport.cam.elevation) \
            if self.viewport.model is not None else None
        try:
            if self.sim is not None and hasattr(self.sim, "close"):
                self.sim.close()
            mt = self.cfg.model_type
            if mt == "validation_rig":
                self.sim = ValidationRigSim(self.cfg)
            elif mt == "custom" and self.cfg.custom.path:
                xml = import_custom_model(self.cfg)
                if self._is_bb8_xml(xml):
                    self.sim = BB8BalanceEnv(self.cfg, training=False)
                    self.sim.load_xml(xml)
                    self.sim.reset(seed=int(time.time()) % 100000)
                else:
                    self.sim = GenericSim(xml, self.cfg)
            else:
                self.sim = BB8BalanceEnv(self.cfg, training=False)
                self._reset_bb8()
        except Exception as e:
            self.log(f"Model build failed: {e}\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.critical(self, "Model error", str(e))
            if not isinstance(self.sim, (BB8BalanceEnv, ValidationRigSim, GenericSim)):
                self.cfg.model_type = "bb8"
                self.sim = BB8BalanceEnv(self.cfg, training=False)
                self._reset_bb8()
        self.recorder = TimelineRecorder(int(self.cfg.viewer.record_seconds * self.cfg.robot.control_hz))
        self.telemetry.clear()
        self._attach_viewport(keep_camera=old_cam is not None)
        if old_cam:
            self.viewport.cam.distance, self.viewport.cam.azimuth, self.viewport.cam.elevation = old_cam
        self._populate_tree()
        if self.cfg.viewer.mode == "grid" and not self.trainer.running:
            self._build_local_grid()
        if isinstance(self.sim, ValidationRigSim):
            self.rig_info.setText(f"Sphere R={self.cfg.robot.ball_radius_m:.3f} m, wheel r={self.cfg.robot.wheel_radius_m * 1000:.1f} mm, "
                                  f"zenith {self.cfg.robot.zenith_angle_deg:.1f}°. Predicted ratio R/r = "
                                  f"{self.cfg.robot.ball_radius_m / self.cfg.robot.wheel_radius_m:.2f}")
        self.statusBar().showMessage(f"Model ready: {self.cfg.model_type}  zenith {self.cfg.robot.zenith_angle_deg:.1f}°", 4000)

    @staticmethod
    def _is_bb8_xml(xml: str) -> bool:
        return all(f'name="{n}"' in xml for n in ("ball", "head", "imu", "motor1", "target", "ball_geom"))

    def _attach_viewport(self, keep_camera=False):
        bid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY,
                                "ball" if not isinstance(self.sim, ValidationRigSim) else "test_ball")
        self.viewport.set_model(self.sim.model, self.sim.data, keep_camera=keep_camera, follow_body=bid)
        self._model_version = getattr(self.sim, "model_version", 0)
        if isinstance(self.sim, BB8BalanceEnv) and self.sim.hfield_id >= 0:
            self.viewport.hfield_upload = self.sim.hfield_id

    def _reset_bb8(self):
        env: BB8BalanceEnv = self.sim
        opts = {"target": (self.tx.value(), self.ty.value()) if hasattr(self, "tx") else None,
                "target_enabled": self.target_cb.isChecked() if hasattr(self, "target_cb") else False}
        if opts["target"] is None:
            opts.pop("target")
        o, info = env.reset(seed=int(time.time() * 1000) % 2 ** 31, options=opts)
        self.last_obs_full = info["obs_full"]
        if env.model_version != getattr(self, "_model_version", -1):
            self._attach_viewport(keep_camera=True)
            self._populate_tree()
        elif env.hfield_dirty and env.hfield_id >= 0:
            self.viewport.hfield_upload = env.hfield_id
        env.hfield_dirty = False

    def reset_episode(self):
        if isinstance(self.sim, BB8BalanceEnv):
            self._reset_bb8()
        else:
            self.sim.reset()
        self.recorder.clear()
        self._go_live()

    def manual_push(self):
        if isinstance(self.sim, BB8BalanceEnv):
            az = np.random.uniform(0, 2 * np.pi)
            mag = max(1.0, self.cfg.disturbance.magnitude_n.nominal())
            self.sim.apply_push([mag * math.cos(az), mag * math.sin(az), 0], 0.1, self.cfg.disturbance.target_body)
            self.log(f"Manual push {mag:.1f} N @ {math.degrees(az):.0f}°")

    def toggle_sim(self):
        self.sim_running = not self.sim_running
        if self.sim_running:
            self._go_live()
        else:
            self._capture_live()
        txt = "⏸ Pause Sim" if self.sim_running else "▶ Run Sim"
        self.play_act.setText(txt)
        self.tl_play.setText("⏸" if self.sim_running else "▶")
        self._last_tick = time.perf_counter()

    def _capture_live(self):
        d = self.sim.data
        self._live_state = (d.qpos.copy(), d.qvel.copy(), d.mocap_pos.copy(), d.time, d.act.copy())

    def _scrub_begin(self):
        if self.sim_running:
            self.toggle_sim()

    def _scrub(self, idx: int):
        if self.sim_running or not len(self.recorder) or idx >= len(self.recorder):
            return
        self.scrubbing = True
        fr = self.recorder.frames[idx]
        m, d = self.sim.model, self.sim.data
        if len(fr["qpos"]) == m.nq:
            d.qpos[:] = fr["qpos"]
            if len(fr["mocap"]) == len(d.mocap_pos):
                d.mocap_pos[:] = fr["mocap"]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
        self.viewport.overlay = "SCRUBBING (paused)"
        self.viewport.render_now()
        vals = fr["vals"]
        self.telemetry.set_cursor(fr["t"])
        self.tl_label.setText(f"t = {fr['t']:.2f} s  tilt {vals.get('tilt_deg', 0):.1f}°")

    def _go_live(self):
        if self._live_state is not None:
            m, d = self.sim.model, self.sim.data
            q, qv, mc, t, act = self._live_state
            if len(q) == m.nq:
                d.qpos[:], d.qvel[:], d.mocap_pos[:], d.act[:] = q, qv, mc, act
                d.time = t
                mujoco.mj_forward(m, d)
            self._live_state = None
        self.scrubbing = False
        if not self.sim_running:
            self.sim_running = True
            self.play_act.setText("⏸ Pause Sim")
            self.tl_play.setText("⏸")
        self._last_tick = time.perf_counter()

    def _policy_action(self, obs_full: np.ndarray, n_act: int) -> np.ndarray:
        if self.policy is None:
            return np.zeros(n_act)
        try:
            a = np.asarray(self.policy(obs_full), dtype=np.float64)
        except Exception as e:
            self.log(f"Policy error ({e}); falling back to zero action")
            self.policy = None
            return np.zeros(n_act)
        out = np.zeros(n_act)
        out[:min(n_act, len(a))] = a[:n_act]
        return out

    def _sim_step(self):
        s = self.sim
        if isinstance(s, BB8BalanceEnv):
            a = self._policy_action(self.last_obs_full, s.action_space.shape[0])
            o, r, term, trunc, info = s.step(a)
            self.last_obs_full = info["obs_full"]
            ob = info["obs_full"]
            vals = {"tilt_deg": info["tilt_deg"], "roll_deg": math.degrees(ob[0]), "pitch_deg": math.degrees(ob[1]),
                    "yaw_deg": math.degrees(ob[2]), "gx": ob[3], "gy": ob[4], "gz": ob[5], "ax": ob[6], "ay": ob[7],
                    "az": ob[8], "cmd1": info["motor_cmd"][0], "cmd2": info["motor_cmd"][1],
                    "cmd3": info["motor_cmd"][2], "enc1": ob[9], "enc2": ob[10], "enc3": ob[11],
                    "tq1": info["motor_torque"][0], "tq2": info["motor_torque"][1], "tq3": info["motor_torque"][2],
                    "reward": r, "ball_x": info["ball_xy"][0], "ball_y": info["ball_xy"][1],
                    "target_x": info["target_xy"][0], "target_y": info["target_xy"][1],
                    "target_dist": info["target_dist"], "slip1": float(info["slip"][0]),
                    "slip2": float(info["slip"][1]), "slip3": float(info["slip"][2]),
                    "push_n": float(np.linalg.norm(info["push_force"])),
                    "missed_steps": float(info["missed_steps"].sum())}
            self.recorder.add(info["time"], s.data.qpos, s.data.mocap_pos, vals)
            if term or trunc:
                st = info.get("episode_stats", {})
                self.log(f"Episode end: {'FALL' if st.get('fallen') else 'timeout'}  survival "
                         f"{st.get('survival_s', 0):.2f}s  return {st.get('return', 0):.1f}")
                if self.cfg.episode.auto_reset_on_fall or trunc:
                    self._reset_bb8()
                    self.recorder.clear()
                else:
                    self.toggle_sim()
        elif isinstance(s, ValidationRigSim):
            t = s.step(self.rig_cmd, self.cfg.robot.frame_skip * self.cfg.robot.physics_timestep_s)
            vals = {k: float(v) for k, v in t.items() if k != "time"}
            self.recorder.add(t["time"], s.data.qpos, s.data.mocap_pos, vals)
            self.viewport.overlay = (f"cmd {t['cmd_rads']:.1f} rad/s | enc {t['enc_vel_rads']:.2f} | torque "
                                     f"{t['motor_torque_nm']:.3f}/{t['torque_limit_nm']:.3f} N·m | sphere "
                                     f"{t['ball_speed_meas_rads']:.2f} vs {t['ball_speed_pred_rads']:.2f} rad/s | "
                                     f"slip {t['slip_ratio'] * 100:.1f}%")
        elif isinstance(s, GenericSim):
            a = self._policy_action(s.observe(), 3) if s.n_motor else np.zeros(3)
            s.step(a)
            self.recorder.add(s.data.time, s.data.qpos, s.data.mocap_pos, {"t_sim": s.data.time})

    def _tick(self):
        now = time.perf_counter()
        dt = min(now - self._last_tick, 0.1)
        self._last_tick = now
        mode = self.cfg.viewer.mode
        if self.sim_running and not self.scrubbing and self.sim is not None and mode != "headless":
            ctrl_dt = self.cfg.robot.frame_skip * self.cfg.robot.physics_timestep_s
            self._acc += dt * self.speed
            n = 0
            while self._acc >= ctrl_dt and n < 40:
                try:
                    if mode == "single":
                        self._sim_step()
                    else:
                        self._grid_step()
                except Exception as e:
                    self.log(f"Simulation error: {e}\n{traceback.format_exc()}")
                    self.toggle_sim()
                    break
                self._acc -= ctrl_dt
                n += 1
            if n >= 40:
                self._acc = 0.0
        if mode == "single":
            if isinstance(self.sim, BB8BalanceEnv):
                if self.sim.model_version != self._model_version:
                    self._attach_viewport(keep_camera=True)
                if not self.scrubbing:
                    self.viewport.overlay = (f"{self.fps:4.0f} FPS | t {self.sim.data.time:6.2f}s | tilt "
                                             f"{math.degrees(self.sim.head_tilt()):5.1f}° | policy: {self.policy_source}"
                                             f"{' | target ON' if self.sim.target_enabled else ''}")
            self.viewport.render_now()
        elif mode == "grid" and not self.trainer.running:
            for i, env in enumerate(self.grid_envs):
                self.grid.update_slot(i, env.get_render_state(), model=env.model, data=env.data)
        self._fps_frames += 1
        if now - self._fps_t > 1.0:
            self.fps = self._fps_frames / (now - self._fps_t)
            self._fps_frames, self._fps_t = 0, now
        if now - self._last_plot > 0.1:
            self._last_plot = now
            if not self.scrubbing and mode == "single":
                self.telemetry.refresh(self.recorder.arrays(), live=self.sim_running)
                n = len(self.recorder)
                self.slider.blockSignals(True)
                self.slider.setRange(0, max(0, n - 1))
                if self.sim_running:
                    self.slider.setValue(max(0, n - 1))
                self.slider.blockSignals(False)
                if n:
                    self.tl_label.setText(f"t = {self.recorder.frames[-1]['t']:.2f} s  ({n} frames)")
            self.train_plots.refresh()

    # --------------------------------------------------------------- modes
    def set_mode(self, mode: str):
        self.cfg.viewer.mode = mode
        self.mode_actions[mode].setChecked(True)
        if mode == "single":
            self.stack.setCurrentWidget(self.viewport)
            self._release_local_grid()
        elif mode == "grid":
            self.stack.setCurrentWidget(self.grid)
            self.grid.set_count(self.grid_count.value())
            if not self.trainer.running:
                self._build_local_grid()
        else:
            self.stack.setCurrentWidget(self.headless_lbl)
            self.viewport.release()
            self.grid.release()
            self._release_local_grid()
        self.trainer.send(cmd="set_visual", on=(mode == "grid"), count=self.grid_count.value())
        self.statusBar().showMessage(f"Mode: {mode}", 3000)

    def _grid_count_changed(self, n):
        self.cfg.viewer.grid_count = n
        if self.cfg.viewer.mode == "grid":
            self.set_mode("grid")

    def _build_local_grid(self):
        self._release_local_grid()
        if self.cfg.model_type != "bb8":
            return
        n = self.grid_count.value()
        if len(self.grid.slots) != n:
            self.grid.set_count(n)
        for i in range(n):
            env = BB8BalanceEnv(self.cfg, seed=1000 + i, training=False)
            env.reset(seed=int(time.time()) + i)
            env._last_obs = env._observe()
            self.grid_envs.append(env)

    def _release_local_grid(self):
        for e in self.grid_envs:
            e.close()
        self.grid_envs = []

    def _grid_step(self):
        for env in self.grid_envs:
            a = self._policy_action(env._last_obs, env.action_space.shape[0])
            o, r, term, trunc, info = env.step(a)
            env._last_obs = info["obs_full"]
            if term or trunc:
                env.reset()
                env._last_obs = env._observe()

    # -------------------------------------------------------------- target
    def _target_changed(self, *_):
        on = self.target_cb.isChecked()
        self.cfg.episode.target_enabled = on
        x, y = self.tx.value(), self.ty.value()
        if isinstance(self.sim, BB8BalanceEnv):
            self.sim.set_target(x, y, on)
        for env in self.grid_envs:
            env.set_target(x, y, on)
        if self.trainer.running:
            self.trainer.send(cmd="set_target", x=x, y=y, enabled=on)

    def on_target_picked(self, x, y):
        self.tx.blockSignals(True)
        self.ty.blockSignals(True)
        self.tx.setValue(x)
        self.ty.setValue(y)
        self.tx.blockSignals(False)
        self.ty.blockSignals(False)
        if not self.target_cb.isChecked():
            self.target_cb.setChecked(True)  # triggers _target_changed
        else:
            self._target_changed()

    # ------------------------------------------------------------ inspector
    def _populate_tree(self):
        self.tree.blockSignals(True)
        self.tree.clear()
        m = self.sim.model
        items = {}
        bold = QtGui.QFont()
        bold.setBold(True)
        for b in list_model_tree(m):
            it = QtWidgets.QTreeWidgetItem([b["name"], str(b["id"])])
            it.setData(0, Qt.UserRole, ("body", b["id"]))
            it.setFont(0, bold)
            for gid, gname in b["geoms"]:
                gi = QtWidgets.QTreeWidgetItem([f"◇ {gname}", str(gid)])
                gi.setData(0, Qt.UserRole, ("geom", gid))
                it.addChild(gi)
            parent = items.get(b["parent"]) if b["id"] != 0 else None
            (parent.addChild(it) if parent is not None else self.tree.addTopLevelItem(it))
            items[b["id"]] = it
        self.tree.expandToDepth(1)
        self.tree.blockSignals(False)
        bodies = [""] + [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(1, m.nbody)]
        joints = [""] + [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "" for i in range(m.njnt)
                         if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        for cbx, items_ in ((self.bind_imu, bodies), (self.bind_ball, bodies)):
            cbx.clear()
            cbx.addItems([x for x in items_ if x is not None])
        for cbx in self.bind_wheels:
            cbx.clear()
            cbx.addItems(joints)

    def _tree_selected(self):
        its = self.tree.selectedItems()
        if not its:
            return
        kind, i = its[0].data(0, Qt.UserRole)
        if kind == "body":
            self.on_body_selected(i, -1, from_tree=True)
        else:
            self.on_body_selected(int(self.sim.model.geom_bodyid[i]), i, from_tree=True)

    def _subtree_geoms(self, body: int) -> set:
        m = self.sim.model
        bodies = {body}
        for b in range(m.nbody):
            p = b
            while p > 0:
                if p == body:
                    bodies.add(b)
                    break
                p = m.body_parentid[p]
        return {g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies}

    def on_body_selected(self, body: int, geom: int, from_tree: bool = False):
        self.selected = (body, geom)
        if body <= 0 and geom < 0:
            self.viewport.highlight_geoms = set()
            self._fill_props(None, None)
            self.viewport.render_now()
            return
        if geom >= 0 and from_tree:
            self.viewport.highlight_geoms = {geom}
        else:
            self.viewport.highlight_geoms = self._subtree_geoms(body) if body > 0 else {geom}
        m = self.sim.model
        if geom >= 0 and (from_tree or body <= 0):
            self._fill_props("geom", geom)
        else:
            self._fill_props("body", body)
        if not from_tree:
            self.cfg_tabs.setCurrentIndex(self.cfg_tabs.count() - 1)
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            hits = self.tree.findItems(name, Qt.MatchExactly | Qt.MatchRecursive, 0)
            if hits:
                self.tree.blockSignals(True)
                self.tree.setCurrentItem(hits[0])
                self.tree.blockSignals(False)
        self.viewport.render_now()

    def _fill_props(self, kind, i):
        while self.prop_form.rowCount():
            self.prop_form.removeRow(0)
        self.prop_edits = {}
        self._prop_target = (kind, i)
        if kind is None:
            self.xml_edit.setPlainText("")
            return
        m = self.sim.model
        if kind == "body":
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            fields = {"mass": [m.body_mass[i]], "pos": m.body_pos[i], "ipos": m.body_ipos[i],
                      "inertia": m.body_inertia[i]}
            tag = "body"
        else:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
            fields = {"friction": m.geom_friction[i], "size": m.geom_size[i], "pos": m.geom_pos[i],
                      "rgba": m.geom_rgba[i]}
            tag = "geom"
        self.prop_form.addRow(QtWidgets.QLabel(f"<b>{kind}</b>: {name}"))
        for k, val in fields.items():
            e = QtWidgets.QLineEdit(" ".join(f"{float(x):.6g}" for x in np.atleast_1d(val)))
            self.prop_form.addRow(k, e)
            self.prop_edits[k] = e
        self._prop_name = name
        xml = getattr(self.sim, "xml", "")
        el = self._find_xml_element(xml, tag, name)
        self.xml_edit.setPlainText(ET.tostring(el, encoding="unicode") if el is not None
                                   else f"<!-- <{tag} name='{name}'> not found in source XML -->")

    @staticmethod
    def _find_xml_element(xml: str, tag: str, name: str):
        try:
            root = ET.fromstring(xml)
        except Exception:
            return None
        for el in root.iter(tag):
            if el.get("name") == name:
                return el
        return None

    def apply_properties(self):
        kind, i = getattr(self, "_prop_target", (None, None))
        if kind is None:
            return
        m = self.sim.model
        try:
            for k, e in self.prop_edits.items():
                vals = np.array([float(x) for x in e.text().split()])
                arr = getattr(m, f"{kind}_{k}")
                if vals.size != np.atleast_1d(arr[i]).size:
                    raise ValueError(f"{k}: expected {np.atleast_1d(arr[i]).size} values")
                if isinstance(self.sim, BB8BalanceEnv):
                    self.sim.set_override(kind, self._prop_name, k, vals if vals.size > 1 else vals[0])
                else:
                    arr[i] = vals if vals.size > 1 else vals[0]
            mujoco.mj_setConst(m, self.sim.data)
            mujoco.mj_forward(m, self.sim.data)
            self.log(f"Applied live properties to {kind} '{self._prop_name}' (persist across resets)")
            self.viewport.render_now()
        except Exception as ex:
            QtWidgets.QMessageBox.warning(self, "Invalid value", str(ex))

    def apply_xml(self):
        text = self.xml_edit.toPlainText().strip()
        src = getattr(self.sim, "xml", "")
        try:
            if text.startswith("<mujoco"):
                new_xml = text
            else:
                snippet = ET.fromstring(text)
                root = ET.fromstring(src)
                parent_map = {c: p for p in root.iter() for c in p}
                kind, i = getattr(self, "_prop_target", (None, None))
                target = None
                for el in root.iter(snippet.tag):
                    if el.get("name") == snippet.get("name") or el.get("name") == getattr(self, "_prop_name", None):
                        target = el
                        break
                if target is None:
                    raise ValueError("Could not locate the element to replace (name changed?)")
                par = parent_map[target]
                idx = list(par).index(target)
                par.remove(target)
                par.insert(idx, snippet)
                new_xml = ET.tostring(root, encoding="unicode")
            self._load_xml(new_xml)
            self.log("XML applied and model recompiled")
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "XML error", str(ex))

    def _load_xml(self, xml: str):
        if isinstance(self.sim, BB8BalanceEnv) and self._is_bb8_xml(xml):
            self.sim.load_xml(xml)
            self._reset_bb8()
        else:
            self.sim = GenericSim(xml, self.cfg)
        self.recorder.clear()
        self._attach_viewport(keep_camera=True)
        self._populate_tree()

    def show_full_xml(self):
        self.xml_edit.setPlainText(getattr(self.sim, "xml", ""))

    def revert_xml(self):
        if isinstance(self.sim, BB8BalanceEnv):
            self.sim.load_xml(None)
        self.rebuild_sim()

    def apply_bindings(self):
        if self.cfg.model_type != "custom" or not self.cfg.custom.path:
            QtWidgets.QMessageBox.information(self, "Bindings", "Bindings apply to imported custom models.")
            return
        c = self.cfg.custom
        c.imu_body = self.bind_imu.currentText()
        c.ball_body = self.bind_ball.currentText()
        c.wheel_joints = [cb.currentText() for cb in self.bind_wheels]
        self.rebuild_sim()
        self.log(f"Bound IMU='{c.imu_body}', ball='{c.ball_body}', wheels={c.wheel_joints}")

    # ------------------------------------------------------- model switcher
    def _model_changed(self, idx: int):
        if idx == 2:
            if not self.cfg.custom.path and not self.import_model():
                self.model_combo.setCurrentIndex(1)
                return
            self.cfg.model_type = "custom"
        else:
            self.cfg.model_type = ["validation_rig", "bb8"][idx]
        if self.cfg.model_type != "bb8" and self.cfg.viewer.mode == "grid":
            self.set_mode("single")
        self.rebuild_sim()

    def import_model(self) -> bool:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import robot model", "",
                                                        "Robot models (*.xml *.urdf *.stl *.obj);;All files (*)")
        if not path:
            return False
        c = self.cfg.custom
        c.path = path
        ext = os.path.splitext(path)[1].lower()
        if ext in (".stl", ".obj"):
            mode, ok = QtWidgets.QInputDialog.getItem(
                self, "Mesh import", "Use mesh as:", ["BB-8 head shell on parametric robot (visual)",
                                                      "Standalone free body"], 0, False)
            if not ok:
                return False
            c.import_mode = "head_shell" if mode.startswith("BB-8") else "standalone"
            scale, ok = QtWidgets.QInputDialog.getItem(self, "Mesh units", "Mesh units:",
                                                       ["millimetres (scale 0.001)", "metres (scale 1)"], 0, False)
            c.mesh_scale = 0.001 if scale.startswith("milli") else 1.0
        else:
            c.import_mode = "standalone"
        c.imu_body, c.ball_body, c.wheel_joints = "", "", ["", "", ""]
        self.cfg.model_type = "custom"
        self.model_combo.setCurrentIndex(2)
        self.rebuild_sim()
        self.log(f"Imported {path}. Bind IMU/ball/wheel joints in the Inspector tab to drive it with a policy.")
        return True

    # -------------------------------------------------------------- policy
    def set_policy(self, source: str):
        ws = os.path.abspath(self.cfg.workspace_dir)
        try:
            if source == "zero":
                self.policy = None
            elif source == "initial":
                self.policy = PolicyRunner(onnx_path=ensure_initial_policy(self.cfg))
            elif source == "best":
                p = os.path.join(ws, "best_policy.onnx")
                if not os.path.isfile(p):
                    self.log("No best_policy.onnx yet - using initial policy until training produces one")
                    p = ensure_initial_policy(self.cfg)
                self.policy = PolicyRunner(onnx_path=p)
            elif source == "file":
                p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load ONNX policy", ws, "ONNX (*.onnx)")
                if not p:
                    return
                self.policy = PolicyRunner(onnx_path=p)
            self.policy_source = source
            self.log(f"Viewport policy: {source}" + (f" ({self.policy.source})" if self.policy else ""))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Policy error", str(e))

    # ------------------------------------------------------------- training
    def start_training(self):
        if self.trainer.running:
            return
        if self.cfg.model_type != "bb8":
            QtWidgets.QMessageBox.information(self, "Training", "Training uses the Parametric BB-8 model. "
                                                                "Switch the model selector to 'Parametric BB-8'.")
            return
        snap_path = os.path.join(os.path.abspath(self.cfg.workspace_dir), "initial_policy.json")
        need_new = True
        if os.path.isfile(snap_path):
            s = PolicySnapshot.load_json(snap_path)
            from config_and_distributions import active_obs_names
            need_new = (s.layer_sizes()[1:-1] != list(self.cfg.network.hidden_layers) or
                        s.obs_names != active_obs_names(self.cfg.network.obs_mask) or
                        s.n_act != sum(self.cfg.network.action_mask))
        if need_new:
            ensure_initial_policy(self.cfg, force=True)
            self.log("Generated new initial_policy.onnx for the current architecture")
        self.train_plots.clear()
        self.trainer.start(self.cfg, resume_path=self.resume_path, session_dir=self.loaded_session_dir)
        self.trainer.send(cmd="set_visual", on=self.cfg.viewer.mode == "grid", count=self.grid_count.value())
        if self.cfg.viewer.mode == "grid":
            self._release_local_grid()
            self.grid.set_count(self.grid_count.value())
        self.progress.setRange(0, self.cfg.training.total_timesteps)
        self.log(f"Training started ({self.cfg.training.algorithm}, {self.cfg.training.n_envs} envs, "
                 f"{len(allowed_cores(self.cfg.resources.cpu_percent))} cores)"
                 + (f", resuming {self.resume_path}" if self.resume_path else ""))
        self.resume_path = None
        self._update_train_buttons()

    def pause_training(self):
        self.trainer.pause()
        self.train_lbl.setText("Pausing (checkpoint)…")

    def resume_training(self):
        self.trainer.resume()

    def stop_training(self):
        self.trainer.stop()
        self.train_lbl.setText("Stopping (saving final checkpoint)…")

    def _update_train_buttons(self):
        run = self.trainer.running
        self.btn_start.setEnabled(not run)
        for b in (self.btn_pause, self.btn_resume, self.btn_stop):
            b.setEnabled(run)

    def _poll_trainer(self):
        if self.trainer.msg_q is None:
            return
        latest_frames = None
        for m in self.trainer.poll():
            t = m.get("type")
            if t == "frames":
                latest_frames = m if latest_frames is None else {**m, "xml": {**latest_frames["xml"], **m["xml"]}}
            elif t == "episode":
                self.train_plots.add(m)
            elif t == "status":
                if "timesteps" in m:
                    self.progress.setValue(min(m["timesteps"], self.progress.maximum()))
                fps = f"  {m['fps']:.0f} steps/s" if "fps" in m else ""
                self.train_lbl.setText(f"{m.get('state', '')}  {m.get('timesteps', 0):,} steps{fps}")
            elif t == "best":
                self.log(f"★ New best policy (mean return {m['mean_return']:.1f} @ {m['timesteps']:,}) -> {m['path']}")
                if self.policy_source == "best":
                    self.set_policy("best")
            elif t == "checkpoint":
                self.log(f"Checkpoint [{m['tag']}] {m['path']}")
            elif t == "log":
                self.log(m["text"])
            elif t == "resources":
                self._worker_res = m
            elif t == "error":
                self.log("TRAINING ERROR:\n" + m["trace"])
                QtWidgets.QMessageBox.critical(self, "Training error", m["trace"][-1500:])
            elif t == "finished":
                self.log("Training worker finished")
                QTimer.singleShot(500, self._trainer_finished)
        if latest_frames and self.cfg.viewer.mode == "grid":
            for i, st in enumerate(latest_frames["states"]):
                self.grid.update_slot(i, st, xml=latest_frames["xml"].get(i))
        self._update_train_buttons()

    def _trainer_finished(self):
        if self.trainer.proc is not None and not self.trainer.proc.is_alive():
            self.trainer.proc.join(timeout=1)
        self._update_train_buttons()
        if self.cfg.viewer.mode == "grid":
            self._build_local_grid()

    def save_session_dialog(self):
        ws = os.path.abspath(self.cfg.workspace_dir)
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Save session into folder",
                                                       os.path.join(ws, "sessions"))
        if not d:
            return
        target = os.path.join(d, f"session_{timestamp()}") if os.path.basename(d) == "sessions" else d
        if self.trainer.running:
            self.trainer.send(cmd="save")
            time.sleep(0.2)
        extra = {}
        src = self.trainer.session_dir
        if src and os.path.isdir(src) and os.path.abspath(src) != os.path.abspath(target):
            os.makedirs(target, exist_ok=True)
            for f in ("latest_model.zip", "best_model.zip", "training_progress.csv"):
                if os.path.isfile(os.path.join(src, f)):
                    shutil.copy2(os.path.join(src, f), os.path.join(target, f))
            extra["source_session"] = src
        save_session(self.cfg, target, extra)
        os.makedirs(target, exist_ok=True)
        self.recorder.to_csv(os.path.join(target, "interactive_telemetry.csv"))
        self.log(f"Session saved -> {target}")

    def load_session_dialog(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Load session folder",
                                                       os.path.join(os.path.abspath(self.cfg.workspace_dir), "sessions"))
        if not d:
            return
        try:
            cfg, model = load_session(d)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load session", str(e))
            return
        theme = self.cfg.viewer.theme
        self.cfg = cfg
        self.cfg.viewer.theme = theme
        self.resume_path = model
        self.loaded_session_dir = d
        self._rebuild_config_ui()
        self.rebuild_sim()
        self.log(f"Session loaded from {d}" + (f"; 'Start' resumes from {model}" if model else ""))

    # --------------------------------------------------------------- export
    def _best_or_initial(self) -> Optional[str]:
        ws = os.path.abspath(self.cfg.workspace_dir)
        p = os.path.join(ws, "best_policy.onnx")
        if os.path.isfile(p):
            return p
        r = QtWidgets.QMessageBox.question(self, "No best policy yet",
                                           "No trained best_policy.onnx found. Use the cold-start initial policy?")
        return ensure_initial_policy(self.cfg) if r == QtWidgets.QMessageBox.Yes else None

    def export_best_onnx(self):
        if self.trainer.running:
            self.trainer.send(cmd="export_best")
            time.sleep(0.5)
        src = self._best_or_initial()
        if not src:
            return
        dst, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export best policy", f"bb8_policy_{timestamp()}.onnx",
                                                       "ONNX (*.onnx)")
        if dst:
            shutil.copy2(src, dst)
            js = os.path.splitext(src)[0] + ".json"
            if os.path.isfile(js):
                shutil.copy2(js, os.path.splitext(dst)[0] + ".json")
            self.log(f"Exported {src} -> {dst}")

    def _current_snapshot(self) -> Optional[PolicySnapshot]:
        src = self._best_or_initial()
        if not src:
            return None
        js = os.path.splitext(src)[0] + ".json"
        return PolicySnapshot.load_json(js) if os.path.isfile(js) else load_onnx_snapshot(src)

    def export_cpp(self):
        snap = self._current_snapshot()
        if snap is None:
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Export C++ policy to folder",
                                                       os.path.join(os.path.abspath(self.cfg.workspace_dir), "exports"))
        if d:
            files = generate_cpp(snap, d)
            self.log(f"C++ policy written: {files['cpp']} , {files['h']}")

    def export_csv(self):
        dst, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export telemetry", f"bb8_telemetry_{timestamp()}.csv",
                                                       "CSV (*.csv)")
        if not dst:
            return
        self.recorder.to_csv(dst)
        tp = os.path.splitext(dst)[0] + "_training_curves.csv"
        self.train_plots.to_csv(tp)
        src = os.path.join(self.trainer.session_dir or "", "training_progress.csv")
        if self.trainer.session_dir and os.path.isfile(src):
            shutil.copy2(src, os.path.splitext(dst)[0] + "_training_progress.csv")
        self.log(f"Telemetry exported -> {dst} (+ training curves)")

    def export_bundle(self):
        snap = self._current_snapshot()
        if snap is None:
            return
        out = QtWidgets.QFileDialog.getExistingDirectory(self, "Bundle output folder",
                                                         os.path.join(os.path.abspath(self.cfg.workspace_dir), "exports"))
        if not out:
            return
        tmp = tempfile.mkdtemp(prefix="bb8_bundle_")
        onnx_p = export_onnx(snap, os.path.join(tmp, "bb8_policy.onnx"))
        cpp = generate_cpp(snap, tmp)
        tel = self.recorder.to_csv(os.path.join(tmp, "telemetry.csv"))
        curves = os.path.join(tmp, "training_curves.csv")
        self.train_plots.to_csv(curves)
        xml_p = os.path.join(tmp, "bb8_model.xml")
        with open(xml_p, "w") as f:
            f.write(getattr(self.sim, "xml", ""))
        prog = os.path.join(self.trainer.session_dir or "", "training_progress.csv")
        files = {"policy/bb8_policy.onnx": onnx_p, "policy/bb8_policy.json": snap.save_json(os.path.join(tmp, "p.json")),
                 "cpp/bb8_policy_inference.cpp": cpp["cpp"], "cpp/bb8_policy_inference.h": cpp["h"],
                 "telemetry/telemetry.csv": tel, "telemetry/training_curves.csv": curves,
                 "telemetry/training_progress.csv": prog if os.path.isfile(prog) else None,
                 "model/bb8_model.xml": xml_p}
        z = create_experiment_bundle(out, self.cfg, files, {"README.txt": (
            "BB-8 experiment bundle.\npolicy/: ONNX + JSON snapshot (load with onnxruntime on Jetson)\n"
            "cpp/: zero-dependency inference (compile with deployment_jetson/CMakeLists.txt)\n"
            "model/: exact MuJoCo MJCF used\ntelemetry/: time series + training curves\n"
            "config_snapshot.json: every randomisation & hyper-parameter\n")})
        shutil.rmtree(tmp, ignore_errors=True)
        self.log(f"Experiment bundle -> {z}")
        QtWidgets.QMessageBox.information(self, "Bundle exported", z)

    # ---------------------------------------------------- resources / ESC
    def _cpu_changed(self, txt):
        self.cfg.resources.cpu_percent = int(txt.rstrip("%"))
        self.log(f"CPU governor: {txt} -> {len(allowed_cores(self.cfg.resources.cpu_percent))} of "
                 f"{os.cpu_count()} cores (applies to the next training start)")

    def _resource_monitor(self):
        if psutil is None:
            return
        vm = psutil.virtual_memory()
        rss = process_tree_rss_mb()
        cores = len(allowed_cores(self.cfg.resources.cpu_percent))
        self.res_label.setText(f"  RAM {vm.percent:.0f}% (app {rss:,.0f} MB) | CPU {psutil.cpu_percent():.0f}% | "
                               f"cores {cores}/{os.cpu_count()}  ")
        if vm.percent > self.cfg.resources.ram_ceiling_percent:
            self.res_label.setStyleSheet(f"color: {THEMES[self.cfg.viewer.theme]['danger']}; font-weight: 700;")
            if self.trainer.running and not getattr(self, "_ram_alarm", False):
                self.trainer.send(cmd="save")
                self.log(f"RAM {vm.percent:.0f}% exceeds ceiling -> checkpoint requested")
                self._ram_alarm = True
        else:
            self.res_label.setStyleSheet("")
            self._ram_alarm = False

    def emergency_reset(self):
        """ESC: halt everything immediately, then optionally restore factory defaults."""
        self.timer.stop()
        self.sim_running = False
        self.trainer.kill()
        self._release_local_grid()
        self.grid.release()
        self.viewport.release()
        self._update_train_buttons()
        self.train_lbl.setText("Killed")
        self.log("⛔ EMERGENCY STOP: simulation halted, training workers killed, queues flushed, renderers released")
        r = QtWidgets.QMessageBox.question(self, "Emergency stop",
                                           "All simulations and workers were stopped.\n\n"
                                           "Reset ALL simulation parameters to factory defaults?",
                                           QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                           QtWidgets.QMessageBox.No)
        if r == QtWidgets.QMessageBox.Yes:
            theme, ws = self.cfg.viewer.theme, self.cfg.workspace_dir
            self.cfg = default_config()
            self.cfg.viewer.theme, self.cfg.workspace_dir = theme, ws
            self.resume_path = None
            self.loaded_session_dir = None
            self._rebuild_config_ui()
            self.log("All parameters reset to defaults")
        self.rebuild_sim()
        self.set_mode(self.cfg.viewer.mode)
        self.sim_running = True
        self.play_act.setText("⏸ Pause Sim")
        self.tl_play.setText("⏸")
        self._last_tick = time.perf_counter()
        self.timer.start()

    def _rebuild_config_ui(self):
        idx = self.cfg_tabs.currentIndex()
        old = self.left_dock.widget()
        self._build_config_tabs()
        if old is not None:
            old.deleteLater()
        self.cfg_tabs.setCurrentIndex(idx)
        self.model_combo.setCurrentIndex({"validation_rig": 0, "bb8": 1, "custom": 2}[self.cfg.model_type])
        self.cpu_combo.setCurrentText(f"{self.cfg.resources.cpu_percent}%")
        self.ram_spin.setValue(self.cfg.resources.ram_ceiling_percent)
        self.grid_count.setValue(self.cfg.viewer.grid_count)

    # ---------------------------------------------------------------- misc
    def apply_theme(self, theme: str):
        self.cfg.viewer.theme = theme
        QtWidgets.QApplication.instance().setStyleSheet(make_stylesheet(theme))
        pg.setConfigOption("foreground", THEMES[theme]["plot_fg"])
        self.telemetry.apply_theme(theme)
        self.train_plots.apply_theme(theme)
        self.theme_btn.setText("☾ Dark" if theme == "light" else "☀ Light")
        self.theme_btn.setChecked(theme == "light")

    def log(self, text: str):
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        if hasattr(self, "log_box"):
            self.log_box.appendPlainText(line)
        print(line, flush=True)

    def closeEvent(self, ev):
        self.timer.stop()
        if self.trainer.running:
            r = QtWidgets.QMessageBox.question(self, "Quit", "Training is running. Save a checkpoint and stop it?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if r == QtWidgets.QMessageBox.Yes:
                self.trainer.stop()
                t0 = time.time()
                while self.trainer.running and time.time() - t0 < 15:
                    QtWidgets.QApplication.processEvents()
                    time.sleep(0.05)
        self.trainer.kill()
        self._release_local_grid()
        self.grid.release()
        self.viewport.release()
        super().closeEvent(ev)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=APP_NAME)
    ap.add_argument("--config", help="load an AppConfig JSON")
    ap.add_argument("--workspace", help="workspace directory (default ./workspace)")
    ap.add_argument("--mode", choices=["single", "grid", "headless"])
    ap.add_argument("--screenshot", help="render the window to PNG after N seconds and quit (CI / docs)")
    ap.add_argument("--screenshot-delay", type=float, default=3.0)
    ap.add_argument("--self-test", action="store_true",
                    help="headless check of physics, ONNX/C++ export and a short multi-process training run")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test(args.workspace)
    cfg = AppConfig.load(args.config) if args.config else default_config()
    if args.workspace:
        cfg.workspace_dir = args.workspace
    elif getattr(sys, "frozen", False):  # packaged .exe: keep the workspace next to the executable
        cfg.workspace_dir = os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "workspace")
    if args.mode:
        cfg.viewer.mode = args.mode
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    pg.setConfigOptions(antialias=True)
    win = MainWindow(cfg)
    win.show()
    if args.screenshot:
        def snap():
            win.grab().save(args.screenshot)
            win.close()
            app.quit()
        QTimer.singleShot(int(args.screenshot_delay * 1000), snap)
    ret = app.exec()
    shutdown_qt(app, win)
    return ret


def self_test(workspace: Optional[str] = None) -> int:
    """Used by build_exe.bat to verify a packaged .exe (frozen multiprocessing included)."""
    ws = workspace or os.path.join(tempfile.gettempdir(), f"bb8_selftest_{os.getpid()}")
    cfg = default_config()
    cfg.workspace_dir = ws
    t0 = time.time()
    try:
        env = BB8BalanceEnv(cfg)
        env.reset(seed=0)
        for _ in range(50):
            env.step(env.action_space.sample())
        print(f"[1/3] physics OK ({cfg.robot.ball_radius_m * 200:.0f} cm ball)", flush=True)
        ensure_initial_policy(cfg)
        generate_cpp(PolicySnapshot.load_json(os.path.join(ws, "initial_policy.json")), os.path.join(ws, "exports"))
        print("[2/3] cold-start ONNX + C++ export OK", flush=True)
        tc = cfg.training
        tc.n_envs, tc.vec_env, tc.n_steps, tc.batch_size, tc.n_epochs = 2, "subproc", 256, 128, 1
        tc.total_timesteps, tc.best_window_episodes = 1024, 2
        cfg.viewer.mode = "headless"
        ctl = TrainingController()
        ctl.start(cfg)
        msgs = []
        while time.time() - t0 < 600:
            msgs += ctl.poll()
            if any(m["type"] == "finished" for m in msgs):
                break
            time.sleep(0.2)
        ctl.kill()
        errs = [m for m in msgs if m["type"] == "error"]
        if errs or not any(m["type"] == "episode" for m in msgs):
            print("[3/3] training FAILED:\n" + (errs[0]["trace"] if errs else "no episodes / timeout"), flush=True)
            return 1
        print(f"[3/3] multi-process training OK ({sum(m['type'] == 'episode' for m in msgs)} episodes)", flush=True)
        print(f"SELF-TEST PASSED in {time.time() - t0:.0f} s  (workspace {ws})", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


def shutdown_qt(app, win) -> None:
    """Destroy widgets while the interpreter is alive (avoids pyqtgraph/PySide atexit teardown errors)."""
    win.deleteLater()
    app.processEvents()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


if __name__ == "__main__":
    sys.exit(main())
