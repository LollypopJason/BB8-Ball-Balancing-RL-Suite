"""End-to-end GUI exercise with an offscreen Qt platform.

Run:  QT_QPA_PLATFORM=offscreen MUJOCO_GL=osmesa python tests/test_gui_offscreen.py
(without a GL library the viewport shows 'Renderer unavailable' but all logic still runs)
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402

import main_gui as G  # noqa: E402
from config_and_distributions import default_config  # noqa: E402
from mjcf_generator import generate_bb8_xml  # noqa: E402

TMP = tempfile.mkdtemp(prefix="bb8_gui_")


def pump(app, seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.005)


def wait_until(app, cond, timeout=20.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    return cond()


def patch_dialogs(paths):
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (paths.pop(0), ""))
    QtWidgets.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: TMP)
    QtWidgets.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (paths.pop(0), ""))
    QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)
    QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: None)
    QtWidgets.QMessageBox.critical = staticmethod(lambda *a, **k: print("CRITICAL:", a[1:3]))
    QtWidgets.QMessageBox.warning = staticmethod(lambda *a, **k: print("WARNING:", a[1:3]))
    QtWidgets.QInputDialog.getItem = staticmethod(lambda p, t, l, items, *a, **k: (items[0], True))


def main():
    from PySide6 import QtCore
    QtCore.QStandardPaths.setTestModeEnabled(True)      # never touch the user's real settings
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cfg = default_config()
    cfg.workspace_dir = os.path.join(TMP, "ws")
    win = G.MainWindow(cfg)
    win.show()
    pump(app, 1.0)
    env = win.sim
    assert isinstance(env, G.BB8BalanceEnv)
    assert len(win.recorder) > 0, "simulation did not advance"

    # live target
    win.on_target_picked(0.5, 0.3)
    assert np.allclose(win.sim.target, [0.5, 0.3]) and win.sim.target_enabled

    # inspector: select head, edit mass, persists across reset
    head = win.sim.head_bid
    win.on_body_selected(head, -1)
    assert win.viewport.highlight_geoms, "no highlight"
    win.prop_edits["mass"].setText("1.234")
    win.apply_properties()
    win.reset_episode()
    assert abs(win.sim.model.body_mass[win.sim.head_bid] - 1.234) < 1e-9, win.sim.model.body_mass[win.sim.head_bid]

    # XML edit of a geom + recompile
    gid = win.sim.model.geom("head_dome").id
    win.on_body_selected(head, gid, from_tree=True)
    xml = win.xml_edit.toPlainText()
    assert "head_dome" in xml
    win.xml_edit.setPlainText(xml.replace('material="head_mat"', 'material="head_mat" rgba="0.2 0.3 0.9 1"'))
    win.apply_xml()
    assert win.sim.xml_override and 'rgba="0.2 0.3 0.9 1"' in win.sim.xml
    win.revert_xml()
    assert win.sim.xml_override is None

    # zenith slider -> procedural rebuild
    win.zenith_slider.setValue(400)
    assert wait_until(app, lambda: abs(win.sim.frames[0].zenith - np.radians(40)) < 1e-6)

    # pause / scrub / live
    assert wait_until(app, lambda: len(win.recorder) > 5)
    win.toggle_sim()
    n = len(win.recorder)
    assert n > 5
    win.slider.setValue(n // 2)
    assert win.scrubbing
    win._go_live()
    assert win.sim_running and not win.scrubbing

    # validation rig
    win._model_changed(0)
    win.rig_slider.setValue(150)
    assert wait_until(app, lambda: len(win.recorder) > 3)
    assert isinstance(win.sim, G.ValidationRigSim)
    assert "enc_vel_rads" in win.recorder.arrays()
    win._model_changed(1)

    # theme toggle
    win.apply_theme("light")
    win.apply_theme("dark")

    # local grid
    win.grid_count.setValue(4)
    win.set_mode("grid")
    pump(app, 0.8)
    assert len(win.grid_envs) == 4

    # training streamed into the grid
    win.cfg.training.n_envs = 2
    win.cfg.training.vec_env = "dummy"
    win.cfg.training.n_steps = 256
    win.cfg.training.batch_size = 128
    win.cfg.training.n_epochs = 1
    win.cfg.training.total_timesteps = 1536
    win.cfg.training.best_window_episodes = 2
    win.cfg.training.bc_samples = 4000          # tiny warm start for the test
    win.cfg.training.bc_epochs = 3
    win.cfg.training.bc_dagger_rounds = 1
    win.cfg.training.value_warmup_epochs = 2
    win.start_training()
    acks = []
    assert wait_until(app, lambda: any("Warm start done" in b.text() for b in [win.log_box.document().lastBlock()])
                      or "warm-start" not in win.train_lbl.text() and win.train_plots.steps, 240)
    if win.trainer.running:
        win._request_worker("save", ("checkpoint",), lambda ok: acks.append(ok))
        wait_until(app, lambda: acks, 60)
        assert acks in ([True], [False]), acks
    t0 = time.time()
    while (win.trainer.running or win.trainer.msg_q is not None and not win.train_plots.steps) and time.time() - t0 < 180:
        pump(app, 0.2)
        if not win.trainer.running:
            break
    pump(app, 1.0)
    assert win.train_plots.steps, "no training episodes received"
    assert any(s["model"] is not None for s in win.grid.slots), "no streamed frames"

    # exports
    patch_dialogs([os.path.join(TMP, "best.onnx"), os.path.join(TMP, "tel.csv")])
    win.export_best_onnx()
    assert os.path.isfile(os.path.join(TMP, "best.onnx"))
    win.export_cpp()
    assert os.path.isfile(os.path.join(TMP, "bb8_policy_inference.cpp"))
    win.set_mode("single")
    pump(app, 0.5)
    win.export_csv()
    assert os.path.isfile(os.path.join(TMP, "tel.csv"))
    win.export_bundle()
    assert any(f.startswith("bb8_experiment_") and f.endswith(".zip") for f in os.listdir(TMP))

    # policy driving
    win.set_policy("best")
    pump(app, 0.5)
    assert win.policy is not None

    # imports: MJCF (recognised as BB-8), STL head shell, URDF
    xml_p = os.path.join(TMP, "exported_bb8.xml")
    with open(xml_p, "w") as f:
        f.write(generate_bb8_xml(default_config()))
    stl_p = os.path.join(TMP, "shell.stl")
    with open(stl_p, "w") as f:
        f.write("solid s\n")
        pts = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]], float)
        for tri in ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)):
            f.write("facet normal 0 0 0\nouter loop\n")
            for i in tri:
                f.write("vertex %g %g %g\n" % tuple(pts[i]))
            f.write("endloop\nendfacet\n")
        f.write("endsolid s\n")
    urdf_p = os.path.join(TMP, "bot.urdf")
    with open(urdf_p, "w") as f:
        f.write("""<robot name="bot"><link name="base"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01"
        izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial><collision><geometry><box size="0.2 0.2 0.1"/></geometry>
        </collision></link><link name="wheel"><inertial><mass value="0.1"/><inertia ixx="0.001" iyy="0.001"
        izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial><collision><geometry><cylinder radius="0.05" length="0.02"/>
        </geometry></collision></link><joint name="w1" type="continuous"><parent link="base"/><child link="wheel"/>
        <origin xyz="0 0.15 0"/><axis xyz="0 1 0"/></joint></robot>""")
    patch_dialogs([xml_p, stl_p, urdf_p])
    assert win.import_model() and isinstance(win.sim, G.BB8BalanceEnv)
    assert win.import_model() and isinstance(win.sim, G.BB8BalanceEnv) and "head_shell_import" in win.sim.xml
    assert win.import_model() and isinstance(win.sim, G.GenericSim)
    win.bind_imu.setCurrentText("base")
    win.bind_wheels[0].setCurrentText("w1")
    win.apply_bindings()
    assert isinstance(win.sim, G.GenericSim) and win.sim.has_imu and win.sim.n_motor == 1
    pump(app, 0.5)
    win._model_changed(1)

    # ---------------- first-batch features -----------------------------------------------------
    win._model_changed(1)
    # classical controller balances in the viewport (default policy source)
    win.set_policy("classical")
    win.reset_episode()
    assert wait_until(app, lambda: win.recorder.frames and win.recorder.frames[-1]["t"] > 3.0, 120)
    assert max(f["vals"]["tilt_deg"] for f in win.recorder.frames) < 10.0, "classical controller fell in the GUI"
    # menus
    titles = [a.text() for a in win.menuBar().actions()]
    assert {"&File", "&View", "&Tools", "&Help"} <= set(titles), titles
    # validation: invalid hidden layers are rejected and marked
    before = list(win.cfg.network.hidden_layers)
    win.hidden_edit.setText("64, abc")
    win.hidden_edit.editingFinished.emit()
    assert win.cfg.network.hidden_layers == before and "e5484d" in win.hidden_edit.styleSheet()
    # tooltips everywhere
    n_tips = sum(1 for w in win.cfg_tabs.findChildren(QtWidgets.QWidget) if w.toolTip())
    assert n_tips > 60, n_tips
    # presets save / open + reset tab
    preset = os.path.join(TMP, "preset.json")
    win.cfg.robot.zenith_angle_deg = 38.0
    win.cfg.save(preset)
    win.reset_sections(["robot"])
    assert abs(win.cfg.robot.zenith_angle_deg - 31.0) < 1e-9
    win.open_preset(preset)
    assert abs(win.cfg.robot.zenith_angle_deg - 38.0) < 1e-9
    # plugins: enable all examples -> scene, panel tab, controller entry
    win.cfg.plugins.enabled = ["wind_gusts.py", "slalom_course.py", "power_meter_panel.py", "sine_sweep_controller.py",
                               "heading_hold_reward.py"]
    win._refresh_plugins(rebuild=True)
    assert "obst_cone_0" in win.sim.xml
    assert any(win.right_tabs.tabText(i).endswith("Power meter") for i in range(win.right_tabs.count()))
    assert win.policy_combo.findData("plugin:Sine sweep") >= 0
    win.set_policy("plugin:Sine sweep")
    pump(app, 0.5)
    win.cfg.plugins.enabled = []
    win._refresh_plugins(rebuild=True)
    win.set_policy("classical")
    # verification dialog: prediction + compare + apply
    dlg = G.VerificationDialog(win)
    for i in range(dlg.list.count()):
        if dlg.list.item(i).data(QtCore.Qt.UserRole) == "spin_in_place":
            dlg.list.setCurrentRow(i)
    dlg.compute_prediction()
    dlg.inputs["head_yaw_rate_rads"].setValue(dlg.pred["head_yaw_rate_rads"] * 1.05)
    dlg.compare()
    assert dlg.result.item(0, 4).text() == "PASS"
    for i in range(dlg.list.count()):
        if dlg.list.item(i).data(QtCore.Qt.UserRole) == "measure_robot":
            dlg.list.setCurrentRow(i)
    dlg.inputs["head_mass_kg"].setValue(2.6)
    QtWidgets.QMessageBox.information = staticmethod(lambda *a, **k: None)
    dlg.apply()
    assert abs(win.cfg.robot.head_mass_kg - 2.6) < 1e-9
    dlg.close()
    # overlays, camera presets, screenshot, video
    for k, a in win.overlay_actions.items():
        a.setChecked(True)
    for name in G.CAMERA_PRESETS:
        G.apply_camera_preset(win.viewport, name)
    win.manual_push()
    pump(app, 0.5)
    win.screenshot()
    shots = [f for f in os.listdir(os.path.join(win.cfg.workspace_dir, "exports")) if f.startswith("screenshot_")]
    assert shots or win.viewport.error, "no screenshot written"
    patch_dialogs([os.path.join(TMP, "episode.mp4")])
    win.export_video()
    assert os.path.isfile(os.path.join(TMP, "episode.mp4")) or win.viewport.error
    # Jetson package
    win.export_jetson_package()
    assert any(f.startswith("bb8_jetson_package_") for f in os.listdir(TMP))

    # headless + ESC factory reset
    win.set_mode("headless")
    pump(app, 0.3)
    win.zenith_slider.setValue(350)
    pump(app, 0.4)
    win.emergency_reset()
    assert abs(win.cfg.robot.zenith_angle_deg - 31.0) < 1e-9 and not win.trainer.running
    pump(app, 0.5)
    win.grab().save(os.path.join(TMP, "final.png"))
    win.close()
    G.shutdown_qt(app, win)
    print("GUI exercise OK ->", TMP)


if __name__ == "__main__":
    main()
