# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for "BB-8 RL Suite & Deployment Studio".
# Used by build_exe.bat (Windows); works unchanged on Linux/macOS:  pyinstaller --noconfirm bb8_studio.spec
#
# Environment switches (set by build_exe.bat):
#   BB8_ONEFILE=1   single self-extracting .exe (slower start-up: torch is unpacked to %TEMP% each launch)
#   BB8_CONSOLE=1   keep a console window (shows logs / tracebacks - useful for debugging)
import os

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ONEFILE = os.environ.get("BB8_ONEFILE", "0") == "1"
CONSOLE = os.environ.get("BB8_CONSOLE", "0") == "1"
NAME = "BB8_RL_Studio"

datas, binaries, hiddenimports = [], [], []
# packages with native plugins / data files that static analysis misses
for pkg in ("mujoco", "glfw", "onnxruntime", "stable_baselines3"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
# pyqtgraph loads its Qt-binding templates dynamically; skip examples (they start a Qt app), jupyter & opengl
datas += collect_data_files("pyqtgraph", excludes=["**/examples/**"])
hiddenimports += collect_submodules(
    "pyqtgraph", filter=lambda m: not m.startswith(("pyqtgraph.examples", "pyqtgraph.jupyter", "pyqtgraph.opengl")))
# PyOpenGL picks its platform backend (WGL / GLX / EGL / OSMesa) at runtime
hiddenimports += collect_submodules("OpenGL", filter=lambda m: m.startswith(("OpenGL.platform", "OpenGL.raw.GL",
                                                                             "OpenGL.raw.WGL", "OpenGL.raw.GLX",
                                                                             "OpenGL.raw.EGL", "OpenGL.raw.osmesa",
                                                                             "OpenGL.arrays", "OpenGL.GL")))
hiddenimports += collect_submodules("onnx", filter=lambda m: not m.startswith(("onnx.backend.test", "onnx.reference")))
hiddenimports += ["config_and_distributions", "mjcf_generator", "bb8_gym_env", "rl_onnx_cpp_engine",
                  "psutil", "stable_baselines3.ppo", "stable_baselines3.sac",
                  "stable_baselines3.common.vec_env.subproc_vec_env"]
datas += [("deployment_jetson", "deployment_jetson"), ("README.md", "."), ("requirements.txt", ".")]

a = Analysis(
    ["main_gui.py"],
    pathex=[SPECPATH],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "jupyter", "notebook", "PyQt5", "PyQt6", "PySide2",
              "tensorboard", "triton", "torch.utils.tensorboard", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

if ONEFILE:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name=NAME, console=CONSOLE, upx=False,
              runtime_tmpdir=None)
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=NAME, console=CONSOLE, upx=False)
    coll = COLLECT(exe, a.binaries, a.datas, name=NAME, upx=False)
