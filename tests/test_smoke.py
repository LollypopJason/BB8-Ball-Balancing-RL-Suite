"""Headless smoke tests: python -m pytest tests -q   (or python tests/test_smoke.py)"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_and_distributions import DP, AppConfig, default_config, create_experiment_bundle  # noqa: E402
from bb8_gym_env import BB8BalanceEnv, ValidationRigSim  # noqa: E402
from classical_controller import controller_from_config  # noqa: E402
from mjcf_generator import omniwheel_envelope  # noqa: E402
from rl_onnx_cpp_engine import (PolicyRunner, build_cold_start_snapshot, ensure_initial_policy,  # noqa: E402
                                export_onnx, generate_cpp, load_onnx_snapshot, TrainingController)


def _cfg(tmp):
    c = default_config()
    c.workspace_dir = tmp
    return c


def test_distributions():
    rng = np.random.default_rng(0)
    assert DP.const(3.0).sample(rng) == 3.0
    u = [DP.uniform(1, 2).sample(rng) for _ in range(500)]
    assert 1 <= min(u) and max(u) <= 2
    n = DP.normal(0, 1, lo=-0.5, hi=0.5)
    assert all(-0.5 <= n.sample(rng) <= 0.5 for _ in range(200))
    c = default_config()
    c2 = AppConfig.from_json(c.to_json())
    assert c2.to_dict() == c.to_dict()


def test_env_modes_and_terrain():
    for omni in ("virtual", "roller"):
        c = default_config()
        c.robot.omniwheel_model = omni
        c.terrain.rough_heightfield = True
        c.disturbance.enabled = True
        c.randomization.ball.dent_depth_m = DP.uniform(0.0, 0.004)
        c.episode.target_enabled = True
        env = BB8BalanceEnv(c, seed=0)
        o, _ = env.reset(seed=1)
        assert o.shape == env.observation_space.shape
        for _ in range(50):
            o, r, term, trunc, info = env.step(env.action_space.sample())
            assert np.all(np.isfinite(o))
            if term or trunc:
                env.reset()
        env.set_target(0.5, -0.5)
        env.close()


def test_validation_rig():
    rig = ValidationRigSim()
    rc = rig.cfg.robot
    for cmd in (3.0, 10.0):
        rig.reset()
        ratios = []
        for k in range(150):
            t = rig.step(cmd, 0.01)
            if k >= 50:   # after spin-up
                ratios.append(-t["enc_vel_rads"] / t["ball_speed_meas_rads"])
        assert abs(t["slip_ratio"]) < 0.25, t
        # roller transmission ratio (wheel speed / sphere speed) stays within 5 % of R / r
        ratio = float(np.mean(ratios)) * rc.wheel_radius_m / rc.ball_radius_m
        assert abs(ratio - 1.0) < 0.05, (cmd, ratio)


def test_roller_envelope():
    rc = default_config().robot
    env_r = omniwheel_envelope(rc, rc.wheel_radius_m)
    ripple = 1.0 - env_r.min() / rc.wheel_radius_m
    assert abs(env_r.max() - rc.wheel_radius_m) < 1e-9 and ripple < 0.015, ripple


def _balance(omni: str, seconds: float, **cfg_kw):
    c = default_config()
    c.robot.omniwheel_model = omni
    c.terrain.off_ideal_nominal = True     # nominal physics
    c.episode.max_episode_s = seconds + 1.0
    for k, v in cfg_kw.items():
        setattr(c.robot, k, v)
    env = BB8BalanceEnv(c, seed=0, training=False)
    ctrl = controller_from_config(c)
    _, info = env.reset(seed=0)
    ctrl.reset()
    max_tilt, max_wheel = 0.0, 0.0
    for _ in range(int(round(seconds / env.dt_ctrl))):
        _, _, term, trunc, info = env.step(ctrl(info["obs_full"]))
        max_tilt = max(max_tilt, info["tilt_deg"])
        max_wheel = max(max_wheel, float(np.abs(info["enc_vel"]).max()))
        assert not (term or info["fallen"]), (omni, info["time"], max_tilt)
    env.close()
    return max_tilt, max_wheel, float(info["missed_steps"].sum())


def test_classical_controller_virtual():
    max_tilt, _, missed = _balance("virtual", 5.0)
    assert max_tilt < 5.0 and missed == 0, (max_tilt, missed)


def test_classical_controller_roller():
    """High-fidelity roller model: the classical controller must balance like on the virtual model
    (no wheel spin-ups / stepper saturation from roller chatter or the head hopping off the rollers)."""
    max_tilt, max_wheel, missed = _balance("roller", 15.0)
    assert max_tilt < 5.0, max_tilt
    assert max_wheel < 8.0, max_wheel          # ~2-3 rad/s needed; the old model spiked to 20-45 rad/s
    assert missed < 1000, missed               # micro-steps over 15 s (old model: >10 000 within 2 s)


def test_onnx_and_cpp_roundtrip():
    tmp = tempfile.mkdtemp()
    c = _cfg(tmp)
    p = ensure_initial_policy(c)
    assert os.path.isfile(p)
    snap = build_cold_start_snapshot(c, seed=3)
    # make the outputs non-trivial
    snap.weights[-1] = (np.asarray(snap.weights[-1]) * 100).tolist()
    op = export_onnx(snap, os.path.join(tmp, "p.onnx"))
    s2 = load_onnx_snapshot(op)
    x = np.random.default_rng(0).normal(size=(5, snap.n_obs)) / np.asarray(snap.obs_scale)
    assert np.allclose(s2.forward(x), snap.forward(x), atol=1e-6)
    files = generate_cpp(snap, tmp)
    gxx = shutil.which("g++")
    if gxx:
        main = os.path.join(tmp, "main.cpp")
        with open(main, "w") as f:
            f.write('#include "bb8_policy_inference.h"\n#include <cstdio>\nint main(){float o[%d];'
                    'for(int i=0;i<%d;i++) if(scanf("%%f",&o[i])!=1) return 1; float a[%d];'
                    'bb8::policy_forward(o,a); for(int i=0;i<%d;i++) printf("%%.7f\\n",a[i]); return 0;}'
                    % (snap.n_obs, snap.n_obs, snap.n_act, snap.n_act))
        exe = os.path.join(tmp, "t")
        subprocess.check_call([gxx, "-std=c++11", "-O2", "-Wall", "-Werror", "-o", exe, main, files["cpp"]])
        out = subprocess.run([exe], input=" ".join(f"{v:.9g}" for v in x[0]), capture_output=True, text=True).stdout
        y = np.array([float(v) for v in out.split()])
        assert np.allclose(y, snap.forward(x[0]), atol=1e-5), (y, snap.forward(x[0]))
    r = PolicyRunner(onnx_path=op)
    env = BB8BalanceEnv(c)
    env.reset(seed=0)
    a = r(env._observe())
    assert a.shape == (3,)
    z = create_experiment_bundle(tmp, c, {"policy.onnx": op, "missing.csv": None})
    assert os.path.isfile(z)


def test_short_training():
    tmp = tempfile.mkdtemp()
    c = _cfg(tmp)
    c.training.n_envs = 2
    c.training.vec_env = "dummy"
    c.training.n_steps = 256
    c.training.batch_size = 128
    c.training.n_epochs = 2
    c.training.total_timesteps = 2048
    c.training.best_window_episodes = 2
    c.viewer.mode = "grid"
    tcn = TrainingController()
    tcn.start(c)
    msgs, t0 = [], time.time()
    while time.time() - t0 < 240:
        msgs += tcn.poll()
        if any(m["type"] == "finished" for m in msgs):
            break
        time.sleep(0.2)
    tcn.kill()
    kinds = {m["type"] for m in msgs}
    errs = [m for m in msgs if m["type"] == "error"]
    assert not errs, errs[0]["trace"]
    assert "episode" in kinds and "frames" in kinds and "checkpoint" in kinds, kinds
    assert os.path.isfile(os.path.join(tmp, "best_policy.onnx"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            fn()
            print(f"{name}: OK ({time.time() - t:.1f}s)")
