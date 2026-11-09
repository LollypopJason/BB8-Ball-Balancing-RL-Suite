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
    for _ in range(100):
        t = rig.step(10.0, 0.01)
    assert abs(t["slip_ratio"]) < 0.25, t


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
    c.training.warm_start = "none"          # warm start has its own test
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



def test_classical_controller_balances_and_cpp_matches():
    from classical_controller import controller_from_config, generate_controller_cpp
    c = default_config()
    c.disturbance.enabled = True
    c.episode.max_episode_s = 10
    env = BB8BalanceEnv(c, training=False)
    env.reset(seed=3)
    obs = env._observe()
    ctrl = controller_from_config(c)
    O, A = [], []
    for _ in range(env.max_steps):
        a = ctrl(obs)
        O.append(obs.copy())
        A.append(a)
        _, _, term, trunc, info = env.step(a)
        obs = info["obs_full"]
        assert not term, f"classical controller fell at t={info['time']:.2f}s"
        if trunc:
            break
    gxx = shutil.which("g++")
    if not gxx:
        return
    d = tempfile.mkdtemp()
    files = generate_controller_cpp(controller_from_config(c), d)
    main = os.path.join(d, "m.cpp")
    with open(main, "w") as f:
        f.write('#include "bb8_policy_inference.h"\n#include <cstdio>\n'
                'int main(){float o[20],a[3];bb8::reset_controller();'
                'while(true){for(int i=0;i<20;i++) if(scanf("%f",&o[i])!=1) return 0;'
                'bb8::policy_forward(o,a); printf("%.7f %.7f %.7f\\n",a[0],a[1],a[2]);}}')
    exe = os.path.join(d, "t")
    subprocess.check_call([gxx, "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror", "-o", exe, main, files["cpp"]])
    inp = "\n".join(" ".join(f"{v:.9g}" for v in o) for o in O[:400])
    out = subprocess.run([exe], input=inp, capture_output=True, text=True).stdout
    Y = np.array([[float(x) for x in line.split()] for line in out.strip().splitlines()])
    assert np.abs(Y - np.array(A[:400])).max() < 2e-3, np.abs(Y - np.array(A[:400])).max()



def test_warm_start_and_curriculum():
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from bb8_gym_env import make_env
    from rl_onnx_cpp_engine import SB3PolicyFn, behaviour_clone, collect_teacher_dataset, evaluate_policy, snapshot_from_sb3
    c = default_config()
    venv = DummyVecEnv([make_env(c.to_dict())])
    m = PPO("MlpPolicy", venv, policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=nn.Tanh))
    data = collect_teacher_dataset(c, 30000, 0.25, workers=2)
    behaviour_clone(m, data, epochs=20, value_epochs=3)
    extra = collect_teacher_dataset(c, 8000, 0.25, workers=2, seed=5, student=snapshot_from_sb3(m, c))
    assert len(extra["X"]) >= 8000
    ev = evaluate_policy(SB3PolicyFn(m, c), c, episodes=2, difficulty=0.0, max_seconds=8)
    assert ev["falls"] == 0, ev
    env = BB8BalanceEnv(c)
    env.set_difficulty(0.0)
    env.reset(seed=1)
    assert abs(env.p["ball_mass"] - c.randomization.ball.mass_kg.nominal()) < 1e-9


def test_plugins():
    from plugins import discover_plugins, load_plugins
    found = {p["file"] for p in discover_plugins()}
    assert {"wind_gusts.py", "heading_hold_reward.py", "slalom_course.py", "sine_sweep_controller.py",
            "power_meter_panel.py"} <= found, found
    c = default_config()
    c.plugins.enabled = sorted(found)
    c.plugins.settings = {"wind_gusts": {"mean_force_n": 2.5}}
    reg = load_plugins(c, force=True)
    assert not reg.errors, reg.errors
    assert "Sine sweep" in reg.controllers and reg.panels and reg.rewards
    env = BB8BalanceEnv(c, training=False)
    assert "obst_cone_0" in env.xml
    env.reset(seed=0)
    assert env.target_enabled                          # slalom sets the first gate
    _, _, _, _, info = env.step(np.zeros(3))
    assert "plugin:heading_hold" in info["reward_terms"]
    assert abs(env.data.xfrc_applied[env.head_bid, 0]) > 0.5     # wind force applied
    sweep = reg.controllers["Sine sweep"].fn(c)
    assert np.asarray(sweep(np.zeros(20))).shape == (3,)



def test_jetson_package():
    import json
    import zipfile
    from deployment_package import build_jetson_package
    tmp = tempfile.mkdtemp()
    c = default_config()
    c.robot.zenith_angle_deg = 38.0
    snap = build_cold_start_snapshot(c)
    onnx_p = export_onnx(snap, os.path.join(tmp, "p.onnx"))
    z = build_jetson_package(c, tmp, onnx_p)
    names = zipfile.ZipFile(z).namelist()
    for need in ("bb8_jetson/robot_hw_config.json", "bb8_jetson/policy/best_policy.onnx",
                 "bb8_jetson/policy/cpp/bb8_policy_inference.cpp", "bb8_jetson/classical/cpp/bb8_policy_inference.cpp",
                 "bb8_jetson/classical_controller.py", "bb8_jetson/jetson_control_loop.py", "bb8_jetson/DEPLOY.md"):
        assert need in names, need
    hw = json.loads(zipfile.ZipFile(z).read("bb8_jetson/robot_hw_config.json"))
    assert hw["zenith_angle_deg"] == 38.0 and hw["ball_radius_m"] == c.robot.ball_radius_m
    assert hw["tmc_uart_ports"]            # wiring fields preserved


def test_verification_examples():
    import verification_examples as V
    c = default_config()
    for ex in V.CATALOG:
        pred = ex.predict(c)
        assert all(k in pred for k in (f.key for f in ex.fields)), ex.key
    rows = V.compare(V.BY_KEY["spin_in_place"], {"head_yaw_rate_rads": -2.0},
                     V.BY_KEY["spin_in_place"].predict(c))
    assert rows[0]["ok"] is False                       # wrong sign must fail
    changes = V.BY_KEY["imu_noise"].apply(c, {"gyro_std_rads": 0.02})
    assert changes and abs(c.randomization.imu.gyro_noise_std_rads.high - 0.04) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            fn()
            print(f"{name}: OK ({time.time() - t:.1f}s)")
