#!/usr/bin/env python3
"""
jetson_control_loop.py - BB-8 real-time policy runner for the NVIDIA Jetson Nano
=================================================================================

Pipeline (100 Hz by default):

    Arduino Nano 33 IoT (LSM6DS3 + Mahony)  --I2C 0x42-->  |
    3x AS5600 magnetic encoders via TCA9548A --I2C 0x70--> |  Jetson Nano
                                                            |  obs -> policy (ONNX Runtime or
                                                            |  exported C++ .so) -> wheel rad/s
    3x TMC2208 (UART VACTUAL velocity mode)  <--UART------  |
    JK42HS40-1704-83AF steppers + omniwheels

Backends
--------
* ``--policy best_policy.onnx``         ONNX Runtime (CPU or CUDA EP on JetPack)
* ``--cpp-lib libbb8_policy.so``         exported zero-dependency C++ (built with CMakeLists.txt)
* ``--sil``                              software-in-the-loop: the *same* loop drives the MuJoCo
                                         simulator from this repository instead of hardware
                                         (validates obs assembly, units, signs, odometry).
* ``--dry-run``                          hardware read-only: motors are never commanded.

Dependencies on the Jetson:  pip3 install numpy smbus2 pyserial onnxruntime  (JetPack wheel for GPU)
Run (real-time priority recommended):  sudo chrt -f 80 python3 jetson_control_loop.py --policy best_policy.onnx
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import signal
import struct
import sys
import time
from typing import Dict, List, Optional

import numpy as np

# Full observation channel order - MUST match config_and_distributions.ALL_OBS_CHANNELS
ALL_OBS_CHANNELS = [
    "imu_roll_rad", "imu_pitch_rad", "imu_yaw_rad",
    "imu_gx_rads", "imu_gy_rads", "imu_gz_rads",
    "imu_ax_ms2", "imu_ay_ms2", "imu_az_ms2",
    "enc1_vel_rads", "enc2_vel_rads", "enc3_vel_rads",
    "enc1_pos_rad", "enc2_pos_rad", "enc3_pos_rad",
    "target_dx_m", "target_dy_m",
    "prev_motor1_cmd", "prev_motor2_cmd", "prev_motor3_cmd",
]


def wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ============================================================================
# Geometry / odometry (identical maths to mjcf_generator.omniwheel_jacobian)
# ============================================================================
def omniwheel_jacobian(R: float, r: float, zenith_deg: float, azimuths_deg) -> np.ndarray:
    th = math.radians(zenith_deg)
    rows = []
    for az in azimuths_deg:
        a = math.radians(az)
        m = np.array([-math.cos(th) * math.cos(a), -math.cos(th) * math.sin(a), math.sin(th)])
        rows.append((R / r) * m)
    return np.stack(rows)


class Odometry:
    """Ball position from wheel encoders + IMU gyro/yaw (no-slip rolling assumption).
    wheel_speeds = J @ w_rel,  w_rel = head-relative-to-ball angular velocity (head frame)."""

    def __init__(self, cfg: dict):
        self.R = cfg["ball_radius_m"]
        self.Jinv = np.linalg.inv(omniwheel_jacobian(self.R, cfg["wheel_radius_m"], cfg["zenith_angle_deg"],
                                                     cfg["wheel_azimuths_deg"]))
        self.pos = np.zeros(2)

    def update(self, wheel_speeds: np.ndarray, gyro: np.ndarray, yaw: float, dt: float) -> np.ndarray:
        w_rel = self.Jinv @ wheel_speeds          # head relative to ball
        w_ball = gyro - w_rel                      # ball angular velocity in the head frame
        v_head = np.array([w_ball[1] * self.R, -w_ball[0] * self.R])
        c, s = math.cos(yaw), math.sin(yaw)
        self.pos += dt * np.array([c * v_head[0] - s * v_head[1], s * v_head[0] + c * v_head[1]])
        return self.pos


# ============================================================================
# Hardware drivers
# ============================================================================
class ArduinoIMU:
    """Arduino Nano 33 IoT I2C slave (see arduino_nano33iot_imu_i2c.ino)."""

    def __init__(self, bus, addr: int):
        self.bus, self.addr = bus, addr
        self.last_seq = None
        self.stale = 0

    def calibrate(self):
        self.bus.write_byte(self.addr, 0x01)
        time.sleep(1.3)
        self.bus.write_byte(self.addr, 0x02)

    def read(self) -> Dict[str, np.ndarray]:
        from smbus2 import i2c_msg
        msg = i2c_msg.read(self.addr, 24)
        self.bus.i2c_rdwr(msg)
        b = bytes(msg)
        c = 0
        for x in b[:23]:
            c ^= x
        if b[0] != 0xB8 or c != b[23]:
            raise IOError("IMU packet checksum/magic error")
        v = struct.unpack_from("<9h", b, 2)
        status = b[20] | (b[21] << 8)
        self.stale = self.stale + 1 if b[1] == self.last_seq else 0
        self.last_seq = b[1]
        return {"rpy": np.array(v[0:3]) / 1e4, "gyro": np.array(v[3:6]) / 1e3, "acc": np.array(v[6:9]) / 1e3,
                "ok": bool(status & 1), "age_ms": b[22]}


class AS5600Encoders:
    """3x AS5600 (fixed address 0x36) behind a TCA9548A I2C multiplexer."""

    def __init__(self, bus, mux_addr: int, channels: List[int], addr: int = 0x36, signs=(1, 1, 1)):
        self.bus, self.mux, self.ch, self.addr = bus, mux_addr, channels, addr
        self.signs = np.asarray(signs, float)

    def read_angles(self) -> np.ndarray:
        out = np.zeros(3)
        for i, ch in enumerate(self.ch):
            self.bus.write_byte(self.mux, 1 << ch)
            hi, lo = self.bus.read_i2c_block_data(self.addr, 0x0C, 2)   # RAW ANGLE (12 bit)
            out[i] = ((hi & 0x0F) << 8 | lo) * (2 * math.pi / 4096.0)
        return out * self.signs


class EncoderEstimator:
    """Unwraps absolute single-turn angles and differentiates with a first-order low-pass."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.prev = None
        self.unwrapped = np.zeros(3)
        self.vel = np.zeros(3)

    def update(self, angles: np.ndarray, dt: float):
        if self.prev is None:
            self.prev = angles.copy()
            self.unwrapped = angles.copy()
            return self.vel, wrap_pi(self.unwrapped)
        d = wrap_pi(angles - self.prev)
        self.prev = angles.copy()
        self.unwrapped += d
        self.vel = self.alpha * (d / dt) + (1 - self.alpha) * self.vel
        return self.vel.copy(), wrap_pi(self.unwrapped)


def tmc_crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        for _ in range(8):
            if (crc >> 7) ^ (byte & 0x01):
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
            byte >>= 1
    return crc


class TMC2208UART:
    """TMC2208 single-wire UART (PDN_UART) driven in VACTUAL velocity mode: the driver's
    internal step generator produces the micro-steps - no jittery Linux GPIO pulses."""
    GCONF, IHOLD_IRUN, VACTUAL, CHOPCONF = 0x00, 0x10, 0x22, 0x6C
    MRES = {256: 0, 128: 1, 64: 2, 32: 3, 16: 4, 8: 5, 4: 6, 2: 7, 1: 8}

    def __init__(self, port: str, baud: int, fclk: float, microsteps: int, irun: int, ihold: int):
        import serial
        self.ser = serial.Serial(port, baud, timeout=0.01)
        self.fclk = fclk
        # GCONF: en_spreadcycle (torque at speed) | pdn_disable (UART on PDN) | mstep_reg_select | multistep_filt
        self.write(self.GCONF, (1 << 2) | (1 << 6) | (1 << 7) | (1 << 8))
        self.write(self.CHOPCONF, (0x10000053 & ~(0xF << 24)) | (self.MRES[microsteps] << 24))
        self.write(self.IHOLD_IRUN, (ihold & 31) | ((irun & 31) << 8) | (4 << 16))
        self.write(self.VACTUAL, 0)

    def write(self, reg: int, value: int):
        d = bytes([0x05, 0x00, reg | 0x80]) + (value & 0xFFFFFFFF).to_bytes(4, "big")
        self.ser.write(d + bytes([tmc_crc8(d)]))
        self.ser.reset_input_buffer()   # single-wire: discard our own echo

    def set_velocity(self, usteps_per_s: float):
        v = int(round(usteps_per_s * (1 << 24) / self.fclk))
        v = max(-(1 << 23) + 1, min((1 << 23) - 1, v))
        self.write(self.VACTUAL, v & 0xFFFFFF)

    def close(self):
        try:
            self.set_velocity(0)
            self.ser.close()
        except Exception:
            pass


class RealHardware:
    def __init__(self, cfg: dict, dry_run: bool):
        from smbus2 import SMBus
        self.cfg = cfg
        self.bus = SMBus(cfg["i2c_bus"])
        self.imu = ArduinoIMU(self.bus, cfg["imu_i2c_addr"])
        self.enc = AS5600Encoders(self.bus, cfg["encoder_mux_addr"], cfg["encoder_mux_channels"],
                                  cfg["encoder_i2c_addr"], cfg["encoder_direction_sign"])
        self.dry = dry_run
        self.drivers = []
        if not dry_run:
            if cfg["motor_driver_mode"] != "uart":
                raise SystemExit("Python runner supports UART/VACTUAL only; use the C++ runner for STEP/DIR")
            self.drivers = [TMC2208UART(p, cfg["tmc_uart_baud"], cfg["tmc_fclk_hz"], cfg["microsteps"],
                                        cfg["tmc_irun"], cfg["tmc_ihold"]) for p in cfg["tmc_uart_ports"]]
        self.steps_per_rad = cfg["full_steps_per_rev"] * cfg["microsteps"] / (2 * math.pi)
        self.signs = np.asarray(cfg["motor_direction_sign"], float)

    def startup(self):
        print("Calibrating IMU gyro bias - keep the robot still...")
        self.imu.calibrate()

    def read(self):
        imu = self.imu.read()
        return imu, self.enc.read_angles()

    def command(self, wheel_rads: np.ndarray):
        if self.dry:
            return
        for drv, w, s in zip(self.drivers, wheel_rads, self.signs):
            drv.set_velocity(s * w * self.steps_per_rad)

    def stop(self):
        for d in self.drivers:
            d.close()


class SILHardware:
    """Software-in-the-loop: the MuJoCo BB-8 env plays the role of the robot."""

    def __init__(self, cfg: dict, repo_root: str, sim_config: Optional[str]):
        sys.path.insert(0, repo_root)
        from bb8_gym_env import BB8BalanceEnv
        from config_and_distributions import AppConfig, default_config
        c = AppConfig.load(sim_config) if sim_config else default_config()
        c.network.obs_mask = {k: True for k in c.network.obs_mask}
        c.robot.ball_radius_m = cfg["ball_radius_m"]
        c.robot.wheel_diameter_m = 2 * cfg["wheel_radius_m"]
        c.robot.zenith_angle_deg = cfg["zenith_angle_deg"]
        c.robot.wheel_azimuths_deg = list(cfg["wheel_azimuths_deg"])
        c.robot.control_hz = cfg["control_hz"]
        c.episode.target_enabled = bool(cfg.get("target_enabled", False))
        self.env = BB8BalanceEnv(c, training=False)
        self.env.reset(seed=0, options={"target": (cfg["target_x_m"], cfg["target_y_m"])})
        self.max_speed = c.robot.stepper_max_speed_rads

    def startup(self):
        pass

    def read(self):
        r = self.env.last_raw
        return {"rpy": r["rpy"], "gyro": r["gyro"], "acc": r["acc"], "ok": True, "age_ms": 0}, \
            wrap_pi(self.env.enc_pos) % (2 * math.pi)

    def command(self, wheel_rads: np.ndarray):
        _, _, term, trunc, info = self.env.step(np.clip(wheel_rads / self.max_speed, -1, 1))
        self.fallen = bool(info["fallen"])

    def stop(self):
        self.env.close()


# ============================================================================
# Policies
# ============================================================================
class OnnxPolicy:
    def __init__(self, path: str):
        import onnx
        import onnxruntime as ort
        md = {p.key: p.value for p in onnx.load(path).metadata_props}
        self.obs_names = json.loads(md["obs_names"])
        self.action_names = json.loads(md["action_names"])
        self.action_mode = md.get("action_mode", "velocity")
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=prov)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.sess.run(["action"], {"obs_raw": obs[None, :].astype(np.float32)})[0][0]


class CppPolicy:
    """Exported C++ compiled as a shared library (bb8_policy_forward_c)."""

    def __init__(self, lib: str, meta_path: str):
        self.lib = ctypes.CDLL(lib)
        self.lib.bb8_policy_forward_c.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        if meta_path.endswith(".json"):
            meta = json.load(open(meta_path))
            self.obs_names, self.action_names = meta["obs_names"], meta["action_names"]
        else:
            import onnx
            md = {p.key: p.value for p in onnx.load(meta_path).metadata_props}
            self.obs_names, self.action_names = json.loads(md["obs_names"]), json.loads(md["action_names"])
        self.action_mode = "velocity"
        self.out = (ctypes.c_float * len(self.action_names))()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (ctypes.c_float * len(obs))(*obs.astype(np.float32))
        self.lib.bb8_policy_forward_c(x, self.out)
        return np.array(self.out[:], dtype=np.float32)


# ============================================================================
# Main loop
# ============================================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="BB-8 Jetson Nano policy runner")
    ap.add_argument("--config", default=os.path.join(here, "robot_hw_config.json"))
    ap.add_argument("--policy", help="ONNX policy (default from config)")
    ap.add_argument("--cpp-lib", help="shared library built from bb8_policy_inference.cpp")
    ap.add_argument("--policy-meta", help="ONNX or JSON snapshot giving the input order for --cpp-lib")
    ap.add_argument("--sil", action="store_true", help="software-in-the-loop with the MuJoCo simulator")
    ap.add_argument("--sim-config", help="AppConfig JSON for --sil (e.g. config_snapshot.json from a bundle)")
    ap.add_argument("--dry-run", action="store_true", help="never command the motors")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0 = run forever)")
    ap.add_argument("--target", type=float, nargs=2, metavar=("X", "Y"), help="waypoint in metres")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    if args.target:
        cfg["target_enabled"], cfg["target_x_m"], cfg["target_y_m"] = True, args.target[0], args.target[1]
    if args.cpp_lib:
        policy = CppPolicy(args.cpp_lib, args.policy_meta or os.path.join(here, cfg["policy_onnx"]))
    else:
        p = args.policy or cfg["policy_onnx"]
        policy = OnnxPolicy(p if os.path.isabs(p) or os.path.exists(p) else os.path.join(here, p))
    idx = np.array([ALL_OBS_CHANNELS.index(n) for n in policy.obs_names])
    act_idx = [int(n[-1]) - 1 for n in policy.action_names]
    print(f"Policy inputs ({len(idx)}): {policy.obs_names}\nPolicy outputs: {policy.action_names}")

    hw = SILHardware(cfg, os.path.dirname(here), args.sim_config) if args.sil else RealHardware(cfg, args.dry_run)
    hw.startup()
    dt = 1.0 / cfg["control_hz"]
    enc_est = EncoderEstimator(cfg["encoder_velocity_lpf_alpha"])
    odo = Odometry(cfg)
    target = np.array([cfg["target_x_m"], cfg["target_y_m"]])
    max_w = cfg["max_wheel_speed_rads"]
    prev_cmd = np.zeros(3)
    cutoff = math.radians(cfg["tilt_cutoff_deg"])
    running = True

    def _sig(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    log_f = open(os.path.join(here, cfg["log_csv"]) if not os.path.isabs(cfg["log_csv"]) else cfg["log_csv"], "w",
                 newline="")
    log = csv.writer(log_f)
    log.writerow(["t", "loop_ms"] + ALL_OBS_CHANNELS + ["cmd1_rads", "cmd2_rads", "cmd3_rads", "odo_x", "odo_y"])
    t0 = time.perf_counter()
    next_t = t0
    k, overruns = 0, 0
    try:
        while running:
            t_loop = time.perf_counter()
            imu, angles = hw.read()
            enc_v, enc_p = enc_est.update(angles, dt)
            rpy, gyro, acc = imu["rpy"], imu["gyro"], imu["acc"]
            pos = odo.update(enc_v, gyro, rpy[2], dt)
            if cfg.get("target_enabled"):
                e = target - pos
                n = np.linalg.norm(e)
                if n > 2.0:
                    e *= 2.0 / n
                c, s = math.cos(rpy[2]), math.sin(rpy[2])
                tgt = np.array([c * e[0] + s * e[1], -s * e[0] + c * e[1]])
            else:
                tgt = np.zeros(2)
            obs_full = np.concatenate([rpy, gyro, acc, enc_v, enc_p, tgt, prev_cmd]).astype(np.float32)
            a = np.clip(policy(obs_full[idx]), -1.0, 1.0)
            act = np.zeros(3)
            for j, i in enumerate(act_idx):
                act[i] = a[j]
            tilt = math.acos(max(-1.0, min(1.0, math.cos(rpy[0]) * math.cos(rpy[1]))))
            wheel_cmd = act * max_w
            wheel_cmd[np.abs(wheel_cmd) < cfg["deadband_rads"]] = 0.0
            if tilt > cutoff or not imu["ok"]:
                wheel_cmd[:] = 0.0
                print(f"SAFETY STOP: tilt {math.degrees(tilt):.1f} deg / imu_ok={imu['ok']}")
                hw.command(wheel_cmd)
                break
            hw.command(wheel_cmd)
            prev_cmd = act
            loop_ms = (time.perf_counter() - t_loop) * 1e3
            log.writerow([f"{t_loop - t0:.4f}", f"{loop_ms:.3f}"] + [f"{v:.5g}" for v in obs_full] +
                         [f"{v:.4f}" for v in wheel_cmd] + [f"{pos[0]:.4f}", f"{pos[1]:.4f}"])
            k += 1
            t_run = k * dt if args.sil else t_loop - t0
            if k % int(cfg["control_hz"]) == 0:
                truth = ""
                if args.sil:
                    bp = hw.env.ball_pos()
                    truth = f" true=({bp[0]:+.2f},{bp[1]:+.2f})"
                print(f"t={t_run:6.1f}s tilt={math.degrees(tilt):5.2f}deg loop={loop_ms:5.2f}ms "
                      f"odo=({pos[0]:+.2f},{pos[1]:+.2f}){truth} overruns={overruns}")
            if args.seconds and t_run > args.seconds:
                break
            if getattr(hw, "fallen", False):
                print("SIL: robot fell")
                break
            next_t += dt
            sleep = next_t - time.perf_counter()
            if args.sil:
                continue                          # SIL runs as fast as possible
            if sleep > 0:
                time.sleep(sleep)
            else:
                overruns += 1
                next_t = time.perf_counter()
    finally:
        hw.command(np.zeros(3))
        hw.stop()
        log_f.close()
        print(f"Stopped after {k} cycles, {overruns} overruns. Log: {log_f.name}")


if __name__ == "__main__":
    main()
