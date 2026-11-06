"""
rl_onnx_cpp_engine.py
=====================

Neural-network, training and export engine.

* ``PolicySnapshot``            - framework-neutral description of the deterministic actor
                                  (raw-unit input scaling, tanh hidden layers, output squash).
* ``build_cold_start_snapshot`` - tanh MLP with orthogonal / Xavier init (numpy, no torch).
* ``ensure_initial_policy``     - writes ``initial_policy.onnx`` (+ .json) if the workspace
                                  has no ``.onnx`` yet, so training always starts from a
                                  valid, inspectable ONNX artefact.
* ``export_onnx``               - hand-built ONNX graph (onnx.helper, opset 17); verified
                                  against the numpy reference forward pass.
* ``generate_cpp``              - zero-dependency ``bb8_policy_inference.cpp/.h`` with the
                                  exact weights and tanh forward pass and a hardware-named API.
* ``snapshot_from_sb3`` / ``load_snapshot_into_sb3`` - bridge to Stable-Baselines3 PPO / SAC.
* ``training_worker_main``      - training subprocess (pause / resume / stop, auto-save by
                                  steps & minutes, best-model tracking + auto ONNX export,
                                  RAM ceiling checkpointing, CPU core governor, live
                                  render-state streaming for the 8-agent grid viewer).
* ``TrainingController``        - GUI-side handle that spawns / kills the worker tree.
"""
from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from config_and_distributions import (ACTION_CHANNELS, ALL_OBS_CHANNELS, ALL_OBS_SCALES, OBS_GROUPS, AppConfig,
                                      active_action_indices, active_obs_indices, ensure_workspace, timestamp)

# ============================================================================
# Policy snapshot (framework neutral)
# ============================================================================
@dataclass
class PolicySnapshot:
    obs_names: List[str]                       # active raw-unit inputs, in order
    obs_scale: List[float]                     # raw -> normalised multiplier
    weights: List[List[List[float]]]           # per layer W[out][in]
    biases: List[List[float]]                  # per layer b[out]
    hidden_activation: str = "tanh"
    output_activation: str = "clip"            # "clip" (PPO mean) | "tanh" (SAC squash)
    action_names: List[str] = field(default_factory=lambda: list(ACTION_CHANNELS))
    action_mode: str = "velocity"              # "velocity" | "torque"
    action_scale: float = 45.0                 # rad/s (velocity) or N*m (torque) at |a| = 1
    steps_per_rad: float = 509.3
    obs_clip: float = 10.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return len(self.obs_names)

    @property
    def n_act(self) -> int:
        return len(self.biases[-1])

    def layer_sizes(self) -> List[int]:
        return [self.n_obs] + [len(b) for b in self.biases]

    def forward(self, obs_raw: np.ndarray) -> np.ndarray:
        """Reference forward pass. ``obs_raw``: (n_obs,) or (N, n_obs) physical units."""
        x = np.atleast_2d(np.asarray(obs_raw, dtype=np.float32)) * np.asarray(self.obs_scale, np.float32)
        x = np.clip(x, -self.obs_clip, self.obs_clip)
        n = len(self.weights)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ np.asarray(W, np.float32).T + np.asarray(b, np.float32)
            if i < n - 1:
                x = np.tanh(x)
        x = np.tanh(x) if self.output_activation == "tanh" else np.clip(x, -1.0, 1.0)
        return x if np.ndim(obs_raw) > 1 else x[0]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "PolicySnapshot":
        return cls(**json.loads(s))

    def save_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return path

    @classmethod
    def load_json(cls, path: str) -> "PolicySnapshot":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())


def _io_spec(cfg: AppConfig):
    idx = active_obs_indices(cfg.network.obs_mask)
    names = [ALL_OBS_CHANNELS[i] for i in idx]
    scale = [float(ALL_OBS_SCALES[i]) for i in idx]
    acts = [ACTION_CHANNELS[i] for i in active_action_indices(cfg.network.action_mask)]
    return names, scale, acts


def _action_scale(cfg: AppConfig) -> float:
    return cfg.robot.stepper_holding_torque_nm if cfg.network.action_mode == "torque" else cfg.robot.stepper_max_speed_rads


def build_cold_start_snapshot(cfg: AppConfig, seed: int = 0) -> PolicySnapshot:
    """Tanh MLP, orthogonal (gain sqrt(2) hidden, 0.01 output - SB3 convention) or
    Xavier-uniform weights, zero biases + tiny random jitter."""
    rng = np.random.default_rng(seed)
    names, scale, acts = _io_spec(cfg)
    sizes = [len(names)] + list(cfg.network.hidden_layers) + [len(acts)]
    Ws, bs = [], []
    for li in range(len(sizes) - 1):
        n_in, n_out = sizes[li], sizes[li + 1]
        last = li == len(sizes) - 2
        if cfg.network.init == "xavier":
            lim = math.sqrt(6.0 / (n_in + n_out)) * (0.01 if last else 1.0)
            W = rng.uniform(-lim, lim, size=(n_out, n_in))
        else:
            a = rng.normal(size=(max(n_out, n_in), min(n_out, n_in)))
            q, r = np.linalg.qr(a)
            q = q * np.sign(np.diag(r))
            W = q if n_out >= n_in else q.T
            W = W[:n_out, :n_in] * (0.01 if last else math.sqrt(2.0))
        Ws.append(W.astype(np.float32).tolist())
        bs.append((rng.normal(0, 1e-3, size=n_out)).astype(np.float32).tolist())
    return PolicySnapshot(names, scale, Ws, bs, "tanh", "clip", acts, cfg.network.action_mode, _action_scale(cfg),
                          cfg.robot.steps_per_rad, meta={"source": "cold_start", "init": cfg.network.init,
                                                         "created": timestamp()})


# ============================================================================
# ONNX export (hand-built graph -> no torch.onnx / onnxscript dependency)
# ============================================================================
def export_onnx(snap: PolicySnapshot, path: str, verify: bool = True) -> str:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    nodes, inits = [], []
    inits.append(numpy_helper.from_array(np.asarray(snap.obs_scale, np.float32)[None, :], "obs_scale"))
    inits.append(numpy_helper.from_array(np.array(-snap.obs_clip, np.float32), "obs_lo"))
    inits.append(numpy_helper.from_array(np.array(snap.obs_clip, np.float32), "obs_hi"))
    inits.append(numpy_helper.from_array(np.array(-1.0, np.float32), "act_lo"))
    inits.append(numpy_helper.from_array(np.array(1.0, np.float32), "act_hi"))
    inits.append(numpy_helper.from_array(np.array(snap.action_scale, np.float32), "act_scale"))
    nodes.append(helper.make_node("Mul", ["obs_raw", "obs_scale"], ["x_scaled"], name="normalise"))
    nodes.append(helper.make_node("Clip", ["x_scaled", "obs_lo", "obs_hi"], ["h0"], name="obs_clip"))
    cur = "h0"
    n = len(snap.weights)
    for i, (W, b) in enumerate(zip(snap.weights, snap.biases)):
        inits.append(numpy_helper.from_array(np.asarray(W, np.float32), f"W{i}"))
        inits.append(numpy_helper.from_array(np.asarray(b, np.float32), f"b{i}"))
        out = f"z{i}"
        nodes.append(helper.make_node("Gemm", [cur, f"W{i}", f"b{i}"], [out], transB=1, name=f"dense{i}"))
        if i < n - 1:
            nodes.append(helper.make_node("Tanh", [out], [f"h{i + 1}"], name=f"tanh{i}"))
            cur = f"h{i + 1}"
        else:
            cur = out
    if snap.output_activation == "tanh":
        nodes.append(helper.make_node("Tanh", [cur], ["action"], name="squash"))
    else:
        nodes.append(helper.make_node("Clip", [cur, "act_lo", "act_hi"], ["action"], name="squash"))
    out_name = "motor_cmd_torque_nm" if snap.action_mode == "torque" else "motor_cmd_rads"
    nodes.append(helper.make_node("Mul", ["action", "act_scale"], [out_name], name="to_physical"))
    graph = helper.make_graph(
        nodes, "bb8_policy",
        [helper.make_tensor_value_info("obs_raw", TensorProto.FLOAT, ["batch", snap.n_obs])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["batch", snap.n_act]),
         helper.make_tensor_value_info(out_name, TensorProto.FLOAT, ["batch", snap.n_act])],
        inits)
    model = helper.make_model(graph, producer_name="bb8-rl-suite", opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # broad ONNX Runtime compatibility (JetPack 4.6 ships ORT 1.10-1.11)
    meta = {"obs_names": json.dumps(snap.obs_names), "obs_scale": json.dumps(snap.obs_scale),
            "action_names": json.dumps(snap.action_names), "action_mode": snap.action_mode,
            "action_scale": str(snap.action_scale), "steps_per_rad": str(snap.steps_per_rad),
            "layer_sizes": json.dumps(snap.layer_sizes()), "hidden_activation": snap.hidden_activation,
            "output_activation": snap.output_activation, "meta": json.dumps(snap.meta)}
    for k, v in meta.items():
        p = model.metadata_props.add()
        p.key, p.value = k, v
    onnx.checker.check_model(model)
    onnx.save(model, path)
    if verify:
        verify_onnx(snap, path)
    return path


def verify_onnx(snap: PolicySnapshot, path: str, n: int = 64, tol: float = 1e-4) -> float:
    try:
        import onnxruntime as ort
    except ImportError:
        return float("nan")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = (np.random.default_rng(0).normal(size=(n, snap.n_obs)) / np.asarray(snap.obs_scale)).astype(np.float32)
    y_ref = snap.forward(x)
    y = sess.run(["action"], {"obs_raw": x})[0]
    err = float(np.max(np.abs(y - y_ref)))
    if err > tol:
        raise RuntimeError(f"ONNX verification failed: max abs err {err:.2e}")
    return err


def load_onnx_snapshot(path: str) -> Optional[PolicySnapshot]:
    """Rebuild a PolicySnapshot from an ONNX file exported by this suite."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path)
    md = {p.key: p.value for p in m.metadata_props}
    if "obs_names" not in md:
        return None
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    Ws, bs, i = [], [], 0
    while f"W{i}" in init:
        Ws.append(init[f"W{i}"].tolist())
        bs.append(init[f"b{i}"].tolist())
        i += 1
    return PolicySnapshot(json.loads(md["obs_names"]), json.loads(md["obs_scale"]), Ws, bs,
                          md.get("hidden_activation", "tanh"), md.get("output_activation", "clip"),
                          json.loads(md["action_names"]), md.get("action_mode", "velocity"),
                          float(md.get("action_scale", 45.0)), float(md.get("steps_per_rad", 509.3)),
                          meta=json.loads(md.get("meta", "{}")))


def ensure_initial_policy(cfg: AppConfig, force: bool = False) -> str:
    """Cold-start rule: if no .onnx exists in the workspace, create ``initial_policy.onnx``."""
    ws = ensure_workspace(cfg)
    existing = [f for f in os.listdir(ws) if f.endswith(".onnx")]
    path = os.path.join(ws, "initial_policy.onnx")
    if existing and not force:
        return os.path.join(ws, "initial_policy.onnx" if "initial_policy.onnx" in existing else existing[0])
    snap = build_cold_start_snapshot(cfg, seed=cfg.training.seed)
    export_onnx(snap, path)
    snap.save_json(os.path.join(ws, "initial_policy.json"))
    return path


# ============================================================================
# Policy runners for the GUI / evaluation
# ============================================================================
class PolicyRunner:
    """Run a policy (snapshot or ONNX file) on the *full raw* observation vector."""

    def __init__(self, snap: Optional[PolicySnapshot] = None, onnx_path: Optional[str] = None):
        self.sess = None
        if onnx_path:
            snap = load_onnx_snapshot(onnx_path) or snap
            try:
                import onnxruntime as ort
                self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            except Exception:
                self.sess = None
        if snap is None:
            raise ValueError("No policy snapshot available")
        self.snap = snap
        self.idx = np.array([ALL_OBS_CHANNELS.index(n) for n in snap.obs_names], dtype=np.int64)
        self.source = onnx_path or snap.meta.get("source", "snapshot")

    def __call__(self, obs_full_raw: np.ndarray) -> np.ndarray:
        x = np.asarray(obs_full_raw, np.float32)[self.idx][None, :]
        if self.sess is not None:
            return self.sess.run(["action"], {"obs_raw": x})[0][0]
        return self.snap.forward(x)[0]


# ============================================================================
# C++ code generation
# ============================================================================
def _f(v: float) -> str:
    """C++ float literal that always parses (e.g. 1 -> 1.0f, 1e-05 -> 1e-05f)."""
    t = f"{float(v):.9g}"
    if not any(ch in t for ch in ".en"):
        t += ".0"
    return t + "f"


def _cpp_array(name: str, arr: np.ndarray) -> str:
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 1:
        body = ", ".join(_f(v) for v in arr)
        return f"static const float {name}[{arr.shape[0]}] = {{{body}}};"
    rows = ",\n    ".join("{" + ", ".join(_f(v) for v in r) + "}" for r in arr)
    return f"static const float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n    {rows}\n}};"


def generate_cpp(snap: PolicySnapshot, out_dir: str, basename: str = "bb8_policy_inference") -> Dict[str, str]:
    """Write ``<basename>.h`` and ``<basename>.cpp`` (C++11, <cmath> only)."""
    os.makedirs(out_dir, exist_ok=True)
    torque = snap.action_mode == "torque"
    unit_doc = {g.channels[i]: g.units for g in OBS_GROUPS for i in range(len(g.channels))}
    act_all = list(ACTION_CHANNELS)
    out_suffix = "torque_nm" if torque else "step_rate"
    fn = "compute_motor_torques" if torque else "compute_stepper_commands"
    in_args = ",\n    ".join(f"float {c}" for c in ALL_OBS_CHANNELS)
    out_args = ",\n    ".join(f"float* out_{a}_{out_suffix}" for a in act_all)
    rads_args = "" if torque else ",\n    " + ",\n    ".join(f"float* out_{a}_rads = nullptr" for a in act_all)
    arg_docs = "\n".join(
        f" *   {c:<18s} [{unit_doc[c]}]{'' if c in snap.obs_names else '  (masked: not used by this policy)'}"
        for c in ALL_OBS_CHANNELS)
    sizes = snap.layer_sizes()
    h = f"""// Auto-generated by BB-8 RL Suite & Deployment Studio - DO NOT EDIT BY HAND
// Policy: {sizes} tanh MLP, output={snap.output_activation}, action_mode={snap.action_mode}
// Zero-dependency (C++11 + <cmath>). Thread-safe, no heap allocation.
#ifndef BB8_POLICY_INFERENCE_H
#define BB8_POLICY_INFERENCE_H

namespace bb8 {{

constexpr int   kNumPolicyInputs   = {snap.n_obs};
constexpr int   kNumPolicyOutputs  = {snap.n_act};
constexpr float kActionScale       = {_f(snap.action_scale)};   // {'N*m' if torque else 'rad/s'} at |action| = 1
constexpr float kStepsPerRad       = {_f(snap.steps_per_rad)};   // 200 full steps * microsteps / (2*pi)

/** Raw policy: active inputs in physical units (order below) -> normalised actions [-1, 1].
 *  Input order: {", ".join(snap.obs_names)}
 *  Output order: {", ".join(snap.action_names)} */
void policy_forward(const float* obs_raw, float* action_norm);

/** Hardware-level entry point. All sensor channels are listed (masked ones are ignored).
{arg_docs}
 *  Outputs: {'motor torque [N*m]' if torque else 'signed stepper step rate [micro-steps/s] (sign = DIR pin)'}
 *  for motors whose action is masked the output is 0. */
void {fn}(
    {in_args},
    {out_args}{rads_args});

}}  // namespace bb8

extern "C" void bb8_policy_forward_c(const float* obs_raw, float* action_norm);

#endif  // BB8_POLICY_INFERENCE_H
"""
    layers_cpp = []
    for i, (W, b) in enumerate(zip(snap.weights, snap.biases)):
        layers_cpp.append(_cpp_array(f"kW{i}", np.asarray(W)))
        layers_cpp.append(_cpp_array(f"kB{i}", np.asarray(b)))
    max_w = max(sizes)
    fwd = []
    n = len(snap.weights)
    for i in range(n):
        n_in, n_out = sizes[i], sizes[i + 1]
        src = "x" if i == 0 else ("bufA" if i % 2 == 1 else "bufB")
        dst = "action_norm" if i == n - 1 else ("bufA" if i % 2 == 0 else "bufB")
        if i == n - 1:
            act = ("std::tanh(acc)" if snap.output_activation == "tanh" else "(acc > 1.f ? 1.f : (acc < -1.f ? -1.f : acc))")
        else:
            act = "std::tanh(acc)"
        fwd.append(f"""    // layer {i}: {n_in} -> {n_out}{' (output)' if i == n - 1 else ' (tanh)'}
    for (int o = 0; o < {n_out}; ++o) {{
        float acc = kB{i}[o];
        for (int k = 0; k < {n_in}; ++k) acc += kW{i}[o][k] * {src}[k];
        {dst}[o] = {act};
    }}""")
    gather = "\n".join(f"    obs[{j}] = {name};" for j, name in enumerate(snap.obs_names))
    outs = []
    for j, a in enumerate(act_all):
        if a in snap.action_names:
            k = snap.action_names.index(a)
            if torque:
                outs.append(f"    *out_{a}_{out_suffix} = act[{k}] * kActionScale;")
            else:
                outs.append(f"    {{ const float w = act[{k}] * kActionScale;\n"
                            f"      *out_{a}_{out_suffix} = w * kStepsPerRad;\n"
                            f"      if (out_{a}_rads) *out_{a}_rads = w; }}")
        else:
            outs.append(f"    *out_{a}_{out_suffix} = 0.f;" + ("" if torque else f" if (out_{a}_rads) *out_{a}_rads = 0.f;"))
    params_sig = ", ".join(f"float {c}" for c in ALL_OBS_CHANNELS) + ", " + \
        ", ".join(f"float* out_{a}_{out_suffix}" for a in act_all) + \
        ("" if torque else ", " + ", ".join(f"float* out_{a}_rads" for a in act_all))
    unused = "\n".join(f"    (void){c};" for c in ALL_OBS_CHANNELS if c not in snap.obs_names)
    cpp = f"""// Auto-generated by BB-8 RL Suite & Deployment Studio - DO NOT EDIT BY HAND
// Network: {sizes}  hidden=tanh  output={snap.output_activation}
// Source: {json.dumps(snap.meta)}
#include "{basename}.h"
#include <cmath>

namespace bb8 {{
namespace {{
{_cpp_array("kObsScale", np.asarray(snap.obs_scale))}
static const float kObsClip = {_f(snap.obs_clip)};
{chr(10).join(layers_cpp)}
}}  // namespace

void policy_forward(const float* obs_raw, float* action_norm) {{
    float x[{snap.n_obs}];
    for (int i = 0; i < {snap.n_obs}; ++i) {{
        float v = obs_raw[i] * kObsScale[i];
        x[i] = v > kObsClip ? kObsClip : (v < -kObsClip ? -kObsClip : v);
    }}
    float bufA[{max_w}];
    float bufB[{max_w}];
    (void)bufA; (void)bufB;
{chr(10).join(fwd)}
}}

void {fn}({params_sig}) {{
{unused}
    float obs[{snap.n_obs}];
{gather}
    float act[{snap.n_act}];
    policy_forward(obs, act);
{chr(10).join(outs)}
}}

}}  // namespace bb8

extern "C" void bb8_policy_forward_c(const float* obs_raw, float* action_norm) {{
    bb8::policy_forward(obs_raw, action_norm);
}}
"""
    hp = os.path.join(out_dir, basename + ".h")
    cp = os.path.join(out_dir, basename + ".cpp")
    with open(hp, "w") as f:
        f.write(h)
    with open(cp, "w") as f:
        f.write(cpp)
    return {"h": hp, "cpp": cp}


# ============================================================================
# Stable-Baselines3 bridge
# ============================================================================
def _linear_layers(seq) -> list:
    import torch.nn as nn
    return [m for m in seq.modules() if isinstance(m, nn.Linear)]


def snapshot_from_sb3(model, cfg: AppConfig, meta: Optional[dict] = None) -> PolicySnapshot:
    names, scale, acts = _io_spec(cfg)
    pol = model.policy
    if hasattr(pol, "actor") and hasattr(pol.actor, "mu"):  # SAC
        lins = _linear_layers(pol.actor.latent_pi) + [pol.actor.mu]
        out_act = "tanh"
    else:  # PPO / A2C
        lins = _linear_layers(pol.mlp_extractor.policy_net) + [pol.action_net]
        out_act = "clip"
    Ws = [l.weight.detach().cpu().numpy().astype(np.float32).tolist() for l in lins]
    bs = [l.bias.detach().cpu().numpy().astype(np.float32).tolist() for l in lins]
    m = {"source": f"sb3_{type(model).__name__}", "timesteps": int(model.num_timesteps), "created": timestamp()}
    m.update(meta or {})
    return PolicySnapshot(names, scale, Ws, bs, "tanh", out_act, acts, cfg.network.action_mode, _action_scale(cfg),
                          cfg.robot.steps_per_rad, meta=m)


def load_snapshot_into_sb3(model, snap: PolicySnapshot) -> bool:
    """Copy cold-start weights into the SB3 actor (if the architecture matches)."""
    import torch
    pol = model.policy
    if hasattr(pol, "actor") and hasattr(pol.actor, "mu"):
        lins = _linear_layers(pol.actor.latent_pi) + [pol.actor.mu]
    else:
        lins = _linear_layers(pol.mlp_extractor.policy_net) + [pol.action_net]
    if len(lins) != len(snap.weights):
        return False
    for l, W, b in zip(lins, snap.weights, snap.biases):
        if tuple(l.weight.shape) != np.asarray(W).shape:
            return False
    with torch.no_grad():
        for l, W, b in zip(lins, snap.weights, snap.biases):
            l.weight.copy_(torch.as_tensor(np.asarray(W, np.float32)))
            l.bias.copy_(torch.as_tensor(np.asarray(b, np.float32)))
    return True


# ============================================================================
# Resource governor helpers
# ============================================================================
def allowed_cores(cpu_percent: int) -> List[int]:
    n = os.cpu_count() or 1
    k = max(1, int(round(n * max(1, min(100, cpu_percent)) / 100.0)))
    return list(range(k))


def apply_resource_limits(cpu_percent: int) -> int:
    cores = allowed_cores(cpu_percent)
    os.environ["OMP_NUM_THREADS"] = str(len(cores))
    os.environ["MKL_NUM_THREADS"] = str(len(cores))
    try:
        import psutil
        p = psutil.Process()
        if hasattr(p, "cpu_affinity"):
            p.cpu_affinity(cores)  # inherited by SubprocVecEnv workers (Linux / Windows)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(max(1, len(cores)))
    except Exception:
        pass
    return len(cores)


def process_tree_rss_mb(pid: Optional[int] = None) -> float:
    import psutil
    try:
        p = psutil.Process(pid or os.getpid())
        procs = [p] + p.children(recursive=True)
        return sum(pp.memory_info().rss for pp in procs if pp.is_running()) / 1e6
    except Exception:
        return 0.0


# ============================================================================
# Training worker (runs in its own process)
# ============================================================================
def _send(q, msg: dict) -> None:
    try:
        q.put_nowait(msg)
    except Exception:
        pass


def training_worker_main(cfg_dict: dict, msg_q, cmd_q, pause_ev, stop_ev, resume_path: Optional[str] = None,
                         session_dir: Optional[str] = None) -> None:
    cfg = AppConfig.from_dict(cfg_dict)
    n_threads = apply_resource_limits(cfg.resources.cpu_percent)
    try:
        _training_loop(cfg, msg_q, cmd_q, pause_ev, stop_ev, resume_path, session_dir, n_threads)
    except Exception:
        _send(msg_q, {"type": "error", "trace": traceback.format_exc()})
    finally:
        _send(msg_q, {"type": "finished"})


def _training_loop(cfg: AppConfig, msg_q, cmd_q, pause_ev, stop_ev, resume_path, session_dir, n_threads):
    import psutil
    import torch.nn as nn
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    from bb8_gym_env import make_env

    ws = ensure_workspace(cfg)
    session_dir = session_dir or os.path.join(ws, "sessions", f"session_{timestamp()}")
    os.makedirs(session_dir, exist_ok=True)
    ckpt_dir = os.path.join(ws, "checkpoints")
    cfg.save(os.path.join(session_dir, "config.json"))
    init_onnx = ensure_initial_policy(cfg)
    tc = cfg.training
    cd = cfg.to_dict()
    fns = [make_env(cd, rank=i, seed=tc.seed) for i in range(tc.n_envs)]
    use_subproc = tc.vec_env == "subproc" and tc.n_envs > 1
    venv = SubprocVecEnv(fns, start_method="spawn") if use_subproc else DummyVecEnv(fns)
    venv = VecMonitor(venv)
    _send(msg_q, {"type": "log", "text": f"Training {tc.algorithm} with {tc.n_envs} envs "
                                         f"({'subproc' if use_subproc else 'dummy'}), {n_threads} CPU threads, "
                                         f"session {session_dir}"})
    algo = PPO if tc.algorithm.upper() == "PPO" else SAC
    hidden = list(cfg.network.hidden_layers)
    if algo is PPO:
        pk = dict(net_arch=dict(pi=hidden, vf=hidden), activation_fn=nn.Tanh, ortho_init=True,
                  log_std_init=tc.log_std_init)
        kwargs = dict(learning_rate=tc.learning_rate, n_steps=tc.n_steps, batch_size=tc.batch_size,
                      n_epochs=tc.n_epochs, gamma=tc.gamma, gae_lambda=tc.gae_lambda, ent_coef=tc.ent_coef,
                      clip_range=tc.clip_range)
    else:
        pk = dict(net_arch=dict(pi=hidden, qf=[256, 256]), activation_fn=nn.Tanh)
        kwargs = dict(learning_rate=tc.learning_rate, batch_size=tc.batch_size, gamma=tc.gamma,
                      buffer_size=tc.sac_buffer_size, train_freq=1, gradient_steps=1)
    if resume_path and os.path.isfile(resume_path):
        model = algo.load(resume_path, env=venv, device=tc.device)
        _send(msg_q, {"type": "log", "text": f"Resumed from {resume_path} @ {model.num_timesteps} steps"})
    else:
        model = algo("MlpPolicy", venv, policy_kwargs=pk, seed=tc.seed, device=tc.device, verbose=0, **kwargs)
        js = os.path.splitext(init_onnx)[0] + ".json"
        snap0 = PolicySnapshot.load_json(js) if os.path.isfile(js) else load_onnx_snapshot(init_onnx)
        ok = snap0 is not None and load_snapshot_into_sb3(model, snap0)
        _send(msg_q, {"type": "log", "text": f"Cold-start weights from {os.path.basename(init_onnx)}: "
                                             f"{'loaded' if ok else 'architecture mismatch, using fresh init'}"})

    class GovernorCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.t_last_save = time.time()
            self.steps_last_save = 0
            self.returns: List[float] = []
            self.best = -math.inf
            self.t_best = 0.0
            self.visual = cfg.viewer.mode != "headless"
            self.grid_count = cfg.viewer.grid_count
            self.t_frame = 0.0
            self.t_res = 0.0
            self.t_status = 0.0
            self.env_versions: Dict[int, int] = {}
            self.progress_path = os.path.join(session_dir, "training_progress.csv")
            new = not os.path.isfile(self.progress_path)
            self.progress_f = open(self.progress_path, "a", newline="")
            self.progress_w = csv.writer(self.progress_f)
            if new:
                self.progress_w.writerow(["timesteps", "wall_s", "episode_return", "survival_s", "mean_tilt_deg",
                                          "max_tilt_deg", "final_target_dist", "fallen", "missed_steps"])
            self.t0 = time.time()
            self.fps_steps, self.fps_t = 0, time.time()

        # -- helpers ----------------------------------------------------
        def save(self, tag: str) -> str:
            path = os.path.join(ckpt_dir, f"ckpt_{tag}_{self.num_timesteps}.zip")
            self.model.save(path)
            self.model.save(os.path.join(session_dir, "latest_model.zip"))
            self.progress_f.flush()
            _send(msg_q, {"type": "checkpoint", "path": path, "tag": tag, "timesteps": self.num_timesteps})
            return path

        def export_best(self, mean_ret: float) -> None:
            snap = snapshot_from_sb3(self.model, cfg, {"mean_return": mean_ret})
            onnx_p = os.path.join(ws, "best_policy.onnx")
            export_onnx(snap, onnx_p)
            snap.save_json(os.path.join(ws, "best_policy.json"))
            self.model.save(os.path.join(session_dir, "best_model.zip"))
            _send(msg_q, {"type": "best", "path": onnx_p, "mean_return": mean_ret, "timesteps": self.num_timesteps})

        def handle_commands(self) -> bool:
            while True:
                try:
                    c = cmd_q.get_nowait()
                except queue.Empty:
                    return True
                kind = c.get("cmd")
                if kind == "set_visual":
                    self.visual = bool(c.get("on", True))
                    self.grid_count = int(c.get("count", self.grid_count))
                elif kind == "save":
                    self.save("manual")
                elif kind == "set_target":
                    self.training_env.env_method("set_target", c["x"], c["y"], c.get("enabled", True))
                elif kind == "export_best":
                    self.export_best(float(np.mean(self.returns[-tc.best_window_episodes:]) if self.returns else 0))
                elif kind == "stop":
                    return False

        def stream_frames(self) -> None:
            n = min(self.grid_count, tc.n_envs)
            idx = list(range(n))
            states = self.training_env.env_method("get_render_state", indices=idx)
            xmls = {}
            for i, st in zip(idx, states):
                if self.env_versions.get(i) != st["model_version"]:
                    xmls[i] = self.training_env.get_attr("xml", indices=[i])[0]
                    self.env_versions[i] = st["model_version"]
            _send(msg_q, {"type": "frames", "states": states, "xml": xmls, "timesteps": self.num_timesteps})

        # -- SB3 hook ---------------------------------------------------
        def _on_step(self) -> bool:
            now = time.time()
            if not self.handle_commands() or stop_ev.is_set():
                return False
            if pause_ev.is_set():
                self.save("pause")
                _send(msg_q, {"type": "status", "state": "paused", "timesteps": self.num_timesteps})
                while pause_ev.is_set() and not stop_ev.is_set():
                    if not self.handle_commands():
                        return False
                    time.sleep(0.05)
                _send(msg_q, {"type": "status", "state": "running", "timesteps": self.num_timesteps})
                self.fps_t, self.fps_steps = time.time(), self.num_timesteps
            for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
                st = info.get("episode_stats") if done else None
                if st:
                    self.returns.append(st["return"])
                    row = [self.num_timesteps, round(now - self.t0, 2), st["return"], st["survival_s"],
                           st["mean_tilt_deg"], st["max_tilt_deg"], st["final_target_dist"], int(st["fallen"]),
                           st["missed_steps"]]
                    self.progress_w.writerow(row)
                    _send(msg_q, {"type": "episode", "timesteps": self.num_timesteps, **st})
            w = tc.best_window_episodes
            if len(self.returns) >= w and now - self.t_best > 20.0:
                mr = float(np.mean(self.returns[-w:]))
                if mr > self.best:  # rate-limited best-model export (ONNX + JSON + SB3 zip)
                    self.best, self.t_best = mr, now
                    self.export_best(mr)
            if (self.num_timesteps - self.steps_last_save >= tc.autosave_every_steps or
                    now - self.t_last_save >= tc.autosave_every_minutes * 60):
                self.save("auto")
                self.t_last_save, self.steps_last_save = now, self.num_timesteps
            if now - self.t_res > 2.0:
                self.t_res = now
                vm = psutil.virtual_memory()
                _send(msg_q, {"type": "resources", "ram_percent": vm.percent, "rss_mb": process_tree_rss_mb(),
                              "cpu_percent": psutil.cpu_percent()})
                if vm.percent > cfg.resources.ram_ceiling_percent:
                    p = self.save("ram_ceiling")
                    _send(msg_q, {"type": "log", "text": f"RAM {vm.percent:.0f}% > ceiling "
                                                         f"{cfg.resources.ram_ceiling_percent:.0f}% -> checkpoint {p}"})
                    if cfg.resources.pause_on_ram_ceiling:
                        pause_ev.set()
            if self.visual and now - self.t_frame > 1 / 25.0:
                self.t_frame = now
                self.stream_frames()
            if now - self.t_status > 1.0:
                fps = (self.num_timesteps - self.fps_steps) / max(now - self.fps_t, 1e-6)
                self.fps_steps, self.fps_t, self.t_status = self.num_timesteps, now, now
                _send(msg_q, {"type": "status", "state": "running", "timesteps": self.num_timesteps, "fps": fps,
                              "total": tc.total_timesteps})
            return True

        def _on_training_end(self) -> None:
            self.save("final")
            self.progress_f.close()

    cb = GovernorCallback()
    _send(msg_q, {"type": "status", "state": "running", "timesteps": model.num_timesteps})
    model.learn(total_timesteps=tc.total_timesteps, callback=cb, reset_num_timesteps=resume_path is None,
                progress_bar=False)
    if cb.returns and cb.best == -math.inf:
        cb.export_best(float(np.mean(cb.returns[-tc.best_window_episodes:])))
    venv.close()
    _send(msg_q, {"type": "status", "state": "stopped", "timesteps": model.num_timesteps})


class TrainingController:
    """GUI-side process manager. All heavy work happens in the spawned worker."""

    def __init__(self):
        self.ctx = mp.get_context("spawn")
        self.proc: Optional[mp.Process] = None
        self.msg_q = None
        self.cmd_q = None
        self.pause_ev = None
        self.stop_ev = None
        self.session_dir: Optional[str] = None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def start(self, cfg: AppConfig, resume_path: Optional[str] = None, session_dir: Optional[str] = None) -> None:
        if self.running:
            raise RuntimeError("Training already running")
        cores = allowed_cores(cfg.resources.cpu_percent)
        os.environ["OMP_NUM_THREADS"] = str(len(cores))
        self.msg_q = self.ctx.Queue(maxsize=2000)
        self.cmd_q = self.ctx.Queue()
        self.pause_ev = self.ctx.Event()
        self.stop_ev = self.ctx.Event()
        ws = ensure_workspace(cfg)
        self.session_dir = session_dir or os.path.join(ws, "sessions", f"session_{timestamp()}")
        self.proc = self.ctx.Process(target=training_worker_main, name="bb8-trainer", daemon=False,
                                     args=(cfg.to_dict(), self.msg_q, self.cmd_q, self.pause_ev, self.stop_ev,
                                           resume_path, self.session_dir))
        # The worker never renders: start it with MuJoCo's GL backend disabled and WITHOUT
        # re-importing the GUI's __main__ module (keeps Qt/OpenGL libraries out of the
        # trainer: faster start-up, less RAM, no GL <-> torch/triton LLVM symbol clashes).
        main = sys.modules.get("__main__")
        saved_file = getattr(main, "__file__", None)
        saved_spec = getattr(main, "__spec__", None)
        saved_gl = os.environ.get("MUJOCO_GL")
        try:
            os.environ["MUJOCO_GL"] = "disable"
            if main is not None and saved_file and getattr(main, "__name__", "") == "__main__":
                del main.__file__
                main.__spec__ = None
            self.proc.start()
        finally:
            if main is not None and saved_file:
                main.__file__ = saved_file
                main.__spec__ = saved_spec
            if saved_gl is None:
                os.environ.pop("MUJOCO_GL", None)
            else:
                os.environ["MUJOCO_GL"] = saved_gl

    def pause(self):
        if self.pause_ev:
            self.pause_ev.set()

    def resume(self):
        if self.pause_ev:
            self.pause_ev.clear()

    def send(self, **cmd):
        if self.cmd_q is not None and self.running:
            self.cmd_q.put(cmd)

    def stop(self, timeout: float = 20.0) -> None:
        """Graceful stop: the worker saves a final checkpoint."""
        if not self.running:
            return
        self.stop_ev.set()
        self.pause_ev.clear()

    def poll(self, max_msgs: int = 500) -> List[dict]:
        out = []
        if self.msg_q is None:
            return out
        for _ in range(max_msgs):
            try:
                out.append(self.msg_q.get_nowait())
            except (queue.Empty, OSError, ValueError, EOFError):
                break
        return out

    def kill(self) -> None:
        """Emergency stop (ESC): terminate the whole worker process tree immediately,
        flush queues and release OS resources."""
        if self.proc is not None:
            try:
                import psutil
                if self.proc.pid and psutil.pid_exists(self.proc.pid):
                    parent = psutil.Process(self.proc.pid)
                    kids = parent.children(recursive=True)
                    for k in kids:
                        try:
                            k.kill()
                        except Exception:
                            pass
                    psutil.wait_procs(kids, timeout=2)
            except Exception:
                pass
            try:
                self.proc.kill()
                self.proc.join(timeout=3)
            except Exception:
                pass
        for q in (self.msg_q, self.cmd_q):
            if q is not None:
                try:
                    while True:
                        q.get_nowait()
                except Exception:
                    pass
                try:
                    q.close()
                    q.cancel_join_thread()
                except Exception:
                    pass
        self.proc = self.msg_q = self.cmd_q = self.pause_ev = self.stop_ev = None


def save_session(cfg: AppConfig, session_dir: str, extra: Optional[dict] = None) -> str:
    os.makedirs(session_dir, exist_ok=True)
    cfg.save(os.path.join(session_dir, "config.json"))
    meta = {"saved": timestamp(), **(extra or {})}
    with open(os.path.join(session_dir, "session.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return session_dir


def load_session(session_dir: str):
    cfg = AppConfig.load(os.path.join(session_dir, "config.json"))
    model = None
    for name in ("latest_model.zip", "best_model.zip"):
        p = os.path.join(session_dir, name)
        if os.path.isfile(p):
            model = p
            break
    return cfg, model


if __name__ == "__main__":
    from config_and_distributions import default_config
    c = default_config()
    p = ensure_initial_policy(c, force=True)
    s = PolicySnapshot.load_json(os.path.join(os.path.dirname(p), "initial_policy.json"))
    print("cold start ->", p, "layers", s.layer_sizes(), "verify err", verify_onnx(s, p))
    print(generate_cpp(s, os.path.join(c.workspace_dir, "exports")))
