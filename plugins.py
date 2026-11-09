"""
plugins.py - add-on system for the BB-8 RL Suite
================================================

Drop a ``*.py`` file into the ``plugins/`` folder, enable it in *Tools -> Plugin Manager*
(or list it in ``AppConfig.plugins.enabled``) and it is loaded in the GUI **and** in every
training subprocess. A plugin is a plain Python file::

    PLUGIN_INFO = {"name": "Wind gusts", "version": "1.0", "description": "Random wind on the head"}

    def register(api):
        strength = api.setting("strength_n", 3.0, "mean gust force [N]")
        def on_step(env):
            env.data.xfrc_applied[env.head_bid, 0] += strength
        api.on_step(on_step)

Available hooks (``api`` methods):

=====================  ===========================================================================
``reward_term``        ``fn(env) -> float`` added to the reward every control step (x weight)
``on_reset``           ``fn(env)`` after every episode reset (after domain randomisation)
``on_step``            ``fn(env)`` before every control step's physics (apply forces, move targets)
``scene_xml``          ``fn(cfg) -> (asset_xml, worldbody_xml)`` injected into the generated MJCF
``controller``         ``factory(cfg) -> callable(obs_full) -> 3 actions`` (GUI policy source)
``gui_panel``          ``factory(main_window) -> QWidget`` shown as a tab in the telemetry dock
``gui_action``         ``fn(main_window)`` added to the *Tools -> Plugins* menu
``setting``            declare a user-editable setting (stored in ``cfg.plugins.settings``)
=====================  ===========================================================================

Observation channels are intentionally *not* pluggable: they are the contract with the ONNX /
C++ / Jetson code (see ``config_and_distributions.OBS_GROUPS``).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

PLUGIN_DIR_NAME = "plugins"


def app_root() -> str:
    """Folder next to the .exe when frozen, else the source folder."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def plugin_dirs(cfg=None) -> List[str]:
    dirs = []
    d = getattr(getattr(cfg, "plugins", None), "directory", "") if cfg is not None else ""
    if d:
        dirs.append(d if os.path.isabs(d) else os.path.join(app_root(), d))
    dirs.append(os.path.join(app_root(), PLUGIN_DIR_NAME))
    if getattr(sys, "frozen", False):  # examples bundled inside the exe
        dirs.append(os.path.join(getattr(sys, "_MEIPASS", app_root()), PLUGIN_DIR_NAME))
    out = []
    for x in dirs:
        x = os.path.abspath(x)
        if x not in out:
            out.append(x)
    return out


def discover_plugins(cfg=None) -> List[Dict[str, str]]:
    """List available plugin files with their PLUGIN_INFO (without executing register)."""
    found, seen = [], set()
    for d in plugin_dirs(cfg):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".py") or f.startswith("_") or f in seen:
                continue
            seen.add(f)
            path = os.path.join(d, f)
            info = {"file": f, "path": path, "name": f[:-3], "version": "", "description": ""}
            try:
                mod = _import_file(path)
                info.update({k: str(v) for k, v in getattr(mod, "PLUGIN_INFO", {}).items()})
            except Exception as e:  # broken plugin: still list it so the user sees the error
                info["description"] = f"LOAD ERROR: {e}"
                info["error"] = traceback.format_exc()
            found.append(info)
    return found


def _import_file(path: str):
    name = "bb8_plugin_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class _Hook:
    plugin: str
    fn: Callable
    name: str = ""
    weight: float = 1.0
    disabled: bool = False


@dataclass
class PluginRegistry:
    rewards: List[_Hook] = field(default_factory=list)
    resets: List[_Hook] = field(default_factory=list)
    steps: List[_Hook] = field(default_factory=list)
    scenes: List[_Hook] = field(default_factory=list)
    controllers: Dict[str, _Hook] = field(default_factory=dict)
    panels: List[_Hook] = field(default_factory=list)
    actions: List[_Hook] = field(default_factory=list)
    settings: Dict[str, Dict[str, Tuple[Any, str]]] = field(default_factory=dict)   # plugin -> key -> (default, doc)
    loaded: List[Dict[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.rewards or self.resets or self.steps or self.scenes or self.controllers or self.panels
                    or self.actions)

    # ---- safe invocation ------------------------------------------------
    def _call(self, hook: _Hook, *args, default=None):
        if hook.disabled:
            return default
        try:
            return hook.fn(*args)
        except Exception:
            hook.disabled = True
            msg = f"[plugin {hook.plugin}] hook '{hook.name or hook.fn.__name__}' disabled after error:\n" \
                  f"{traceback.format_exc()}"
            self.errors.append(msg)
            print(msg, file=sys.stderr)
            return default

    def reward_terms(self, env) -> Dict[str, float]:
        out = {}
        for h in self.rewards:
            v = self._call(h, env, default=0.0)
            out[f"plugin:{h.name}"] = float(h.weight) * float(v or 0.0)
        return out

    def run_reset(self, env) -> None:
        for h in self.resets:
            self._call(h, env)

    def run_step(self, env) -> None:
        for h in self.steps:
            self._call(h, env)

    def scene_fragments(self, cfg) -> Tuple[str, str]:
        assets, bodies = [], []
        for h in self.scenes:
            r = self._call(h, cfg, default=("", ""))
            if r:
                assets.append(r[0] or "")
                bodies.append(r[1] or "")
        return "\n    ".join(a for a in assets if a), "\n    ".join(b for b in bodies if b)


class PluginAPI:
    """Object handed to each plugin's ``register(api)``."""

    def __init__(self, registry: PluginRegistry, plugin: str, cfg=None):
        self._r, self._p, self._cfg = registry, plugin, cfg

    @property
    def cfg(self):
        return self._cfg

    def setting(self, key: str, default: Any, description: str = "") -> Any:
        self._r.settings.setdefault(self._p, {})[key] = (default, description)
        store = getattr(getattr(self._cfg, "plugins", None), "settings", None)
        if store is None:
            return default
        val = store.get(self._p, {}).get(key, default)
        try:
            return type(default)(val) if default is not None and not isinstance(default, bool) else val
        except (TypeError, ValueError):
            return default

    def reward_term(self, name: str, fn: Callable, weight: float = 1.0) -> None:
        w = getattr(getattr(self._cfg, "plugins", None), "reward_weights", {}).get(name, weight)
        self._r.rewards.append(_Hook(self._p, fn, name, float(w)))

    def on_reset(self, fn: Callable) -> None:
        self._r.resets.append(_Hook(self._p, fn))

    def on_step(self, fn: Callable) -> None:
        self._r.steps.append(_Hook(self._p, fn))

    def scene_xml(self, fn: Callable) -> None:
        self._r.scenes.append(_Hook(self._p, fn))

    def controller(self, name: str, factory: Callable) -> None:
        self._r.controllers[name] = _Hook(self._p, factory, name)

    def gui_panel(self, name: str, factory: Callable) -> None:
        self._r.panels.append(_Hook(self._p, factory, name))

    def gui_action(self, name: str, fn: Callable) -> None:
        self._r.actions.append(_Hook(self._p, fn, name))


_CACHE: Dict[tuple, PluginRegistry] = {}
EMPTY = PluginRegistry()


def resolve_plugin_path(entry: str, cfg=None) -> Optional[str]:
    if os.path.isabs(entry) and os.path.isfile(entry):
        return entry
    for d in plugin_dirs(cfg):
        p = os.path.join(d, entry if entry.endswith(".py") else entry + ".py")
        if os.path.isfile(p):
            return p
    return None


def load_plugins(cfg, force: bool = False) -> PluginRegistry:
    """Registry for ``cfg.plugins.enabled`` (cached per process; reloaded when files change)."""
    pc = getattr(cfg, "plugins", None)
    enabled = list(getattr(pc, "enabled", []) or [])
    if not enabled:
        return EMPTY
    paths = [resolve_plugin_path(e, cfg) for e in enabled]
    key = tuple((p, os.path.getmtime(p) if p else 0) for p in paths) + \
        (repr(sorted((getattr(pc, "settings", {}) or {}).items())), repr(sorted((getattr(pc, "reward_weights", {}) or {}).items())))
    if not force and key in _CACHE:
        return _CACHE[key]
    reg = PluginRegistry()
    for entry, path in zip(enabled, paths):
        if not path:
            reg.errors.append(f"plugin '{entry}' not found in {plugin_dirs(cfg)}")
            continue
        try:
            mod = _import_file(path)
            name = os.path.splitext(os.path.basename(path))[0]
            if not hasattr(mod, "register"):
                raise AttributeError("plugin has no register(api) function")
            mod.register(PluginAPI(reg, name, cfg))
            info = {"file": os.path.basename(path), "path": path, "name": name}
            info.update({k: str(v) for k, v in getattr(mod, "PLUGIN_INFO", {}).items()})
            reg.loaded.append(info)
        except Exception:
            reg.errors.append(f"plugin '{entry}' failed to load:\n{traceback.format_exc()}")
    _CACHE[key] = reg
    return reg
