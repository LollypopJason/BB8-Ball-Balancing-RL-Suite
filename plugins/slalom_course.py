"""Example plugin: slalom course with gates + automatic waypoint sequencing.

* scene_xml  adds cones to the arena (they collide with the ball and the head).
* on_reset   enables the target and sets it to the first gate.
* on_step    advances to the next gate when the ball is within the reach radius.
Enable it together with "Episode & Target -> Enable target endpoint" style training.
"""
import numpy as np

PLUGIN_INFO = {"name": "Slalom course", "version": "1.0",
               "description": "Cones + automatic waypoint route through the gates (multi-target driving)."}


def register(api):
    n = int(api.setting("gates", 4, "number of gates"))
    spacing = api.setting("spacing_m", 1.0, "distance between gates [m]")
    offset = api.setting("offset_m", 0.5, "sideways slalom amplitude [m]")
    gates = [np.array([spacing * (i + 1), offset * (1 if i % 2 == 0 else -1)]) for i in range(n)]
    progress = {}

    def scene(cfg):
        cones = []
        for i, g in enumerate(gates):
            cx, cy = g[0], g[1] + (0.45 if g[1] > 0 else -0.45)
            cones.append(f'<geom name="obst_cone_{i}" type="cylinder" size="0.05 0.12" pos="{cx:.3f} {cy:.3f} 0.12" '
                         f'rgba="1 0.45 0.05 1" contype="1" conaffinity="5"/>')
            cones.append(f'<geom name="slalom_gate_{i}" type="cylinder" size="0.06 0.001" pos="{g[0]:.3f} {g[1]:.3f} 0.001" '
                         f'rgba="0.2 0.9 1 0.6" contype="0" conaffinity="0" group="1"/>')
        return "", "\n    ".join(cones)

    def on_reset(env):
        progress[id(env)] = 0
        env.set_target(*gates[0], enabled=True)

    def on_step(env):
        i = progress.get(id(env), 0)
        if i < len(gates) and np.linalg.norm(env.ball_pos()[:2] - gates[i]) < env.cfg.episode.target_reach_radius_m * 2:
            i += 1
            progress[id(env)] = i
            if i < len(gates):
                env.set_target(*gates[i], enabled=True)

    api.scene_xml(scene)
    api.on_reset(on_reset)
    api.on_step(on_step)
