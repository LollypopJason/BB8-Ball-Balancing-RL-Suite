# Plugins

Drop a `*.py` file here and enable it in **Tools → Plugin Manager**. Enabled plugins are loaded in the
GUI *and* in every training subprocess. See the docstring of `../plugins.py` for the full API.

| File | Hooks used | What it does |
|---|---|---|
| `wind_gusts.py` | `on_reset`, `on_step` | continuous random wind on the head (robustness training) |
| `heading_hold_reward.py` | `reward_term` | keeps the eye pointing forward (reward shaping example) |
| `slalom_course.py` | `scene_xml`, `on_reset`, `on_step` | cones + automatic waypoint route through gates |
| `sine_sweep_controller.py` | `controller` | open-loop chirp for motor / encoder verification |
| `power_meter_panel.py` | `gui_panel` | live motor power & energy tab |

Minimal template:

```python
PLUGIN_INFO = {"name": "My plugin", "version": "1.0", "description": "What it does"}

def register(api):
    k = api.setting("gain", 1.0, "shown and editable in the Plugin Manager")
    api.reward_term("my_term", lambda env: -k * abs(env.last_raw["gyro"][2]))
```
