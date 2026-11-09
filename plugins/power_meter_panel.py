"""Example plugin: GUI panel with live mechanical motor power and energy (battery sizing).

Shows how to add a custom tab to the telemetry dock. Mechanical power per motor = torque x speed.
"""
PLUGIN_INFO = {"name": "Power meter panel", "version": "1.0",
               "description": "Live motor torque x speed power, peak and energy per episode (GUI tab)."}


def register(api):
    def panel(win):
        from PySide6 import QtCore, QtWidgets
        w = QtWidgets.QWidget()
        lay = QtWidgets.QFormLayout(w)
        labels = {k: QtWidgets.QLabel("-") for k in ("p1", "p2", "p3", "total", "peak", "energy")}
        for k, txt in (("p1", "Motor 1 [W]"), ("p2", "Motor 2 [W]"), ("p3", "Motor 3 [W]"),
                       ("total", "Total mech. power [W]"), ("peak", "Peak [W]"), ("energy", "Energy this run [J]")):
            lay.addRow(txt, labels[k])
        note = QtWidgets.QLabel("Mechanical output only; electrical draw of TMC2208 + steppers is typically 3-6x higher.")
        note.setWordWrap(True)
        lay.addRow(note)
        st = {"peak": 0.0, "energy": 0.0, "t": None}

        def tick():
            env = getattr(win, "sim", None)
            if env is None or not hasattr(env, "motor_torque") or not hasattr(env, "enc_vel"):
                return
            p = [abs(float(t * v)) for t, v in zip(env.motor_torque, env.enc_vel)]
            tot = sum(p)
            t = float(env.data.time)
            if st["t"] is not None and t > st["t"]:
                st["energy"] += tot * (t - st["t"])
            elif st["t"] is not None and t < st["t"]:
                st["energy"], st["peak"] = 0.0, 0.0
            st["t"] = t
            st["peak"] = max(st["peak"], tot)
            for i in range(3):
                labels[f"p{i + 1}"].setText(f"{p[i]:.2f}")
            labels["total"].setText(f"{tot:.2f}")
            labels["peak"].setText(f"{st['peak']:.2f}")
            labels["energy"].setText(f"{st['energy']:.1f}")

        timer = QtCore.QTimer(w, interval=200, timeout=tick)
        timer.start()
        return w

    api.gui_panel("Power meter", panel)
