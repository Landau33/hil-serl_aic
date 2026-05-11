"""Live monitor for scripted cable-insertion intervention.

Run in a separate terminal while record_demos.py / run_scripted_insertion.py
is up. Subscribes to /scripted_intervention/status and refreshes the latest
phase / errors / stuck flags in place.

Usage:
    python -m examples.experiments.aic_cable_insertion.monitor_scripted_intervention
    # or
    python examples/experiments/aic_cable_insertion/monitor_scripted_intervention.py
"""
from __future__ import annotations

import json
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


TOPIC = "/scripted_intervention/status"
REFRESH_HZ = 10.0

CSI_CLEAR = "\x1b[2J\x1b[H"
CSI_BOLD = "\x1b[1m"
CSI_RESET = "\x1b[0m"
CSI_GREEN = "\x1b[32m"
CSI_YELLOW = "\x1b[33m"
CSI_RED = "\x1b[31m"
CSI_DIM = "\x1b[2m"


class StatusMonitor(Node):
    def __init__(self) -> None:
        super().__init__("scripted_intervention_monitor")
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._latest_time: float | None = None
        self._sub = self.create_subscription(String, TOPIC, self._on_status, 10)

    def _on_status(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self._lock:
            self._latest = data
            self._latest_time = time.monotonic()

    def snapshot(self) -> tuple[dict | None, float | None]:
        with self._lock:
            return self._latest, self._latest_time


def _color(value: float, *, good: float, warn: float) -> str:
    if value <= good:
        return CSI_GREEN
    if value <= warn:
        return CSI_YELLOW
    return CSI_RED


def _bool_tag(label: str, flag: bool, *, true_color: str = CSI_RED) -> str:
    color = true_color if flag else CSI_DIM
    return f"{color}{label}={flag}{CSI_RESET}"


def render(status: dict | None, age: float | None) -> str:
    lines = [f"{CSI_BOLD}Scripted intervention status{CSI_RESET}  ({TOPIC})"]
    if status is None:
        lines.append("")
        lines.append(f"{CSI_DIM}waiting for first message...{CSI_RESET}")
        return "\n".join(lines)

    active = bool(status.get("active", False))
    phase = str(status.get("phase", "?"))
    xy = float(status.get("lateral_error_norm", 0.0))
    z = float(abs(status.get("axial_error", 0.0)))
    ang = float(status.get("angular_error_norm", 0.0))
    align_stuck = bool(status.get("align_stuck", False))
    stuck = bool(status.get("stuck", False))
    stuck_steps = int(status.get("stuck_steps", 0))
    complete = bool(status.get("complete", False))

    active_str = (
        f"{CSI_GREEN}ON{CSI_RESET}" if active else f"{CSI_DIM}off{CSI_RESET}"
    )
    phase_color = {
        "align": CSI_YELLOW, "pause": CSI_DIM, "insert": CSI_GREEN,
    }.get(phase, "")
    complete_str = (
        f"{CSI_GREEN}done{CSI_RESET}" if complete else f"{CSI_DIM}-{CSI_RESET}"
    )

    age_str = (
        f"{CSI_DIM}{age:.2f}s ago{CSI_RESET}" if age is not None else ""
    )

    lines += [
        "",
        f"  active : {active_str}    phase : {phase_color}{phase}{CSI_RESET}    {complete_str}",
        "",
        f"  lateral : {_color(xy, good=0.0015, warn=0.005)}{xy*1000:7.3f} mm{CSI_RESET}",
        f"  axial   : {_color(z, good=0.0015, warn=0.005)}{z*1000:7.3f} mm{CSI_RESET}",
        f"  ang err : {_color(ang, good=0.05, warn=0.15)}{ang:7.4f} rad{CSI_RESET}",
        "",
        f"  {_bool_tag('align_stuck', align_stuck, true_color=CSI_YELLOW)}    "
        f"{_bool_tag('stuck', stuck)}    "
        f"{CSI_DIM}stuck_steps={stuck_steps}{CSI_RESET}",
        "",
        f"  {age_str}",
        f"  {CSI_DIM}Ctrl-C to exit{CSI_RESET}",
    ]
    return "\n".join(lines)


def main() -> int:
    rclpy.init()
    node = StatusMonitor()
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True,
    )
    spin_thread.start()
    period = 1.0 / REFRESH_HZ
    try:
        while rclpy.ok():
            status, t = node.snapshot()
            age = time.monotonic() - t if t is not None else None
            sys.stdout.write(CSI_CLEAR)
            sys.stdout.write(render(status, age))
            sys.stdout.write("\n")
            sys.stdout.flush()
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
