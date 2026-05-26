from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

SCRIPTED_INTERVENTION_CODE_VERSION = "slim_v12_revert_lift_trigger"

# Hardcoded tuning constants (rarely changed in practice).
INSERT_START_XY_TOL_M = 0.0015
INSERT_START_ORIENT_TOL_RAD = 0.05
STUCK_WINDOW_STEPS = 12
STUCK_Z_PROGRESS_M = 0.0005
STUCK_RECOVER_M = 0.002
STUCK_SEARCH_PERIOD = 8
STUCK_SEARCH_RAMP = 6
# Spiral + pulse engage as soon as stuck is detected; the lateral motion is
# fine even far above the port because the relax window below keeps axial
# from dragging the cable across the rim while searching.
STUCK_SEARCH_MAX_AXIAL_M = 0.020
STUCK_PULSE_VEL_M_S = 0.01
STUCK_PULSE_STEPS = 4
STUCK_PULSE_PERIOD = 16
# While stuck, zero out axial push for STUCK_RELAX_STEPS out of every
# STUCK_RELAX_PERIOD search steps. This releases contact friction so the
# spiral search + lateral pulse can actually reposition the cable.
STUCK_RELAX_STEPS = 2
STUCK_RELAX_PERIOD = 8
# Constant-speed lift: retreat at this velocity (m/s) regardless of distance
# to clearance. Lift has no overshoot risk (going away from port), and the
# robot's low-bandwidth velocity tracking means PD-decayed speeds near the
# target threshold are too low to actually move the cable.
LIFT_VELOCITY_M_S = 0.05
# Scale align lateral velocity by min(1, lateral_err / ALIGN_DECAY_RADIUS_M)
# so we don't overshoot the insert-start tolerance.
ALIGN_DECAY_RADIUS_M = 0.005
ANG_GAIN_MULT = 2.0
WARN_INTERVAL_SEC = 2.0


@dataclass(frozen=True)
class ScriptedInterventionConfig:
    toggle_key: str = "0"
    base_frame: str = "base_link"
    tip_frame: str = "cable_1/sc_tip_link"
    port_frame: str = "task_board/sc_port_0/sc_port_base_link"
    # PD gains. xy = perpendicular to port_z; z = along port_z.
    align_linear_gain: float = 8.0
    align_angular_gain: float = 1.5
    insert_xy_gain: float = 12.0
    insert_z_gain: float = 40.0
    insert_angular_gain: float = 0.8
    # Velocity caps.
    max_linear_velocity: float = 0.01
    max_angular_velocity: float = 0.04
    # Insert phase: scales lateral+axial caps relative to max_linear_velocity.
    insert_linear_velocity_scale: float = 1.0 / 3.0
    # Minimum directional velocities while inserting (overrides PD when below).
    min_insert_z_velocity: float = 0.002
    min_insert_xy_correction_velocity: float = 0.002
    min_insert_angular_correction_velocity: float = 0.006
    # Output scaling (policy action ↔ velocity).
    action_scale_linear: float = 0.005
    action_scale_angular: float = 0.02
    # Deadbands and final completion tolerances.
    linear_deadband_m: float = 0.001
    angular_deadband_rad: float = 0.03
    xy_align_tolerance_m: float = 0.001
    z_insert_tolerance_m: float = 0.0015
    orientation_align_tolerance_rad: float = 0.05
    # Lift safety: retreat along -port_z if too close + laterally off.
    safe_axial_clearance_m: float = 0.010
    lift_lateral_threshold_m: float = 0.003
    # Aggressive insert gains when stuck.
    aggressive_insert_xy_gain: float = 30.0
    aggressive_insert_z_gain: float = 25.0
    aggressive_insert_angular_gain: float = 1.2
    aggressive_max_linear_velocity: float = 0.03
    aggressive_xy_velocity_scale: float = 1.0
    aggressive_z_velocity_scale: float = 0.5
    aggressive_min_insert_z_velocity: float = 0.003
    # Spiral search amplitude while stuck.
    stuck_search_linear_velocity: float = 0.005
    # JSONL per-step log path. Empty disables.
    log_path: str = "/tmp/scripted_intervention.log.jsonl"


def _pose_from_transform(tf_msg: Any) -> tuple[np.ndarray, np.ndarray]:
    t = tf_msg.transform.translation
    r = tf_msg.transform.rotation
    return (
        np.array([t.x, t.y, t.z], dtype=np.float64),
        np.array([r.x, r.y, r.z, r.w], dtype=np.float64),
    )


def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= max_norm or n <= 1e-12:
        return v
    return v * (max_norm / n)


def _apply_min_directional_velocity(
    velocity: np.ndarray,
    error: np.ndarray,
    min_velocity: float,
    velocity_limit: float,
) -> None:
    min_velocity = min(max(min_velocity, 0.0), max(velocity_limit, 0.0))
    if min_velocity <= 0.0:
        return
    error_norm = float(np.linalg.norm(error))
    if error_norm <= 1e-9:
        return
    direction = error / error_norm
    current = float(np.dot(velocity, direction))
    if current < min_velocity:
        velocity += (min_velocity - current) * direction
        velocity[:] = _clip_norm(velocity, velocity_limit)


class ScriptedCableInsertionIntervention:
    """Three-phase scripted insertion controller.

    align  : zero axial velocity, drive lateral + angular to within tolerance.
    lift   : when too close to port AND laterally off, retreat along -port_z.
    insert : descend along +port_z; lateral + angular PD stay on; stuck
             recovery (aggressive gains + spiral search + lateral pulse)
             activates when axial progress stalls.

    insert is terminal. Completion → emit zero velocity.
    """

    def __init__(
        self,
        *,
        tf_buffer: Any,
        logger: Any,
        time_type: Any,
        transform_exception: type[Exception],
        config: ScriptedInterventionConfig,
    ):
        self._tf_buffer = tf_buffer
        self._logger = logger
        self._Time = time_type
        self._TransformException = transform_exception
        self._config = config
        self._active = False
        self._last_warning_time = 0.0
        self._log_file = None
        self._log_step = 0
        self._reset_state()

    def _reset_state(self) -> None:
        self._phase = "align"
        self._last_angular_error = np.zeros(3)
        self._last_lateral_error_norm = 0.0
        self._last_axial_error = 0.0
        self._best_axial_err_abs = np.inf
        self._stuck_steps = 0
        self._stuck_active = False
        self._stuck_search_steps = 0

    # -- Public API -----------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active

    @property
    def toggle_key(self) -> str:
        return self._config.toggle_key

    def set_active(self, active: bool) -> bool:
        if self._active == active:
            return self._active
        self._active = active
        self._reset_state()
        if active:
            self._truncate_log()
        self._logger.info(
            f"Scripted intervention {'enabled' if active else 'disabled'}: "
            f"{self._config.tip_frame} -> {self._config.port_frame}"
        )
        return self._active

    def toggle(self) -> bool:
        return self.set_active(not self._active)

    def start(self) -> None:
        self.set_active(True)

    def stop(self) -> None:
        self.set_active(False)

    def status(self) -> dict[str, Any]:
        return {
            "active": self._active,
            "phase": self._phase,
            "lateral_error_norm": float(self._last_lateral_error_norm),
            "axial_error": float(self._last_axial_error),
            "angular_error_norm": float(np.linalg.norm(self._last_angular_error)),
            "stuck": self._stuck_active,
            "stuck_steps": int(self._stuck_steps),
            "complete": self.is_complete(),
        }

    def is_complete(self) -> bool:
        cfg = self._config
        return (
            self._phase == "insert"
            and self._last_lateral_error_norm <= cfg.xy_align_tolerance_m
            and self._last_axial_error <= cfg.z_insert_tolerance_m
            and float(np.linalg.norm(self._last_angular_error))
            <= cfg.orientation_align_tolerance_rad
        )

    # -- Main loop ------------------------------------------------------------

    def get_action(self) -> np.ndarray | None:
        if not self._active:
            return None

        cfg = self._config

        try:
            now = self._Time()
            port_tf = self._tf_buffer.lookup_transform(
                cfg.base_frame, cfg.port_frame, now
            )
            tip_tf = self._tf_buffer.lookup_transform(
                cfg.base_frame, cfg.tip_frame, now
            )
        except self._TransformException as exc:
            self._maybe_warn(f"Scripted intervention TF lookup failed: {exc}")
            return np.zeros((6,), dtype=np.float32)

        port_pos, port_quat = _pose_from_transform(port_tf)
        tip_pos, tip_quat = _pose_from_transform(tip_tf)
        port_rot = Rotation.from_quat(port_quat)
        port_x, port_y, port_z = port_rot.as_matrix().T  # columns = body axes in base

        position_error = port_pos - tip_pos
        angular_error = (port_rot * Rotation.from_quat(tip_quat).inv()).as_rotvec()

        axial_error = float(position_error @ port_z)
        lateral_error_raw = (position_error @ port_x) * port_x + (
            position_error @ port_y
        ) * port_y
        lateral_error_norm = float(np.linalg.norm(lateral_error_raw))
        ang_err_norm = float(np.linalg.norm(angular_error))

        self._last_angular_error = angular_error.copy()
        self._last_lateral_error_norm = lateral_error_norm
        self._last_axial_error = axial_error

        insert_start_aligned = (
            lateral_error_norm <= INSERT_START_XY_TOL_M
            and ang_err_norm <= INSERT_START_ORIENT_TOL_RAD
        )

        # Phase transitions. insert is terminal.
        if self._phase == "align":
            if (
                lateral_error_norm > cfg.lift_lateral_threshold_m
                and axial_error < cfg.safe_axial_clearance_m
            ):
                self._phase = "lift"
                self._logger.info("Scripted intervention: align -> lift")
            elif insert_start_aligned:
                self._phase = "insert"
                self._best_axial_err_abs = abs(axial_error)
                self._stuck_steps = 0
                self._stuck_active = False
                self._stuck_search_steps = 0
                self._logger.info("Scripted intervention: align -> insert")
        elif self._phase == "lift" and axial_error >= cfg.safe_axial_clearance_m:
            self._phase = "align"
            self._logger.info("Scripted intervention: lift -> align")

        # Deadband-zeroed PD inputs (raw values kept for stuck path / lift).
        lateral_pd = (
            lateral_error_raw
            if lateral_error_norm >= cfg.linear_deadband_m
            else np.zeros(3)
        )
        axial_pd = axial_error if axial_error > cfg.z_insert_tolerance_m else 0.0
        angular_pd = (
            angular_error if ang_err_norm >= cfg.angular_deadband_rad else np.zeros(3)
        )

        if self._phase == "lift":
            linear_v, angular_v = self._lift_action(axial_error, port_z)
        elif self._phase == "align":
            linear_v, angular_v = self._align_action(lateral_pd, angular_pd)
        else:
            linear_v, angular_v = self._insert_action(
                lateral_pd=lateral_pd,
                lateral_raw=lateral_error_raw,
                axial_pd=axial_pd,
                axial_raw=axial_error,
                angular_pd=angular_pd,
                stuck_detection_aligned=insert_start_aligned,
                port_z=port_z,
            )

        complete = self.is_complete()
        if complete:
            linear_v = np.zeros(3)
            angular_v = np.zeros(3)

        self._log_step_state(
            complete=complete,
            position_error=position_error,
            lateral_error_raw=lateral_error_raw,
            lateral_error_norm=lateral_error_norm,
            axial_error=axial_error,
            angular_error=angular_error,
            port_x=port_x,
            port_y=port_y,
            port_z=port_z,
            linear_v=linear_v,
            angular_v=angular_v,
        )

        if complete:
            return np.zeros((6,), dtype=np.float32)

        lin_action = linear_v / max(cfg.action_scale_linear, 1e-9)
        ang_action = angular_v / max(cfg.action_scale_angular, 1e-9)
        return np.concatenate((lin_action, ang_action)).astype(np.float32)

    # -- Phase actions --------------------------------------------------------

    def _lift_action(
        self, axial_error: float, port_z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Constant speed until past clearance; phase machine flips to align
        # the moment we cross the threshold.
        if axial_error >= self._config.safe_axial_clearance_m:
            return np.zeros(3), np.zeros(3)
        return -port_z * LIFT_VELOCITY_M_S, np.zeros(3)

    def _align_action(
        self,
        lateral_pd: np.ndarray,
        angular_pd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self._config
        linear_v = _clip_norm(
            lateral_pd * cfg.align_linear_gain, cfg.max_linear_velocity
        )
        if ALIGN_DECAY_RADIUS_M > 0.0:
            decay = min(1.0, float(np.linalg.norm(lateral_pd)) / ALIGN_DECAY_RADIUS_M)
            linear_v = linear_v * decay
        ang_limit = cfg.max_angular_velocity * ANG_GAIN_MULT
        angular_v = _clip_norm(
            angular_pd * cfg.align_angular_gain * ANG_GAIN_MULT, ang_limit
        )
        return linear_v, angular_v

    def _insert_action(
        self,
        *,
        lateral_pd: np.ndarray,
        lateral_raw: np.ndarray,
        axial_pd: float,
        axial_raw: float,
        angular_pd: np.ndarray,
        stuck_detection_aligned: bool,
        port_z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self._config

        # Stuck state machine — only meaningful while aligned enough to descend.
        if stuck_detection_aligned:
            self._update_insert_stuck(axial_raw)
        else:
            self._stuck_active = False
            self._stuck_steps = 0
            self._stuck_search_steps = 0

        # Gain selection.
        if self._stuck_active:
            lat_gain = cfg.aggressive_insert_xy_gain
            ax_gain = cfg.aggressive_insert_z_gain
            ang_gain = cfg.aggressive_insert_angular_gain
            lin_limit = cfg.aggressive_max_linear_velocity
            lat_limit = cfg.aggressive_xy_velocity_scale * lin_limit
            ax_limit = cfg.aggressive_z_velocity_scale * lin_limit
            min_ax_v = cfg.aggressive_min_insert_z_velocity
        else:
            lat_gain = cfg.insert_xy_gain
            ax_gain = cfg.insert_z_gain
            ang_gain = cfg.insert_angular_gain
            lin_limit = cfg.max_linear_velocity
            lat_limit = ax_limit = cfg.insert_linear_velocity_scale * lin_limit
            min_ax_v = cfg.min_insert_z_velocity

        # Lateral PD + min directional velocity.
        linear_v = _clip_norm(lateral_pd * lat_gain, lat_limit)
        _apply_min_directional_velocity(
            linear_v,
            lateral_pd,
            min(
                cfg.min_insert_xy_correction_velocity,
                float(np.linalg.norm(lateral_pd)) * lat_gain,
                lat_limit,
            ),
            lat_limit,
        )

        # Axial PD: only descend while lateral + angular are within the
        # insert-start tolerance. If we overshoot during descent, hold axial
        # at zero and let the lateral PD pull the tip back over the port
        # before pushing further down.
        if stuck_detection_aligned and axial_pd > 0.0:
            axial_v = float(np.clip(axial_pd * ax_gain, -ax_limit, ax_limit))
            if abs(axial_pd) >= cfg.z_insert_tolerance_m:
                axial_v = float(np.sign(axial_pd)) * max(abs(axial_v), min_ax_v)
            linear_v = linear_v + axial_v * port_z

        # Angular PD + min directional.
        ang_limit = cfg.max_angular_velocity * ANG_GAIN_MULT
        angular_v = _clip_norm(angular_pd * ang_gain * ANG_GAIN_MULT, ang_limit)
        _apply_min_directional_velocity(
            angular_v,
            angular_pd,
            cfg.min_insert_angular_correction_velocity * ANG_GAIN_MULT,
            ang_limit,
        )

        # Stuck additions: spiral search + lateral pulse. Only when close enough.
        if self._stuck_active and stuck_detection_aligned:
            self._stuck_search_steps += 1
            # Periodically zero axial push so contact friction releases and the
            # spiral/pulse below can actually move the cable laterally.
            if self._stuck_search_steps % STUCK_RELAX_PERIOD < STUCK_RELAX_STEPS:
                axial_now = float(linear_v @ port_z)
                linear_v = linear_v - axial_now * port_z
            if axial_raw <= STUCK_SEARCH_MAX_AXIAL_M:
                self._add_spiral_search(
                    lateral_raw, linear_v, lat_limit, lin_limit, port_z
                )
                self._add_lateral_pulse(lateral_raw, linear_v, lin_limit)

        return linear_v, angular_v

    # -- Stuck helpers --------------------------------------------------------

    def _update_insert_stuck(self, axial_raw: float) -> None:
        current = abs(axial_raw)
        progress = self._best_axial_err_abs - current
        if self._stuck_active:
            if progress >= STUCK_RECOVER_M:
                self._best_axial_err_abs = current
                self._stuck_steps = 0
                self._stuck_active = False
                self._stuck_search_steps = 0
                self._logger.info("Scripted intervention recovered from stuck")
            elif progress > 0.0:
                self._best_axial_err_abs = current
            return
        if progress >= STUCK_Z_PROGRESS_M:
            self._best_axial_err_abs = current
            self._stuck_steps = 0
            return
        self._stuck_steps += 1
        if self._stuck_steps >= STUCK_WINDOW_STEPS:
            self._stuck_active = True
            self._logger.warn(
                "Scripted intervention insertion stalled; aggressive mode on"
            )

    def _add_spiral_search(
        self,
        lateral_raw: np.ndarray,
        linear_v: np.ndarray,
        lat_limit: float,
        lin_limit: float,
        port_z: np.ndarray,
    ) -> None:
        cfg = self._config
        if cfg.stuck_search_linear_velocity <= 0.0:
            return
        n = self._stuck_search_steps
        phase = 2.0 * np.pi * (n % STUCK_SEARCH_PERIOD) / STUCK_SEARCH_PERIOD
        ramp = min(1.0, n / STUCK_SEARCH_RAMP)

        # Basis (e1, e2) spanning port-perpendicular plane, e1 aligned with residual.
        in_plane = lateral_raw - (lateral_raw @ port_z) * port_z
        n1 = float(np.linalg.norm(in_plane))
        if n1 > 1e-6:
            e1 = in_plane / n1
        else:
            cand = np.array([1.0, 0.0, 0.0])
            e1 = cand - (cand @ port_z) * port_z
            if float(np.linalg.norm(e1)) < 1e-6:
                cand = np.array([0.0, 1.0, 0.0])
                e1 = cand - (cand @ port_z) * port_z
            e1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
        e2 = np.cross(port_z, e1)
        e2 = e2 / max(float(np.linalg.norm(e2)), 1e-9)

        amp = ramp * cfg.stuck_search_linear_velocity
        bias = amp * (np.cos(phase) * e1 + np.sin(phase) * e2)
        limit = min(lin_limit, lat_limit + amp)
        linear_v[:] = _clip_norm(linear_v + bias, limit)

    def _add_lateral_pulse(
        self, lateral_raw: np.ndarray, linear_v: np.ndarray, lin_limit: float
    ) -> None:
        if STUCK_PULSE_VEL_M_S <= 0.0 or STUCK_PULSE_STEPS <= 0:
            return
        if self._stuck_search_steps % STUCK_PULSE_PERIOD >= STUCK_PULSE_STEPS:
            return
        norm = float(np.linalg.norm(lateral_raw))
        if norm < 1e-9:
            return
        bias = (lateral_raw / norm) * STUCK_PULSE_VEL_M_S
        linear_v[:] = _clip_norm(linear_v + bias, lin_limit)

    # -- Logging --------------------------------------------------------------

    def _truncate_log(self) -> None:
        if not self._config.log_path:
            return
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
        self._log_file = None
        self._log_step = 0
        try:
            open(self._config.log_path, "w").close()
        except Exception:
            pass

    def _log_step_state(
        self,
        *,
        complete: bool,
        position_error: np.ndarray,
        lateral_error_raw: np.ndarray,
        lateral_error_norm: float,
        axial_error: float,
        angular_error: np.ndarray,
        port_x: np.ndarray,
        port_y: np.ndarray,
        port_z: np.ndarray,
        linear_v: np.ndarray,
        angular_v: np.ndarray,
    ) -> None:
        if not self._config.log_path:
            return
        self._log_step += 1
        try:
            if self._log_file is None:
                self._log_file = open(self._config.log_path, "a", buffering=1)
            record = {
                "t": time.time(),
                "code_version": SCRIPTED_INTERVENTION_CODE_VERSION,
                "step": self._log_step,
                "phase": self._phase,
                "complete": bool(complete),
                "lat_err_norm_mm": round(lateral_error_norm * 1000.0, 3),
                "ax_err_mm": round(axial_error * 1000.0, 3),
                "ang_err_norm_rad": round(float(np.linalg.norm(angular_error)), 4),
                "pos_err_base_mm": [
                    round(float(v) * 1000.0, 3) for v in position_error
                ],
                "lat_err_base_mm": [
                    round(float(v) * 1000.0, 3) for v in lateral_error_raw
                ],
                "ang_err_xyz_rad": [round(float(v), 4) for v in angular_error],
                "port_x_axis": [round(float(v), 3) for v in port_x],
                "port_y_axis": [round(float(v), 3) for v in port_y],
                "port_z_axis": [round(float(v), 3) for v in port_z],
                "lin_v_base_mm_s": [round(float(v) * 1000.0, 3) for v in linear_v],
                "ang_v_xyz_rad_s": [round(float(v), 4) for v in angular_v],
                "stuck": bool(self._stuck_active),
                "stuck_steps": int(self._stuck_steps),
                "stuck_search_steps": int(self._stuck_search_steps),
            }
            self._log_file.write(json.dumps(record) + "\n")
        except Exception:
            self._log_file = None

    def _maybe_warn(self, message: str) -> None:
        now = time.time()
        if now - self._last_warning_time >= WARN_INTERVAL_SEC:
            self._logger.warn(message)
            self._last_warning_time = now
