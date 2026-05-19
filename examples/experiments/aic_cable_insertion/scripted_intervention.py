from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


# --- Quaternion math (xyzw) ---------------------------------------------------


def _normalize_quaternion_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def _quat_inverse_xyzw(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion_xyzw(quat)
    return np.array([-x, -y, -z, w], dtype=np.float64)


def _quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = _normalize_quaternion_xyzw(q1)
    x2, y2, z2, w2 = _normalize_quaternion_xyzw(q2)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _quat_to_rotvec_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quaternion_xyzw(quat)
    if quat[3] < 0.0:
        quat = -quat
    vector = quat[:3]
    vector_norm = np.linalg.norm(vector)
    if vector_norm < 1e-9:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, quat[3])
    return vector / vector_norm * angle


def _rotate_vector_by_quat_xyzw(vector: np.ndarray, quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quaternion_xyzw(quat)
    q_vec = quat[:3]
    q_w = quat[3]
    vector = np.asarray(vector, dtype=np.float64)
    return (
        vector
        + 2.0 * q_w * np.cross(q_vec, vector)
        + 2.0 * np.cross(q_vec, np.cross(q_vec, vector))
    )


def _pose_from_transform(tf_msg: Any) -> tuple[np.ndarray, np.ndarray]:
    t = tf_msg.transform.translation
    r = tf_msg.transform.rotation
    return (
        np.array([t.x, t.y, t.z], dtype=np.float64),
        np.array([r.x, r.y, r.z, r.w], dtype=np.float64),
    )


# --- Config -------------------------------------------------------------------


@dataclass(frozen=True)
class ScriptedInterventionConfig:
    toggle_key: str = "0"
    base_frame: str = "base_link"
    tip_frame: str = "cable_1/sc_tip_link"
    port_frame: str = "task_board/sc_port_0/sc_port_base_link"
    # PD gains. xy = perpendicular to port_z (lateral plane).
    # z         = along port_z (insertion / "axial" direction).
    align_linear_gain: float = 8.0
    align_angular_gain: float = 1.5
    insert_xy_gain: float = 12.0
    insert_z_gain: float = 40.0
    insert_angular_gain: float = 0.8
    # Velocity caps
    max_linear_velocity: float = 0.01
    max_angular_velocity: float = 0.04
    # Insert phase scaling (slows axial descent)
    insert_linear_velocity_scale: float = 1.0 / 3.0
    min_insert_z_velocity: float = 0.002
    min_insert_xy_correction_velocity: float = 0.002
    min_insert_angular_correction_velocity: float = 0.006
    # Action output scaling
    action_scale_linear: float = 0.005
    action_scale_angular: float = 0.02
    # Deadbands and tolerances
    linear_deadband_m: float = 0.001
    angular_deadband_rad: float = 0.03
    xy_align_tolerance_m: float = 0.001
    z_insert_tolerance_m: float = 0.0015
    orientation_align_tolerance_rad: float = 0.05
    # Lift safety: if tip is on the entry side, but too close to the port AND
    # laterally far off, lift along -port_z to a safe clearance before letting
    # align translate horizontally — avoids dragging the cable across the port edge.
    safe_axial_clearance_m: float = 0.010
    # Hysteresis on the lift <-> align transition: enter lift only when axial
    # error is well below the safe clearance, but exit at the full safe value.
    # Prevents oscillation when ax_err parks right on the boundary.
    lift_trigger_axial_clearance_m: float = 0.005
    lift_lateral_threshold_m: float = 0.003
    # Align stall detection (xy/lateral progress)
    align_stuck_window_steps: int = 8
    align_stuck_xy_progress_threshold_m: float = 0.0002
    align_stuck_xy_min_velocity: float = 0.01
    # Lateral velocity decay near target in align (suppresses overshoot).
    # |v| is scaled by min(1, lateral_error_norm / align_decay_radius_m).
    # Set 0 to disable.
    align_decay_radius_m: float = 0.005
    # Insert stall detection + aggressive recovery.
    # Hysteresis: a small `stuck_z_progress_threshold_m` decides whether progress
    # this step counts as "still descending" (stuck stays False). Once stuck has
    # latched, leaving the stuck state requires cumulative progress >=
    # `stuck_recover_progress_threshold_m` so the aggressive mode + search wiggle
    # are not aborted by sub-millimeter blips that the normal-gain path can't
    # follow through on. Best-error is still updated on any forward progress so
    # the recover threshold compares against the deepest point reached, not the
    # point at which stuck triggered.
    stuck_window_steps: int = 12
    stuck_z_progress_threshold_m: float = 0.0005
    stuck_recover_progress_threshold_m: float = 0.002
    aggressive_insert_xy_gain: float = 30.0
    aggressive_insert_z_gain: float = 25.0
    aggressive_insert_angular_gain: float = 1.2
    aggressive_max_linear_velocity: float = 0.03
    aggressive_xy_velocity_scale: float = 1.0
    aggressive_z_velocity_scale: float = 0.5
    aggressive_min_insert_z_velocity: float = 0.003
    stuck_target_xy_min_velocity: float = 0.018
    stuck_directional_linear_boost: float = 1.8
    stuck_directional_angular_boost: float = 1.8
    persistent_angular_stuck_window_steps: int = 4
    persistent_angular_progress_threshold_rad: float = 0.002
    persistent_angular_boost: float = 2.6
    stuck_search_linear_velocity: float = 0.005
    stuck_search_angular_velocity: float = 0.000
    stuck_search_period_steps: int = 8
    stuck_search_ramp_steps: int = 6
    # Lateral pulse while stuck: periodically inject a strong directional
    # velocity along lateral_error_raw (toward target xy), bypassing the
    # deadband. Designed to break sim mesh contact overlap caused by a
    # sub-millimeter horizontal misalignment that the normal PD path
    # (zeroed by the deadband once inside tolerance) cannot push through.
    # Set stuck_lateral_pulse_velocity to 0 to disable.
    stuck_lateral_pulse_velocity: float = 0.01
    stuck_lateral_pulse_steps: int = 4
    stuck_lateral_pulse_period_steps: int = 16
    warn_interval_sec: float = 2.0
    # Per-step state log. Empty string disables. Append-only JSONL.
    log_path: str = "/tmp/scripted_intervention.log.jsonl"
    log_every_n_steps: int = 1


@dataclass(frozen=True)
class _InsertGains:
    lateral_gain: float
    axial_gain: float
    angular_gain: float
    linear_velocity_limit: float
    lateral_velocity_limit: float
    axial_velocity_limit: float
    min_insert_axial_velocity: float


# --- Main class ---------------------------------------------------------------


class ScriptedCableInsertionIntervention:
    """Three-phase scripted intervention.

    lift   : safety retreat. If at startup the tip sits on the port-entry side
             (axial_error > 0) but too close to the port AND laterally far from
             the hole, move along -port_z to a safe clearance before doing any
             horizontal motion. Avoids scraping the cable across the port edge.
    align  : drive lateral + 3-axis angular error to within deadzone.
             Axial (along port_z) velocity is held at zero — no vertical motion.
    insert : descend along port_z while continuing lateral + angular correction.
             Axial velocity is gated on alignment: drift outside deadzone halts
             descent until lateral + angular are back within tolerance.

    Transitions: lift <-> align <-> insert; insert is terminal (no reversion).
    Once is_complete() is true the controller emits zero velocity.
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
        self._last_position_error = np.zeros(3, dtype=np.float64)
        self._last_angular_error = np.zeros(3, dtype=np.float64)
        self._last_abs_angular_error = np.zeros(3, dtype=np.float64)
        self._last_lateral_error_norm = 0.0
        self._last_axial_error = 0.0
        self._best_insert_axial_error_abs = np.inf
        self._best_align_lateral_error_norm = np.inf
        self._align_stuck_steps = 0
        self._align_stuck_active = False
        self._insert_stuck_steps = 0
        self._stuck_active = False
        self._stuck_search_steps = 0
        self._persistent_angular_stuck_steps = np.zeros(3, dtype=np.int32)

    # -- Public API ------------------------------------------------------------

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
        state = "enabled" if self._active else "disabled"
        self._logger.info(
            f"Scripted intervention {state}: "
            f"{self._config.tip_frame} -> {self._config.port_frame}"
        )
        return self._active

    def _truncate_log(self) -> None:
        cfg = self._config
        if not cfg.log_path:
            return
        try:
            if self._log_file is not None:
                self._log_file.close()
        except Exception:
            pass
        self._log_file = None
        self._log_step = 0
        try:
            open(cfg.log_path, "w").close()
        except Exception:
            pass

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
            "align_stuck": self._align_stuck_active,
            "stuck": self._stuck_active,
            "stuck_steps": int(self._insert_stuck_steps),
            "complete": self.is_complete(),
        }

    def is_complete(self) -> bool:
        cfg = self._config
        return (
            self._phase == "insert"
            and self._last_lateral_error_norm <= cfg.xy_align_tolerance_m
            and abs(self._last_axial_error) <= cfg.z_insert_tolerance_m
            and float(np.linalg.norm(self._last_angular_error))
            <= cfg.orientation_align_tolerance_rad
        )

    def get_action(self) -> np.ndarray | None:
        if not self._active:
            return None

        try:
            now = self._Time()
            port_tf = self._tf_buffer.lookup_transform(
                self._config.base_frame,
                self._config.port_frame,
                now,
            )
            tip_tf = self._tf_buffer.lookup_transform(
                self._config.base_frame,
                self._config.tip_frame,
                now,
            )
        except self._TransformException as exc:
            self._maybe_warn(f"Scripted intervention TF lookup failed: {exc}")
            return np.zeros((6,), dtype=np.float32)

        port_pos, port_quat = _pose_from_transform(port_tf)
        tip_pos, tip_quat = _pose_from_transform(tip_tf)
        position_error = port_pos - tip_pos
        quat_error = _quat_multiply_xyzw(port_quat, _quat_inverse_xyzw(tip_quat))
        angular_error = _quat_to_rotvec_xyzw(quat_error)
        port_x_axis = _rotate_vector_by_quat_xyzw(
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            port_quat,
        )
        port_y_axis = _rotate_vector_by_quat_xyzw(
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            port_quat,
        )
        port_z_axis = _rotate_vector_by_quat_xyzw(
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            port_quat,
        )

        # Decompose position error into axial (along port_z) and lateral (perpendicular).
        axial_error = float(np.dot(position_error, port_z_axis))
        lateral_error_raw = (
            float(np.dot(position_error, port_x_axis)) * port_x_axis
            + float(np.dot(position_error, port_y_axis)) * port_y_axis
        )
        lateral_error = lateral_error_raw.copy()
        lateral_error_norm = float(np.linalg.norm(lateral_error))

        cfg = self._config
        self._last_position_error = position_error.copy()
        self._last_angular_error = angular_error.copy()
        self._last_lateral_error_norm = lateral_error_norm
        self._last_axial_error = axial_error

        lateral_aligned = lateral_error_norm <= cfg.xy_align_tolerance_m
        orientation_aligned = (
            float(np.linalg.norm(angular_error)) <= cfg.orientation_align_tolerance_rad
        )
        aligned = lateral_aligned and orientation_aligned

        # Phase transitions:
        #   align -> lift   if entry-side AND too close to port AND laterally off
        #   lift  -> align  once axial clearance is safe again
        #   align -> insert once both lateral and orientation are within tolerance
        # insert is terminal (no reversion).
        if self._phase == "align":
            if (
                lateral_error_norm > cfg.lift_lateral_threshold_m
                and axial_error < cfg.lift_trigger_axial_clearance_m
            ):
                self._phase = "lift"
                self._logger.info(
                    "Scripted intervention phase switch: align -> lift "
                    "(tip too close to port + laterally off)"
                )
            elif aligned:
                self._phase = "insert"
                self._best_insert_axial_error_abs = abs(axial_error)
                self._insert_stuck_steps = 0
                self._stuck_active = False
                self._stuck_search_steps = 0
                self._persistent_angular_stuck_steps.fill(0)
                self._logger.info("Scripted intervention phase switch: align -> insert")
        elif self._phase == "lift" and axial_error >= cfg.safe_axial_clearance_m:
            self._phase = "align"
            self._logger.info("Scripted intervention phase switch: lift -> align")

        # Apply deadbands to PD inputs
        if lateral_error_norm < cfg.linear_deadband_m:
            lateral_error = np.zeros(3, dtype=np.float64)
        if abs(axial_error) < cfg.z_insert_tolerance_m:
            axial_error_pd: float = 0.0
        else:
            axial_error_pd = axial_error
        if float(np.linalg.norm(angular_error)) < cfg.angular_deadband_rad:
            angular_error = np.zeros(3, dtype=np.float64)

        if self._phase == "lift":
            linear_velocity, angular_velocity = self._lift_action(
                axial_error=axial_error,
                port_z_axis=port_z_axis,
            )
        elif self._phase == "align":
            linear_velocity, angular_velocity = self._align_action(
                lateral_error=lateral_error,
                lateral_error_raw=lateral_error_raw,
                lateral_error_norm=lateral_error_norm,
                lateral_aligned=lateral_aligned,
                angular_error=angular_error,
            )
        else:  # insert
            linear_velocity, angular_velocity = self._insert_action(
                lateral_error=lateral_error,
                lateral_error_raw=lateral_error_raw,
                lateral_aligned=lateral_aligned,
                axial_error_pd=axial_error_pd,
                axial_error_raw=axial_error,
                aligned=aligned,
                angular_error=angular_error,
                port_z_axis=port_z_axis,
            )

        self._last_abs_angular_error = np.abs(angular_error)
        # z-axis reached target → emit zeros instead of trickle PD output.
        complete = self.is_complete()
        if complete:
            linear_velocity = np.zeros(3, dtype=np.float64)
            angular_velocity = np.zeros(3, dtype=np.float64)
        self._log_step_state(
            phase=self._phase,
            complete=complete,
            position_error=position_error,
            lateral_error_raw=lateral_error_raw,
            lateral_error_norm=lateral_error_norm,
            axial_error=axial_error,
            angular_error=angular_error,
            port_x_axis=port_x_axis,
            port_y_axis=port_y_axis,
            port_z_axis=port_z_axis,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )
        if complete:
            return np.zeros((6,), dtype=np.float32)
        linear_action = linear_velocity / max(cfg.action_scale_linear, 1e-9)
        angular_action = angular_velocity / max(cfg.action_scale_angular, 1e-9)
        return np.concatenate((linear_action, angular_action), axis=0).astype(
            np.float32
        )

    def _log_step_state(
        self,
        *,
        phase: str,
        complete: bool,
        position_error: np.ndarray,
        lateral_error_raw: np.ndarray,
        lateral_error_norm: float,
        axial_error: float,
        angular_error: np.ndarray,
        port_x_axis: np.ndarray,
        port_y_axis: np.ndarray,
        port_z_axis: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> None:
        cfg = self._config
        if not cfg.log_path:
            return
        self._log_step += 1
        if self._log_step % max(int(cfg.log_every_n_steps), 1) != 0:
            return
        try:
            if self._log_file is None:
                self._log_file = open(cfg.log_path, "a", buffering=1)
            record = {
                "t": time.time(),
                "step": self._log_step,
                "phase": phase,
                "complete": bool(complete),
                "active": bool(self._active),
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
                "port_x_axis": [round(float(v), 3) for v in port_x_axis],
                "port_y_axis": [round(float(v), 3) for v in port_y_axis],
                "port_z_axis": [round(float(v), 3) for v in port_z_axis],
                "lin_v_base_mm_s": [
                    round(float(v) * 1000.0, 3) for v in linear_velocity
                ],
                "ang_v_xyz_rad_s": [round(float(v), 4) for v in angular_velocity],
                "align_stuck": bool(self._align_stuck_active),
                "stuck": bool(self._stuck_active),
                "stuck_steps": int(self._insert_stuck_steps),
                "stuck_search_steps": int(self._stuck_search_steps),
            }
            self._log_file.write(json.dumps(record) + "\n")
        except Exception:
            self._log_file = None

    # -- Lift phase -----------------------------------------------------------

    def _lift_action(
        self,
        *,
        axial_error: float,
        port_z_axis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pure axial retreat along -port_z. No port-frame lateral, no angular.

        Velocity is built as a scalar multiple of port_z_axis, so any component
        perpendicular to port_z (i.e. "horizontal" in the port frame) is exactly
        zero. The published twist is in base frame, so a tilted port will yield
        small base-frame x/y components — they are still purely axial in port
        frame, which is what we want.
        """
        cfg = self._config
        clearance_deficit = cfg.safe_axial_clearance_m - axial_error
        speed = float(
            np.clip(
                cfg.align_linear_gain * clearance_deficit,
                0.0,
                cfg.max_linear_velocity,
            )
        )
        linear_velocity = -port_z_axis * speed
        angular_velocity = np.zeros(3, dtype=np.float64)
        # Reset stall counters that don't apply during lift
        self._align_stuck_steps = 0
        self._align_stuck_active = False
        self._persistent_angular_stuck_steps.fill(0)
        return linear_velocity, angular_velocity

    # -- Align phase ----------------------------------------------------------

    def _align_action(
        self,
        *,
        lateral_error: np.ndarray,
        lateral_error_raw: np.ndarray,
        lateral_error_norm: float,
        lateral_aligned: bool,
        angular_error: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self._config
        self._update_align_stuck_state(
            lateral_error_norm=lateral_error_norm,
            lateral_aligned=lateral_aligned,
        )
        linear_velocity = np.clip(
            lateral_error * cfg.align_linear_gain,
            -cfg.max_linear_velocity,
            cfg.max_linear_velocity,
        )
        if cfg.align_decay_radius_m > 0.0:
            decay = min(1.0, lateral_error_norm / max(cfg.align_decay_radius_m, 1e-9))
            linear_velocity = linear_velocity * decay
        if self._align_stuck_active:
            self._apply_min_directional_velocity(
                velocity=linear_velocity,
                error=lateral_error_raw,
                min_velocity=cfg.align_stuck_xy_min_velocity,
                velocity_limit=cfg.max_linear_velocity,
            )
        angular_velocity = np.clip(
            angular_error * cfg.align_angular_gain,
            -cfg.max_angular_velocity,
            cfg.max_angular_velocity,
        )
        self._persistent_angular_stuck_steps.fill(0)
        return linear_velocity, angular_velocity

    def _update_align_stuck_state(
        self,
        *,
        lateral_error_norm: float,
        lateral_aligned: bool,
    ) -> None:
        cfg = self._config
        if lateral_aligned:
            self._best_align_lateral_error_norm = lateral_error_norm
            self._align_stuck_steps = 0
            self._align_stuck_active = False
            return
        if (
            self._best_align_lateral_error_norm - lateral_error_norm
            >= cfg.align_stuck_xy_progress_threshold_m
        ):
            self._best_align_lateral_error_norm = lateral_error_norm
            self._align_stuck_steps = 0
            if self._align_stuck_active:
                self._logger.info("Scripted intervention recovered from align stall")
            self._align_stuck_active = False
            return
        self._align_stuck_steps += 1
        if self._align_stuck_steps >= cfg.align_stuck_window_steps:
            if not self._align_stuck_active:
                self._logger.warn(
                    "Scripted intervention detected align stall; "
                    "boosting lateral correction"
                )
            self._align_stuck_active = True

    # -- Insert phase ---------------------------------------------------------

    def _insert_action(
        self,
        *,
        lateral_error: np.ndarray,
        lateral_error_raw: np.ndarray,
        lateral_aligned: bool,
        axial_error_pd: float,
        axial_error_raw: float,
        aligned: bool,
        angular_error: np.ndarray,
        port_z_axis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self._config
        self._align_stuck_steps = 0
        self._align_stuck_active = False
        # Stuck progress is only meaningful while we are actually descending.
        if aligned:
            self._update_insert_stuck_state(axial_error=axial_error_raw)
        else:
            self._stuck_active = False
            self._insert_stuck_steps = 0
            self._stuck_search_steps = 0
            self._persistent_angular_stuck_steps.fill(0)
        gains = self._select_insert_gains()

        # Lateral PD (always active in insert too)
        linear_velocity = np.clip(
            lateral_error * gains.lateral_gain,
            -gains.lateral_velocity_limit,
            gains.lateral_velocity_limit,
        )
        self._apply_min_directional_velocity(
            velocity=linear_velocity,
            error=lateral_error,
            min_velocity=cfg.min_insert_xy_correction_velocity,
            velocity_limit=gains.lateral_velocity_limit,
        )
        # Axial descent — always on in insert phase. Pure PD + min-velocity
        # floor, decoupled from lateral/angular alignment quality so z speed
        # stays steady even when horizontal correction is mid-flight.
        axial_velocity = float(
            np.clip(
                axial_error_pd * gains.axial_gain,
                -gains.axial_velocity_limit,
                gains.axial_velocity_limit,
            )
        )
        if abs(axial_error_pd) >= cfg.z_insert_tolerance_m:
            axial_velocity = float(np.sign(axial_error_pd)) * max(
                abs(axial_velocity),
                gains.min_insert_axial_velocity,
            )
        linear_velocity += axial_velocity * port_z_axis

        angular_velocity = np.clip(
            angular_error * gains.angular_gain,
            -0.5 * cfg.max_angular_velocity,
            0.5 * cfg.max_angular_velocity,
        )
        self._apply_min_directional_velocity(
            velocity=angular_velocity,
            error=angular_error,
            min_velocity=cfg.min_insert_angular_correction_velocity,
            velocity_limit=0.5 * cfg.max_angular_velocity,
        )

        if self._stuck_active and aligned:
            self._stuck_search_steps += 1
            self._apply_stuck_directional_boost(
                lateral_error=lateral_error,
                lateral_error_raw=lateral_error_raw,
                lateral_aligned=lateral_aligned,
                angular_error=angular_error,
                gains=gains,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
            )
            self._add_stuck_search_velocity(
                lateral_error_raw=lateral_error_raw,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                lateral_velocity_limit=gains.lateral_velocity_limit,
                max_linear_velocity=gains.linear_velocity_limit,
                port_z_axis=port_z_axis,
            )
            self._apply_stuck_lateral_pulse(
                lateral_error_raw=lateral_error_raw,
                linear_velocity=linear_velocity,
                gains=gains,
            )

        return linear_velocity, angular_velocity

    def _apply_stuck_lateral_pulse(
        self,
        *,
        lateral_error_raw: np.ndarray,
        linear_velocity: np.ndarray,
        gains: _InsertGains,
    ) -> None:
        """Periodic lateral nudge toward target while stuck.

        Bypasses the deadband: uses ``lateral_error_raw`` direction so
        sub-millimeter horizontal misalignment (that the PD path zeros out)
        still produces a clear lateral push that can break sim mesh contact
        overlap. On for ``stuck_lateral_pulse_steps`` out of every
        ``stuck_lateral_pulse_period_steps`` steps.
        """
        cfg = self._config
        if cfg.stuck_lateral_pulse_velocity <= 0.0:
            return
        period = max(int(cfg.stuck_lateral_pulse_period_steps), 1)
        pulse_width = max(int(cfg.stuck_lateral_pulse_steps), 0)
        if pulse_width <= 0:
            return
        phase_step = self._stuck_search_steps % period
        if phase_step >= pulse_width:
            return
        err_norm = float(np.linalg.norm(lateral_error_raw))
        if err_norm < 1e-9:
            return
        direction = lateral_error_raw / err_norm
        bias = direction * cfg.stuck_lateral_pulse_velocity
        limit = gains.linear_velocity_limit
        linear_velocity[:] = np.clip(linear_velocity + bias, -limit, limit)

    def _update_insert_stuck_state(self, *, axial_error: float) -> None:
        cfg = self._config
        current = abs(axial_error)
        progress = self._best_insert_axial_error_abs - current

        if self._stuck_active:
            if progress >= cfg.stuck_recover_progress_threshold_m:
                self._best_insert_axial_error_abs = current
                self._insert_stuck_steps = 0
                self._stuck_active = False
                self._stuck_search_steps = 0
                self._logger.info("Scripted intervention recovered from stuck state")
            elif progress > 0.0:
                self._best_insert_axial_error_abs = current
            return

        if progress >= cfg.stuck_z_progress_threshold_m:
            self._best_insert_axial_error_abs = current
            self._insert_stuck_steps = 0
            return
        self._insert_stuck_steps += 1
        if self._insert_stuck_steps >= cfg.stuck_window_steps:
            self._stuck_active = True
            self._logger.warn(
                "Scripted intervention detected insertion stall; "
                "switching to aggressive push"
            )

    def _select_insert_gains(self) -> _InsertGains:
        cfg = self._config
        if self._stuck_active:
            limit = cfg.aggressive_max_linear_velocity
            return _InsertGains(
                lateral_gain=cfg.aggressive_insert_xy_gain,
                axial_gain=cfg.aggressive_insert_z_gain,
                angular_gain=cfg.aggressive_insert_angular_gain,
                linear_velocity_limit=limit,
                lateral_velocity_limit=cfg.aggressive_xy_velocity_scale * limit,
                axial_velocity_limit=cfg.aggressive_z_velocity_scale * limit,
                min_insert_axial_velocity=cfg.aggressive_min_insert_z_velocity,
            )
        limit = cfg.max_linear_velocity
        scaled = cfg.insert_linear_velocity_scale * limit
        return _InsertGains(
            lateral_gain=cfg.insert_xy_gain,
            axial_gain=cfg.insert_z_gain,
            angular_gain=cfg.insert_angular_gain,
            linear_velocity_limit=limit,
            lateral_velocity_limit=scaled,
            axial_velocity_limit=scaled,
            min_insert_axial_velocity=cfg.min_insert_z_velocity,
        )

    def _apply_stuck_directional_boost(
        self,
        *,
        lateral_error: np.ndarray,
        lateral_error_raw: np.ndarray,
        lateral_aligned: bool,
        angular_error: np.ndarray,
        gains: _InsertGains,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> None:
        cfg = self._config
        abs_angular_error = np.abs(angular_error)
        angular_progress = self._last_abs_angular_error - abs_angular_error
        for axis in range(3):
            if (
                abs_angular_error[axis] >= cfg.angular_deadband_rad
                and angular_progress[axis]
                < cfg.persistent_angular_progress_threshold_rad
            ):
                self._persistent_angular_stuck_steps[axis] += 1
            else:
                self._persistent_angular_stuck_steps[axis] = 0

        lateral_max = float(np.max(np.abs(lateral_error)))
        angular_max = float(np.max(abs_angular_error))
        lateral_severity = lateral_max / max(cfg.linear_deadband_m, 1e-9)
        angular_severity = angular_max / max(cfg.angular_deadband_rad, 1e-9)
        persistent_axis = int(np.argmax(self._persistent_angular_stuck_steps))
        persistent_active = (
            self._persistent_angular_stuck_steps[persistent_axis]
            >= cfg.persistent_angular_stuck_window_steps
        )

        if persistent_active and abs_angular_error[persistent_axis] > 0.0:
            self._boost_angular_axis(
                axis=persistent_axis,
                angular_error=angular_error,
                angular_velocity=angular_velocity,
                base_gain=gains.angular_gain,
                boost=cfg.persistent_angular_boost,
            )
        elif (
            lateral_severity >= angular_severity
            and lateral_max > 0.0
            and not lateral_aligned
        ):
            axis = int(np.argmax(np.abs(lateral_error)))
            limit = min(
                gains.linear_velocity_limit,
                gains.lateral_velocity_limit * cfg.stuck_directional_linear_boost,
            )
            linear_velocity[axis] = float(
                np.clip(
                    lateral_error[axis]
                    * gains.lateral_gain
                    * cfg.stuck_directional_linear_boost,
                    -limit,
                    limit,
                )
            )
        elif angular_max > 0.0:
            axis = int(np.argmax(np.abs(angular_error)))
            self._boost_angular_axis(
                axis=axis,
                angular_error=angular_error,
                angular_velocity=angular_velocity,
                base_gain=gains.angular_gain,
                boost=cfg.stuck_directional_angular_boost,
            )

        if not lateral_aligned:
            self._apply_min_directional_velocity(
                velocity=linear_velocity,
                error=lateral_error_raw,
                min_velocity=cfg.stuck_target_xy_min_velocity,
                velocity_limit=gains.lateral_velocity_limit,
            )

    def _boost_angular_axis(
        self,
        *,
        axis: int,
        angular_error: np.ndarray,
        angular_velocity: np.ndarray,
        base_gain: float,
        boost: float,
    ) -> None:
        cfg = self._config
        limit = min(cfg.max_angular_velocity, 0.5 * cfg.max_angular_velocity * boost)
        angular_velocity[axis] = float(
            np.clip(
                angular_error[axis] * base_gain * boost,
                -limit,
                limit,
            )
        )

    # -- Common helpers -------------------------------------------------------

    def _add_stuck_search_velocity(
        self,
        *,
        lateral_error_raw: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
        lateral_velocity_limit: float,
        max_linear_velocity: float,
        port_z_axis: np.ndarray,
    ) -> None:
        """Spiral lateral search while stuck.

        Pure translation in the port-perpendicular plane: the PD lateral term
        continues driving the residual offset toward zero, and this routine
        adds an expanding circular sweep on top so a jammed-off-axis cable can
        find the hole. Radius grows linearly with cycle count, capped to avoid
        flying out. Angular bias is opt-in via `stuck_search_angular_velocity`
        — default 0 (no angle wiggle while stuck).
        """
        cfg = self._config
        if (
            cfg.stuck_search_linear_velocity <= 0.0
            and cfg.stuck_search_angular_velocity <= 0.0
        ):
            return
        period = max(int(cfg.stuck_search_period_steps), 4)
        ramp_steps = max(int(cfg.stuck_search_ramp_steps), 1)
        n = self._stuck_search_steps
        phase = 2.0 * np.pi * (n % period) / period
        cycle_count = n // period
        ramp = min(1.0, n / ramp_steps)

        if cfg.stuck_search_linear_velocity > 0.0:
            # Orthonormal basis (e1, e2) spanning the port-perpendicular plane.
            # Prefer aligning e1 with the residual lateral offset so the spiral
            # "starts" from the current PD direction. Fall back to world x/y if
            # the residual is degenerate.
            raw_in_plane = (
                lateral_error_raw
                - float(np.dot(lateral_error_raw, port_z_axis)) * port_z_axis
            )
            e1_norm = float(np.linalg.norm(raw_in_plane))
            if e1_norm > 1e-6:
                e1 = raw_in_plane / e1_norm
            else:
                candidate = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                e1 = candidate - float(np.dot(candidate, port_z_axis)) * port_z_axis
                if float(np.linalg.norm(e1)) < 1e-6:
                    candidate = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    e1 = candidate - float(np.dot(candidate, port_z_axis)) * port_z_axis
                e1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
            e2 = np.cross(port_z_axis, e1)
            e2 = e2 / max(float(np.linalg.norm(e2)), 1e-9)

            # Radius grows with every completed cycle, capped at 4x base.
            radius_scale = ramp * min(1.0 + 0.5 * cycle_count, 4.0)
            search_linear = radius_scale * cfg.stuck_search_linear_velocity
            bias = search_linear * (np.cos(phase) * e1 + np.sin(phase) * e2)
            limit = min(max_linear_velocity, lateral_velocity_limit + search_linear)
            linear_velocity[:] = np.clip(linear_velocity + bias, -limit, limit)

        if cfg.stuck_search_angular_velocity > 0.0:
            search_angular = ramp * cfg.stuck_search_angular_velocity
            angular_bias = search_angular * np.array(
                [np.sin(phase), np.cos(phase), 0.5 * np.sin(2.0 * phase)],
                dtype=np.float64,
            )
            angular_velocity[:] = np.clip(
                angular_velocity + angular_bias,
                -cfg.max_angular_velocity,
                cfg.max_angular_velocity,
            )

    @staticmethod
    def _apply_min_directional_velocity(
        *,
        velocity: np.ndarray,
        error: np.ndarray,
        min_velocity: float,
        velocity_limit: float,
    ) -> None:
        min_velocity = min(
            max(float(min_velocity), 0.0), max(float(velocity_limit), 0.0)
        )
        if min_velocity <= 0.0:
            return
        error_norm = float(np.linalg.norm(error))
        if error_norm <= 1e-9:
            return
        direction = error / error_norm
        direction_velocity = float(np.dot(velocity, direction))
        if direction_velocity < min_velocity:
            velocity += (min_velocity - direction_velocity) * direction
            velocity[:] = np.clip(velocity, -velocity_limit, velocity_limit)

    def _maybe_warn(self, message: str) -> None:
        now = time.time()
        if now - self._last_warning_time >= self._config.warn_interval_sec:
            self._logger.warn(message)
            self._last_warning_time = now
