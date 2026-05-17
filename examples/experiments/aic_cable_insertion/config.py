import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from experiments.aic_cable_insertion.scripted_intervention import (
    _pose_from_transform,
    _quat_inverse_xyzw,
    _quat_multiply_xyzw,
    _quat_to_rotvec_xyzw,
    _rotate_vector_by_quat_xyzw,
)
from experiments.aic_cable_insertion.wrapper import AICCableInsertionEnv
from experiments.config import DefaultTrainingConfig


@dataclass(frozen=True)
class EnvConfig:
    """AIC task settings for HIL-SERL training."""

    image_width: int = 128
    image_height: int = 128
    image_keys: tuple[str, ...] = ("left_camera", "center_camera", "right_camera")
    classifier_keys: tuple[str, ...] = ("left_camera", "center_camera", "right_camera")
    proprio_keys: tuple[str, ...] = (
        "tcp_pose",
        "tcp_vel",
        "tcp_error",
        "joint_positions",
        "joint_velocities",
        "joint_efforts",
        "wrist_force",
        "wrist_torque",
    )

    action_scale_linear: float = 0.01
    action_scale_angular: float = 0.05
    control_frame_id: str = "base_link"
    max_episode_length: int = 500
    policy_control_period_sec: float = 0.10
    # Geometric reward classifier thresholds.
    # Position is decomposed along port_z: lateral = perpendicular plane, axial = along port_z.
    reward_lateral_tolerance_m: float = 0.002
    reward_axial_tolerance_m: float = 0.0008
    reward_angle_tolerance_rad: float = 0.04
    reward_consecutive_frames: int = 2
    display_image: bool = True
    observation_timeout_sec: float = 1.0
    post_reset_settle_sec: float = 1.0

    use_sim_time: bool = True
    suppress_ros_warnings: bool = True
    observation_topic: str = "observations_masked_roi"
    pose_command_topic: str = "/aic_controller/pose_commands"
    change_target_mode_service: str = "/aic_controller/change_target_mode"
    tare_force_torque_service: str = "/aic_controller/tare_force_torque_sensor"
    reset_joints_service: str = "/scoring/reset_joints"

    enable_tare_on_reset: bool = True
    enable_joint_reset: bool = True
    require_manual_reset_ack: bool = False
    reset_prompt: str = "Reset episode state if needed, then press Enter to continue..."
    home_joint_names: tuple[str, ...] = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    home_joint_positions: tuple[float, ...] = (0.6, -1.3, -1.9, -1.57, 1.57, 0.6)

    enable_keyboard_intervention: bool = True
    intervention_linear_velocity: float = 0.02
    intervention_angular_velocity: float = 0.05
    reset_resume_key: str = "r"
    enable_scripted_intervention: bool = True
    scripted_intervention_toggle_key: str = "0"
    scripted_intervention_tip_frame: str = os.environ.get(
        "AIC_SCRIPTED_TIP_FRAME",
        "cable_1/sc_tip_link",
    )
    scripted_intervention_port_frame: str = os.environ.get(
        "AIC_SCRIPTED_PORT_FRAME",
        "task_board/sc_port_0/sc_port_base_link",
    )
    scripted_intervention_align_linear_gain: float = 8.0
    scripted_intervention_insert_xy_gain: float = 12.0
    scripted_intervention_insert_z_gain: float = 40.0
    scripted_intervention_align_angular_gain: float = 1.5
    scripted_intervention_insert_angular_gain: float = 1.5
    scripted_intervention_insert_linear_velocity_scale: float = 1.0 / 3.0
    scripted_intervention_min_insert_z_velocity: float = 0.004
    scripted_intervention_min_insert_xy_correction_velocity: float = 0.003
    scripted_intervention_min_insert_angular_correction_velocity: float = 0.01
    scripted_intervention_linear_deadband_m: float = 0.0005
    scripted_intervention_angular_deadband_rad: float = 0.01
    scripted_intervention_xy_align_tolerance_m: float = 0.0005
    scripted_intervention_z_insert_tolerance_m: float = 0.00015
    scripted_intervention_orientation_align_tolerance_rad: float = 0.03
    scripted_intervention_safe_axial_clearance_m: float = 0.015
    scripted_intervention_lift_trigger_axial_clearance_m: float = 0.010
    scripted_intervention_lift_lateral_threshold_m: float = 0.002
    scripted_intervention_align_stuck_window_steps: int = 8
    scripted_intervention_align_stuck_xy_progress_threshold_m: float = 0.0002
    scripted_intervention_align_stuck_xy_min_velocity: float = 0.01
    scripted_intervention_stuck_window_steps: int = 2
    scripted_intervention_stuck_z_progress_threshold_m: float = 0.0001
    scripted_intervention_stuck_recover_progress_threshold_m: float = 0.002
    scripted_intervention_aggressive_insert_xy_gain: float = 30.0
    scripted_intervention_aggressive_insert_z_gain: float = 25.0
    scripted_intervention_aggressive_insert_angular_gain: float = 3
    scripted_intervention_aggressive_max_linear_velocity: float = 0.04
    scripted_intervention_aggressive_xy_velocity_scale: float = 1.0
    scripted_intervention_aggressive_z_velocity_scale: float = 1.0
    scripted_intervention_aggressive_min_insert_z_velocity: float = 0.008
    scripted_intervention_stuck_target_xy_min_velocity: float = 0.012
    scripted_intervention_stuck_directional_linear_boost: float = 2.4
    scripted_intervention_stuck_directional_angular_boost: float = 2.2
    scripted_intervention_persistent_angular_stuck_window_steps: int = 4
    scripted_intervention_persistent_angular_progress_threshold_rad: float = 0.002
    scripted_intervention_persistent_angular_boost: float = 3.0
    scripted_intervention_stuck_search_linear_velocity: float = 0.005
    scripted_intervention_stuck_search_angular_velocity: float = 0.0
    scripted_intervention_stuck_search_period_steps: int = 8
    scripted_intervention_stuck_search_ramp_steps: int = 6


class TrainConfig(DefaultTrainingConfig):
    batch_size = int(os.environ.get("AIC_BATCH_SIZE", "64"))
    cta_ratio = int(os.environ.get("AIC_CTA_RATIO", "1"))
    image_keys = ["left_camera", "center_camera", "right_camera"]
    classifier_keys = ["left_camera", "center_camera", "right_camera"]
    proprio_keys = [
        "tcp_pose",
        "tcp_vel",
        "tcp_error",
        "joint_positions",
        "joint_velocities",
        "joint_efforts",
        "wrist_force",
        "wrist_torque",
    ]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        base_env = AICCableInsertionEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = SERLObsWrapper(base_env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        if classifier:
            live = getattr(base_env, "_live", None)
            env = AICCoordinateRewardClassifierWrapper(
                env,
                tf_buffer=getattr(live, "_tf_buffer", None),
                time_type=getattr(live, "_Time", None),
                transform_exception=getattr(live, "_TransformException", Exception),
                base_frame=base_env.config.control_frame_id,
                tip_frame=base_env.config.scripted_intervention_tip_frame,
                port_frame=base_env.config.scripted_intervention_port_frame,
                lateral_tolerance_m=base_env.config.reward_lateral_tolerance_m,
                axial_tolerance_m=base_env.config.reward_axial_tolerance_m,
                angle_tolerance_rad=base_env.config.reward_angle_tolerance_rad,
                consecutive_frames=base_env.config.reward_consecutive_frames,
            )

        return env


class AICCoordinateRewardClassifierWrapper(gym.Wrapper):
    """Geometric success detector.

    Reads tip and port poses from TF and computes, in the port frame:
      - lateral error: ||(port_pos - tip_pos) projected onto plane ⟂ port_z||
      - axial   error: |(port_pos - tip_pos) · port_z|   (along insertion axis)
      - angle   error: ||rotvec(port_quat * tip_quat^-1)||
    Reward = 1 and done = True once all three are within their respective
    tolerances for `consecutive_frames` consecutive steps. Any miss resets
    the streak. If TF is unavailable (e.g. fake env), the streak never
    advances.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        tf_buffer,
        time_type,
        transform_exception: type[Exception],
        base_frame: str,
        tip_frame: str,
        port_frame: str,
        lateral_tolerance_m: float,
        axial_tolerance_m: float,
        angle_tolerance_rad: float,
        consecutive_frames: int,
    ):
        super().__init__(env)
        self._tf_buffer = tf_buffer
        self._Time = time_type
        self._TransformException = transform_exception
        self._base_frame = base_frame
        self._tip_frame = tip_frame
        self._port_frame = port_frame
        self._lateral_tol = float(lateral_tolerance_m)
        self._axial_tol = float(axial_tolerance_m)
        self._ang_tol = float(angle_tolerance_rad)
        self._needed = max(int(consecutive_frames), 1)
        self._met_streak = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._met_streak = 0
        info["succeed"] = 0
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        lat_err, ax_err, ang_err, ok = self._compute_errors()
        info["reward_lateral_error_m"] = lat_err
        info["reward_axial_error_m"] = ax_err
        info["reward_angle_error_rad"] = ang_err
        if ok:
            self._met_streak += 1
        else:
            self._met_streak = 0
        info["reward_met_streak"] = self._met_streak
        if self._met_streak >= self._needed:
            reward = 1.0
            done = True
            info["succeed"] = 1
        else:
            info["succeed"] = 0
        return obs, reward, done, truncated, info

    def _compute_errors(self) -> tuple[float, float, float, bool]:
        if self._tf_buffer is None or self._Time is None:
            return float("inf"), float("inf"), float("inf"), False
        try:
            now = self._Time()
            port_tf = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._port_frame,
                now,
            )
            tip_tf = self._tf_buffer.lookup_transform(
                self._base_frame,
                self._tip_frame,
                now,
            )
        except self._TransformException:
            return float("inf"), float("inf"), float("inf"), False
        port_pos, port_quat = _pose_from_transform(port_tf)
        tip_pos, tip_quat = _pose_from_transform(tip_tf)
        position_error = port_pos - tip_pos
        port_z_axis = _rotate_vector_by_quat_xyzw(
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            port_quat,
        )
        axial_error = float(np.dot(position_error, port_z_axis))
        lateral_error = float(
            np.linalg.norm(position_error - axial_error * port_z_axis)
        )
        quat_err = _quat_multiply_xyzw(port_quat, _quat_inverse_xyzw(tip_quat))
        ang_err = float(np.linalg.norm(_quat_to_rotvec_xyzw(quat_err)))
        ok = (
            lateral_error <= self._lateral_tol
            and abs(axial_error) <= self._axial_tol
            and ang_err <= self._ang_tol
        )
        return lateral_error, abs(axial_error), ang_err, ok
