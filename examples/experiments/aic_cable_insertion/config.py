import os
from dataclasses import dataclass

from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from experiments.aic_cable_insertion.wrapper import AICCableInsertionEnv
from experiments.config import DefaultTrainingConfig


@dataclass(frozen=True)
class EnvConfig:
    """AIC task settings for HIL-SERL training."""

    image_width: int = 128
    image_height: int = 128
    image_keys: tuple[str, ...] = ()
    classifier_keys: tuple[str, ...] = ()
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
    action_scale_angular: float = 0.06
    control_frame_id: str = "base_link"
    max_episode_length: int = 6000
    policy_control_period_sec: float = 0.10
    display_image: bool = True
    observation_timeout_sec: float = 1.0
    post_reset_settle_sec: float = 1.0

    use_sim_time: bool = True
    observation_topic: str = "observations"
    pose_command_topic: str = "/aic_controller/pose_commands"
    change_target_mode_service: str = "/aic_controller/change_target_mode"
    tare_force_torque_service: str = "/aic_controller/tare_force_torque_sensor"
    reset_joints_service: str = "/scoring/reset_joints"

    enable_tare_on_reset: bool = True
    enable_joint_reset: bool = True
    require_manual_reset_ack: bool = False
    reset_prompt: str = (
        "Reset episode state if needed, then press Enter to continue..."
    )
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
    intervention_linear_velocity: float = 0.01
    intervention_angular_velocity: float = 0.06
    reset_resume_key: str = "r"

    ground_truth_base_frame: str = "base_link"
    task_board_frame: str = "task_board"
    cable_name: str = "cable_0"
    plug_name: str = "sfp_module"
    target_module_name: str = "nic_card_mount_0"
    port_name: str = "sfp_port_1"
    reward_source_frame: str = "cable_0/sfp_tip_link"
    reward_target_frame: str = "task_board/nic_card_mount_0/sfp_port_1_link"
    reward_target_entrance_frame: str = "task_board/nic_card_mount_0/sfp_port_1_link_entrance"
    insertion_xy_tolerance_m: float = 0.005
    xy_distance_penalty_start_m: float = 0.005
    xy_distance_penalty_per_cm: float = 0.2
    angle_penalty_degrees_per_step: float = 1.0
    angle_penalty_per_3deg_per_sec: float = 0.003
    angle_expected_relative_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


class TrainConfig(DefaultTrainingConfig):
    image_keys = []
    classifier_keys = []
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
        env = AICCableInsertionEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        return env
