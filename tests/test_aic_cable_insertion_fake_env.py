from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pytest


pytest.importorskip("gymnasium")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from experiments.aic_cable_insertion.wrapper import AICCableInsertionEnv
from experiments.aic_cable_insertion.wrapper import _compute_angle_penalty
from experiments.aic_cable_insertion.wrapper import _compute_depth_delta_reward
from experiments.aic_cable_insertion.wrapper import _compute_depth_reward
from experiments.aic_cable_insertion.wrapper import _compute_xy_distance_penalty


@dataclass(frozen=True)
class _TestEnvConfig:
    image_width: int = 128
    image_height: int = 128
    action_scale_linear: float = 0.01
    action_scale_angular: float = 0.06
    max_episode_length: int = 5
    observation_timeout_sec: float = 1.0
    policy_control_period_sec: float = 0.10
    use_sim_time: bool = True
    observation_topic: str = "observations"
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
    enable_keyboard_intervention: bool = False
    intervention_linear_velocity: float = 0.01
    intervention_angular_velocity: float = 0.06


def _make_env():
    return AICCableInsertionEnv(
        fake_env=True,
        save_video=False,
        config=_TestEnvConfig(),
    )


def test_fake_env_reset_returns_valid_observation():
    env = _make_env()
    try:
        obs, info = env.reset(seed=0)

        assert set(obs.keys()) == {"state", "images"}
        assert obs["state"]["tcp_pose"].shape == (7,)
        assert obs["state"]["tcp_vel"].shape == (6,)
        assert obs["state"]["tcp_error"].shape == (6,)
        assert obs["state"]["joint_positions"].shape == (7,)
        assert obs["state"]["joint_velocities"].shape == (7,)
        assert obs["state"]["joint_efforts"].shape == (7,)
        assert obs["state"]["wrist_force"].shape == (3,)
        assert obs["state"]["wrist_torque"].shape == (3,)
        assert obs["images"]["left_camera"].shape == (128, 128, 3)
        assert obs["images"]["center_camera"].shape == (128, 128, 3)
        assert obs["images"]["right_camera"].shape == (128, 128, 3)
        assert obs["images"]["left_camera"].dtype == np.uint8
        assert info == {"succeed": 0}
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_fake_env_step_updates_state_and_truncates():
    env = _make_env()
    try:
        obs, _ = env.reset(seed=0)
        initial_pose = obs["state"]["tcp_pose"].copy()

        action = np.array([1.0, -0.5, 0.25, 0.2, -0.1, 0.3], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(action)

        assert reward == 0.0
        assert done is False
        assert truncated is False
        assert info == {}
        assert not np.allclose(initial_pose[:3], obs["state"]["tcp_pose"][:3])

        for _ in range(env.config.max_episode_length - 1):
            obs, reward, done, truncated, info = env.step(action)

        assert truncated is True
        assert done is False
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_depth_reward_is_linear_from_entrance_to_target():
    port = np.array([0.0, 0.0, 0.00], dtype=np.float32)
    entrance = np.array([0.0, 0.0, 0.02], dtype=np.float32)

    assert np.isclose(
        _compute_depth_reward(
        plug_position=np.array([0.0, 0.0, 0.02], dtype=np.float32),
        port_position=port,
        port_entrance_position=entrance,
        ),
        0.0,
    )
    assert np.isclose(
        _compute_depth_reward(
        plug_position=np.array([0.0, 0.0, 0.015], dtype=np.float32),
        port_position=port,
        port_entrance_position=entrance,
        ),
        0.25,
    )
    assert np.isclose(
        _compute_depth_reward(
        plug_position=np.array([0.0, 0.0, 0.009], dtype=np.float32),
        port_position=port,
        port_entrance_position=entrance,
        ),
        0.55,
    )
    assert np.isclose(
        _compute_depth_reward(
        plug_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        port_position=port,
        port_entrance_position=entrance,
        ),
        1.0,
    )


def test_angle_penalty_is_applied_every_three_degrees_per_axis():
    penalty, euler_deg = _compute_angle_penalty(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    assert penalty == 0.0
    assert np.allclose(euler_deg, np.zeros((3,), dtype=np.float32))

    half_angle_rad = np.deg2rad(6.0 / 2.0)
    quat_x_6_deg = np.array(
        [np.sin(half_angle_rad), 0.0, 0.0, np.cos(half_angle_rad)],
        dtype=np.float32,
    )
    penalty, euler_deg = _compute_angle_penalty(
        quat_x_6_deg,
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    assert np.isclose(penalty, -0.0002)
    assert np.isclose(euler_deg[0], 6.0, atol=1e-4)

    penalty, euler_deg = _compute_angle_penalty(
        quat_x_6_deg,
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        expected_relative_quaternion_xyzw=quat_x_6_deg,
    )
    assert penalty == 0.0
    assert np.allclose(euler_deg, np.zeros((3,), dtype=np.float32), atol=1e-4)


def test_depth_delta_reward_only_rewards_new_max_depth():
    assert np.isclose(_compute_depth_delta_reward(0.5, 0.2), 0.3)
    assert np.isclose(_compute_depth_delta_reward(0.5, 0.5), 0.0)
    assert np.isclose(_compute_depth_delta_reward(0.2, 0.5), 0.0)
    assert np.isclose(_compute_depth_delta_reward(0.45, 0.5), 0.0)
    assert np.isclose(_compute_depth_delta_reward(0.7, 0.5), 0.2)


def test_xy_distance_penalty_only_applies_outside_threshold():
    penalty, distance = _compute_xy_distance_penalty(
        source_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        target_position=np.array([0.003, 0.004, 0.0], dtype=np.float32),
        start_distance_m=0.005,
        penalty_per_cm=0.2,
    )
    assert np.isclose(distance, 0.005, atol=1e-6)
    assert np.isclose(penalty, 0.0)

    penalty, distance = _compute_xy_distance_penalty(
        source_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        target_position=np.array([0.0, 0.015, 0.0], dtype=np.float32),
        start_distance_m=0.005,
        penalty_per_cm=0.2,
    )
    assert np.isclose(distance, 0.015, atol=1e-6)
    assert np.isclose(penalty, -0.2)
