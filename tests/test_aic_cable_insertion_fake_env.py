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
