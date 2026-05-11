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
from experiments.aic_cable_insertion.scripted_intervention import (
    ScriptedCableInsertionIntervention,
    ScriptedInterventionConfig,
)


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
    intervention_angular_velocity: float = 0.05
    enable_scripted_intervention: bool = False
    scripted_intervention_toggle_key: str = "0"
    scripted_intervention_tip_frame: str = "cable_1/sc_tip_link"
    scripted_intervention_port_frame: str = "task_board/sc_port_1/sc_port_base_link"
    scripted_intervention_align_linear_gain: float = 8.0
    scripted_intervention_insert_xy_gain: float = 12.0
    scripted_intervention_insert_z_gain: float = 40.0
    scripted_intervention_align_angular_gain: float = 1.5
    scripted_intervention_insert_angular_gain: float = 0.8
    scripted_intervention_min_insert_z_velocity: float = 0.004
    scripted_intervention_linear_deadband_m: float = 0.0015
    scripted_intervention_angular_deadband_rad: float = 0.03
    scripted_intervention_xy_align_tolerance_m: float = 0.0025
    scripted_intervention_z_insert_tolerance_m: float = 0.0015
    scripted_intervention_orientation_align_tolerance_rad: float = 0.05


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
        assert info == {"succeed": 0}
        assert not np.allclose(initial_pose[:3], obs["state"]["tcp_pose"][:3])

        for _ in range(env.config.max_episode_length - 1):
            obs, reward, done, truncated, info = env.step(action)

        assert truncated is True
        assert done is False
        assert env.observation_space.contains(obs)
    finally:
        env.close()


class _DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warn(self, message):
        self.messages.append(("warn", message))


class _DummyTransform:
    def __init__(self, translation, rotation):
        self.transform = type("TransformContainer", (), {})()
        self.transform.translation = type("Translation", (), {})()
        self.transform.rotation = type("Rotation", (), {})()
        self.transform.translation.x = translation[0]
        self.transform.translation.y = translation[1]
        self.transform.translation.z = translation[2]
        self.transform.rotation.x = rotation[0]
        self.transform.rotation.y = rotation[1]
        self.transform.rotation.z = rotation[2]
        self.transform.rotation.w = rotation[3]


class _DummyTfBuffer:
    def __init__(self, transforms):
        self._transforms = transforms

    def lookup_transform(self, target_frame, source_frame, now):
        del target_frame, now
        return self._transforms[source_frame]


def test_scripted_intervention_toggle_and_action():
    logger = _DummyLogger()
    config = ScriptedInterventionConfig()
    # Port placed above tip by more than safe_axial_clearance_m so the lift
    # safety phase doesn't trigger; we want to exercise align here.
    tf_buffer = _DummyTfBuffer(
        {
            config.port_frame: _DummyTransform(
                translation=(0.10, 0.00, 0.011),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
            config.tip_frame: _DummyTransform(
                translation=(0.08, -0.01, 0.00),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        }
    )
    controller = ScriptedCableInsertionIntervention(
        tf_buffer=tf_buffer,
        logger=logger,
        time_type=lambda: None,
        transform_exception=RuntimeError,
        config=config,
    )

    assert controller.get_action() is None
    assert controller.active is False

    controller.toggle()
    action = controller.get_action()

    assert controller.active is True
    assert action is not None
    np.testing.assert_allclose(
        action[:3],
        np.array([2.0, 2.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(action[3:], np.zeros(3, dtype=np.float32))


def test_scripted_intervention_start_stop_semantics():
    logger = _DummyLogger()
    config = ScriptedInterventionConfig()
    tf_buffer = _DummyTfBuffer(
        {
            config.port_frame: _DummyTransform(
                translation=(0.10, 0.00, 0.00),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
            config.tip_frame: _DummyTransform(
                translation=(0.08, -0.01, 0.00),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        }
    )
    controller = ScriptedCableInsertionIntervention(
        tf_buffer=tf_buffer,
        logger=logger,
        time_type=lambda: None,
        transform_exception=RuntimeError,
        config=config,
    )

    controller.start()
    assert controller.active is True
    assert controller.get_action() is not None

    controller.stop()
    assert controller.active is False
    assert controller.get_action() is None


def test_scripted_intervention_align_phase_uses_soft_xy_gain_near_deadband():
    logger = _DummyLogger()
    config = ScriptedInterventionConfig(
        align_linear_gain=8.0,
        linear_deadband_m=0.0005,
        xy_align_tolerance_m=0.0005,
    )
    tf_buffer = _DummyTfBuffer(
        {
            config.port_frame: _DummyTransform(
                translation=(0.10, 0.00, 0.00),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
            config.tip_frame: _DummyTransform(
                translation=(0.0992, 0.00, 0.00),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        }
    )
    controller = ScriptedCableInsertionIntervention(
        tf_buffer=tf_buffer,
        logger=logger,
        time_type=lambda: None,
        transform_exception=RuntimeError,
        config=config,
    )
    controller.toggle()

    action = controller.get_action()

    expected_x_action = 0.0008 * config.align_linear_gain / config.action_scale_linear
    assert 0.0 < action[0] < config.max_linear_velocity / config.action_scale_linear
    np.testing.assert_allclose(action[0], expected_x_action, rtol=1e-6)


def test_scripted_intervention_align_stuck_boosts_horizontal_correction():
    logger = _DummyLogger()
    config = ScriptedInterventionConfig(
        align_linear_gain=1.0,
        linear_deadband_m=0.0005,
        xy_align_tolerance_m=0.0003,
        align_stuck_window_steps=2,
        align_stuck_xy_progress_threshold_m=0.001,
        align_stuck_xy_min_velocity=0.01,
    )
    tf_buffer = _DummyTfBuffer(
        {
            config.port_frame: _DummyTransform(
                translation=(0.10, 0.00, 0.00),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
            config.tip_frame: _DummyTransform(
                translation=(0.0992, 0.00, 0.00),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        }
    )
    controller = ScriptedCableInsertionIntervention(
        tf_buffer=tf_buffer,
        logger=logger,
        time_type=lambda: None,
        transform_exception=RuntimeError,
        config=config,
    )
    controller.toggle()

    first_action = controller.get_action()
    second_action = controller.get_action()
    third_action = controller.get_action()

    assert controller.status()["align_stuck"] is True
    assert 0.0 < first_action[0] < third_action[0]
    np.testing.assert_allclose(second_action[0], first_action[0], rtol=1e-6)
    assert (
        third_action[0]
        >= config.align_stuck_xy_min_velocity / config.action_scale_linear
    )


def test_scripted_intervention_inserts_z_only_after_alignment():
    logger = _DummyLogger()
    config = ScriptedInterventionConfig()

    align_tf_buffer = _DummyTfBuffer(
        {
            config.port_frame: _DummyTransform(
                translation=(0.10, 0.00, 0.03),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
            config.tip_frame: _DummyTransform(
                translation=(0.08, -0.01, 0.00),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        }
    )
    controller = ScriptedCableInsertionIntervention(
        tf_buffer=align_tf_buffer,
        logger=logger,
        time_type=lambda: None,
        transform_exception=RuntimeError,
        config=config,
    )
    controller.toggle()

    align_action = controller.get_action()
    np.testing.assert_allclose(
        align_action[:3],
        np.array([2.0, 2.0, 0.0], dtype=np.float32),
    )

    insert_tf_buffer = _DummyTfBuffer(
        {
            config.port_frame: _DummyTransform(
                translation=(0.10, 0.00, 0.03),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
            config.tip_frame: _DummyTransform(
                translation=(0.099, 0.001, 0.00),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        }
    )
    controller._tf_buffer = insert_tf_buffer
    controller._phase = "insert"
    insert_action = controller.get_action()

    np.testing.assert_allclose(
        insert_action[:2], np.zeros(2, dtype=np.float32), atol=1e-6
    )
    assert insert_action[2] > 0.0


def test_scripted_intervention_insert_phase_applies_minimum_z_push():
    logger = _DummyLogger()
    config = ScriptedInterventionConfig()
    tf_buffer = _DummyTfBuffer(
        {
            config.port_frame: _DummyTransform(
                translation=(0.10, 0.00, 0.0018),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
            config.tip_frame: _DummyTransform(
                translation=(0.099, 0.001, 0.00),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        }
    )
    controller = ScriptedCableInsertionIntervention(
        tf_buffer=tf_buffer,
        logger=logger,
        time_type=lambda: None,
        transform_exception=RuntimeError,
        config=config,
    )
    controller.toggle()
    controller._phase = "insert"
    action = controller.get_action()

    expected_z_action = config.min_insert_z_velocity / config.action_scale_linear
    assert action[2] >= expected_z_action


def test_scripted_intervention_stuck_mode_increases_xy_but_limits_z():
    logger = _DummyLogger()
    config = ScriptedInterventionConfig(
        stuck_window_steps=2,
        stuck_z_progress_threshold_m=0.01,
    )
    tf_buffer = _DummyTfBuffer(
        {
            config.port_frame: _DummyTransform(
                translation=(0.10, 0.00, 0.02),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
            config.tip_frame: _DummyTransform(
                translation=(0.099, 0.001, 0.00),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        }
    )
    controller = ScriptedCableInsertionIntervention(
        tf_buffer=tf_buffer,
        logger=logger,
        time_type=lambda: None,
        transform_exception=RuntimeError,
        config=config,
    )
    controller.toggle()
    controller._phase = "insert"

    first_action = controller.get_action()
    second_action = controller.get_action()
    third_action = controller.get_action()

    assert controller.status()["stuck"] is True
    assert abs(third_action[0]) >= abs(second_action[0])
    assert abs(third_action[1]) >= abs(second_action[1])
    assert abs(third_action[2]) <= (
        config.aggressive_z_velocity_scale
        * config.aggressive_max_linear_velocity
        / config.action_scale_linear
    )


def test_scripted_intervention_stuck_search_uses_raw_target_direction_in_deadband():
    logger = _DummyLogger()
    config = ScriptedInterventionConfig(
        linear_deadband_m=0.001,
        xy_align_tolerance_m=0.001,
        stuck_window_steps=1,
        stuck_z_progress_threshold_m=0.01,
        stuck_search_linear_velocity=0.004,
        stuck_search_angular_velocity=0.0,
        stuck_search_period_steps=4,
        stuck_search_ramp_steps=1,
    )
    tf_buffer = _DummyTfBuffer(
        {
            config.port_frame: _DummyTransform(
                translation=(0.10, 0.00, 0.02),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
            config.tip_frame: _DummyTransform(
                translation=(0.0998, 0.00, 0.00),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        }
    )
    controller = ScriptedCableInsertionIntervention(
        tf_buffer=tf_buffer,
        logger=logger,
        time_type=lambda: None,
        transform_exception=RuntimeError,
        config=config,
    )
    controller.toggle()
    controller._phase = "insert"

    first_action = controller.get_action()
    second_action = controller.get_action()

    assert controller.status()["stuck"] is True
    np.testing.assert_allclose(first_action[0], 0.0, atol=1e-6)
    assert second_action[0] > 0.0


def test_scripted_intervention_insert_halts_z_when_lateral_drifts():
    """Insert phase must hold axial descent if lateral drifts out of tolerance."""
    logger = _DummyLogger()
    config = ScriptedInterventionConfig(
        xy_align_tolerance_m=0.0005,
    )
    tf_buffer = _DummyTfBuffer(
        {
            config.port_frame: _DummyTransform(
                translation=(0.10, 0.00, 0.02),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
            config.tip_frame: _DummyTransform(
                translation=(0.097, 0.00, 0.00),  # 3 mm lateral drift
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
        }
    )
    controller = ScriptedCableInsertionIntervention(
        tf_buffer=tf_buffer,
        logger=logger,
        time_type=lambda: None,
        transform_exception=RuntimeError,
        config=config,
    )
    controller.toggle()
    controller._phase = "insert"

    action = controller.get_action()

    # axial (z) must be held; lateral correction must still drive x.
    np.testing.assert_allclose(action[2], 0.0, atol=1e-6)
    assert action[0] > 0.0


def test_scripted_intervention_insert_phase_keeps_xy_and_angle_within_deadband():
    logger = _DummyLogger()
    angle_error = 0.012
    config = ScriptedInterventionConfig(
        insert_xy_gain=1.0,
        insert_angular_gain=0.1,
        linear_deadband_m=0.0005,
        angular_deadband_rad=0.01,
        min_insert_xy_correction_velocity=0.002,
        min_insert_angular_correction_velocity=0.006,
        stuck_window_steps=100,
    )
    tf_buffer = _DummyTfBuffer(
        {
            config.port_frame: _DummyTransform(
                translation=(0.10, 0.00, 0.002),
                rotation=(0.0, 0.0, 0.0, 1.0),
            ),
            config.tip_frame: _DummyTransform(
                translation=(0.0994, 0.00, 0.00),
                rotation=(
                    0.0,
                    0.0,
                    -np.sin(angle_error / 2.0),
                    np.cos(angle_error / 2.0),
                ),
            ),
        }
    )
    controller = ScriptedCableInsertionIntervention(
        tf_buffer=tf_buffer,
        logger=logger,
        time_type=lambda: None,
        transform_exception=RuntimeError,
        config=config,
    )
    controller.toggle()
    controller._phase = "insert"

    action = controller.get_action()

    assert controller.status()["stuck"] is False
    assert (
        action[0]
        >= config.min_insert_xy_correction_velocity / config.action_scale_linear
    )
    assert (
        action[5]
        >= config.min_insert_angular_correction_velocity / config.action_scale_angular
    )
