from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


def _image_space(height: int, width: int) -> gym.spaces.Box:
    return gym.spaces.Box(
        low=0,
        high=255,
        shape=(height, width, 3),
        dtype=np.uint8,
    )


def _vector_space(dim: int) -> gym.spaces.Box:
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(dim,),
        dtype=np.float32,
    )


@dataclass
class _FakeTaskState:
    step_count: int = 0
    state: np.ndarray | None = None


class AICCableInsertionEnv(gym.Env):
    """AIC cable insertion environment for HIL-SERL training."""

    metadata = {"render_modes": []}

    def __init__(self, fake_env: bool, save_video: bool, config: Any):
        super().__init__()
        self.fake_env = fake_env
        self.save_video = save_video
        self.config = config
        self._episode_step = 0
        self._fake = _FakeTaskState()
        self._last_action = np.zeros((6,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": _vector_space(7),
                        "tcp_vel": _vector_space(6),
                        "tcp_error": _vector_space(6),
                        "joint_positions": _vector_space(7),
                        "joint_velocities": _vector_space(7),
                        "joint_efforts": _vector_space(7),
                        "wrist_force": _vector_space(3),
                        "wrist_torque": _vector_space(3),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        "left_camera": _image_space(
                            self.config.image_height,
                            self.config.image_width,
                        ),
                        "center_camera": _image_space(
                            self.config.image_height,
                            self.config.image_width,
                        ),
                        "right_camera": _image_space(
                            self.config.image_height,
                            self.config.image_width,
                        ),
                    }
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        if not self.fake_env:
            self._live = self._build_live_backend()
        else:
            self._live = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_step = 0
        self._last_action[:] = 0.0

        if self.fake_env:
            self._fake = self._sample_fake_initial_state()
            obs = self._build_fake_observation()
            return obs, {"succeed": 0}

        self._live.reset_task()
        obs = self._live.get_observation(timeout_sec=self.config.observation_timeout_sec)
        return obs, {"succeed": 0}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(6)
        self._episode_step += 1
        self._last_action = action

        if self.fake_env:
            obs, info = self._step_fake(action)
        else:
            obs, info = self._step_live(action)

        reward = 0.0
        done = False
        truncated = self._episode_step >= self.config.max_episode_length
        return obs, reward, done, truncated, info

    def close(self):
        if self._live is not None:
            self._live.close()

    def _step_fake(self, action: np.ndarray):
        assert self._fake.state is not None
        noise = self.np_random.normal(0.0, 0.01, size=self._fake.state.shape).astype(np.float32)
        action_effect = np.zeros_like(self._fake.state)
        action_effect[:3] = action[:3] * self.config.action_scale_linear
        action_effect[7:10] = action[:3] * 0.1
        action_effect[13:16] = action[3:6] * 0.1
        self._fake.state = self._fake.state + noise + action_effect
        self._fake.step_count += 1

        obs = self._build_fake_observation()
        info = {}
        return obs, info

    def _step_live(self, action: np.ndarray):
        info = self._live.apply_action(action)
        obs = self._live.get_observation(timeout_sec=self.config.observation_timeout_sec)
        return obs, info

    def _sample_fake_initial_state(self) -> _FakeTaskState:
        base = np.zeros((46,), dtype=np.float32)
        base[:3] = self.np_random.uniform(low=-0.02, high=0.02, size=(3,)).astype(np.float32)
        base[3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return _FakeTaskState(step_count=0, state=base)

    def _build_fake_observation(self):
        assert self._fake.state is not None
        flat = self._fake.state
        return {
            "state": {
                "tcp_pose": flat[0:7].copy(),
                "tcp_vel": flat[7:13].copy(),
                "tcp_error": flat[13:19].copy(),
                "joint_positions": flat[19:26].copy(),
                "joint_velocities": flat[26:33].copy(),
                "joint_efforts": flat[33:40].copy(),
                "wrist_force": flat[40:43].copy(),
                "wrist_torque": flat[43:46].copy(),
            },
            "images": {
                "left_camera": self._fake_image(0),
                "center_camera": self._fake_image(1),
                "right_camera": self._fake_image(2),
            },
        }

    def _fake_image(self, camera_index: int) -> np.ndarray:
        img = np.zeros(
            (self.config.image_height, self.config.image_width, 3),
            dtype=np.uint8,
        )
        band = (camera_index + 1) * 32
        img[..., camera_index] = band
        cursor = int((self._episode_step * 3 + camera_index * 11) % self.config.image_width)
        img[:, max(0, cursor - 2): min(self.config.image_width, cursor + 2), :] = 255
        return img

    def _build_live_backend(self):
        return _AICLiveBackend(self.config)


class _AICLiveBackend:
    """ROS-backed AIC environment adapter for HIL-SERL."""

    def __init__(self, config: Any):
        self.config = config

        import cv2
        import rclpy
        from aic_control_interfaces.msg import MotionUpdate, TargetMode, TrajectoryGenerationMode
        from aic_control_interfaces.srv import ChangeTargetMode
        from aic_engine_interfaces.srv import ResetJoints
        from aic_model_interfaces.msg import Observation
        from geometry_msgs.msg import Twist, Vector3, Wrench
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from std_srvs.srv import Trigger

        self._cv2 = cv2
        self._rclpy = rclpy
        self._MotionUpdate = MotionUpdate
        self._TargetMode = TargetMode
        self._TrajectoryGenerationMode = TrajectoryGenerationMode
        self._ChangeTargetMode = ChangeTargetMode
        self._ResetJoints = ResetJoints
        self._Observation = Observation
        self._Twist = Twist
        self._Vector3 = Vector3
        self._Wrench = Wrench
        self._Trigger = Trigger
        self._initialized_rclpy = False

        if not rclpy.ok():
            rclpy.init(args=None)
            self._initialized_rclpy = True

        self._node = Node(
            "aic_hil_serl_env",
            parameter_overrides=[
                Parameter("use_sim_time", value=bool(self.config.use_sim_time)),
            ],
        )
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self._obs_lock = threading.Lock()
        self._obs_event = threading.Event()
        self._latest_obs = None
        self._obs_seq = 0

        self._observation_sub = self._node.create_subscription(
            self._Observation,
            self.config.observation_topic,
            self._observation_callback,
            10,
        )
        self._motion_pub = self._node.create_publisher(
            self._MotionUpdate,
            self.config.pose_command_topic,
            10,
        )
        self._change_target_mode_client = self._node.create_client(
            self._ChangeTargetMode,
            self.config.change_target_mode_service,
        )
        self._tare_client = self._node.create_client(
            self._Trigger,
            self.config.tare_force_torque_service,
        )
        self._reset_joints_client = self._node.create_client(
            self._ResetJoints,
            self.config.reset_joints_service,
        )

        self._intervention = self._make_intervention()
        self._ensure_cartesian_mode()
        self._wait_for_initial_observation()

    def reset_task(self):
        self._publish_zero_twist()
        time.sleep(0.2)

        if self.config.enable_tare_on_reset:
            self._maybe_tare_force_torque_sensor()

        if self.config.enable_joint_reset:
            self._maybe_reset_joints()

        if self.config.require_manual_reset_ack:
            input(self.config.reset_prompt)

        time.sleep(self.config.post_reset_settle_sec)
        self._publish_zero_twist()
        self.get_observation(timeout_sec=self.config.observation_timeout_sec, require_new=True)

    def apply_action(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(6)
        info = {}
        commanded_action = action

        if self._intervention is not None:
            intervene_action = self._intervention.get_action()
            if intervene_action is not None:
                commanded_action = intervene_action
                info["intervene_action"] = intervene_action.copy()

        self._publish_action(commanded_action)
        time.sleep(self.config.policy_control_period_sec)
        return info

    def get_observation(self, timeout_sec: float | None = None, require_new: bool = False):
        last_seq = self._obs_seq if require_new else None
        if require_new:
            deadline = None if timeout_sec is None else time.time() + timeout_sec
            while True:
                with self._obs_lock:
                    if self._latest_obs is not None and self._obs_seq > last_seq:
                        return self._clone_obs(self._latest_obs)
                remaining = None if deadline is None else max(0.0, deadline - time.time())
                if remaining is not None and remaining == 0.0:
                    raise TimeoutError("Timed out waiting for a fresh AIC observation")
                self._obs_event.wait(timeout=remaining if remaining is not None else 0.1)
                self._obs_event.clear()
        else:
            if not self._obs_event.wait(timeout=timeout_sec):
                raise TimeoutError("Timed out waiting for an AIC observation")
            with self._obs_lock:
                return self._clone_obs(self._latest_obs)

    def close(self):
        try:
            self._publish_zero_twist()
        except Exception:
            pass

        if self._intervention is not None:
            self._intervention.close()

        self._executor.shutdown()
        self._node.destroy_node()
        if self._initialized_rclpy and self._rclpy.ok():
            self._rclpy.shutdown()

    def _make_intervention(self):
        if not self.config.enable_keyboard_intervention:
            return None
        try:
            return _KeyboardIntervention()
        except Exception as exc:
            self._node.get_logger().warn(
                f"Keyboard intervention disabled: {exc}"
            )
            return None

    def _observation_callback(self, msg):
        obs = self._adapt_observation(msg)
        with self._obs_lock:
            self._latest_obs = obs
            self._obs_seq += 1
            self._obs_event.set()

    def _adapt_observation(self, obs_msg):
        return {
            "state": {
                "tcp_pose": np.asarray(
                    [
                        obs_msg.controller_state.tcp_pose.position.x,
                        obs_msg.controller_state.tcp_pose.position.y,
                        obs_msg.controller_state.tcp_pose.position.z,
                        obs_msg.controller_state.tcp_pose.orientation.x,
                        obs_msg.controller_state.tcp_pose.orientation.y,
                        obs_msg.controller_state.tcp_pose.orientation.z,
                        obs_msg.controller_state.tcp_pose.orientation.w,
                    ],
                    dtype=np.float32,
                ),
                "tcp_vel": np.asarray(
                    [
                        obs_msg.controller_state.tcp_velocity.linear.x,
                        obs_msg.controller_state.tcp_velocity.linear.y,
                        obs_msg.controller_state.tcp_velocity.linear.z,
                        obs_msg.controller_state.tcp_velocity.angular.x,
                        obs_msg.controller_state.tcp_velocity.angular.y,
                        obs_msg.controller_state.tcp_velocity.angular.z,
                    ],
                    dtype=np.float32,
                ),
                "tcp_error": np.asarray(
                    list(obs_msg.controller_state.tcp_error[:6]),
                    dtype=np.float32,
                ),
                "joint_positions": self._joint_array(obs_msg.joint_states.position),
                "joint_velocities": self._joint_array(obs_msg.joint_states.velocity),
                "joint_efforts": self._joint_array(obs_msg.joint_states.effort),
                "wrist_force": np.asarray(
                    [
                        obs_msg.wrist_wrench.wrench.force.x,
                        obs_msg.wrist_wrench.wrench.force.y,
                        obs_msg.wrist_wrench.wrench.force.z,
                    ],
                    dtype=np.float32,
                ),
                "wrist_torque": np.asarray(
                    [
                        obs_msg.wrist_wrench.wrench.torque.x,
                        obs_msg.wrist_wrench.wrench.torque.y,
                        obs_msg.wrist_wrench.wrench.torque.z,
                    ],
                    dtype=np.float32,
                ),
            },
            "images": {
                "left_camera": self._extract_image(obs_msg.left_image),
                "center_camera": self._extract_image(obs_msg.center_image),
                "right_camera": self._extract_image(obs_msg.right_image),
            },
        }

    def _extract_image(self, image_msg):
        img = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            image_msg.height,
            image_msg.width,
            3,
        )
        if image_msg.encoding.lower() == "rgb8":
            img = img[..., ::-1]
        img = self._cv2.resize(
            img,
            (self.config.image_width, self.config.image_height),
            interpolation=self._cv2.INTER_AREA,
        )
        return img

    @staticmethod
    def _joint_array(values):
        joint_array = np.zeros((7,), dtype=np.float32)
        available = min(7, len(values))
        if available > 0:
            joint_array[:available] = np.asarray(values[:available], dtype=np.float32)
        return joint_array

    def _publish_action(self, action: np.ndarray):
        linear = action[:3] * self.config.action_scale_linear
        angular = action[3:6] * self.config.action_scale_angular
        twist = self._Twist(
            linear=self._Vector3(x=float(linear[0]), y=float(linear[1]), z=float(linear[2])),
            angular=self._Vector3(x=float(angular[0]), y=float(angular[1]), z=float(angular[2])),
        )
        self._motion_pub.publish(self._velocity_motion_update(twist))

    def _publish_zero_twist(self):
        zero_twist = self._Twist(
            linear=self._Vector3(x=0.0, y=0.0, z=0.0),
            angular=self._Vector3(x=0.0, y=0.0, z=0.0),
        )
        self._motion_pub.publish(self._velocity_motion_update(zero_twist))

    def _velocity_motion_update(self, twist):
        msg = self._MotionUpdate()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self.config.control_frame_id
        msg.velocity = twist
        msg.target_stiffness = np.diag([85.0, 85.0, 85.0, 85.0, 85.0, 85.0]).flatten()
        msg.target_damping = np.diag([75.0, 75.0, 75.0, 75.0, 75.0, 75.0]).flatten()
        msg.feedforward_wrench_at_tip = self._Wrench(
            force=self._Vector3(x=0.0, y=0.0, z=0.0),
            torque=self._Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.trajectory_generation_mode.mode = self._TrajectoryGenerationMode.MODE_VELOCITY
        return msg

    def _ensure_cartesian_mode(self):
        if not self._change_target_mode_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                f"Service unavailable: {self.config.change_target_mode_service}"
            )
        req = self._ChangeTargetMode.Request()
        req.target_mode.mode = self._TargetMode.MODE_CARTESIAN
        response = self._call_service(self._change_target_mode_client, req, timeout_sec=5.0)
        if response is None or not response.success:
            raise RuntimeError("Unable to switch AIC controller to cartesian mode")

    def _maybe_tare_force_torque_sensor(self):
        if not self._tare_client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().warn(
                f"Tare service unavailable: {self.config.tare_force_torque_service}"
            )
            return
        response = self._call_service(self._tare_client, self._Trigger.Request(), timeout_sec=3.0)
        if response is None or not response.success:
            self._node.get_logger().warn("Force-torque tare request failed")

    def _maybe_reset_joints(self):
        if not self._reset_joints_client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().warn(
                f"Reset service unavailable: {self.config.reset_joints_service}"
            )
            return
        req = self._ResetJoints.Request()
        req.joint_names = list(self.config.home_joint_names)
        req.initial_positions = list(self.config.home_joint_positions)
        response = self._call_service(self._reset_joints_client, req, timeout_sec=10.0)
        if response is None or not response.success:
            message = "unknown error" if response is None else response.message
            self._node.get_logger().warn(f"Reset joints request failed: {message}")

    def _wait_for_initial_observation(self):
        self.get_observation(timeout_sec=max(1.0, self.config.observation_timeout_sec))

    @staticmethod
    def _clone_obs(obs):
        return {
            "state": {k: v.copy() for k, v in obs["state"].items()},
            "images": {k: v.copy() for k, v in obs["images"].items()},
        }

    @staticmethod
    def _call_service(client, request, timeout_sec: float):
        future = client.call_async(request)
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if future.done():
                return future.result()
            time.sleep(0.05)
        return None


class _KeyboardIntervention:
    """Minimal keyboard teleoperation for HIL-SERL demo collection."""

    _KEY_MAPPINGS = {
        "d": np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
        "a": np.array([-1, 0, 0, 0, 0, 0], dtype=np.float32),
        "w": np.array([0, -1, 0, 0, 0, 0], dtype=np.float32),
        "s": np.array([0, 1, 0, 0, 0, 0], dtype=np.float32),
        "r": np.array([0, 0, -1, 0, 0, 0], dtype=np.float32),
        "f": np.array([0, 0, 1, 0, 0, 0], dtype=np.float32),
        "W": np.array([0, 0, 0, 1, 0, 0], dtype=np.float32),
        "S": np.array([0, 0, 0, -1, 0, 0], dtype=np.float32),
        "A": np.array([0, 0, 0, 0, -1, 0], dtype=np.float32),
        "D": np.array([0, 0, 0, 0, 1, 0], dtype=np.float32),
        "e": np.array([0, 0, 0, 0, 0, 1], dtype=np.float32),
        "q": np.array([0, 0, 0, 0, 0, -1], dtype=np.float32),
    }

    def __init__(self):
        from pynput import keyboard

        self._active_keys: set[str] = set()
        self._lock = threading.Lock()
        self._listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._listener.start()

    def get_action(self):
        with self._lock:
            if not self._active_keys:
                return None
            action = np.zeros((6,), dtype=np.float32)
            for key in self._active_keys:
                if key in self._KEY_MAPPINGS:
                    action += self._KEY_MAPPINGS[key]
        action = np.clip(action, -1.0, 1.0)
        if not np.any(action):
            return None
        return action

    def close(self):
        self._listener.stop()

    def _on_key_press(self, key):
        try:
            if hasattr(key, "char") and key.char is not None:
                with self._lock:
                    self._active_keys.add(key.char)
        except AttributeError:
            return

    def _on_key_release(self, key):
        try:
            if hasattr(key, "char") and key.char is not None:
                with self._lock:
                    self._active_keys.discard(key.char)
        except AttributeError:
            return
