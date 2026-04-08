#!/usr/bin/env python3

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np


HIL_SERL_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = HIL_SERL_ROOT / "examples"
SERL_LAUNCHER_ROOT = HIL_SERL_ROOT / "serl_launcher"

for path in (EXAMPLES_ROOT, SERL_LAUNCHER_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.aic_cable_insertion.config import EnvConfig
from experiments.aic_cable_insertion.wrapper import _compute_angle_penalty
from experiments.aic_cable_insertion.wrapper import _compute_depth_delta_reward
from experiments.aic_cable_insertion.wrapper import _compute_depth_reward
from experiments.aic_cable_insertion.wrapper import _compute_xy_distance_penalty
from experiments.aic_cable_insertion.wrapper import _euler_xyz_degrees_to_quat_xyzw


def main() -> int:
    try:
        from pynput import keyboard
    except ImportError as exc:
        raise RuntimeError("pynput requires a graphical session. Set DISPLAY or run under X11.") from exc
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from tf2_ros import Buffer, TransformException, TransformListener

    config = EnvConfig()
    rclpy.init(args=None)

    class RewardDebugNode(Node):
        def __init__(self):
            super().__init__("aic_reward_debug")
            self.set_parameters(
                [Parameter("use_sim_time", Parameter.Type.BOOL, bool(config.use_sim_time))]
            )
            self._tf_buffer = None
            self._tf_listener = None
            self._max_depth_reward = 0.0
            self._running_total_reward = 0.0
            self._started = False
            self._printed_waiting = False
            self._toggle_lock = threading.Lock()
            self._expected_relative_quaternion = _euler_xyz_degrees_to_quat_xyzw(
                config.angle_expected_relative_euler_deg
            )
            self.create_timer(config.policy_control_period_sec, self._on_timer)

        def _disconnect_tf(self):
            if self._tf_listener is not None:
                for attr_name in ("tf_sub", "tf_static_sub"):
                    sub = getattr(self._tf_listener, attr_name, None)
                    if sub is not None:
                        try:
                            self.destroy_subscription(sub)
                        except Exception:
                            pass
                self._tf_listener = None
            self._tf_buffer = None

        def _lookup(self, source_frame: str):
            if self._tf_buffer is None:
                raise RuntimeError("TF lookup requested before recording started.")
            return self._tf_buffer.lookup_transform(
                config.ground_truth_base_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )

        def toggle_recording(self):
            with self._toggle_lock:
                self._started = not self._started
                self._printed_waiting = False
                if self._started:
                    if self._tf_buffer is None:
                        self._tf_buffer = Buffer()
                        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)
                    self._max_depth_reward = 0.0
                    self._running_total_reward = 0.0
                    print("Reward recording started.", flush=True)
                else:
                    self._disconnect_tf()
                    print("Reward recording stopped.", flush=True)

        def _on_timer(self):
            if not self._started:
                if not self._printed_waiting:
                    print("Waiting for 'r' to start reward recording...", flush=True)
                    self._printed_waiting = True
                return
            try:
                source_tf = self._lookup(config.reward_source_frame)
                target_tf = self._lookup(config.reward_target_frame)
                target_entrance_tf = self._lookup(config.reward_target_entrance_frame)
            except TransformException as exc:
                self.get_logger().warn(f"TF lookup failed: {exc}")
                return

            source_position = np.array(
                [
                    source_tf.transform.translation.x,
                    source_tf.transform.translation.y,
                    source_tf.transform.translation.z,
                ],
                dtype=np.float32,
            )
            source_quaternion = np.array(
                [
                    source_tf.transform.rotation.x,
                    source_tf.transform.rotation.y,
                    source_tf.transform.rotation.z,
                    source_tf.transform.rotation.w,
                ],
                dtype=np.float32,
            )
            target_position = np.array(
                [
                    target_tf.transform.translation.x,
                    target_tf.transform.translation.y,
                    target_tf.transform.translation.z,
                ],
                dtype=np.float32,
            )
            target_quaternion = np.array(
                [
                    target_tf.transform.rotation.x,
                    target_tf.transform.rotation.y,
                    target_tf.transform.rotation.z,
                    target_tf.transform.rotation.w,
                ],
                dtype=np.float32,
            )
            target_entrance_position = np.array(
                [
                    target_entrance_tf.transform.translation.x,
                    target_entrance_tf.transform.translation.y,
                    target_entrance_tf.transform.translation.z,
                ],
                dtype=np.float32,
            )

            depth_reward = _compute_depth_reward(
                plug_position=source_position,
                port_position=target_position,
                port_entrance_position=target_entrance_position,
                xy_tolerance_m=config.insertion_xy_tolerance_m,
            )
            depth_delta_reward = _compute_depth_delta_reward(
                current_depth_reward=depth_reward,
                max_depth_reward=self._max_depth_reward,
            )
            angle_penalty, euler_deg = _compute_angle_penalty(
                source_quaternion_xyzw=source_quaternion,
                target_quaternion_xyzw=target_quaternion,
                expected_relative_quaternion_xyzw=self._expected_relative_quaternion,
                degrees_per_step=config.angle_penalty_degrees_per_step,
                penalty_per_bucket_per_step=config.angle_penalty_per_bucket_per_step,
            )
            xy_distance_penalty, xy_distance = _compute_xy_distance_penalty(
                source_position=source_position,
                target_position=target_position,
                start_distance_m=config.xy_distance_penalty_start_m,
                penalty_per_cm=config.xy_distance_penalty_per_cm,
            )
            total_reward = (
                depth_delta_reward
                + angle_penalty
                + xy_distance_penalty
            )

            self._max_depth_reward = max(self._max_depth_reward, depth_reward)
            self._running_total_reward += total_reward

            print(
                " ".join(
                    [
                        f"step_reward={total_reward:+.3f}",
                        f"total_reward={self._running_total_reward:+.3f}",
                        (
                            "angle_deg="
                            f"[{euler_deg[0]:.2f}, {euler_deg[1]:.2f}, {euler_deg[2]:.2f}]"
                        ),
                    ]
                ),
                flush=True,
            )

    node = RewardDebugNode()
    def on_press(key):
        try:
            if hasattr(key, "char") and key.char == "r":
                node.toggle_recording()
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        node._disconnect_tf()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
