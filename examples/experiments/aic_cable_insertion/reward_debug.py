#!/usr/bin/env python3

from __future__ import annotations

import sys
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
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)
            self._max_depth_reward = 0.0
            self._expected_relative_quaternion = _euler_xyz_degrees_to_quat_xyzw(
                config.angle_expected_relative_euler_deg
            )
            self.create_timer(0.2, self._on_timer)

        def _lookup(self, source_frame: str):
            return self._tf_buffer.lookup_transform(
                config.ground_truth_base_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )

        def _on_timer(self):
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
                penalty_per_3deg_per_sec=config.angle_penalty_per_3deg_per_sec,
                control_period_sec=config.policy_control_period_sec,
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

            print(f"total_reward={total_reward:+.3f}", flush=True)

    node = RewardDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
