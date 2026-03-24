from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


def _image_space(height: int, width: int) -> gym.spaces.Box:
    """创建图像观测空间。
    
    Args:
        height: 图像高度（像素）
        width: 图像宽度（像素）
        
    Returns:
        gym.spaces.Box: 形状为 (height, width, 3) 的 RGB 图像空间，值域 [0, 255]
    """
    return gym.spaces.Box(
        low=0,
        high=255,
        shape=(height, width, 3),
        dtype=np.uint8,
    )


def _vector_space(dim: int) -> gym.spaces.Box:
    """创建向量观测空间。
    
    Args:
        dim: 向量维度
        
    Returns:
        gym.spaces.Box: 形状为 (dim,) 的浮点数向量空间，值域 [-∞, +∞]
    """
    return gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(dim,),
        dtype=np.float32,
    )


@dataclass
class _FakeTaskState:
    """仿真任务状态数据类。
    
    用于在没有真实机器人时模拟环境状态。
    
    Attributes:
        step_count: 当前步数计数
        state: 状态向量（46 维）
    """
    step_count: int = 0
    state: np.ndarray | None = None


class AICCableInsertionEnv(gym.Env):
    """AIC 电缆插入环境，用于 HIL-SERL 训练。
    
    该类实现了 OpenAI Gym 接口，支持两种模式：
    1. 仿真模式 (fake_env=True): 使用简化的物理模型生成虚拟观测
    2. 真实模式 (fake_env=False): 通过 ROS2 与真实机器人或仿真器交互
    
    观测空间包含：
    - 状态信息：TCP 位姿/速度/误差、关节位置/速度/力矩、腕部力/力矩
    - 图像信息：左/中/右三个相机视角
    
    动作空间：6 维连续动作（3 维线速度 + 3 维角速度），范围 [-1, 1]
    """

    metadata = {"render_modes": []}

    def __init__(self, fake_env: bool, save_video: bool, config: Any):
        """初始化环境。
        
        Args:
            fake_env: 是否使用仿真模式
            save_video: 是否保存视频
            config: 配置对象，包含图像尺寸、话题名称等参数
        """
        super().__init__()
        self.fake_env = fake_env
        self.save_video = save_video
        self.config = config
        self._episode_step = 0
        self._fake = _FakeTaskState()
        self._last_action = np.zeros((6,), dtype=np.float32)

        # 定义观测空间：包含机器人状态和三目视觉图像
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": _vector_space(7),      # TCP 位姿 (x,y,z,qx,qy,qz,qw)
                        "tcp_vel": _vector_space(6),       # TCP 速度 (线速度 + 角速度)
                        "tcp_error": _vector_space(6),     # TCP 跟踪误差
                        "joint_positions": _vector_space(7),   # 7 个关节位置
                        "joint_velocities": _vector_space(7),  # 7 个关节速度
                        "joint_efforts": _vector_space(7),     # 7 个关节力矩
                        "wrist_force": _vector_space(3),       # 腕部力传感器 (Fx,Fy,Fz)
                        "wrist_torque": _vector_space(3),      # 腕部力传感器 (Tx,Ty,Tz)
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
        # 定义动作空间：6 维笛卡尔速度命令
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        # 根据模式初始化后端
        if not self.fake_env:
            self._live = self._build_live_backend()
        else:
            self._live = None

    def reset(self, *, seed=None, options=None):
        """重置环境到初始状态。
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            tuple: (观测值, 信息字典)
        """
        super().reset(seed=seed)
        self._episode_step = 0
        self._last_action[:] = 0.0

        if self.fake_env:
            self._fake = self._sample_fake_initial_state()
            obs = self._build_fake_observation()
            return obs, {"succeed": 0}

        # 真实环境：调用后端重置并获取初始观测
        self._live.reset_task()
        obs = self._live.get_observation(timeout_sec=self.config.observation_timeout_sec)
        return obs, {"succeed": 0}

    def step(self, action):
        """执行一步环境交互。
        
        Args:
            action: 6 维动作向量
            
        Returns:
            tuple: (观测值，奖励，是否结束，是否截断，信息字典)
        """
        action = np.asarray(action, dtype=np.float32).reshape(6)
        self._episode_step += 1
        self._last_action = action

        if self.fake_env:
            obs, info = self._step_fake(action)
        else:
            obs, info = self._step_live(action)

        reward = 0.0  # 稀疏奖励，由上层处理
        done = False
        truncated = self._episode_step >= self.config.max_episode_length
        return obs, reward, done, truncated, info

    def close(self):
        """关闭环境并释放资源。"""
        if self._live is not None:
            self._live.close()

    def _step_fake(self, action: np.ndarray):
        """仿真模式下的步进逻辑。
        
        简化模型：动作直接叠加噪声影响状态
        
        Args:
            action: 6 维动作向量
            
        Returns:
            tuple: (观测值，信息字典)
        """
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
        """真实模式下的步进逻辑。
        
        Args:
            action: 6 维动作向量
            
        Returns:
            tuple: (观测值，信息字典)
        """
        info = self._live.apply_action(action)
        obs = self._live.get_observation(timeout_sec=self.config.observation_timeout_sec)
        return obs, info

    def _sample_fake_initial_state(self) -> _FakeTaskState:
        """采样仿真的初始状态。
        
        Returns:
            _FakeTaskState: 初始状态，TCP 位于原点附近，四元数为单位四元数
        """
        base = np.zeros((46,), dtype=np.float32)
        base[:3] = self.np_random.uniform(low=-0.02, high=0.02, size=(3,)).astype(np.float32)
        base[3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return _FakeTaskState(step_count=0, state=base)

    def _build_fake_observation(self):
        """构建仿真观测值。
        
        Returns:
            dict: 包含状态和图像的观测字典
        """
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
        """生成仿真相机图像。
        
        生成带有简单图案的假图像用于测试
        
        Args:
            camera_index: 相机索引 (0=左，1=中，2=右)
            
        Returns:
            np.ndarray: RGB 图像数组
        """
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
        """构建真实 ROS2 后端。
        
        Returns:
            _AICLiveBackend: ROS2 通信后端实例
        """
        return _AICLiveBackend(self.config)


class _AICLiveBackend:
    """ROS2 支持的 AIC 环境适配器，用于 HIL-SERL。
    
    该类负责：
    1. 与 AIC 控制器进行 ROS2 通信（订阅观测、发布动作）
    2. 管理 TF 坐标系变换
    3. 处理键盘干预（人工遥操作）
    4. 提供线程安全的观测缓存机制
    """

    def __init__(self, config: Any):
        """初始化 ROS2 后端。
        
        Args:
            config: 配置对象，包含话题名、服务名、超时等参数
        """
        self.config = config

        import cv2
        import rclpy
        from aic_control_interfaces.msg import MotionUpdate, TargetMode, TrajectoryGenerationMode
        from aic_control_interfaces.srv import ChangeTargetMode
        from aic_model_interfaces.msg import Observation
        from geometry_msgs.msg import Twist, Vector3, Wrench
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from std_srvs.srv import Trigger

        # 保存 ROS2 相关类型引用
        self._cv2 = cv2
        self._rclpy = rclpy
        self._MotionUpdate = MotionUpdate
        self._TargetMode = TargetMode
        self._TrajectoryGenerationMode = TrajectoryGenerationMode
        self._ChangeTargetMode = ChangeTargetMode
        self._ResetJoints = None
        try:
            from aic_engine_interfaces.srv import ResetJoints
        except ImportError:
            self._node_reset_joints_available = False
        else:
            self._ResetJoints = ResetJoints
            self._node_reset_joints_available = True
        self._Observation = Observation
        self._Twist = Twist
        self._Vector3 = Vector3
        self._Wrench = Wrench
        self._Trigger = Trigger
        self._initialized_rclpy = False

        # 初始化 rclpy（如果尚未初始化）
        if not rclpy.ok():
            rclpy.init(args=None)
            self._initialized_rclpy = True

        # 创建 ROS2 节点
        self._node = Node(
            "aic_hil_serl_env",
            parameter_overrides=[
                Parameter("use_sim_time", value=bool(self.config.use_sim_time)),
            ],
        )
        # 启动多线程执行器用于处理回调
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # 观测缓存相关锁和事件
        self._obs_lock = threading.Lock()
        self._obs_event = threading.Event()
        self._latest_obs = None
        self._obs_seq = 0

        # 创建订阅者和发布者
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
        # 创建服务客户端
        self._change_target_mode_client = self._node.create_client(
            self._ChangeTargetMode,
            self.config.change_target_mode_service,
        )
        self._tare_client = self._node.create_client(
            self._Trigger,
            self.config.tare_force_torque_service,
        )
        self._reset_joints_client = None
        if self._node_reset_joints_available:
            self._reset_joints_client = self._node.create_client(
                self._ResetJoints,
                self.config.reset_joints_service,
            )
        else:
            self._node.get_logger().warn(
                "aic_engine_interfaces.srv.ResetJoints unavailable; joint reset disabled."
            )

        # 初始化键盘干预并切换到笛卡尔模式
        self._intervention = self._make_intervention()
        self._ensure_cartesian_mode()
        self._wait_for_initial_observation()

    def reset_task(self):
        """重置任务到初始状态。
        
        执行步骤：
        1. 停止机器人运动
        2. 可选：力传感器归零
        3. 可选：关节复位
        4. 可选：等待人工确认
        5. 稳定后获取新观测
        """
        self._publish_zero_twist()
        time.sleep(0.2)

        if self.config.enable_tare_on_reset:
            self._maybe_tare_force_torque_sensor()

        if self.config.enable_joint_reset and self._reset_joints_client is not None:
            self._maybe_reset_joints()

        if self.config.require_manual_reset_ack:
            input(self.config.reset_prompt)

        time.sleep(self.config.post_reset_settle_sec)
        self._publish_zero_twist()
        self.get_observation(timeout_sec=self.config.observation_timeout_sec, require_new=True)

    def apply_action(self, action: np.ndarray):
        """应用动作命令。
        
        Args:
            action: 6 维动作向量（线速度 + 角速度）
            
        Returns:
            dict: 信息字典，可能包含干预动作
        """
        action = np.asarray(action, dtype=np.float32).reshape(6)
        info = {}
        commanded_action = action

        # 如果启用了键盘干预，优先使用人工控制
        if self._intervention is not None:
            intervene_action = self._intervention.get_action()
            if intervene_action is not None:
                commanded_action = intervene_action
                info["intervene_action"] = intervene_action.copy()

        self._publish_action(commanded_action)
        time.sleep(self.config.policy_control_period_sec)
        return info

    def get_observation(self, timeout_sec: float | None = None, require_new: bool = False):
        """获取机器人观测。
        
        Args:
            timeout_sec: 超时时间（秒）
            require_new: 是否要求获取新的观测（而非缓存）
            
        Returns:
            dict: 观测字典
            
        Raises:
            TimeoutError: 超时时抛出异常
        """
        last_seq = self._obs_seq if require_new else None
        if require_new:
            # 等待新观测的逻辑
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
            # 可接受缓存观测的逻辑
            if not self._obs_event.wait(timeout=timeout_sec):
                raise TimeoutError("Timed out waiting for an AIC observation")
            with self._obs_lock:
                return self._clone_obs(self._latest_obs)

    def close(self):
        """关闭后端并清理 ROS2 资源。"""
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
        """创建键盘干预对象。
        
        Returns:
            _KeyboardIntervention or None: 键盘干预实例或 None
        """
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
        """观测消息回调函数。
        
        Args:
            msg: Observation 消息
        """
        obs = self._adapt_observation(msg)
        with self._obs_lock:
            self._latest_obs = obs
            self._obs_seq += 1
            self._obs_event.set()

    def _adapt_observation(self, obs_msg):
        """适配 ROS2 观测消息为 Gym 格式。
        
        Args:
            obs_msg: Observation ROS2 消息
            
        Returns:
            dict: Gym 格式的观测字典
        """
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
        """从 ROS2 图像消息中提取并处理图像。
        
        Args:
            image_msg: ROS2 Image 消息
            
        Returns:
            np.ndarray: 处理后的 RGB 图像
        """
        img = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
            image_msg.height,
            image_msg.width,
            3,
        )
        # RGB 转 BGR（OpenCV 格式）
        if image_msg.encoding.lower() == "rgb8":
            img = img[..., ::-1]
        # 缩放到配置尺寸
        img = self._cv2.resize(
            img,
            (self.config.image_width, self.config.image_height),
            interpolation=self._cv2.INTER_AREA,
        )
        return img

    @staticmethod
    def _joint_array(values):
        """将关节值列表转换为固定长度数组。
        
        Args:
            values: 关节值列表
            
        Returns:
            np.ndarray: 7 维关节数组，不足补零
        """
        joint_array = np.zeros((7,), dtype=np.float32)
        available = min(7, len(values))
        if available > 0:
            joint_array[:available] = np.asarray(values[:available], dtype=np.float32)
        return joint_array

    def _publish_action(self, action: np.ndarray):
        """发布动作命令到机器人。
        
        Args:
            action: 6 维动作向量（归一化）
        """
        linear = action[:3] * self.config.action_scale_linear
        angular = action[3:6] * self.config.action_scale_angular
        twist = self._Twist(
            linear=self._Vector3(x=float(linear[0]), y=float(linear[1]), z=float(linear[2])),
            angular=self._Vector3(x=float(angular[0]), y=float(angular[1]), z=float(angular[2])),
        )
        self._motion_pub.publish(self._velocity_motion_update(twist))

    def _publish_zero_twist(self):
        """发布零速度命令以停止机器人。"""
        zero_twist = self._Twist(
            linear=self._Vector3(x=0.0, y=0.0, z=0.0),
            angular=self._Vector3(x=0.0, y=0.0, z=0.0),
        )
        self._motion_pub.publish(self._velocity_motion_update(zero_twist))

    def _velocity_motion_update(self, twist):
        """构建速度模式的 MotionUpdate 消息。
        
        Args:
            twist: Twist 消息（线速度 + 角速度）
            
        Returns:
            MotionUpdate: 运动更新消息
        """
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
        """确保控制器处于笛卡尔模式。
        
        Raises:
            RuntimeError: 服务不可用或切换失败时抛出异常
        """
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
        """尝试对力传感器进行归零（如果服务可用）。"""
        if not self._tare_client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().warn(
                f"Tare service unavailable: {self.config.tare_force_torque_service}"
            )
            return
        response = self._call_service(self._tare_client, self._Trigger.Request(), timeout_sec=3.0)
        if response is None or not response.success:
            self._node.get_logger().warn("Force-torque tare request failed")

    def _maybe_reset_joints(self):
        """尝试重置关节到预设位置（如果服务可用）。"""
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
        """等待初始观测到达。"""
        self.get_observation(timeout_sec=max(1.0, self.config.observation_timeout_sec))

    @staticmethod
    def _clone_obs(obs):
        """深拷贝观测字典以避免引用问题。
        
        Args:
            obs: 原始观测字典
            
        Returns:
            dict: 深拷贝后的观测字典
        """
        return {
            "state": {k: v.copy() for k, v in obs["state"].items()},
            "images": {k: v.copy() for k, v in obs["images"].items()},
        }

    @staticmethod
    def _call_service(client, request, timeout_sec: float):
        """同步调用 ROS2 服务。
        
        Args:
            client: ROS2 服务客户端
            request: 请求对象
            timeout_sec: 超时时间（秒）
            
        Returns:
            Response or None: 服务响应或超时返回 None
        """
        future = client.call_async(request)
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if future.done():
                return future.result()
            time.sleep(0.05)
        return None


class _KeyboardIntervention:
    """最小化的键盘遥操作接口，用于 HIL-SERL 演示数据采集。
    
    支持按键：
    - D/A: X 轴正负方向
    - W/S: Y 轴正负方向
    - R/F: Z 轴正负方向
    - W/S(大写): Rx 旋转
    - A/D(大写): Ry 旋转
    - E/Q: Rz 旋转
    """

    _KEY_MAPPINGS = {
        "d": np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),  # +X
        "a": np.array([-1, 0, 0, 0, 0, 0], dtype=np.float32), # -X
        "w": np.array([0, -1, 0, 0, 0, 0], dtype=np.float32), # +Y
        "s": np.array([0, 1, 0, 0, 0, 0], dtype=np.float32),  # -Y
        "r": np.array([0, 0, -1, 0, 0, 0], dtype=np.float32), # +Z
        "f": np.array([0, 0, 1, 0, 0, 0], dtype=np.float32),  # -Z
        "W": np.array([0, 0, 0, 1, 0, 0], dtype=np.float32),  # +Rx
        "S": np.array([0, 0, 0, -1, 0, 0], dtype=np.float32), # -Rx
        "A": np.array([0, 0, 0, 0, -1, 0], dtype=np.float32), # +Ry
        "D": np.array([0, 0, 0, 0, 1, 0], dtype=np.float32),  # -Ry
        "e": np.array([0, 0, 0, 0, 0, 1], dtype=np.float32),  # +Rz
        "q": np.array([0, 0, 0, 0, 0, -1], dtype=np.float32), # -Rz
    }

    def __init__(self):
        """初始化键盘监听器。"""
        from pynput import keyboard

        self._active_keys: set[str] = set()
        self._lock = threading.Lock()
        self._listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._listener.start()

    def get_action(self):
        """获取当前按键组合对应的动作。
        
        Returns:
            np.ndarray or None: 6 维动作向量，无按键时返回 None
        """
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
        """停止键盘监听器。"""
        self._listener.stop()

    def _on_key_press(self, key):
        """按键按下回调。
        
        Args:
            key: 按键对象
        """
        try:
            if hasattr(key, "char") and key.char is not None:
                with self._lock:
                    self._active_keys.add(key.char)
        except AttributeError:
            return

    def _on_key_release(self, key):
        """按键释放回调。
        
        Args:
            key: 按键对象
        """
        try:
            if hasattr(key, "char") and key.char is not None:
                with self._lock:
                    self._active_keys.discard(key.char)
        except AttributeError:
            return
