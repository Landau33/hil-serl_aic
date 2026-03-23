import copy
import time
from franka_env.utils.rotations import euler_2_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
import requests
from pynput import keyboard

from franka_env.envs.franka_env import FrankaEnv

class RAMEnv(FrankaEnv):
    """RAM（随机存取存储器）插入环境。
    
    继承自 FrankaEnv，专为 RAM 条插入任务设计的机器人环境。
    支持键盘干预（F1 键触发重新抓取）和自定义重置逻辑。
    """
    
    def __init__(self, **kwargs):
        """初始化 RAM 环境。
        
        Args:
            **kwargs: 传递给父类 FrankaEnv 的参数
        """
        super().__init__(**kwargs)
        self.should_regrasp = False  # 是否需要重新抓取的标志

        def on_press(key):
            """键盘按下回调函数。
            
            Args:
                key: 按下的键
            """
            if str(key) == "Key.f1":
                self.should_regrasp = True  # F1 键触发重新抓取标志

        listener = keyboard.Listener(
            on_press=on_press)
        listener.start()  # 启动键盘监听器

    def go_to_reset(self, joint_reset=False):
        """移动到重置位置。
        
        在基类定义的基础上，增加 Z 轴偏移以避免与物体碰撞。
        
        Args:
            joint_reset: 是否执行关节重置
        """
        # 使用柔顺模式进行耦合重置
        self._update_currpos()  # 更新当前位置
        self._send_pos_command(self.currpos)  # 发送位置命令保持当前位置
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)  # 切换到精度模式

        # 向上提起，避免碰撞
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = self.resetpos[2] + 0.04  # Z 轴抬高 4cm
        self.interpolate_move(reset_pose, timeout=1)  # 插值移动到目标位置

        # 如果需要，执行关节重置
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # 执行笛卡尔坐标系重置
        if self.randomreset:  # 在 XY 平面随机化重置位置
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)  # 欧拉角转四元数
            self._send_pos_command(reset_pose)
        else:
            reset_pose = self.resetpos.copy()
            self._send_pos_command(reset_pose)
        time.sleep(0.5)

        # 切换回柔顺模式
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)


    def regrasp(self):
        """重新抓取 RAM 条。
        
        用于在抓取失败或需要调整抓取姿态时重新执行抓取流程。
        包含人工交互步骤，需要操作员手动放置 RAM 条。
        """
        # 使用柔顺模式进行耦合重置
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)  # 精度模式

        # 向上提起
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = self.resetpos[2] + 0.04  # Z 轴抬高
        self.interpolate_move(reset_pose, timeout=1)

        # 人工交互：释放夹爪
        input("Press enter to release gripper...")
        self._send_gripper_command(1.0)  # 打开夹爪
        
        # 人工交互：放置 RAM 并重新抓取
        input("Place RAM in holder and press enter to grasp...")
        
        # 移动到抓取位置上方
        top_pose = self.config.GRASP_POSE.copy()
        top_pose[2] += 0.05  # 上方 5cm
        top_pose[0] += np.random.uniform(-0.005, 0.005)  # X 方向小幅随机偏移
        self.interpolate_move(top_pose, timeout=1)
        time.sleep(0.5)

        # 下降到抓取位置
        grasp_pose = top_pose.copy()
        grasp_pose[2] -= 0.05  # 下降 5cm
        self.interpolate_move(grasp_pose, timeout=0.5)

        # 闭合夹爪
        requests.post(self.url + "close_gripper_slow")
        self.last_gripper_act = time.time()
        time.sleep(2)  # 等待夹紧

        # 抬起到安全位置
        self.interpolate_move(top_pose, timeout=0.5)
        time.sleep(0.2)

        # 返回到重置位置
        self.interpolate_move(self.config.RESET_POSE, timeout=1)
        time.sleep(0.5)


    def reset(self, joint_reset=False, **kwargs):
        """重置环境到初始状态。
        
        Args:
            joint_reset: 是否执行关节重置
            **kwargs: 其他参数
            
        Returns:
            tuple: (观测值，信息字典)
        """
        self.last_gripper_act = time.time()
        
        # 保存视频录制（如果启用）
        if self.save_video:
            self.save_video_recording()

        # 如果需要重新抓取
        if self.should_regrasp:
            self.regrasp()
            self.should_regrasp = False

        # 恢复机器人状态
        self._recover()
        self.go_to_reset(joint_reset=False)
        self._recover()
        self.curr_path_length = 0  # 重置路径长度计数器

        # 获取初始观测
        self._update_currpos()
        obs = self._get_obs()
        
        # 切换到柔顺模式
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        
        self.terminate = False  # 重置终止标志
        return obs, {}
