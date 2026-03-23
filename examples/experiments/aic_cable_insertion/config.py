import os
from dataclasses import dataclass

import gymnasium as gym
import jax
import jax.numpy as jnp

from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.aic_cable_insertion.wrapper import AICCableInsertionEnv


@dataclass(frozen=True)
class EnvConfig:
    """AIC task settings for HIL-SERL training.

    This is intentionally aligned with the runtime-side AIC HIL-SERL adapter:
    - images: left/center/right wrist cameras, resized to 128x128
    - state: 46-D vector composed from TCP, joints, and wrist wrench signals

    The live ROS bridge is left to the wrapper implementation.
    """

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
    action_scale_angular: float = 0.06
    max_episode_length: int = 100
    policy_control_period_sec: float = 0.10
    reward_classifier_threshold: float = 0.85
    display_image: bool = True
    ros_domain_id: str | None = None
    observation_timeout_sec: float = 1.0


class TrainConfig(DefaultTrainingConfig):
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
        env = AICCableInsertionEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        if classifier:
            classifier_fn = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            env = AICBinaryRewardClassifierWrapper(
                env,
                classifier_fn=classifier_fn,
                confidence_threshold=EnvConfig.reward_classifier_threshold,
            )

        return env


class AICBinaryRewardClassifierWrapper(gym.Wrapper):
    """Binary reward wrapper for AIC cable insertion."""

    def __init__(
        self,
        env: gym.Env,
        classifier_fn: callable,
        confidence_threshold: float,
    ):
        super().__init__(env)
        self._classifier_fn = classifier_fn
        self._confidence_threshold = confidence_threshold

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = 0
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        logits = self._classifier_fn(obs)
        prob = float(jax.nn.sigmoid(jnp.asarray(logits)).reshape(-1)[0])

        info["classifier_prob"] = prob
        info["succeed"] = 0

        if prob >= self._confidence_threshold:
            reward = 1
            done = True
            info["succeed"] = 1

        return obs, reward, done, truncated, info
