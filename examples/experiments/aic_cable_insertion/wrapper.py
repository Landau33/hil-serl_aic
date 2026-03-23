from __future__ import annotations

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
    """AIC cable insertion environment skeleton for HIL-SERL.

    Contract:
    - raw observation matches what SERLObsWrapper expects:
      {"state": {...}, "images": {...}}
    - flattened observation after wrapping matches the runtime-side TestPolicy
      adapter ordering and image keys

    Current status:
    - `fake_env=True` works for shape checks, classifier training, and script wiring
    - live AIC/ROS collection is intentionally left as a concrete integration TODO
    """

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
        obs = self._live.get_observation()
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
    """Placeholder for the actual ROS/AIC online collection bridge."""

    def __init__(self, config: Any):
        self.config = config
        raise NotImplementedError(
            "AIC live HIL-SERL data collection is not wired yet. "
            "Implement the ROS bridge here to stream Observation messages, "
            "publish velocity commands, expose human intervention actions, and "
            "reset the task between episodes."
        )

    def reset_task(self):
        raise NotImplementedError

    def apply_action(self, action: np.ndarray):
        raise NotImplementedError

    def get_observation(self, timeout_sec: float | None = None):
        _ = timeout_sec
        raise NotImplementedError

    def close(self):
        return None
