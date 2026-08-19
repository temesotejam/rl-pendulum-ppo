from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class SensorNoiseConfig:
    enabled: bool = True
    angle_noise_std_deg: float = 0.25
    angle_bias_std_deg: float = 1.0
    gyro_noise_std_dps: float = 0.10
    gyro_bias_std_dps: float = 0.30

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SensorNoiseConfig":
        if not data:
            return cls(enabled=False)
        return cls(
            enabled=bool(data.get("enabled", True)),
            angle_noise_std_deg=float(data.get("angle_noise_std_deg", 0.25)),
            angle_bias_std_deg=float(data.get("angle_bias_std_deg", 1.0)),
            gyro_noise_std_dps=float(data.get("gyro_noise_std_dps", 0.10)),
            gyro_bias_std_dps=float(data.get("gyro_bias_std_dps", 0.30)),
        )

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "enabled": self.enabled,
            "angle_noise_std_deg": self.angle_noise_std_deg,
            "angle_bias_std_deg": self.angle_bias_std_deg,
            "gyro_noise_std_dps": self.gyro_noise_std_dps,
            "gyro_bias_std_dps": self.gyro_bias_std_dps,
        }


class ConsumerImuNoiseWrapper(gym.Wrapper):
    """Make Pendulum observations resemble a modest consumer MEMS IMU.

    The physical simulation itself remains ideal. Only the observation given to
    the agent is corrupted. A fixed bias is sampled once per episode and white
    measurement noise is sampled every observation.

    The wrapped Pendulum observation is still [cos(theta), sin(theta), theta_dot].
    """

    def __init__(self, env: gym.Env, config: SensorNoiseConfig):
        super().__init__(env)
        self.config = config
        self._rng = np.random.default_rng()
        self._angle_bias_rad = 0.0
        self._gyro_bias_rad_s = 0.0

    def _seed_noise(self, seed: int | None) -> None:
        if seed is not None:
            # Keep the sensor RNG deterministic but distinct from the physics RNG.
            self._rng = np.random.default_rng(seed + 1_000_003)

    def _sample_episode_bias(self) -> None:
        if not self.config.enabled:
            self._angle_bias_rad = 0.0
            self._gyro_bias_rad_s = 0.0
            return

        self._angle_bias_rad = math.radians(
            self._rng.normal(0.0, self.config.angle_bias_std_deg)
        )
        self._gyro_bias_rad_s = math.radians(
            self._rng.normal(0.0, self.config.gyro_bias_std_dps)
        )

    def _corrupt(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if not self.config.enabled:
            return observation.copy()

        theta = math.atan2(float(observation[1]), float(observation[0]))
        theta_dot = float(observation[2])

        theta += self._angle_bias_rad + math.radians(
            self._rng.normal(0.0, self.config.angle_noise_std_deg)
        )
        theta_dot += self._gyro_bias_rad_s + math.radians(
            self._rng.normal(0.0, self.config.gyro_noise_std_dps)
        )

        return np.asarray(
            [math.cos(theta), math.sin(theta), theta_dot],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._seed_noise(seed)
        observation, info = self.env.reset(seed=seed, options=options)
        self._sample_episode_bias()
        info = dict(info)
        info["sensor_angle_bias_deg"] = math.degrees(self._angle_bias_rad)
        info["sensor_gyro_bias_dps"] = math.degrees(self._gyro_bias_rad_s)
        return self._corrupt(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["sensor_angle_bias_deg"] = math.degrees(self._angle_bias_rad)
        info["sensor_gyro_bias_dps"] = math.degrees(self._gyro_bias_rad_s)
        return self._corrupt(observation), reward, terminated, truncated, info


def make_pendulum_env(
    sensor_noise: SensorNoiseConfig,
    render_mode: str | None = None,
) -> gym.Env:
    env = gym.make("Pendulum-v1", render_mode=render_mode)
    return ConsumerImuNoiseWrapper(env, sensor_noise)
