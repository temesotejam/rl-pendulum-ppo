from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from .environment import SensorNoiseConfig, make_pendulum_env

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


class PredictPolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True): ...


@dataclass
class EpisodeMetrics:
    episode_return: float
    rms_angle_deg: float
    upright_ratio: float
    rms_angular_velocity: float
    rms_torque: float


@dataclass
class AggregateMetrics:
    mean_return: float
    std_return: float
    rms_angle_deg: float
    upright_ratio: float
    rms_angular_velocity: float
    rms_torque: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _action(policy: PredictPolicy | None, env: gym.Env, observation: np.ndarray) -> np.ndarray:
    if policy is None:
        return np.asarray(env.action_space.sample(), dtype=np.float32)
    action, _ = policy.predict(observation, deterministic=True)
    return np.asarray(action, dtype=np.float32)


def _true_state(env: gym.Env) -> tuple[float, float]:
    """Return the simulator's true angle and angular velocity.

    The policy never receives these values directly when sensor noise is enabled.
    They are used only for objective scoring and plots.
    """
    state = np.asarray(env.unwrapped.state, dtype=np.float64)
    theta = math.atan2(math.sin(float(state[0])), math.cos(float(state[0])))
    return theta, float(state[1])


def evaluate_episode(
    policy: PredictPolicy | None,
    seed: int,
    sensor_noise: SensorNoiseConfig,
    video_path: Path | None = None,
) -> EpisodeMetrics:
    render_mode = "rgb_array" if video_path is not None else None
    env = make_pendulum_env(sensor_noise=sensor_noise, render_mode=render_mode)
    env.action_space.seed(seed + 10_000)
    observation, _ = env.reset(seed=seed)

    frames: list[np.ndarray] = []
    angles: list[float] = []
    angular_velocities: list[float] = []
    torques: list[float] = []
    episode_return = 0.0

    if video_path is not None:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        # `observation` contains simulated sensor error. The policy never sees
        # the true simulator state below.
        action = _action(policy, env, observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)

        theta, theta_dot = _true_state(env)
        angles.append(theta)
        angular_velocities.append(theta_dot)
        torques.append(float(np.asarray(action).reshape(-1)[0]))

        if video_path is not None:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    env.close()

    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=30, macro_block_size=1)

    angle_array = np.asarray(angles, dtype=np.float64)
    angular_velocity_array = np.asarray(angular_velocities, dtype=np.float64)
    torque_array = np.asarray(torques, dtype=np.float64)

    return EpisodeMetrics(
        episode_return=episode_return,
        rms_angle_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(angle_array))))),
        upright_ratio=float(np.mean(np.abs(angle_array) <= np.deg2rad(10.0))),
        rms_angular_velocity=float(np.sqrt(np.mean(np.square(angular_velocity_array)))),
        rms_torque=float(np.sqrt(np.mean(np.square(torque_array)))),
    )


def evaluate_policy(
    policy: PredictPolicy | None,
    seeds: list[int],
    sensor_noise: SensorNoiseConfig,
) -> AggregateMetrics:
    episodes = [
        evaluate_episode(policy, seed=seed, sensor_noise=sensor_noise)
        for seed in seeds
    ]

    returns = np.asarray([item.episode_return for item in episodes], dtype=np.float64)
    return AggregateMetrics(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        rms_angle_deg=float(np.mean([item.rms_angle_deg for item in episodes])),
        upright_ratio=float(np.mean([item.upright_ratio for item in episodes])),
        rms_angular_velocity=float(np.mean([item.rms_angular_velocity for item in episodes])),
        rms_torque=float(np.mean([item.rms_torque for item in episodes])),
    )
