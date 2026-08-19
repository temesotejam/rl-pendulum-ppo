from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.environment import SensorNoiseConfig, make_pendulum_env
from src.evaluation import evaluate_policy


def test_consumer_imu_noise_is_deterministic_for_seed() -> None:
    noise = SensorNoiseConfig(
        enabled=True,
        angle_noise_std_deg=0.25,
        angle_bias_std_deg=1.0,
        gyro_noise_std_dps=0.10,
        gyro_bias_std_dps=0.30,
    )
    env_a = make_pendulum_env(noise)
    env_b = make_pendulum_env(noise)

    obs_a, info_a = env_a.reset(seed=123)
    obs_b, info_b = env_b.reset(seed=123)

    assert np.allclose(obs_a, obs_b)
    assert info_a["sensor_angle_bias_deg"] == info_b["sensor_angle_bias_deg"]
    assert info_a["sensor_gyro_bias_dps"] == info_b["sensor_gyro_bias_dps"]

    action = np.array([0.0], dtype=np.float32)
    next_a, reward_a, *_ = env_a.step(action)
    next_b, reward_b, *_ = env_b.step(action)
    assert np.allclose(next_a, next_b)
    assert reward_a == reward_b

    env_a.close()
    env_b.close()


def test_short_ppo_training_and_evaluation_with_sensor_noise() -> None:
    noise = SensorNoiseConfig(
        enabled=True,
        angle_noise_std_deg=0.25,
        angle_bias_std_deg=1.0,
        gyro_noise_std_dps=0.10,
        gyro_bias_std_dps=0.30,
    )

    def env_factory():
        return make_pendulum_env(noise)

    env = make_vec_env(env_factory, n_envs=1, seed=123)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        gamma=0.9,
        learning_rate=1e-3,
        seed=123,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=128)

    metrics = evaluate_policy(model, seeds=[456], sensor_noise=noise)
    assert np.isfinite(metrics.mean_return)
    assert np.isfinite(metrics.rms_angle_deg)
    assert 0.0 <= metrics.upright_ratio <= 1.0
    env.close()
