from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.evaluation import evaluate_policy


def test_pendulum_environment_step() -> None:
    env = gym.make("Pendulum-v1")
    observation, _ = env.reset(seed=123)
    assert observation.shape == (3,)

    observation, reward, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
    assert observation.shape == (3,)
    assert np.isfinite(reward)
    assert not terminated
    assert isinstance(truncated, bool)
    env.close()


def test_short_ppo_training_and_evaluation() -> None:
    env = make_vec_env("Pendulum-v1", n_envs=1, seed=123)
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

    metrics = evaluate_policy(model, seeds=[456])
    assert np.isfinite(metrics.mean_return)
    assert np.isfinite(metrics.rms_angle_deg)
    assert 0.0 <= metrics.upright_ratio <= 1.0
    env.close()
