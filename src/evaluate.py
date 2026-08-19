from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from stable_baselines3 import PPO

from .environment import SensorNoiseConfig
from .evaluation import evaluate_episode, evaluate_policy

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved Pendulum-v1 PPO model.")
    parser.add_argument("model", type=Path, help="Path to a Stable-Baselines3 .zip checkpoint")
    parser.add_argument("--preset", choices=["quick", "normal", "long"], default="normal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument(
        "--clean-sensors",
        action="store_true",
        help="Disable the simulated IMU noise for this evaluation.",
    )
    return parser.parse_args()


def load_sensor_noise(preset: str, clean_sensors: bool) -> SensorNoiseConfig:
    if clean_sensors:
        return SensorNoiseConfig(enabled=False)
    config_path = REPO_ROOT / "configs" / f"{preset}.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return SensorNoiseConfig.from_dict(config.get("sensor_noise"))


def main() -> None:
    args = parse_args()
    sensor_noise = load_sensor_noise(args.preset, args.clean_sensors)
    model = PPO.load(args.model, device="cpu")
    seeds = [args.seed + index for index in range(args.episodes)]
    metrics = evaluate_policy(model, seeds, sensor_noise=sensor_noise)

    if args.video is not None:
        evaluate_episode(
            model,
            seed=args.seed,
            sensor_noise=sensor_noise,
            video_path=args.video,
        )

    print(
        json.dumps(
            {
                "sensor_noise": sensor_noise.to_dict(),
                "metrics": metrics.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
