from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from .evaluation import evaluate_episode, evaluate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved Pendulum-v1 PPO model.")
    parser.add_argument("model", type=Path, help="Path to a Stable-Baselines3 .zip checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--video", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = PPO.load(args.model, device="cpu")
    seeds = [args.seed + index for index in range(args.episodes)]
    metrics = evaluate_policy(model, seeds)

    if args.video is not None:
        evaluate_episode(model, seed=args.seed, video_path=args.video)

    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
