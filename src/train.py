from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from .evaluation import evaluate_episode, evaluate_policy
from .reporting import create_plots, write_metrics_csv, write_summary

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = [
    ("25_percent", 0.25),
    ("50_percent", 0.50),
    ("75_percent", 0.75),
    ("100_percent", 1.00),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Gymnasium Pendulum-v1.")
    parser.add_argument("--preset", choices=["quick", "normal", "long"], default="normal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def load_config(preset: str) -> dict:
    path = REPO_ROOT / "configs" / f"{preset}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Pendulum is cheap to simulate and runs in multiple subprocesses. Keeping
    # PyTorch to one thread avoids oversubscribing a 4-vCPU GitHub runner.
    torch.set_num_threads(1)


def version_info() -> dict[str, str]:
    packages = ["gymnasium", "stable-baselines3", "torch", "numpy"]
    return {name: importlib.metadata.version(name) for name in packages}


def record_stage(
    records: list[dict],
    stage: str,
    progress: float,
    timesteps: int,
    policy,
    evaluation_seeds: list[int],
    video_seed: int,
    videos_dir: Path,
    video_index: int,
) -> None:
    metrics = evaluate_policy(policy, evaluation_seeds)
    evaluate_episode(
        policy,
        seed=video_seed,
        video_path=videos_dir / f"{video_index:02d}_{stage}.mp4",
    )
    records.append(
        {
            "stage": stage,
            "progress": progress,
            "timesteps": timesteps,
            **metrics.to_dict(),
        }
    )
    print(
        f"[{stage}] steps={timesteps:,} mean_return={metrics.mean_return:.2f} "
        f"rms_angle={metrics.rms_angle_deg:.1f}deg upright={metrics.upright_ratio * 100:.1f}%"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.preset)
    set_global_seed(args.seed)

    output_dir = args.output_dir.resolve()
    models_dir = output_dir / "models"
    videos_dir = output_dir / "videos"
    plots_dir = output_dir / "plots"
    for directory in [output_dir, models_dir, videos_dir, plots_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    evaluation_episodes = int(config["evaluation_episodes"])
    evaluation_seeds = [args.seed + 100 + index for index in range(evaluation_episodes)]
    video_seed = args.seed + 999

    records: list[dict] = []
    record_stage(
        records,
        stage="random",
        progress=0.0,
        timesteps=0,
        policy=None,
        evaluation_seeds=evaluation_seeds,
        video_seed=video_seed,
        videos_dir=videos_dir,
        video_index=0,
    )

    ppo = config["ppo"]
    env = make_vec_env(
        "Pendulum-v1",
        n_envs=int(ppo["n_envs"]),
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(ppo["learning_rate"]),
        n_steps=int(ppo["n_steps"]),
        batch_size=int(ppo["batch_size"]),
        n_epochs=int(ppo["n_epochs"]),
        gamma=float(ppo["gamma"]),
        gae_lambda=float(ppo["gae_lambda"]),
        clip_range=float(ppo["clip_range"]),
        ent_coef=float(ppo["ent_coef"]),
        use_sde=bool(ppo["use_sde"]),
        sde_sample_freq=int(ppo["sde_sample_freq"]),
        seed=args.seed,
        device="cpu",
        verbose=1,
    )

    total_timesteps = int(config["total_timesteps"])
    for video_index, (stage, fraction) in enumerate(CHECKPOINTS, start=1):
        target = int(total_timesteps * fraction)
        remaining = target - model.num_timesteps
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                reset_num_timesteps=False,
                progress_bar=False,
            )

        model.save(models_dir / f"{stage}.zip")
        record_stage(
            records,
            stage=stage,
            progress=fraction,
            timesteps=int(model.num_timesteps),
            policy=model,
            evaluation_seeds=evaluation_seeds,
            video_seed=video_seed,
            videos_dir=videos_dir,
            video_index=video_index,
        )

    env.close()

    write_metrics_csv(records, output_dir / "metrics.csv")
    create_plots(records, plots_dir)
    write_summary(records, output_dir / "summary.md", args.preset, args.seed)

    metadata = {
        "environment": "Pendulum-v1",
        "algorithm": "PPO",
        "preset": args.preset,
        "seed": args.seed,
        "requested_total_timesteps": total_timesteps,
        "actual_final_timesteps": int(records[-1]["timesteps"]),
        "evaluation_seeds": evaluation_seeds,
        "video_seed": video_seed,
        "config": config,
        "versions": version_info(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print((output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
