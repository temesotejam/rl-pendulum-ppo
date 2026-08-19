from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_FIELDS = [
    "stage",
    "progress",
    "timesteps",
    "mean_return",
    "std_return",
    "rms_angle_deg",
    "upright_ratio",
    "rms_angular_velocity",
    "rms_torque",
]


def write_metrics_csv(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in METRIC_FIELDS})


def _save_plot(path: Path, xlabel: str, ylabel: str, title: str) -> None:
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def create_plots(records: list[dict], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    steps = [item["timesteps"] for item in records]

    means = [item["mean_return"] for item in records]
    stds = [item["std_return"] for item in records]
    plt.figure(figsize=(8, 5))
    plt.errorbar(steps, means, yerr=stds, marker="o", capsize=4)
    _save_plot(
        plots_dir / "learning_curve.png",
        "Training timesteps",
        "Mean episode return",
        "Pendulum-v1 PPO learning progress",
    )

    angles = [item["rms_angle_deg"] for item in records]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, angles, marker="o")
    _save_plot(
        plots_dir / "angle_error.png",
        "Training timesteps",
        "RMS angle error [deg]",
        "RMS angle error during evaluation",
    )

    upright = [100.0 * item["upright_ratio"] for item in records]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, upright, marker="o")
    _save_plot(
        plots_dir / "upright_ratio.png",
        "Training timesteps",
        "Time within ±10 deg [%]",
        "Upright time during evaluation",
    )


def write_summary(
    records: list[dict],
    output_path: Path,
    preset: str,
    seed: int,
) -> None:
    best = max(records, key=lambda item: item["mean_return"])
    lines = [
        "# Pendulum PPO training result",
        "",
        f"- Preset: `{preset}`",
        f"- Seed: `{seed}`",
        f"- Best checkpoint: `{best['stage']}`",
        f"- Best mean return: `{best['mean_return']:.2f}` (closer to 0 is better)",
        "",
        "## Evaluation checkpoints",
        "",
        "| Stage | Timesteps | Mean return | RMS angle | Upright ±10° | RMS torque |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in records:
        lines.append(
            "| {stage} | {timesteps:,} | {mean_return:.2f} ± {std_return:.2f} | "
            "{rms_angle_deg:.1f}° | {upright:.1f}% | {rms_torque:.3f} |".format(
                stage=item["stage"],
                timesteps=int(item["timesteps"]),
                mean_return=item["mean_return"],
                std_return=item["std_return"],
                rms_angle_deg=item["rms_angle_deg"],
                upright=100.0 * item["upright_ratio"],
                rms_torque=item["rms_torque"],
            )
        )

    lines.extend(
        [
            "",
            "## How to read this",
            "",
            "`Pendulum-v1` returns are non-positive, so a return closer to 0 is better. "
            "The RMS angle should decrease and the upright ratio should increase as control improves.",
            "",
            "Download the workflow artifact to compare the videos for random, 25%, 50%, 75%, and 100% training progress.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
