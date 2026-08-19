# Pendulum PPO training result

- Preset: `normal`
- Seed: `42`
- Best checkpoint: `50_percent`
- Best mean return: `-175.81` (closer to 0 is better)

## Evaluation checkpoints

| Stage | Timesteps | Mean return | RMS angle | Upright ±10° | RMS torque |
|---|---:|---:|---:|---:|---:|
| random | 0 | -1271.45 ± 314.48 | 127.2° | 1.4% | 1.157 |
| 25_percent | 28,672 | -194.56 ± 142.19 | 43.7° | 76.7% | 0.844 |
| 50_percent | 53,248 | -175.81 ± 123.80 | 41.4° | 78.6% | 0.737 |
| 75_percent | 77,824 | -190.63 ± 137.14 | 43.2° | 78.1% | 0.805 |
| 100_percent | 102,400 | -176.89 ± 124.33 | 41.7° | 79.8% | 0.778 |

## How to read this

`Pendulum-v1` returns are non-positive, so a return closer to 0 is better. The RMS angle should decrease and the upright ratio should increase as control improves.

Download the workflow artifact to compare the videos for random, 25%, 50%, 75%, and 100% training progress.
