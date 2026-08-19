from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a static GitHub Pages dashboard from an RL training result artifact."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-url", default="")
    return parser.parse_args()


def fmt_num(value: str, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    args = parse_args()
    src = args.input.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    with (src / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metadata = json.loads((src / "metadata.json").read_text(encoding="utf-8"))

    if not rows:
        raise RuntimeError("metrics.csv does not contain any evaluation rows")

    best = max(rows[1:] or rows, key=lambda row: float(row["mean_return"]))
    final = rows[-1]
    random_row = rows[0]

    for dirname in ("videos", "plots"):
        source_dir = src / dirname
        if source_dir.exists():
            shutil.copytree(source_dir, out / dirname, dirs_exist_ok=True)

    for filename in ("metrics.csv", "metadata.json", "summary.md"):
        source = src / filename
        if source.exists():
            shutil.copy2(source, out / filename)

    stage_labels = {
        "random": "学習前",
        "25_percent": "25%",
        "50_percent": "50%",
        "75_percent": "75%",
        "100_percent": "100%",
    }
    video_files = [
        ("random", "videos/00_random.mp4"),
        ("25_percent", "videos/01_25_percent.mp4"),
        ("50_percent", "videos/02_50_percent.mp4"),
        ("75_percent", "videos/03_75_percent.mp4"),
        ("100_percent", "videos/04_100_percent.mp4"),
    ]

    table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(stage_labels.get(row['stage'], row['stage']))}</td>"
        f"<td>{int(float(row['timesteps'])):,}</td>"
        f"<td>{fmt_num(row['mean_return'])}</td>"
        f"<td>{fmt_num(row['rms_angle_deg'], 1)}°</td>"
        f"<td>{float(row['upright_ratio']) * 100:.1f}%</td>"
        f"<td>{fmt_num(row['rms_torque'], 3)}</td>"
        "</tr>"
        for row in rows
    )

    video_buttons = "\n".join(
        f'<button class="video-tab{(" active" if index == len(video_files) - 1 else "")}" '
        f'data-src="{path}" data-label="{html.escape(stage_labels[stage])}">'
        f"{html.escape(stage_labels[stage])}</button>"
        for index, (stage, path) in enumerate(video_files)
    )

    run_link = ""
    if args.run_url:
        safe_url = html.escape(args.run_url, quote=True)
        run_link = (
            f'<a class="button secondary" href="{safe_url}">'
            "GitHub Actions の実行を見る</a>"
        )

    sensor = metadata.get("sensor_noise", {})
    best_label = stage_labels.get(best["stage"], best["stage"])

    page = f'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rl-pendulum-ppo | Training Dashboard</title>
<style>
:root {{ color-scheme: dark; --bg:#0b0f14; --panel:#121922; --panel2:#17202b; --text:#eef4fb; --muted:#9eb0c2; --accent:#63d1ff; --good:#68e0a5; --line:#263544; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:linear-gradient(180deg,#091018 0%,var(--bg) 28%); color:var(--text); }}
a {{ color:var(--accent); }}
.wrap {{ width:min(1180px,calc(100% - 32px)); margin:auto; }}
header {{ padding:52px 0 28px; }}
.eyebrow {{ color:var(--accent); font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-size:.78rem; }}
h1 {{ margin:.35rem 0 .5rem; font-size:clamp(2rem,5vw,4.3rem); line-height:1; }}
.lead {{ color:var(--muted); font-size:1.05rem; max-width:820px; line-height:1.7; }}
.actions {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:20px; }}
.button {{ display:inline-block; padding:10px 14px; border-radius:10px; background:var(--accent); color:#06121a; font-weight:750; text-decoration:none; }}
.button.secondary {{ background:var(--panel2); color:var(--text); border:1px solid var(--line); }}
.grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:18px 0 30px; }}
.card {{ background:rgba(18,25,34,.9); border:1px solid var(--line); border-radius:16px; padding:18px; box-shadow:0 16px 45px rgba(0,0,0,.18); }}
.metric-label {{ color:var(--muted); font-size:.82rem; }}
.metric-value {{ margin-top:6px; font-weight:800; font-size:1.55rem; }}
section {{ margin:30px 0 42px; }}
h2 {{ font-size:1.55rem; margin:0 0 14px; }}
.video-shell {{ overflow:hidden; border-radius:16px; border:1px solid var(--line); background:#000; }}
video {{ display:block; width:100%; max-height:650px; background:#000; }}
.video-tabs {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }}
.video-tab {{ border:1px solid var(--line); color:var(--muted); background:var(--panel); padding:9px 13px; border-radius:999px; cursor:pointer; }}
.video-tab.active {{ color:#071019; background:var(--good); border-color:var(--good); font-weight:800; }}
.video-caption {{ color:var(--muted); margin-top:8px; }}
.plots {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }}
.plot img {{ width:100%; display:block; border-radius:10px; background:white; }}
table {{ width:100%; border-collapse:collapse; overflow:hidden; }}
th,td {{ padding:11px 10px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ color:var(--muted); font-size:.8rem; }}
.table-wrap {{ overflow:auto; }}
.note {{ color:var(--muted); line-height:1.7; }}
.code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; background:var(--panel2); border:1px solid var(--line); padding:2px 6px; border-radius:6px; }}
footer {{ border-top:1px solid var(--line); color:var(--muted); padding:24px 0 40px; margin-top:50px; }}
@media (max-width:850px) {{ .grid,.plots {{ grid-template-columns:1fr 1fr; }} }}
@media (max-width:560px) {{ .grid,.plots {{ grid-template-columns:1fr; }} header {{ padding-top:34px; }} }}
</style>
</head>
<body>
<header><div class="wrap">
<div class="eyebrow">Reinforcement Learning Experiment #1</div>
<h1>rl-pendulum-ppo</h1>
<p class="lead">民生用MEMS IMU相当の観測ノイズを含む <strong>Pendulum-v1</strong> を、PPOが学習して制御していく過程をブラウザから確認できるダッシュボードです。最新の成功した学習結果を表示しています。</p>
<div class="actions"><a class="button" href="https://github.com/temesotejam/rl-pendulum-ppo">GitHubリポジトリ</a>{run_link}<a class="button secondary" href="metrics.csv">metrics.csv</a><a class="button secondary" href="metadata.json">metadata.json</a></div>
</div></header>
<main class="wrap">
<div class="grid">
<div class="card"><div class="metric-label">最終ステップ</div><div class="metric-value">{int(float(final['timesteps'])):,}</div></div>
<div class="card"><div class="metric-label">最終 ±10° 滞在率</div><div class="metric-value">{float(final['upright_ratio']) * 100:.1f}%</div></div>
<div class="card"><div class="metric-label">最良 Mean return</div><div class="metric-value">{float(best['mean_return']):.2f}</div><div class="metric-label">{html.escape(best_label)} checkpoint</div></div>
<div class="card"><div class="metric-label">RMS角度の改善</div><div class="metric-value">{float(random_row['rms_angle_deg']):.1f}° → {float(final['rms_angle_deg']):.1f}°</div></div>
</div>
<section>
<h2>学習の進み方を動画で比較</h2>
<div class="video-tabs">{video_buttons}</div>
<div class="video-shell"><video id="training-video" controls playsinline preload="metadata" src="videos/04_100_percent.mp4"></video></div>
<p class="video-caption">表示中: <strong id="video-label">100%</strong>。同じ評価seedで、学習前から100%までを切り替えて比較できます。</p>
</section>
<section>
<h2>学習曲線と制御性能</h2>
<div class="plots">
<div class="card plot"><img src="plots/learning_curve.png" alt="Learning curve"><p class="note">Mean return。0に近いほど良い。</p></div>
<div class="card plot"><img src="plots/angle_error.png" alt="Angle error"><p class="note">RMS角度誤差。小さいほど直立に近い。</p></div>
<div class="card plot"><img src="plots/upright_ratio.png" alt="Upright ratio"><p class="note">±10°以内に滞在した割合。</p></div>
</div>
</section>
<section class="card">
<h2>評価結果</h2>
<div class="table-wrap"><table><thead><tr><th>段階</th><th>Timesteps</th><th>Mean return</th><th>RMS angle</th><th>±10°</th><th>RMS torque</th></tr></thead><tbody>{table_rows}</tbody></table></div>
<p class="note">今回の最良Mean returnは <strong>{html.escape(best_label)}</strong> の {float(best['mean_return']):.2f}。最終checkpointでは±10°滞在率が {float(final['upright_ratio']) * 100:.1f}% です。</p>
</section>
<section class="card">
<h2>実験条件</h2>
<p class="note">Environment: <span class="code">{html.escape(str(metadata.get('environment', '')))}</span> / Algorithm: <span class="code">{html.escape(str(metadata.get('algorithm', '')))}</span> / Preset: <span class="code">{html.escape(str(metadata.get('preset', '')))}</span> / Seed: <span class="code">{html.escape(str(metadata.get('seed', '')))}</span></p>
<p class="note">Sensor noise: angle σ={sensor.get('angle_noise_std_deg', '?')}°、angle bias σ={sensor.get('angle_bias_std_deg', '?')}°、gyro σ={sensor.get('gyro_noise_std_dps', '?')}°/s、gyro bias σ={sensor.get('gyro_bias_std_dps', '?')}°/s。PPOはノイズ付き観測のみを見て、評価はシミュレータ真値で行います。</p>
</section>
</main>
<footer><div class="wrap">Generated automatically from the latest successful <span class="code">Train RL Agent</span> GitHub Actions run.</div></footer>
<script>
const video = document.getElementById('training-video');
const label = document.getElementById('video-label');
for (const button of document.querySelectorAll('.video-tab')) {{
  button.addEventListener('click', () => {{
    document.querySelectorAll('.video-tab').forEach(item => item.classList.remove('active'));
    button.classList.add('active');
    video.src = button.dataset.src;
    label.textContent = button.dataset.label;
    video.load();
  }});
}}
</script>
</body></html>'''

    (out / "index.html").write_text(page, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
