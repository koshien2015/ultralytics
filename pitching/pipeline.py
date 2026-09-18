"""動画から比較ビューアまでを一息で通す。

extract → analyze → compare → viewer を順に呼ぶだけで、新しい計算はしない。
毎回 3〜4 コマンド打つのが面倒なので用意した近道で、各段を個別に実行しても
結果は同じになる。

推論はやり直すと時間がかかるので、出力先に pose.json があれば既定で使い回す。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pitching.adapters.json_adapter import load_pose_json, save_pose_csv, save_pose_json
from pitching.analysis import PitchAnalysis, analyze_pitch
from pitching.comparison.comparator import compare_summaries
from pitching.config import PitchConfig
from pitching.models import PoseSeries
from pitching.reporting.summary import build_summary
from pitching.reporting.writers import write_analysis, write_comparison
from pitching.visualization import charts
from pitching.visualization.viewer import write_viewer


@dataclass
class RunOptions:
    """run の振る舞い。既定はすべて「作る・使い回す」。"""

    output_dir: Path
    force_extract: bool = False
    charts: bool = True
    viewer: bool = True
    overlay_video: bool = False


@dataclass
class RunResult:
    analyses: list[PitchAnalysis]
    outputs: dict[str, Path]
    messages: list[str]


def run(configs: list[PitchConfig], options: RunOptions) -> RunResult:
    """投球ごとに抽出と解析を行い、2球なら比較とビューアまで作る。"""
    if not configs:
        raise ValueError("投球が指定されていません")
    if len(configs) > 2:
        raise ValueError("一度に扱えるのは2投球までです")

    outputs: dict[str, Path] = {}
    messages: list[str] = []
    analyses: list[PitchAnalysis] = []

    for config in configs:
        directory = options.output_dir / config.pitch_id
        series, note = _pose_series(config, directory, options.force_extract)
        messages.append(f"{config.pitch_id}: {note}")

        # 時間の基準はキーポイント側に合わせる。設定の fps とずれていると、
        # 角速度も「最大屈曲→リリース」の秒数も丸ごとずれる。
        if series.fps > 0 and abs(series.fps - config.fps) > 0.01:
            messages.append(f"{config.pitch_id}: fps を {config.fps:g} → {series.fps:g} に合わせた")
            config.fps = series.fps

        analysis = analyze_pitch(series, config)
        analyses.append(analysis)
        written = write_analysis(analysis, directory)
        outputs[f"{config.pitch_id}/metrics"] = written["metrics_json"]

        if options.charts:
            charts.write_pitch_charts(analysis, directory / "charts")
        if options.overlay_video and config.video_path:
            from pitching.visualization.video_overlay import render_overlay_video

            outputs[f"{config.pitch_id}/overlay"] = render_overlay_video(
                analysis, config.video_path, directory / "overlay.mp4"
            )
        for name in analysis.quality["unresolved_events"]:
            messages.append(f"{config.pitch_id}: 未確定のイベント: {name}")

    if len(analyses) == 2:
        report = compare_summaries(*(build_summary(analysis) for analysis in analyses))
        comparison = write_comparison(report, options.output_dir / "comparison")
        outputs["comparison"] = comparison["comparison_json"]
        if options.charts:
            charts.write_comparison_charts(
                analyses[0], analyses[1], options.output_dir / "comparison" / "charts"
            )
        for name in report["unavailable"]:
            messages.append(f"比較できない項目: {name}")

    if options.viewer:
        outputs["viewer"] = write_viewer(analyses, options.output_dir / "viewer.html")

    return RunResult(analyses=analyses, outputs=outputs, messages=messages)


def _pose_series(
    config: PitchConfig, directory: Path, force_extract: bool
) -> tuple[PoseSeries, str]:
    """キーポイントを用意する。既にあれば使い回す。"""
    pose_path = directory / "pose.json"
    if pose_path.is_file() and not force_extract:
        return load_pose_json(pose_path), f"既存のキーポイントを使用（{pose_path}）"

    from pitching.adapters.ultralytics_adapter import extract_pose_series

    series = extract_pose_series(config)
    save_pose_json(series, pose_path)
    save_pose_csv(series, directory / "pose.csv")
    note = f"{len(series)} フレームを抽出"
    if series.notes:
        note = f"{note}（{' / '.join(series.notes)}）"
    return series, note
