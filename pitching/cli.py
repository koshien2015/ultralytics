"""コマンドラインインターフェース。

    python -m pitching extract  --video good.mp4 --config good.yaml --output output/good
    python -m pitching analyze  --pose-data output/good/pose.json --config good.yaml \
                                --output output/good
    python -m pitching compare  --pitch-a output/good/metrics.json \
                                --pitch-b output/bad/metrics.json --output output/comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pitching.adapters.json_adapter import load_pose_json, save_pose_csv, save_pose_json
from pitching.analysis import analyze_pitch
from pitching.comparison.comparator import compare_summaries
from pitching.config import load_pitch_config
from pitching.reporting.summary import config_from_summary
from pitching.reporting.writers import write_analysis, write_comparison
from pitching.visualization import charts


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"エラー: {error}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pitching",
        description="キャップ野球の投球フォームを解析・比較する",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="動画から YOLO Pose のキーポイントを抽出する")
    extract.add_argument("--video", help="入力動画（省略時は設定の video_path）")
    extract.add_argument("--config", required=True, help="投球設定 YAML")
    extract.add_argument("--output", required=True, help="出力ディレクトリ")
    extract.set_defaults(handler=_handle_extract)

    analyze = subparsers.add_parser("analyze", help="キーポイント列を解析する")
    analyze.add_argument("--pose-data", required=True, help="pose.json")
    analyze.add_argument("--config", required=True, help="投球設定 YAML")
    analyze.add_argument("--output", required=True, help="出力ディレクトリ")
    analyze.add_argument("--no-charts", action="store_true", help="グラフを出力しない")
    analyze.add_argument("--overlay-video", action="store_true", help="解析動画も出力する")
    analyze.set_defaults(handler=_handle_analyze)

    compare = subparsers.add_parser("compare", help="2投球を比較する")
    compare.add_argument("--pitch-a", required=True, help="比較元の metrics.json")
    compare.add_argument("--pitch-b", required=True, help="比較先の metrics.json")
    compare.add_argument("--output", required=True, help="出力ディレクトリ")
    compare.add_argument("--no-charts", action="store_true", help="比較グラフを出力しない")
    compare.add_argument(
        "--frames", action="store_true", help="同一イベントの比較画像も出力する（動画が必要）"
    )
    compare.set_defaults(handler=_handle_compare)

    return parser


def _handle_extract(args) -> int:
    from pitching.adapters.ultralytics_adapter import extract_pose_series

    config = load_pitch_config(args.config)
    series = extract_pose_series(config, args.video)

    output_dir = Path(args.output)
    json_path = save_pose_json(series, output_dir / "pose.json")
    csv_path = save_pose_csv(series, output_dir / "pose.csv")

    print(f"{len(series)} フレームを抽出しました")
    for note in series.notes:
        print(f"  注意: {note}")
    print(f"  {json_path}\n  {csv_path}")
    return 0


def _handle_analyze(args) -> int:
    config = load_pitch_config(args.config)
    series = load_pose_json(args.pose_data)
    analysis = analyze_pitch(series, config)

    written = write_analysis(analysis, args.output)
    if not args.no_charts:
        charts.write_pitch_charts(analysis, Path(args.output) / "charts")

    if args.overlay_video:
        from pitching.visualization.video_overlay import render_overlay_video

        video_path = config.video_path or series.video_path
        if not video_path:
            print("警告: video_path が無いため解析動画は出力しません", file=sys.stderr)
        else:
            written["overlay_video"] = render_overlay_video(
                analysis, video_path, Path(args.output) / "overlay.mp4"
            )

    print(f"解析しました: {analysis.pitch_id}")
    for name in analysis.quality["unresolved_events"]:
        print(f"  未確定のイベント: {name}")
    for key, path in written.items():
        print(f"  {key}: {path}")
    return 0


def _handle_compare(args) -> int:
    summary_a = _load_summary(args.pitch_a)
    summary_b = _load_summary(args.pitch_b)
    report = compare_summaries(summary_a, summary_b)

    written = write_comparison(report, args.output)

    if not args.no_charts or args.frames:
        analysis_a = _rebuild_analysis(args.pitch_a, summary_a)
        analysis_b = _rebuild_analysis(args.pitch_b, summary_b)
        if analysis_a is None or analysis_b is None:
            print(
                "警告: pose_raw.json が見つからないためグラフ・比較画像は出力しません",
                file=sys.stderr,
            )
        else:
            if not args.no_charts:
                charts.write_comparison_charts(analysis_a, analysis_b, Path(args.output) / "charts")
            if args.frames:
                from pitching.visualization.comparison_frames import write_comparison_frames

                write_comparison_frames(analysis_a, analysis_b, Path(args.output) / "frames")

    print(f"比較しました: {summary_a.get('pitch_id')} vs {summary_b.get('pitch_id')}")
    for name in report["unavailable"]:
        print(f"  比較できない項目: {name}")
    for key, path in written.items():
        print(f"  {key}: {path}")
    return 0


def _load_summary(path: str) -> dict:
    summary_path = Path(path)
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"metrics.json が見つかりません: {summary_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"metrics.json が不正です: {summary_path}: {error}") from error


def _rebuild_analysis(metrics_path: str, summary: dict):
    """metrics.json の隣のキーポイントJSONから解析をやり直す（比較グラフ用）。

    analyze が書いた pose_raw.json を優先し、無ければ extract が書いた pose.json を使う。
    """
    directory = Path(metrics_path).parent
    for filename in ("pose_raw.json", "pose.json"):
        pose_path = directory / filename
        if pose_path.is_file():
            return analyze_pitch(load_pose_json(pose_path), config_from_summary(summary))
    return None
