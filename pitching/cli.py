"""コマンドラインインターフェース。

    python -m pitching extract  --video good.mp4 --config good.yaml --output output/good
    python -m pitching analyze  --pose-data output/good/pose.json --config good.yaml \
                                --output output/good
    python -m pitching compare  --pitch-a output/good/metrics.json \
                                --pitch-b output/bad/metrics.json --output output/comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pitching.adapters.json_adapter import load_pose_json, save_pose_csv, save_pose_json
from pitching.analysis import analyze_pitch
from pitching.comparison.comparator import compare_summaries
from pitching.config import PitchConfig, load_pitch_config
from pitching.reporting.loader import load_analysis, load_summary, resolve_metrics_path
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

    run = subparsers.add_parser(
        "run",
        help="動画からビューアまで一息で通す（extract → analyze → compare → viewer）",
        description="--video を並べた順に1投球ずつ扱う。イベントの指定は直前の --video に付く。",
    )
    run.add_argument("--video", action=_PitchOption, help="入力動画（投球ごとに1つ）")
    run.add_argument("--config", action=_PitchOption, help="直前の --video に使う設定YAML")
    run.add_argument("--id", action=_PitchOption, help="投球の識別名（既定は動画のファイル名）")
    run.add_argument("--label", action=_PitchOption, help="覚え書き（任意。良否の分類ではない）")
    run.add_argument("--release", type=int, action=_PitchOption, help="リリースのフレーム番号")
    run.add_argument("--contact", type=int, action=_PitchOption, help="踏み出し足接地のフレーム番号")
    run.add_argument("--start", type=int, action=_PitchOption, help="解析区間の開始フレーム")
    run.add_argument("--end", type=int, action=_PitchOption, help="解析区間の終了フレーム")
    run.add_argument("--output", required=True, help="出力ディレクトリ")
    run.add_argument("--hand", choices=("right", "left"), default="right", help="投げ手（共通）")
    run.add_argument(
        "--batter", choices=("right", "left"), default="left",
        help="画像上で打者がいる側（共通）",
    )
    run.add_argument("--fps", type=float, help="fps（省略時は動画から取る）")
    run.add_argument("--pose-model", help="Pose 重み")
    run.add_argument("--detection-model", help="11クラス検出モデル。投手の特定に使う（推奨）")
    run.add_argument("--force-extract", action="store_true", help="既存の pose.json を使わず取り直す")
    run.add_argument("--no-charts", action="store_true", help="グラフを出力しない")
    run.add_argument("--no-viewer", action="store_true", help="ビューアを出力しない")
    run.add_argument("--overlay-video", action="store_true", help="解析動画も出力する")
    run.set_defaults(handler=_handle_run)

    viewer = subparsers.add_parser("viewer", help="棒人間ビューアのHTMLを作る")
    viewer.add_argument("pitch", nargs="+", help="analyze の出力ディレクトリ（最大2つ）")
    viewer.add_argument("--output", help="出力HTML（既定は1つ目の隣の viewer.html）")
    viewer.add_argument("--title", help="見出し")
    viewer.set_defaults(handler=_handle_viewer)

    return parser


class _PitchOption(argparse.Action):
    """--video ごとに設定をまとめる。イベント指定は直前の --video に属する。"""

    def __call__(self, parser, namespace, values, option_string=None):
        pitches = list(getattr(namespace, "pitches", None) or [])
        if self.dest == "video":
            pitches.append({"video": values})
        elif not pitches:
            parser.error(f"{option_string} は --video より後に指定してください")
        else:
            pitches[-1] = {**pitches[-1], self.dest: values}
        namespace.pitches = pitches


def _handle_run(args) -> int:
    from pitching.pipeline import RunOptions, run

    pitches = getattr(args, "pitches", None)
    if not pitches:
        raise ValueError("--video を1つ以上指定してください")

    configs = [_config_for_run(pitch, args) for pitch in pitches]
    result = run(
        configs,
        RunOptions(
            output_dir=Path(args.output),
            force_extract=args.force_extract,
            charts=not args.no_charts,
            viewer=not args.no_viewer,
            overlay_video=args.overlay_video,
        ),
    )

    for message in result.messages:
        print(f"  {message}")
    for key, path in result.outputs.items():
        print(f"  {key}: {path}")
    if "viewer" in result.outputs:
        print(f"\nビューアをブラウザで開いてください: {result.outputs['viewer']}")
    return 0


def _config_for_run(pitch: dict, args) -> PitchConfig:
    """--video と、それに続く指定から PitchConfig を組み立てる。

    --config が指定されていればそれを土台にし、コマンドラインの指定で上書きする。
    """
    video = pitch["video"]
    base = load_pitch_config(pitch["config"]).model_dump() if pitch.get("config") else {}

    overrides = {
        "pitch_id": pitch.get("id") or base.get("pitch_id") or Path(video).stem,
        "video_path": video,
        "throwing_hand": base.get("throwing_hand") or args.hand,
        "batter_direction": base.get("batter_direction") or args.batter,
    }
    if args.fps or base.get("fps"):
        overrides["fps"] = args.fps or base["fps"]
    for key, value in (("start_frame", pitch.get("start")), ("end_frame", pitch.get("end"))):
        if value is not None:
            overrides[key] = value

    events = dict(base.get("events") or {})
    if pitch.get("release") is not None:
        events["release_frame"] = pitch["release"]
    if pitch.get("contact") is not None:
        events["stride_foot_contact_frame"] = pitch["contact"]

    result = dict(base.get("result") or {})
    if pitch.get("label"):
        result["label"] = pitch["label"]

    extraction = dict(base.get("extraction") or {})
    if args.pose_model:
        extraction["pose_model_path"] = args.pose_model
    if args.detection_model:
        extraction["detection_model_path"] = args.detection_model

    return PitchConfig.model_validate(
        {**base, **overrides, "events": events, "result": result, "extraction": extraction}
    )


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
    summary_a = load_summary(resolve_metrics_path(args.pitch_a))
    summary_b = load_summary(resolve_metrics_path(args.pitch_b))
    report = compare_summaries(summary_a, summary_b)

    written = write_comparison(report, args.output)

    if not args.no_charts or args.frames:
        analysis_a = load_analysis(resolve_metrics_path(args.pitch_a))
        analysis_b = load_analysis(resolve_metrics_path(args.pitch_b))
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


def _handle_viewer(args) -> int:
    from pitching.visualization.viewer import write_viewer

    analyses = []
    for target in args.pitch:
        metrics_path = resolve_metrics_path(target)
        analysis = load_analysis(metrics_path)
        if analysis is None:
            raise FileNotFoundError(
                f"キーポイントJSONが見つかりません: {metrics_path.parent}"
            )
        analyses.append(analysis)

    default = resolve_metrics_path(args.pitch[0]).parent / "viewer.html"
    path = write_viewer(analyses, args.output or default, args.title)
    print(f"ビューアを書き出しました: {path}")
    return 0
