"""コマンドの入口。shared/ で python -m temporal_experiment <command> として使う。"""

import argparse
import json
from collections import Counter
from pathlib import Path

from .clips import SPLITS, load_clips, select_truths, unlabelled_frames
from .config import load_config
from .pipeline import run_methods
from .plots import save_all_plots
from .render import render_comparison
from .report import load_estimates, markdown_table, save_estimates, summarize
from .storage import load_detections, load_truth_csv, load_truth_yolo

DEFAULT_CONFIG = Path(__file__).parent / "experiment.yaml"
DEFAULT_RUNS = Path(__file__).parent / "runs"


def _run_dir(args) -> Path:
    return Path(args.runs) / args.video_id


def cmd_infer(args) -> int:
    from .infer import run_inference

    config = load_config(args.config)
    video = Path(args.video)
    weights = Path(args.weights or Path(__file__).parents[1] / config.inference.weights)
    path = run_inference(video, Path(args.runs) / video.stem, weights, config.inference)
    print(f"保存しました: {path}")
    return 0


def _load_run(args) -> tuple:
    meta, records = load_detections(_run_dir(args) / "detections.json")
    config = load_config(args.config).scaled_for(meta["width"])
    return meta, records, config


def cmd_track(args) -> int:
    _meta, records, config = _load_run(args)
    runs = run_methods(records, config)
    save_estimates(_run_dir(args) / "estimates.json", runs)
    for name, run in runs.items():
        counts = Counter(e.status for e in run.estimates)
        print(f"{name}: {dict(counts)} 後処理 {run.elapsed_ms:.1f}ms")
    return 0


def _load_truths(args, meta) -> tuple:
    if args.truth:
        return load_truth_csv(args.truth)
    if args.truth_yolo:
        return load_truth_yolo(args.truth_yolo, args.video_id, meta["width"], meta["height"], 0)
    raise SystemExit("正解データを --truth か --truth-yolo で指定してください")


def cmd_evaluate(args) -> int:
    meta, records, config = _load_run(args)
    all_truths = _load_truths(args, meta)
    clips = load_clips(args.clips)
    if args.truth_yolo:
        unlabelled = unlabelled_frames(all_truths, clips, args.split)
        if unlabelled:
            raise SystemExit(f"区間内にラベルファイルの無いフレームがあります: {unlabelled}")
    truths = select_truths(all_truths, clips, args.split)
    if not truths:
        raise SystemExit(f"split={args.split} の区間に正解データがありません")
    runs = run_methods(records, config)
    summary = {"split": args.split, **summarize(records, runs, truths, config)}
    out_dir = _run_dir(args) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_all_plots(out_dir, runs, truths, summary)
    print(markdown_table(summary))
    return 0


def cmd_video(args) -> int:
    meta, _records, _config = _load_run(args)
    estimates = load_estimates(_run_dir(args) / "estimates.json")
    panels = {"YOLO": estimates["yolo"], "YOLO+linear": estimates["linear"],
              "YOLO+Kalman": estimates["kalman"]}
    out = _run_dir(args) / f"compare_{args.start}_{args.end}.mp4"
    render_comparison(Path(meta["video_path"]), out, panels, args.start, args.end, args.panel_width)
    print(f"保存しました: {out}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="temporal_experiment")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    commands = parser.add_subparsers(dest="command", required=True)

    infer = commands.add_parser("infer", help="全フレームを推論して候補を保存する")
    infer.add_argument("video")
    infer.add_argument("--weights")
    infer.set_defaults(handler=cmd_infer)

    track = commands.add_parser("track", help="保存済みの候補に全手法を適用する")
    track.add_argument("video_id")
    track.set_defaults(handler=cmd_track)

    evaluate = commands.add_parser("evaluate", help="正解データで評価し、指標とグラフを出す")
    evaluate.add_argument("video_id")
    evaluate.add_argument("--truth", type=Path, help="annotate.py 形式の CSV")
    evaluate.add_argument("--truth-yolo", type=Path, help="人が確認済みの YOLO ラベルのディレクトリ")
    evaluate.add_argument("--clips", type=Path, required=True)
    evaluate.add_argument("--split", choices=SPLITS, required=True)
    evaluate.set_defaults(handler=cmd_evaluate)

    video = commands.add_parser("video", help="3手法の比較動画を作る")
    video.add_argument("video_id")
    video.add_argument("--start", type=int, required=True)
    video.add_argument("--end", type=int, required=True)
    video.add_argument("--panel-width", type=int, default=640)
    video.set_defaults(handler=cmd_video)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    return args.handler(args)
