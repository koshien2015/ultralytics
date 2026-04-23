from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pitching.app.run_pipeline import run


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="キャップ野球動画解析パイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ステージ名一覧:
  frame_diff, yolo_detect, fusion, pose_estimation,
  tracking, strike_zone, release_detection, pitch_analysis, rendering

使用例:
  # 通常実行
  python -m pitching.app.cli -i video.mp4 -o output/

  # fusion ステージまで実行して停止
  python -m pitching.app.cli -i video.mp4 -o output/ --stop-after fusion

  # tracking から再実行（前ステージはチェックポイントからロード）
  python -m pitching.app.cli -i video.mp4 -o output/ --resume-from tracking
""",
    )
    parser.add_argument("-i", "--input", required=True, type=Path, help="入力動画ファイル")
    parser.add_argument("-o", "--output-dir", required=True, type=Path, help="出力ディレクトリ")
    parser.add_argument("-c", "--config", type=Path, default=None, help="YAML 設定ファイル（省略時はデフォルト値）")
    parser.add_argument("--resume-from", type=str, default=None, metavar="STAGE",
                        help="このステージから再実行（前ステージはチェックポイントからロード）")
    parser.add_argument("--stop-after", type=str, default=None, metavar="STAGE",
                        help="このステージで停止")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="チェックポイントの保存/ロードを無効にする")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    run(
        video_path=args.input,
        output_dir=args.output_dir,
        config_path=args.config,
        resume_from=args.resume_from,
        stop_after=args.stop_after,
        enable_checkpoint=not args.no_checkpoint,
    )


if __name__ == "__main__":
    main()
