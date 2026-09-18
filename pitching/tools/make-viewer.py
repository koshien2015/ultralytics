#!/usr/bin/env python3
"""棒人間ビューアの HTML を作る。

    python tools/make-viewer.py output/good [output/bad] [-o output/viewer.html]

numpy などが要るので、依存を入れた python で実行すること
（このリポジトリなら pitching/.venv/bin/python）。

analyze が書いたディレクトリ（metrics.json とキーポイントJSONがある場所）を
1つか2つ渡す。動画も再エンコードもせず、開いた瞬間から見られる HTML を1枚作る。

できること:
- 2投球の棒人間を、並べて / 重ねて 比較する
- そろえ方を「進行率0〜100%」「リリース基準」「フレーム番号」から選ぶ
- 連続フレームの差から、速度ベクトルと加速度ベクトル（力の向き）を重ねる

file:// で開くのでサーバは要らない。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pitching.reporting.loader import load_analysis, resolve_metrics_path  # noqa: E402
from pitching.visualization.viewer import ViewerError, write_viewer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="投球フォームの棒人間ビューアを作る")
    parser.add_argument("pitch", nargs="+", help="analyze の出力ディレクトリ（最大2つ）")
    parser.add_argument("-o", "--output", help="出力HTML（既定は1つ目の隣の viewer.html）")
    parser.add_argument("--title", help="見出し")
    args = parser.parse_args(argv)

    if len(args.pitch) > 2:
        print("エラー: 比較できるのは2投球までです", file=sys.stderr)
        return 1

    analyses = []
    for target in args.pitch:
        metrics_path = resolve_metrics_path(target)
        analysis = load_analysis(metrics_path)
        if analysis is None:
            print(
                f"エラー: キーポイントJSON（pose_raw.json / pose.json）が見つかりません: "
                f"{metrics_path.parent}",
                file=sys.stderr,
            )
            return 1
        analyses.append(analysis)

    output = Path(args.output) if args.output else resolve_metrics_path(args.pitch[0]).parent / "viewer.html"
    try:
        path = write_viewer(analyses, output, args.title)
    except (ViewerError, ValueError, FileNotFoundError) as error:
        print(f"エラー: {error}", file=sys.stderr)
        return 1

    print(f"ビューアを書き出しました: {path}")
    print("  ブラウザで開いてください（サーバ不要）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
