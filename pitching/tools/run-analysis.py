#!/usr/bin/env python3
"""キーポイントJSONから、解析・比較・ビューアまでを一息で作る。

    python tools/run-analysis.py

**引数は要らない。**下の「設定」を書き換えて実行する（track.py と同じ流儀）。
コマンドラインで指定したい場合は `python -m pitching run --help` を参照。

必要なのはキーポイントJSONだけで、動画も GPU も要らない。
JSON は track.py（ENABLE_POSE = True）か tools/extract-pose.py が書く。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _package_path import ensure_package_on_path  # noqa: E402

# ============================ 設定 ============================
# 比較する投球。1つでも2つでもよい。パスはこのファイルからの相対でも絶対でもよい。
#
#   pose:    キーポイントJSON（track.py の {動画名}_pose.json など）
#   release: リリースのフレーム番号（動画を見て指定する。Pose からは決めない）
#   contact: 踏み出し足接地のフレーム番号（不明なら None）
#   start / end: 解析する区間（None なら JSON 全体）
#   label:   見分けるための覚え書き（良否の分類ではない）
PITCHES = [
    {
        "pose": "../../shared/input/a_pose.json",
        "release": None,
        "contact": None,
        "start": None,
        "end": None,
        "label": "1球目",
    },
    {
        "pose": "../../shared/input/b_pose.json",
        "release": None,
        "contact": None,
        "start": None,
        "end": None,
        "label": "2球目",
    },
]

OUTPUT_DIR = "../../output/compare"  # 出力先
THROWING_HAND = "right"             # 投げ手: right / left
BATTER_DIRECTION = "right"          # 画像上で打者がいる側: right / left
MAKE_CHARTS = True                  # グラフを出すか
MAKE_VIEWER = True                  # 棒人間ビューアを出すか
# ==============================================================


def resolve(path: str) -> str:
    """このファイルからの相対パスも受け付ける。"""
    target = Path(path)
    return str(target if target.is_absolute() else (Path(__file__).resolve().parent / target))


def build_arguments() -> list[str]:
    """設定を `python -m pitching run` の引数に組み立てる。"""
    arguments = ["--output", resolve(OUTPUT_DIR),
                 "--hand", THROWING_HAND, "--batter", BATTER_DIRECTION]
    if not MAKE_CHARTS:
        arguments.append("--no-charts")
    if not MAKE_VIEWER:
        arguments.append("--no-viewer")

    for pitch in PITCHES:
        if not pitch.get("pose"):
            continue
        arguments += ["--pose", resolve(pitch["pose"])]
        for key, option in (
            ("release", "--release"), ("contact", "--contact"),
            ("start", "--start"), ("end", "--end"),
        ):
            if pitch.get(key) is not None:
                arguments += [option, str(pitch[key])]
        if pitch.get("label"):
            arguments += ["--label", pitch["label"]]
    return arguments


def main() -> int:
    ensure_package_on_path()
    from pitching.cli import main as cli_main

    arguments = build_arguments()
    if "--pose" not in arguments:
        raise SystemExit("PITCHES にキーポイントJSONのパスを設定してください")

    print("設定に従って実行します:")
    print(f"  python -m pitching run {' '.join(arguments)}\n")
    return cli_main(["run", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
