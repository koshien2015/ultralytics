#!/usr/bin/env python3
"""棒人間ビューアの HTML を作る。

    python tools/make-viewer.py                  # 下の「設定」に従う
    python tools/make-viewer.py output/good [output/bad] [-o viewer.html]
    python tools/make-viewer.py clip_pose.json --release 152 --contact 140

渡せるもの（混ぜてよい）:
- analyze の出力ディレクトリ（metrics.json がある場所）
- metrics.json そのもの
- キーポイントJSON（track.py が書く {動画名}_pose.json、extract の pose.json）

キーポイントJSONだけを渡すときは、イベントをここで指定する。
複数の投球に指定するときは、投球を並べた順に対応する（1つだけなら全部に効く）。

`python -m pitching viewer` と中身は同じ。解析パッケージ（pitching/）が要るので、
推論用コンテナで動かすならリポジトリの pitching/ をマウントするか、
PITCHING_ROOT にその親ディレクトリを指定すること。
キーポイントの抽出だけなら tools/extract-pose.py が単体で動く。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _package_path import ensure_package_on_path  # noqa: E402

# ============================ 設定 ============================
# 引数なしで実行したときに使う。ここを書き換えれば毎回コマンドを打たずに済む。
# パスはこのファイルからの相対でも絶対でもよい。
PITCHES = [
    # {"pose": "../../shared/input/a_pose.json", "release": None, "contact": None},
    # {"pose": "../../shared/input/b_pose.json", "release": None, "contact": None},
]
OUTPUT = "../../output/viewer.html"
THROWING_HAND = "right"     # right / left
BATTER_DIRECTION = "right"  # 画像上で打者がいる側
# ==============================================================


def resolve(path: str) -> str:
    """このファイルからの相対パスも受け付ける。"""
    target = Path(path)
    return str(target if target.is_absolute() else (Path(__file__).resolve().parent / target))


def arguments_from_settings() -> list[str]:
    """先頭の設定を CLI の引数に組み立てる。"""
    if not PITCHES:
        raise SystemExit(
            "投球が指定されていません。"
            "引数で渡すか、このファイル先頭の PITCHES を設定してください"
        )

    targets = [resolve(pitch["pose"]) for pitch in PITCHES if pitch.get("pose")]
    arguments = [*targets, "--output", resolve(OUTPUT),
                 "--hand", THROWING_HAND, "--batter", BATTER_DIRECTION]
    for key, option in (("release", "--release"), ("contact", "--contact"),
                        ("start", "--start"), ("end", "--end")):
        values = [pitch.get(key) for pitch in PITCHES]
        if any(value is not None for value in values):
            for value in values:
                arguments += [option, str(value if value is not None else 0)]
    return arguments


def main(argv: list[str] | None = None) -> int:
    ensure_package_on_path()

    from pitching.cli import main as cli_main

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        arguments = arguments_from_settings()
    # -o を CLI 側の --output に合わせる
    arguments = ["--output" if item == "-o" else item for item in arguments]
    return cli_main(["viewer", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
