#!/usr/bin/env python3
"""棒人間ビューアの HTML を作る。

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

import os
import sys
from pathlib import Path

# pitching パッケージの置き場所。環境ごとに変わるので順に試す。
ROOT_CANDIDATES = (
    os.environ.get("PITCHING_ROOT"),
    str(Path(__file__).resolve().parents[2]),  # リポジトリを丸ごと置いた場合
    "/",                                        # /pitching にマウントした場合
)


def ensure_package_on_path() -> None:
    """pitching を import できるようにする。できなければ理由を言って終わる。"""
    for root in ROOT_CANDIDATES:
        if root and (Path(root) / "pitching" / "__init__.py").is_file():
            if root not in sys.path:
                sys.path.insert(0, root)
            return

    try:
        import pitching  # noqa: F401  インストール済みならこれで足りる
    except ImportError:
        raise SystemExit(
            "pitching パッケージが見つかりません。\n"
            "  探した場所: " + ", ".join(str(r) for r in ROOT_CANDIDATES if r) + "\n"
            "  対処: リポジトリの pitching/ をマウントして PITCHING_ROOT にその親を指定するか、\n"
            "        ビューア作成は手元（解析環境）で実行してください。\n"
            "        キーポイントの抽出だけなら tools/extract-pose.py が単体で動きます。"
        )


def main(argv: list[str] | None = None) -> int:
    ensure_package_on_path()

    from pitching.cli import main as cli_main

    arguments = list(sys.argv[1:] if argv is None else argv)
    # -o を CLI 側の --output に合わせる
    arguments = ["--output" if item == "-o" else item for item in arguments]
    return cli_main(["viewer", *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
