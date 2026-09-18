"""tools/*.py から pitching パッケージを import できるようにする補助。

置き場所が環境ごとに変わる（リポジトリ直下 / コンテナの /pitching / インストール済み）
ので、順に探す。extract-pose.py はパッケージを使わないのでこれを読み込まない。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
            "        手元（解析環境）で実行してください。\n"
            "        キーポイントの抽出だけなら tools/extract-pose.py が単体で動きます。"
        )
