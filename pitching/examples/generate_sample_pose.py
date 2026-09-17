"""実動画が無いときに CLI を試すための、合成キーポイント列を書き出すスクリプト。

    python examples/generate_sample_pose.py output/sample

出力された pose.json は `python -m pitching analyze` にそのまま渡せる。
本物の投球ではないので、数値の大小に意味は無い。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pitching.adapters.json_adapter import save_pose_json  # noqa: E402
from pitching.tests.support import make_series  # noqa: E402

# 屈曲 → 伸展 の肘角度（度）。合成データ。
GOOD_ANGLES = [
    175.0, 170.0, 160.0, 148.0, 135.0, 120.0, 108.0, 98.0, 90.0, 85.0,
    82.0, 88.0, 100.0, 118.0, 138.0, 158.0, 172.0, 176.0, 177.0, 177.0,
]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 1

    output_dir = Path(argv[1])
    series = make_series(GOOD_ANGLES, fps=60.0, start=100)
    series.pitch_id = "sample"
    path = save_pose_json(series, output_dir / "pose.json")
    print(f"合成キーポイントを書き出しました: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
