"""キーポイント名で表した骨格の辺。

shared/pose.py の SKELETON_EDGES は COCO17 の添字で書かれているが、
こちらは名前付き辞書を扱うので名前で持つ。頭部の4辺は投球フォームの
確認には不要なので省いている。
"""

from __future__ import annotations

SKELETON_EDGES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


def throwing_arm_edges(side: str) -> tuple[tuple[str, str], ...]:
    """投球側の肩→肘→手首。強調表示に使う。"""
    return ((f"{side}_shoulder", f"{side}_elbow"), (f"{side}_elbow", f"{side}_wrist"))
