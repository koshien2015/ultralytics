"""テスト用の合成キーポイント生成。

実動画を使わずに検証できるよう、人工的な座標系列を組み立てる。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# pitching パッケージのルート（ultralytics/）を import パスに足す
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pitching.models import Keypoint, PoseFrame, PoseSeries  # noqa: E402

SHOULDER = (100.0, 100.0)
ELBOW = (140.0, 120.0)
FOREARM_LENGTH = 45.0


def wrist_for_elbow_angle(angle_deg: float) -> tuple[float, float]:
    """肘角度が angle_deg になる手首の位置を作る。

    肘→肩の向きから angle_deg だけ回した方向に前腕を伸ばす。
    """
    to_shoulder = math.atan2(SHOULDER[1] - ELBOW[1], SHOULDER[0] - ELBOW[0])
    direction = to_shoulder - math.radians(angle_deg)
    return (
        ELBOW[0] + FOREARM_LENGTH * math.cos(direction),
        ELBOW[1] + FOREARM_LENGTH * math.sin(direction),
    )


def make_frame(
    frame_index: int, elbow_angle_deg: float, fps: float = 60.0, confidence: float = 0.9
) -> PoseFrame:
    """指定した肘角度になる1フレームを作る。右投げ想定。"""
    wrist = wrist_for_elbow_angle(elbow_angle_deg)
    points = {
        "nose": (95.0, 70.0),
        "right_shoulder": SHOULDER,
        "left_shoulder": (70.0, 102.0),
        "right_elbow": ELBOW,
        "left_elbow": (60.0, 130.0),
        "right_wrist": wrist,
        "left_wrist": (55.0, 155.0),
        "right_hip": (98.0, 170.0),
        "left_hip": (72.0, 172.0),
        "right_knee": (100.0, 215.0),
        "left_knee": (68.0, 218.0),
        "right_ankle": (102.0, 260.0),
        "left_ankle": (60.0, 262.0),
    }
    return PoseFrame(
        frame_index=frame_index,
        timestamp_sec=frame_index / fps,
        keypoints={name: Keypoint(x, y, confidence) for name, (x, y) in points.items()},
    )


def make_series(angles: list[float], fps: float = 60.0, start: int = 0) -> PoseSeries:
    """肘角度の列から PoseSeries を作る。"""
    return PoseSeries(
        frames=[make_frame(start + offset, angle, fps) for offset, angle in enumerate(angles)],
        fps=fps,
        pitch_id="synthetic",
        adapter="test",
    )
