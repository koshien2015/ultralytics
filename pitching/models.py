"""解析全体で共有するデータ構造。

YOLO にも OpenCV にも依存しない。アダプタがこの形式へ変換し、
解析ロジックはこの形式だけを見る（project.md「入力部分はアダプターとして分離」）。

欠損の表現は1つに統一する: 座標が NaN なら欠損。信頼度は捨てずに残すので、
「低信頼度で落とした」のか「そもそも検出されなかった」のかを後から区別できる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

# 解析で利用する COCO 17 キーポイントの部分集合（project.md「最低限」）
REQUIRED_KEYPOINTS = (
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


@dataclass(frozen=True)
class Keypoint:
    """1点のキーポイント。x/y は画像ピクセル座標（Y軸は下向き）。"""

    x: float
    y: float
    confidence: float

    @property
    def is_valid(self) -> bool:
        """座標が有限値か。欠損は NaN で表す。"""
        return math.isfinite(self.x) and math.isfinite(self.y)

    @classmethod
    def missing(cls, confidence: float = 0.0) -> "Keypoint":
        return cls(float("nan"), float("nan"), confidence)


@dataclass(frozen=True)
class PoseFrame:
    """1フレーム分の姿勢。"""

    frame_index: int
    timestamp_sec: float
    keypoints: dict[str, Keypoint]

    def get(self, name: str) -> Keypoint | None:
        """有効なキーポイントだけを返す。欠損・未検出は None。"""
        keypoint = self.keypoints.get(name)
        if keypoint is None or not keypoint.is_valid:
            return None
        return keypoint

    def confidence_of(self, name: str) -> float:
        keypoint = self.keypoints.get(name)
        return 0.0 if keypoint is None else keypoint.confidence


class Hand(str, Enum):
    RIGHT = "right"
    LEFT = "left"


class Side(str, Enum):
    """画像上の向き。打者がフレームのどちら側にいるか。"""

    LEFT = "left"
    RIGHT = "right"


class EventSource(str, Enum):
    """イベントフレームの確度。値が上のものほど信頼できる。"""

    MANUAL = "manual"
    CAP_TRACKING = "cap_tracking"
    DETECTION_CLASS = "detection_class"
    POSE_HEURISTIC = "pose_heuristic"


# 優先順位。手動指定は常に自動推定を上書きする（project.md のテスト要件）。
EVENT_SOURCE_PRIORITY = {
    EventSource.MANUAL: 3,
    EventSource.CAP_TRACKING: 2,
    EventSource.DETECTION_CLASS: 1,
    EventSource.POSE_HEURISTIC: 0,
}


class EventName(str, Enum):
    PITCH_START = "pitch_start"
    FOOT_CONTACT = "foot_contact"
    MAX_ELBOW_FLEXION = "max_elbow_flexion"
    EXTENSION_START = "extension_start"
    RELEASE = "release"
    PITCH_END = "pitch_end"


@dataclass(frozen=True)
class PitchEvent:
    """イベント1件。frame_index が None なら「決められなかった」。"""

    name: EventName
    frame_index: int | None
    source: EventSource
    note: str = ""

    @property
    def is_resolved(self) -> bool:
        return self.frame_index is not None


@dataclass
class PoseSeries:
    """1投球分の姿勢時系列。frames は frame_index の昇順。"""

    frames: list[PoseFrame]
    fps: float
    pitch_id: str = ""
    video_path: str = ""
    adapter: str = ""
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def frame_indices(self) -> list[int]:
        return [frame.frame_index for frame in self.frames]

    def position_of(self, frame_index: int | None) -> int | None:
        """動画のフレーム番号から、この時系列内の添字を引く。"""
        if frame_index is None:
            return None
        for position, frame in enumerate(self.frames):
            if frame.frame_index == frame_index:
                return position
        return None
