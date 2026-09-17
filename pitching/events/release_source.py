"""リリースフレームの供給元。

Pose だけでリリースを断定しないため、リリースは外部から与えるのが基本。
キャップ検出・追跡データが得られたときに差し込めるよう、
「手首とキャップが分離した最初のフレーム」を返すインターフェースを用意する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from pitching.models import EventSource


@dataclass(frozen=True)
class ReleaseCandidate:
    frame_index: int
    source: EventSource
    note: str = ""


class ReleaseSource(Protocol):
    """リリースフレームを1つ返す供給元。決められなければ None。"""

    def detect(self) -> ReleaseCandidate | None: ...


@dataclass(frozen=True)
class CapTrack:
    """キャップ検出結果1件。座標は手首と同じ画像ピクセル座標系。"""

    frame_index: int
    x: float
    y: float


@dataclass
class CapSeparationRelease:
    """手首とキャップの距離が閾値を超えた最初のフレームをリリースとみなす。

    閾値は身体サイズで正規化した比率で与える（ピクセルは解像度依存のため）。
    """

    cap_tracks: Sequence[CapTrack]
    wrist_positions: dict[int, tuple[float, float]]
    body_scale_px: float | None
    separation_ratio: float = 0.25
    min_consecutive_frames: int = 2

    def detect(self) -> ReleaseCandidate | None:
        if self.body_scale_px is None or self.body_scale_px <= 0:
            return None
        threshold = self.body_scale_px * self.separation_ratio

        streak = 0
        first_frame: int | None = None
        for track in sorted(self.cap_tracks, key=lambda item: item.frame_index):
            wrist = self.wrist_positions.get(track.frame_index)
            if wrist is None:
                streak = 0
                first_frame = None
                continue
            separated = math.hypot(track.x - wrist[0], track.y - wrist[1]) > threshold
            if not separated:
                streak = 0
                first_frame = None
                continue
            streak += 1
            if first_frame is None:
                first_frame = track.frame_index
            if streak >= self.min_consecutive_frames:
                return ReleaseCandidate(
                    frame_index=first_frame,
                    source=EventSource.CAP_TRACKING,
                    note=f"手首から{threshold:.1f}px以上離れた最初のフレーム",
                )
        return None
