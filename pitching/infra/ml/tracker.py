from __future__ import annotations

import logging
from typing import Dict, List, Protocol, Tuple

import numpy as np

from pitching.domain.entities.detection import DetectionSource, FusedDetection
from pitching.domain.entities.track import Track, TrackPoint

logger = logging.getLogger(__name__)


class MultiObjectTracker(Protocol):
    """マルチオブジェクトトラッキングの interface。"""

    def update(
        self, detections: Tuple[FusedDetection, ...], frame_index: int
    ) -> Tuple[Track, ...]:
        ...


class NearestNeighborTracker:
    """
    最近傍マッチングによる簡易トラッカー。
    track.py の軌跡管理ロジックをクラス化したもの。
    """

    def __init__(
        self,
        max_trajectory_length: int = 60,
        fade_frames: int = 60,
        max_match_distance_px: float = 100.0,
    ) -> None:
        self._max_length = max_trajectory_length
        self._fade_frames = fade_frames
        self._max_dist = max_match_distance_px
        self._tracks: Dict[int, Dict] = {}  # {track_id: {points, last_seen}}
        self._next_id = 0

    def update(
        self, detections: Tuple[FusedDetection, ...], frame_index: int
    ) -> Tuple[Track, ...]:
        used: set[int] = set()

        # 既存トラックに最近傍マッチング
        for track_id, state in self._tracks.items():
            points: List[TrackPoint] = state["points"]
            if not points or not detections:
                continue

            last = points[-1]
            best_idx, best_dist = None, float("inf")
            for i, det in enumerate(detections):
                if i in used:
                    continue
                dist = np.sqrt((last.x - det.center_x) ** 2 + (last.y - det.center_y) ** 2)
                if dist < best_dist and dist < self._max_dist:
                    best_dist, best_idx = dist, i

            if best_idx is not None:
                det = detections[best_idx]
                points.append(TrackPoint(
                    frame_index=frame_index,
                    x=det.center_x,
                    y=det.center_y,
                    source=det.source,
                ))
                if len(points) > self._max_length:
                    points.pop(0)
                state["last_seen"] = frame_index
                used.add(best_idx)

        # 新規検出を新しいトラックとして登録
        for i, det in enumerate(detections):
            if i not in used:
                self._tracks[self._next_id] = {
                    "points": [TrackPoint(
                        frame_index=frame_index,
                        x=det.center_x,
                        y=det.center_y,
                        source=det.source,
                    )],
                    "last_seen": frame_index,
                    "class_id": det.class_id,
                }
                self._next_id += 1

        # フェードアウト期限切れトラックを削除
        expired = [
            tid for tid, s in self._tracks.items()
            if frame_index - s["last_seen"] > self._fade_frames
        ]
        for tid in expired:
            del self._tracks[tid]

        # 現在の全トラックを frozen dataclass として返す
        return tuple(
            Track(
                track_id=tid,
                class_id=state.get("class_id", 0),
                points=tuple(state["points"]),
                first_frame=state["points"][0].frame_index,
                last_frame=state["points"][-1].frame_index,
            )
            for tid, state in self._tracks.items()
            if state["points"]
        )
