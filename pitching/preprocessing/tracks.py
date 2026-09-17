"""PoseFrame のリストと、キーポイントごとの配列表現を相互変換する。

前処理は「点ごとの時系列」で扱うほうが自然なので、ここで行列に開く。
フレーム数は全工程で不変。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from pitching.models import Keypoint, PoseFrame, PoseSeries


@dataclass
class KeypointTracks:
    """キーポイント名 → (N, 2) 座標配列 / (N,) 信頼度配列。欠損は NaN。"""

    frame_indices: np.ndarray
    timestamps: np.ndarray
    xy: dict[str, np.ndarray]
    confidence: dict[str, np.ndarray]
    fps: float

    @property
    def length(self) -> int:
        return int(self.frame_indices.size)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.xy.keys())

    def copy(self) -> "KeypointTracks":
        return KeypointTracks(
            frame_indices=self.frame_indices.copy(),
            timestamps=self.timestamps.copy(),
            xy={name: array.copy() for name, array in self.xy.items()},
            confidence={name: array.copy() for name, array in self.confidence.items()},
            fps=self.fps,
        )

    def point(self, name: str, position: int):
        """指定フレームの座標を (x, y) で返す。欠損なら None。"""
        array = self.xy.get(name)
        if array is None or not (0 <= position < self.length):
            return None
        x, y = array[position]
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (float(x), float(y))


def from_series(series: PoseSeries) -> KeypointTracks:
    """PoseSeries を配列表現に開く。"""
    names: list[str] = []
    for frame in series.frames:
        for name in frame.keypoints:
            if name not in names:
                names.append(name)

    length = len(series.frames)
    xy = {name: np.full((length, 2), np.nan) for name in names}
    confidence = {name: np.zeros(length) for name in names}

    for position, frame in enumerate(series.frames):
        for name in names:
            keypoint = frame.keypoints.get(name)
            if keypoint is None:
                continue
            confidence[name][position] = keypoint.confidence
            if keypoint.is_valid:
                xy[name][position] = (keypoint.x, keypoint.y)

    return KeypointTracks(
        frame_indices=np.array([frame.frame_index for frame in series.frames], dtype=int),
        timestamps=np.array([frame.timestamp_sec for frame in series.frames], dtype=float),
        xy=xy,
        confidence=confidence,
        fps=series.fps,
    )


def to_series(tracks: KeypointTracks, template: PoseSeries, adapter_suffix: str = "") -> PoseSeries:
    """配列表現を PoseSeries に戻す。メタ情報は template から引き継ぐ。"""
    frames = []
    for position in range(tracks.length):
        keypoints = {}
        for name in tracks.names:
            x, y = tracks.xy[name][position]
            keypoints[name] = Keypoint(float(x), float(y), float(tracks.confidence[name][position]))
        frames.append(
            PoseFrame(
                frame_index=int(tracks.frame_indices[position]),
                timestamp_sec=float(tracks.timestamps[position]),
                keypoints=keypoints,
            )
        )
    return replace(
        template,
        frames=frames,
        adapter=f"{template.adapter}{adapter_suffix}",
    )
