from __future__ import annotations

from dataclasses import dataclass

from pitching.domain.entities.detection import DetectionSource


@dataclass(frozen=True)
class TrackPoint:
    frame_index: int
    x: float
    y: float
    source: DetectionSource


@dataclass(frozen=True)
class Track:
    track_id: int
    class_id: int
    points: tuple[TrackPoint, ...]
    first_frame: int
    last_frame: int
