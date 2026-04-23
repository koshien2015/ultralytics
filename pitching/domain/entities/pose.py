from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PoseFrame:
    frame_index: int
    person_id: int | None
    keypoints: tuple[Keypoint, ...]
