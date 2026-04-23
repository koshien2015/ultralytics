from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from pitching.domain.entities.bbox import BBox


class DetectionSource(Enum):
    YOLO = auto()
    FRAME_DIFF = auto()
    FUSED_YOLO_PRIMARY = auto()
    FUSED_DIFF_FALLBACK = auto()


@dataclass(frozen=True)
class YoloDetection:
    frame_index: int
    class_id: int
    class_name: str
    bbox: BBox
    confidence: float


@dataclass(frozen=True)
class DiffDetection:
    frame_index: int
    center_x: float
    center_y: float
    area: float


@dataclass(frozen=True)
class FusedDetection:
    frame_index: int
    class_id: int
    bbox: BBox | None
    center_x: float
    center_y: float
    confidence: float
    source: DetectionSource
