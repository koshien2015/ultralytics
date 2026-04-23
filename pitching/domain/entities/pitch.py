from __future__ import annotations

from dataclasses import dataclass

from pitching.domain.entities.detection import DetectionSource


@dataclass(frozen=True)
class ReleaseEvent:
    pitch_id: int
    release_frame: int
    release_time_sec: float


@dataclass(frozen=True)
class PitchTrajectoryPoint:
    frame_index: int
    elapsed_time_sec: float
    x_norm: float
    y_norm: float
    z: float
    source: DetectionSource


@dataclass(frozen=True)
class Pitch:
    pitch_id: int
    release: ReleaseEvent
    trajectory: tuple[PitchTrajectoryPoint, ...]
    is_strike: bool | None
