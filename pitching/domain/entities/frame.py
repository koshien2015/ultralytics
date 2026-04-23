from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameMeta:
    frame_index: int
    timestamp: float
    width: int
    height: int
