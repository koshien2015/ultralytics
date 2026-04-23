from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrikeZone:
    frame_index: int
    left: float
    right: float
    top: float
    bottom: float
    center_x: float
    center_y: float
    zone_width: float

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


@dataclass(frozen=True)
class StrikeZoneSeries:
    zones: tuple[StrikeZone, ...]
    locked_center_x: float | None
    camera_angle_deg: float

    def at(self, frame_index: int) -> StrikeZone | None:
        """指定フレームに最も近いストライクゾーンを返す"""
        if not self.zones:
            return None
        candidates = [z for z in self.zones if z.frame_index <= frame_index]
        return candidates[-1] if candidates else None
