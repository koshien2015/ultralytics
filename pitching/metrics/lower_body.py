"""前脚（踏み出し脚）の膝角度。

踏み出し脚は投球腕の反対側とする（右投げなら左脚）。
"""

from __future__ import annotations

import numpy as np

from pitching.metrics.geometry import joint_angle
from pitching.metrics.sampling import difference, value_at
from pitching.preprocessing.tracks import KeypointTracks


def lead_knee_angle_series(tracks: KeypointTracks, lead_side: str) -> np.ndarray:
    """股関節-膝-足首のなす角（度）。伸びているほど180に近い。"""
    angles = np.full(tracks.length, np.nan)
    for position in range(tracks.length):
        angle = joint_angle(
            tracks.point(f"{lead_side}_hip", position),
            tracks.point(f"{lead_side}_knee", position),
            tracks.point(f"{lead_side}_ankle", position),
        )
        if angle is not None:
            angles[position] = angle
    return angles


def lower_body_features(angles: np.ndarray, positions: dict[str, int | None]) -> dict:
    contact_position = positions.get("foot_contact")
    release_position = positions.get("release")
    at_contact = value_at(angles, contact_position)
    at_release = value_at(angles, release_position)
    return {
        "lead_knee_angle_at_foot_contact_deg": at_contact,
        "lead_knee_angle_at_release_deg": at_release,
        "lead_knee_angle_change_deg": difference(at_release, at_contact),
    }
