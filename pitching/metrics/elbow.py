"""肘角度とその時間変化。

完全伸展に近い状態が約180度になる（肩-肘-手首のなす角）。
"""

from __future__ import annotations

import numpy as np

from pitching.metrics.geometry import derivative, joint_angle
from pitching.metrics.sampling import difference, duration, frame_at, value_at
from pitching.preprocessing.tracks import KeypointTracks


def elbow_angle_series(tracks: KeypointTracks, side: str) -> np.ndarray:
    """投球腕の肘角度（度）。欠損フレームは NaN。"""
    angles = np.full(tracks.length, np.nan)
    for position in range(tracks.length):
        angle = joint_angle(
            tracks.point(f"{side}_shoulder", position),
            tracks.point(f"{side}_elbow", position),
            tracks.point(f"{side}_wrist", position),
        )
        if angle is not None:
            angles[position] = angle
    return angles


def extension_velocity_series(angles: np.ndarray, fps: float) -> np.ndarray:
    """肘伸展角速度（度/秒）。正なら伸展方向。"""
    return derivative(angles, fps)


def elbow_features(
    angles: np.ndarray,
    velocity: np.ndarray,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    positions: dict[str, int | None],
) -> dict:
    """肘まわりの要約特徴量。

    Args:
        positions: イベント名 → 時系列内の添字（未確定なら None）。
    """
    flexion_position = positions.get("max_elbow_flexion")
    release_position = positions.get("release")
    contact_position = positions.get("foot_contact")

    minimum_angle = value_at(angles, flexion_position)
    release_angle = value_at(angles, release_position)

    features = {
        "minimum_elbow_angle_deg": minimum_angle,
        "minimum_elbow_angle_frame": frame_at(frame_indices, flexion_position),
        "elbow_angle_at_foot_contact_deg": value_at(angles, contact_position),
        "elbow_angle_at_release_deg": release_angle,
        "elbow_extension_range_deg": difference(release_angle, minimum_angle),
        "flexion_to_release_time_sec": duration(timestamps, flexion_position, release_position),
    }

    peak_position = _peak_velocity_position(velocity, flexion_position, release_position)
    features["peak_extension_velocity_deg_per_sec"] = value_at(velocity, peak_position)
    features["peak_extension_velocity_frame"] = frame_at(frame_indices, peak_position)
    features["extension_velocity_at_release_deg_per_sec"] = value_at(velocity, release_position)
    return features


def _peak_velocity_position(
    velocity: np.ndarray, start: int | None, end: int | None
) -> int | None:
    """最大屈曲〜リリース区間での最大伸展角速度の位置。区間が無ければ全体から探す。"""
    low = 0 if start is None else start
    high = velocity.size - 1 if end is None else end
    if low > high:
        low, high = 0, velocity.size - 1
    window = velocity[low : high + 1]
    if window.size == 0 or not np.any(np.isfinite(window)):
        return None
    return low + int(np.nanargmax(window))


