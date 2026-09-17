"""体幹まわり。

すべて2D画像上の見かけの角度であり、実際の3D回旋角ではない。
出力にもその旨を残す（project.md「明記してください」）。
"""

from __future__ import annotations

import numpy as np

from pitching.metrics.geometry import angle_difference, direction_angle, line_angle, midpoint
from pitching.metrics.sampling import value_at
from pitching.preprocessing.tracks import KeypointTracks

APPARENT_2D_NOTE = "2D画像上の見かけの角度。3Dの回旋角ではない。"


def torso_series(tracks: KeypointTracks, sign: int) -> dict[str, np.ndarray]:
    """体幹の各種角度系列と、肩中点・股関節中点の座標系列。"""
    length = tracks.length
    trunk_lean = np.full(length, np.nan)
    shoulder_line = np.full(length, np.nan)
    hip_line = np.full(length, np.nan)
    separation = np.full(length, np.nan)
    shoulder_mid = np.full((length, 2), np.nan)
    hip_mid = np.full((length, 2), np.nan)

    for position in range(length):
        left_shoulder = tracks.point("left_shoulder", position)
        right_shoulder = tracks.point("right_shoulder", position)
        left_hip = tracks.point("left_hip", position)
        right_hip = tracks.point("right_hip", position)

        shoulder_center = midpoint(left_shoulder, right_shoulder)
        hip_center = midpoint(left_hip, right_hip)
        if shoulder_center is not None:
            shoulder_mid[position] = shoulder_center
        if hip_center is not None:
            hip_mid[position] = hip_center

        # 股関節中点→肩中点。直立で +90 度（真上）になる。
        # line_angle で畳むと打者側への傾きと反対側への傾きが同じ値になるため、
        # 向きを保つ direction_angle を使う。
        axis = _or_nan(direction_angle(hip_center, shoulder_center, sign))
        trunk_lean[position] = _lean_from_axis(axis)
        shoulder_line[position] = _or_nan(line_angle(left_shoulder, right_shoulder, sign))
        hip_line[position] = _or_nan(line_angle(left_hip, right_hip, sign))
        separation[position] = _or_nan(
            angle_difference(
                None if np.isnan(shoulder_line[position]) else shoulder_line[position],
                None if np.isnan(hip_line[position]) else hip_line[position],
            )
        )

    return {
        "trunk_lean_deg": trunk_lean,
        "shoulder_line_deg": shoulder_line,
        "hip_line_deg": hip_line,
        "shoulder_hip_separation_deg": separation,
        "shoulder_mid_xy": shoulder_mid,
        "hip_mid_xy": hip_mid,
    }


def torso_features(series: dict[str, np.ndarray], positions: dict[str, int | None]) -> dict:
    release_position = positions.get("release")
    contact_position = positions.get("foot_contact")
    return {
        "trunk_lean_at_release_deg": value_at(series["trunk_lean_deg"], release_position),
        "trunk_lean_at_foot_contact_deg": value_at(series["trunk_lean_deg"], contact_position),
        "shoulder_line_at_release_deg": value_at(series["shoulder_line_deg"], release_position),
        "hip_line_at_release_deg": value_at(series["hip_line_deg"], release_position),
        "shoulder_hip_separation_at_release_deg": value_at(
            series["shoulder_hip_separation_deg"], release_position
        ),
        "shoulder_hip_separation_at_foot_contact_deg": value_at(
            series["shoulder_hip_separation_deg"], contact_position
        ),
        "note": APPARENT_2D_NOTE,
    }


def _lean_from_axis(axis_angle: float) -> float:
    """体幹軸の角度を「直立からの傾き」に直す。

    正なら打者方向へ、負なら打者と反対方向へ倒れている。直立で 0。
    """
    if np.isnan(axis_angle):
        return np.nan
    return 90.0 - axis_angle


def _or_nan(value: float | None) -> float:
    return np.nan if value is None else value
