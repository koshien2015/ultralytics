"""前腕・上腕の向き。

肘→手首（前腕）と肩→肘（上腕）のベクトルが、打者方向の水平線に対して
成す角を返す。+90 が真上、0 が打者方向の水平、負が下向き。

「リリース時に前腕が上を向いたままではないか」を比較するための値。
"""

from __future__ import annotations

import numpy as np

from pitching.metrics.geometry import classify_direction, direction_angle
from pitching.metrics.sampling import difference, value_at
from pitching.preprocessing.tracks import KeypointTracks


def forearm_angle_series(tracks: KeypointTracks, side: str, sign: int) -> np.ndarray:
    """肘→手首の方向角（度）。"""
    return _segment_angle_series(tracks, f"{side}_elbow", f"{side}_wrist", sign)


def upper_arm_angle_series(tracks: KeypointTracks, side: str, sign: int) -> np.ndarray:
    """肩→肘の方向角（度）。前腕角と合わせて、手首が肘を追い越す時点を見る。"""
    return _segment_angle_series(tracks, f"{side}_shoulder", f"{side}_elbow", sign)


def wrist_lead_series(tracks: KeypointTracks, side: str, sign: int) -> np.ndarray:
    """手首が肘より打者側にどれだけ出ているか（ピクセル、正なら手首が前）。

    角度どうしの大小では「追い越し」は測れない（原点が違うベクトルなので）。
    前後位置の差で見る。
    """
    lead = np.full(tracks.length, np.nan)
    for position in range(tracks.length):
        elbow = tracks.point(f"{side}_elbow", position)
        wrist = tracks.point(f"{side}_wrist", position)
        if elbow is None or wrist is None:
            continue
        lead[position] = (wrist[0] - elbow[0]) * sign
    return lead


def forearm_features(
    forearm: np.ndarray,
    upper_arm: np.ndarray,
    wrist_lead: np.ndarray,
    positions: dict[str, int | None],
    horizontal_tolerance_deg: float,
    window_frames: int = 3,
) -> dict:
    """前腕・上腕まわりの要約特徴量。"""
    release_position = positions.get("release")
    flexion_position = positions.get("max_elbow_flexion")

    at_release = value_at(forearm, release_position)
    before = value_at(forearm, _shift(release_position, -window_frames, forearm.size))
    after = value_at(forearm, _shift(release_position, window_frames, forearm.size))

    return {
        "forearm_angle_at_release_deg": at_release,
        "forearm_angle_at_max_flexion_deg": value_at(forearm, flexion_position),
        "forearm_angle_before_release_deg": before,
        "forearm_angle_after_release_deg": after,
        "forearm_angle_change_around_release_deg": difference(after, before),
        "forearm_direction_at_release": classify_direction(at_release, horizontal_tolerance_deg),
        "forearm_angle_window_frames": window_frames,
        "upper_arm_angle_at_release_deg": value_at(upper_arm, release_position),
        "upper_arm_angle_at_max_flexion_deg": value_at(upper_arm, flexion_position),
        "wrist_lead_at_release_px": value_at(wrist_lead, release_position),
        "wrist_passes_elbow_frame_offset": _crossing_offset(wrist_lead, flexion_position),
    }


def _segment_angle_series(
    tracks: KeypointTracks, origin_name: str, target_name: str, sign: int
) -> np.ndarray:
    angles = np.full(tracks.length, np.nan)
    for position in range(tracks.length):
        angle = direction_angle(
            tracks.point(origin_name, position), tracks.point(target_name, position), sign
        )
        if angle is not None:
            angles[position] = angle
    return angles


def _crossing_offset(wrist_lead: np.ndarray, start: int | None) -> int | None:
    """最大屈曲後、手首が肘より打者側に出た最初のフレーム差。

    2Dの見かけ上の前後関係なので、タイミングの比較材料としてのみ扱う。
    """
    if start is None:
        return None
    for position in range(start, wrist_lead.size):
        if np.isfinite(wrist_lead[position]) and wrist_lead[position] > 0:
            return position - start
    return None


def _shift(position: int | None, offset: int, size: int) -> int | None:
    if position is None:
        return None
    shifted = position + offset
    return shifted if 0 <= shifted < size else None
