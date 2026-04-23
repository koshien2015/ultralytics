from __future__ import annotations

from pitching.domain.entities.detection import FusedDetection
from pitching.domain.entities.pitch import Pitch, PitchTrajectoryPoint, ReleaseEvent
from pitching.domain.entities.strike_zone import StrikeZoneSeries
from pitching.domain.services.normalizer import normalize_position


def build_pitch(
    release: ReleaseEvent,
    ball_detections: tuple[FusedDetection, ...],
    strike_zone_series: StrikeZoneSeries,
    fps: float,
) -> Pitch:
    """
    リリースイベントとボール検出列から Pitch を構築する。
    リリース前のフレームは除外し、ストライクゾーン通過時点の is_strike を判定する。
    """
    points: list[PitchTrajectoryPoint] = []
    is_strike: bool | None = None

    for det in ball_detections:
        if det.frame_index < release.release_frame:
            continue

        elapsed = (det.frame_index - release.release_frame) / fps
        zone = strike_zone_series.at(det.frame_index)
        if zone is None:
            continue

        x_norm, y_norm = normalize_position(
            det.center_x, det.center_y, zone, strike_zone_series.camera_angle_deg
        )

        points.append(PitchTrajectoryPoint(
            frame_index=det.frame_index,
            elapsed_time_sec=elapsed,
            x_norm=x_norm,
            y_norm=y_norm,
            z=elapsed,
            source=det.source,
        ))

        # ストライク判定はゾーン通過時の最初の点で行う
        if is_strike is None:
            is_strike = zone.contains(det.center_x, det.center_y)

    return Pitch(
        pitch_id=release.pitch_id,
        release=release,
        trajectory=tuple(points),
        is_strike=is_strike,
    )
