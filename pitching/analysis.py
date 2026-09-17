"""1投球分の解析パイプライン。

入力は PoseSeries（どのアダプタが作ったかは問わない）と PitchConfig だけ。
動画にも YOLO にも依存しない。

流れ: 区間切り出し → 信頼度フィルタ → 短い欠損の補間 → 平滑化 →
角度系列 → イベント決定 → 要約特徴量。
生データと平滑化後データの両方を保持する。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from pitching.config import PitchConfig
from pitching.events.detector import resolve_events
from pitching.events.release_source import ReleaseSource
from pitching.metrics import elbow, forearm, lower_body, release as release_metrics, torso
from pitching.metrics.geometry import facing_sign
from pitching.metrics.normalization import body_scale
from pitching.models import EventName, PitchEvent, PoseSeries
from pitching.preprocessing import confidence_filter, interpolation, smoothing
from pitching.preprocessing.tracks import KeypointTracks, from_series, to_series


@dataclass
class PitchAnalysis:
    """解析結果一式。出力・可視化はここから読む。"""

    config: PitchConfig
    raw_series: PoseSeries
    smoothed_series: PoseSeries
    frame_indices: np.ndarray
    timestamps: np.ndarray
    series: dict[str, np.ndarray]
    events: dict[EventName, PitchEvent]
    features: dict
    quality: dict
    progress_percent: np.ndarray

    @property
    def pitch_id(self) -> str:
        return self.config.pitch_id

    def position_of_event(self, name: EventName) -> int | None:
        event = self.events.get(name)
        return None if event is None else self.position_of_event_frame(event.frame_index)

    def position_of_event_frame(self, frame: int | None) -> int | None:
        """元動画のフレーム番号から時系列内の添字を引く。"""
        return _position_of(self.frame_indices, frame)


def analyze_pitch(
    series: PoseSeries,
    config: PitchConfig,
    release_source: ReleaseSource | None = None,
) -> PitchAnalysis:
    """PoseSeries を解析して PitchAnalysis を返す。"""
    windowed = _slice_window(series, config)
    if len(windowed) == 0:
        raise ValueError(
            f"解析区間にフレームがありません（start_frame={config.start_frame}, "
            f"end_frame={config.end_frame}）"
        )

    raw_tracks = from_series(windowed)
    low_confidence = confidence_filter.low_confidence_report(
        raw_tracks, config.preprocessing.confidence_threshold
    )
    filtered = confidence_filter.apply_threshold(
        raw_tracks, config.preprocessing.confidence_threshold
    )
    interpolated, interpolation_report = interpolation.interpolate_short_gaps(
        filtered, config.preprocessing.max_gap_frames
    )
    smoothed = smoothing.smooth_tracks(
        interpolated,
        config.preprocessing.smoothing_window,
        config.preprocessing.smoothing_polyorder,
    )

    sign = facing_sign(config.batter_direction.value)
    throwing_side = config.side_prefix(throwing=True)
    lead_side = config.side_prefix(throwing=False)

    scale_px, scale_report = body_scale(smoothed, config.metrics.normalization)

    torso_map = torso.torso_series(smoothed, sign)
    series_map = _build_series(smoothed, raw_tracks, throwing_side, lead_side, sign, torso_map)

    events = resolve_events(
        config, smoothed.frame_indices, series_map["elbow_angle_deg"], release_source
    )
    positions = {
        name.value: _position_of(smoothed.frame_indices, event.frame_index)
        for name, event in events.items()
    }

    features = _build_features(
        series_map, torso_map, smoothed, positions, throwing_side, lead_side, sign, scale_px, config
    )

    quality = {
        "frames": smoothed.length,
        "confidence_threshold": config.preprocessing.confidence_threshold,
        "low_confidence_intervals": low_confidence,
        "interpolated_gaps": interpolation_report,
        "max_gap_frames": config.preprocessing.max_gap_frames,
        "missing_after_interpolation": _missing_report(interpolated),
        "body_scale": scale_report,
        "unresolved_events": [
            name.value for name, event in events.items() if not event.is_resolved
        ],
    }

    return PitchAnalysis(
        config=config,
        raw_series=windowed,
        smoothed_series=to_series(smoothed, windowed, adapter_suffix="+smoothed"),
        frame_indices=smoothed.frame_indices,
        timestamps=smoothed.timestamps,
        series=series_map,
        events=events,
        features=features,
        quality=quality,
        progress_percent=_progress_percent(
            smoothed.length,
            positions.get(EventName.FOOT_CONTACT.value),
            positions.get(EventName.RELEASE.value),
        ),
    )


def _build_series(
    smoothed: KeypointTracks,
    raw: KeypointTracks,
    throwing_side: str,
    lead_side: str,
    sign: int,
    torso_map: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    elbow_angles = elbow.elbow_angle_series(smoothed, throwing_side)
    return {
        "elbow_angle_deg": elbow_angles,
        "elbow_angle_raw_deg": elbow.elbow_angle_series(raw, throwing_side),
        "elbow_extension_velocity_deg_per_sec": elbow.extension_velocity_series(
            elbow_angles, smoothed.fps
        ),
        "forearm_angle_deg": forearm.forearm_angle_series(smoothed, throwing_side, sign),
        "upper_arm_angle_deg": forearm.upper_arm_angle_series(smoothed, throwing_side, sign),
        "wrist_lead_px": forearm.wrist_lead_series(smoothed, throwing_side, sign),
        "lead_knee_angle_deg": lower_body.lead_knee_angle_series(smoothed, lead_side),
        **{key: value for key, value in torso_map.items() if not key.endswith("_xy")},
    }


def _build_features(
    series_map: dict[str, np.ndarray],
    torso_map: dict[str, np.ndarray],
    smoothed: KeypointTracks,
    positions: dict[str, int | None],
    throwing_side: str,
    lead_side: str,
    sign: int,
    scale_px: float | None,
    config: PitchConfig,
) -> dict:
    return {
        "elbow": elbow.elbow_features(
            series_map["elbow_angle_deg"],
            series_map["elbow_extension_velocity_deg_per_sec"],
            smoothed.timestamps,
            smoothed.frame_indices,
            positions,
        ),
        "forearm": forearm.forearm_features(
            series_map["forearm_angle_deg"],
            series_map["upper_arm_angle_deg"],
            series_map["wrist_lead_px"],
            positions,
            config.metrics.horizontal_tolerance_deg,
        ),
        "torso": torso.torso_features(torso_map, positions),
        "lower_body": lower_body.lower_body_features(series_map["lead_knee_angle_deg"], positions),
        "release_position": release_metrics.release_features(
            smoothed,
            throwing_side,
            lead_side,
            positions.get(EventName.RELEASE.value),
            scale_px,
            sign,
        ),
    }


def _slice_window(series: PoseSeries, config: PitchConfig) -> PoseSeries:
    """設定の start_frame / end_frame で時系列を切り出す。"""
    end = config.end_frame
    frames = [
        frame
        for frame in series.frames
        if frame.frame_index >= config.start_frame and (end is None or frame.frame_index <= end)
    ]
    return replace(series, frames=frames, pitch_id=series.pitch_id or config.pitch_id)


def _progress_percent(length: int, start: int | None, end: int | None) -> np.ndarray:
    """足接地=0%、リリース=100% とした進行率。

    区間外は NaN にする。外挿した値（-33% や 125%）を出すと、
    進行率なのか別の量なのか読み手に判断できないため。
    区間外のフレームはフレーム番号で辿る。
    """
    progress = np.full(length, np.nan)
    if start is None or end is None or end <= start:
        return progress
    span = end - start
    for position in range(start, min(end, length - 1) + 1):
        progress[position] = (position - start) / span * 100.0
    return progress


def _missing_report(tracks: KeypointTracks) -> dict[str, int]:
    report: dict[str, int] = {}
    for name in tracks.names:
        array = tracks.xy[name]
        missing = int(np.sum(~np.isfinite(array[:, 0])))
        if missing:
            report[name] = missing
    return report


def _position_of(frame_indices: np.ndarray, frame: int | None) -> int | None:
    if frame is None:
        return None
    matches = np.flatnonzero(frame_indices == frame)
    return int(matches[0]) if matches.size else None
