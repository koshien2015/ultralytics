"""身体サイズによる正規化。

斜め後方からの2D撮影なので、ピクセル距離を実世界の距離として扱わない。
同一投手・同一撮影条件での相対比較のために、身体のサイズで割るだけに留める。

基準値はフレームごとに揺れるので、投球区間全体の中央値を使う。
"""

from __future__ import annotations

import numpy as np

from pitching.metrics.geometry import distance, midpoint
from pitching.preprocessing.tracks import KeypointTracks

_MEASURES = {
    "shoulder_width": ("left_shoulder", "right_shoulder"),
    "hip_width": ("left_hip", "right_hip"),
}


def body_scale(tracks: KeypointTracks, mode: str) -> tuple[float | None, dict]:
    """正規化に使う基準長（ピクセル）と、その算出根拠を返す。

    Returns:
        (scale, report)。有効フレームが無ければ scale は None。
    """
    if mode == "torso_length":
        values = _torso_length_series(tracks)
    elif mode in _MEASURES:
        first, second = _MEASURES[mode]
        values = _pair_distance_series(tracks, first, second)
    else:
        raise ValueError(f"未知の正規化方式: {mode}")

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, {"mode": mode, "frames_used": 0, "scale_px": None}

    scale = float(np.median(finite))
    report = {
        "mode": mode,
        "frames_used": int(finite.size),
        "frames_total": int(values.size),
        "scale_px": scale,
        "scale_std_px": float(np.std(finite)),
        "note": "ピクセル基準の相対値。実世界の長さではない。",
    }
    if scale <= 0:
        return None, {**report, "scale_px": None}
    return scale, report


def normalize(value: float | None, scale: float | None) -> float | None:
    """ピクセル量を基準長で割る。"""
    if value is None or scale is None or scale <= 0:
        return None
    return value / scale


def _pair_distance_series(tracks: KeypointTracks, first: str, second: str) -> np.ndarray:
    values = np.full(tracks.length, np.nan)
    for position in range(tracks.length):
        values[position] = _or_nan(
            distance(tracks.point(first, position), tracks.point(second, position))
        )
    return values


def _torso_length_series(tracks: KeypointTracks) -> np.ndarray:
    values = np.full(tracks.length, np.nan)
    for position in range(tracks.length):
        shoulder = midpoint(
            tracks.point("left_shoulder", position), tracks.point("right_shoulder", position)
        )
        hip = midpoint(tracks.point("left_hip", position), tracks.point("right_hip", position))
        values[position] = _or_nan(distance(shoulder, hip))
    return values


def _or_nan(value: float | None) -> float:
    return np.nan if value is None else value
