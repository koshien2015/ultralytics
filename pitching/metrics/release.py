"""リリース位置。

ピクセル座標だけでなく、身体サイズで正規化した相対位置も出す。
斜め後方からの2D映像なので、絶対位置の比較には意味が薄い。

キャップ追跡データがあれば、手首ではなくキャップの分離位置も記録する。
"""

from __future__ import annotations

import numpy as np

from pitching.metrics.normalization import normalize
from pitching.preprocessing.tracks import KeypointTracks


def release_features(
    tracks: KeypointTracks,
    throwing_side: str,
    lead_side: str,
    release_position: int | None,
    scale_px: float | None,
    sign: int,
    cap_position_xy: tuple[float, float] | None = None,
) -> dict:
    """リリースフレームにおける投球側手首の位置。"""
    if release_position is None:
        return {"available": False, "reason": "リリースフレームが未確定"}

    wrist = tracks.point(f"{throwing_side}_wrist", release_position)
    if wrist is None:
        return {"available": False, "reason": "リリースフレームの手首が欠損"}

    shoulder_mid = _midpoint(
        tracks.point("left_shoulder", release_position),
        tracks.point("right_shoulder", release_position),
    )
    ankle = tracks.point(f"{lead_side}_ankle", release_position)

    features = {
        "available": True,
        "wrist_x_px": wrist[0],
        "wrist_y_px": wrist[1],
        # 画像座標のままでは左右・上下の意味が読めないので、
        # 打者方向を正、上方向を正に揃えた相対値も出す。
        **_relative(wrist, shoulder_mid, scale_px, sign, "relative_to_shoulder_mid"),
        **_relative(wrist, ankle, scale_px, sign, "relative_to_lead_ankle"),
    }

    if cap_position_xy is not None:
        features["cap_separation_x_px"] = float(cap_position_xy[0])
        features["cap_separation_y_px"] = float(cap_position_xy[1])
        features.update(
            _relative(cap_position_xy, shoulder_mid, scale_px, sign, "cap_relative_to_shoulder_mid")
        )
    return features


def _relative(point, origin, scale_px: float | None, sign: int, prefix: str) -> dict:
    if origin is None:
        return {f"{prefix}_x": None, f"{prefix}_y": None}
    forward_px = (point[0] - origin[0]) * sign
    upward_px = -(point[1] - origin[1])
    return {
        f"{prefix}_x_px": float(forward_px),
        f"{prefix}_y_px": float(upward_px),
        f"{prefix}_x": normalize(float(forward_px), scale_px),
        f"{prefix}_y": normalize(float(upward_px), scale_px),
    }


def _midpoint(first, second):
    if first is None or second is None:
        return None
    return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
