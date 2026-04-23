from __future__ import annotations

import numpy as np

from pitching.domain.entities.strike_zone import StrikeZone


def normalize_position(
    x_pixel: float,
    y_pixel: float,
    strike_zone: StrikeZone,
    camera_angle_deg: float,
) -> tuple[float, float]:
    """
    ピクセル座標をストライクゾーン基準の正規化座標に変換する。

    Returns:
        (x_norm, y_norm)
        x_norm: ストライクゾーン中心からの距離をゾーン幅で正規化（0=中心, ±0.5=端）
        y_norm: 0=下端, 1=上端
    """
    cos_val = np.cos(np.radians(90.0 - camera_angle_deg))
    # cos が 0 に近い場合（真正面）は補正しない
    cos_safe = cos_val if abs(cos_val) > 1e-6 else 1.0

    x_offset = (x_pixel - strike_zone.center_x) / cos_safe
    x_norm = x_offset / strike_zone.zone_width

    strike_height = strike_zone.bottom - strike_zone.top
    y_norm = (strike_zone.bottom - y_pixel) / strike_height if strike_height > 0 else 0.0

    return float(x_norm), float(y_norm)
