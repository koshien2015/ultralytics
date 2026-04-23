from __future__ import annotations

import numpy as np

from pitching.domain.entities.bbox import BBox


def estimate_camera_angle(pitcher_bbox: BBox, catcher_bbox: BBox) -> float:
    """
    投手と捕手のBBoxから画面に対するカメラ角度を推定する。

    Returns:
        angle: 0〜90度。90度=真横（投手-捕手が画面上で縦に並ぶ）
    """
    dx = catcher_bbox.center_x - pitcher_bbox.center_x
    dy = catcher_bbox.y2 - pitcher_bbox.y2  # 足元同士

    angle_from_vertical = np.degrees(np.arctan2(abs(dx), abs(dy)))
    camera_angle = 90.0 - angle_from_vertical
    return float(np.clip(camera_angle, 0.0, 90.0))
