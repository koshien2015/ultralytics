from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def draw_neon_polyline(
    frame: np.ndarray,
    points: List[Tuple[int, int]],
    glow_color: Tuple[int, int, int] = (0, 255, 255),
    core_color: Tuple[int, int, int] = (255, 255, 255),
    glow_thickness: int = 15,
    core_thickness: int = 3,
    glow_blur: int = 25,
    glow_intensity: float = 0.8,
) -> np.ndarray:
    """ネオン風の軌跡をフレームに描画する（track.py の draw_neon_polyline を純粋関数化）。"""
    if len(points) < 2:
        return frame

    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    h, w = frame.shape[:2]

    glow_layer = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.polylines(glow_layer, [pts], isClosed=False, color=glow_color,
                  thickness=glow_thickness, lineType=cv2.LINE_AA)

    # ぼかしてグロウを作る
    if glow_blur % 2 == 0:
        glow_blur += 1  # GaussianBlur は奇数カーネルが必要
    glow_layer = cv2.GaussianBlur(glow_layer, (glow_blur, glow_blur), 0)

    result = cv2.addWeighted(frame, 1.0, glow_layer, glow_intensity, 0)
    cv2.polylines(result, [pts], isClosed=False, color=core_color,
                  thickness=core_thickness, lineType=cv2.LINE_AA)
    return result
