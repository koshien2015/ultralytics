from __future__ import annotations

import math
from collections import deque

from pitching.domain.entities.bbox import BBox
from pitching.domain.entities.detection import (
    DetectionSource,
    DiffDetection,
    FusedDetection,
    YoloDetection,
)

CLASS_BALL = 0


def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _direction_cos(
    recent: tuple[FusedDetection, ...],
    cx: float,
    cy: float,
) -> float:
    """直近2点からの速度ベクトルと新座標への変位のコサイン類似度を返す。"""
    if len(recent) < 2:
        return 1.0  # 実績が少ない場合は制約しない

    prev = recent[-2]
    last = recent[-1]
    vx = last.center_x - prev.center_x
    vy = last.center_y - prev.center_y
    dx = cx - last.center_x
    dy = cy - last.center_y

    v_len = math.sqrt(vx**2 + vy**2)
    d_len = math.sqrt(dx**2 + dy**2)
    if v_len < 1e-6 or d_len < 1e-6:
        return 1.0

    return (vx * dx + vy * dy) / (v_len * d_len)


def fuse_frame(
    frame_index: int,
    yolo_dets: tuple[YoloDetection, ...],
    diff_dets: tuple[DiffDetection, ...],
    recent: tuple[FusedDetection, ...],
    body_bboxes: tuple[BBox, ...],
    yolo_min_conf: float,
    diff_area_min: float,
    diff_area_max: float,
    max_jump_px: float,
    min_direction_cos: float,
) -> FusedDetection | None:
    """
    1フレーム分の YOLO 検出と差分検出を融合し FusedDetection を返す。
    どちらも条件を満たさない場合は None。
    """
    # --- YOLO 優先 ---
    ball_dets = [d for d in yolo_dets if d.class_id == CLASS_BALL and d.confidence >= yolo_min_conf]
    if ball_dets:
        best = max(ball_dets, key=lambda d: d.confidence)
        return FusedDetection(
            frame_index=frame_index,
            class_id=CLASS_BALL,
            bbox=best.bbox,
            center_x=best.bbox.center_x,
            center_y=best.bbox.center_y,
            confidence=best.confidence,
            source=DetectionSource.FUSED_YOLO_PRIMARY,
        )

    # --- 差分フォールバック ---
    last = recent[-1] if recent else None

    for diff in diff_dets:
        # 面積制約
        if not (diff_area_min <= diff.area <= diff_area_max):
            continue

        # 身体 bbox 内の差分を除外（人物の動作誤検出防止）
        if any(bbox.contains(diff.center_x, diff.center_y) for bbox in body_bboxes):
            continue

        # 直前採用座標からの距離制約
        if last is not None:
            dist = _distance(diff.center_x, diff.center_y, last.center_x, last.center_y)
            if dist > max_jump_px:
                continue

        # 軌跡方向の整合性
        cos_sim = _direction_cos(recent, diff.center_x, diff.center_y)
        if cos_sim < min_direction_cos:
            continue

        return FusedDetection(
            frame_index=frame_index,
            class_id=CLASS_BALL,
            bbox=None,
            center_x=diff.center_x,
            center_y=diff.center_y,
            confidence=0.0,
            source=DetectionSource.FUSED_DIFF_FALLBACK,
        )

    return None
