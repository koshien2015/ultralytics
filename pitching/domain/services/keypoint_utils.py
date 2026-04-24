from __future__ import annotations

import math
from typing import Optional, Tuple

from pitching.domain.entities.bbox import BBox
from pitching.domain.entities.pose import Keypoint, PoseFrame


def find_keypoint(frame: PoseFrame, name: str, min_conf: float = 0.3) -> Optional[Keypoint]:
    for kp in frame.keypoints:
        if kp.name == name and kp.confidence >= min_conf:
            return kp
    return None


def angle_between(a: Keypoint, b: Keypoint, c: Keypoint) -> Optional[float]:
    """b を頂点とした a-b-c のなす角 (deg) を返す。"""
    bax, bay = a.x - b.x, a.y - b.y
    bcx, bcy = c.x - b.x, c.y - b.y
    dot = bax * bcx + bay * bcy
    mag_ba = math.hypot(bax, bay)
    mag_bc = math.hypot(bcx, bcy)
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return None
    cos_val = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_val))


def horizontal_angle(a: Keypoint, b: Keypoint) -> float:
    """a → b ベクトルの水平基準角 (deg)。右向きが 0、上向きが -90。"""
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def distance_px(a: Keypoint, b: Keypoint) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def bbox_from_keypoints(frame: PoseFrame, min_conf: float = 0.3) -> Optional[BBox]:
    """信頼度を満たすキーポイントから外接 BBox を返す。"""
    xs = [kp.x for kp in frame.keypoints if kp.confidence >= min_conf]
    ys = [kp.y for kp in frame.keypoints if kp.confidence >= min_conf]
    if not xs:
        return None
    return BBox(min(xs), min(ys), max(xs), max(ys))


def iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0
