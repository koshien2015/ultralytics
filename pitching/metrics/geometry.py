"""角度・距離の基礎計算。

画像座標は Y軸が下向きなので、「上向き」を扱う関数は必ずここで符号を反転する
（project.md「画像座標ではY軸が下向きである点を、角度計算時に明示的に処理」）。
呼び出し側で反転しないこと。

打者方向は `facing_sign` で渡す。画像上で打者が右にいるなら +1、左なら -1。
これにより右投げ・左投げ・カメラの左右反転に関係なく、
「打者方向に対して上向き何度」という同じ意味の値になる。
"""

from __future__ import annotations

import math

import numpy as np

Point = tuple[float, float]

# これ未満のベクトル長は方向が定まらないとみなす（ピクセル）
_MIN_VECTOR_NORM = 1e-6


def as_point(keypoint) -> Point | None:
    """Keypoint / (x, y) / None を Point | None に正規化する。"""
    if keypoint is None:
        return None
    if isinstance(keypoint, tuple):
        x, y = keypoint
    else:
        x, y = keypoint.x, keypoint.y
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (float(x), float(y))


def facing_sign(batter_direction: str) -> int:
    """打者が画像の右にいるなら +1、左なら -1。"""
    normalized = str(batter_direction).lower()
    if normalized == "right":
        return 1
    if normalized == "left":
        return -1
    raise ValueError(f"batter_direction は left / right のいずれか: {batter_direction!r}")


def distance(a, b) -> float | None:
    """2点間のピクセル距離。どちらかが欠損なら None。"""
    point_a, point_b = as_point(a), as_point(b)
    if point_a is None or point_b is None:
        return None
    return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])


def midpoint(a, b) -> Point | None:
    point_a, point_b = as_point(a), as_point(b)
    if point_a is None or point_b is None:
        return None
    return ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)


def joint_angle(a, b, c) -> float | None:
    """点bを頂点とする、ベクトルb→aとb→cのなす角を 0〜180度で返す。

    なす角なので座標系の向き（Y軸の上下）にも左右反転にも影響されない。
    欠損点やゼロ長ベクトルがあれば None。
    """
    point_a, point_b, point_c = as_point(a), as_point(b), as_point(c)
    if point_a is None or point_b is None or point_c is None:
        return None

    vector_a = (point_a[0] - point_b[0], point_a[1] - point_b[1])
    vector_c = (point_c[0] - point_b[0], point_c[1] - point_b[1])
    norm_a = math.hypot(*vector_a)
    norm_c = math.hypot(*vector_c)
    if norm_a < _MIN_VECTOR_NORM or norm_c < _MIN_VECTOR_NORM:
        return None

    cosine = (vector_a[0] * vector_c[0] + vector_a[1] * vector_c[1]) / (norm_a * norm_c)
    # 丸め誤差で |cos| が 1 をわずかに超えることがある
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def direction_angle(origin, target, sign: int) -> float | None:
    """origin→target のベクトルが、打者方向の水平線に対して成す角（度）。

    +90 は真上、0 は打者方向の水平、±180 は打者と反対方向の水平、
    負値は下向き。戻り値は (-180, 180]。
    """
    point_origin, point_target = as_point(origin), as_point(target)
    if point_origin is None or point_target is None:
        return None

    forward = (point_target[0] - point_origin[0]) * sign
    # 画像のYは下向きなので、上向きを正にするために反転する
    upward = -(point_target[1] - point_origin[1])
    if math.hypot(forward, upward) < _MIN_VECTOR_NORM:
        return None
    return math.degrees(math.atan2(upward, forward))


def classify_direction(angle_deg: float | None, horizontal_tolerance: float = 15.0) -> str | None:
    """方向角を 上向き / 水平 / 下向き に分類する。"""
    if angle_deg is None:
        return None
    if angle_deg > horizontal_tolerance:
        return "upward"
    if angle_deg < -horizontal_tolerance:
        return "downward"
    return "horizontal"


def line_angle(a, b, sign: int) -> float | None:
    """2点を結ぶ「線」の水平からの傾き（度、-90〜90）。

    線には向きがないので、a と b を入れ替えても同じ値になるよう畳む。
    正なら打者方向に向かって上がっている。
    """
    angle = direction_angle(a, b, sign)
    if angle is None:
        return None
    if angle > 90.0:
        angle -= 180.0
    elif angle <= -90.0:
        angle += 180.0
    return angle


def angle_difference(first: float | None, second: float | None) -> float | None:
    """2つの線の角度差（-90〜90 に畳む）。"""
    if first is None or second is None:
        return None
    difference = first - second
    while difference > 90.0:
        difference -= 180.0
    while difference <= -90.0:
        difference += 180.0
    return difference


def derivative(values: np.ndarray, fps: float) -> np.ndarray:
    """時系列の時間微分（度/秒）。端は片側差分、NaN は伝播させる。"""
    array = np.asarray(values, dtype=float)
    if array.size < 2 or fps <= 0:
        return np.full(array.shape, np.nan)
    return np.gradient(array, 1.0 / float(fps))
