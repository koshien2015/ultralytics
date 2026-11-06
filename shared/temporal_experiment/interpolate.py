"""オフラインの欠損補完。どちらも未来フレームを参照するので、リアルタイムには使えない。

補完するのは同じ軌道の2つの観測に挟まれた、上限以内の欠損だけ。先頭・末尾には何も作らない。
"""

from dataclasses import replace

from trajectory_fitter import MIN_POINTS_FOR_FIT, TrajectoryPoint, position_at, ransac_fit

from .schema import DETECTED, INTERPOLATED, MISSING, Estimate


def _lerp(a, b, ratio):
    return a + (b - a) * ratio


def _lerp_box(left, right, ratio):
    if left.box is None or right.box is None:
        return None
    return tuple(_lerp(a, b, ratio) for a, b in zip(left.box, right.box))


def _anchor_pairs(estimates, track_id):
    """同じ軌道で隣り合う実検出の添字の組"""
    anchors = [
        i for i, e in enumerate(estimates) if e.status == DETECTED and e.track_id == track_id
    ]
    return zip(anchors, anchors[1:])


def _track_ids(estimates):
    return sorted({e.track_id for e in estimates if e.track_id is not None})


def _is_fillable(estimates, left, right, max_gap):
    between = estimates[left + 1 : right]
    return 0 < len(between) <= max_gap and all(e.status == MISSING for e in between)


def _linear_point(left, right, target, track_id):
    ratio = (target.t - left.t) / (right.t - left.t)
    return replace(
        target,
        status=INTERPOLATED,
        x=_lerp(left.x, right.x, ratio),
        y=_lerp(left.y, right.y, ratio),
        box=_lerp_box(left, right, ratio),
        track_id=track_id,
    )


def linear_fill(estimates, max_gap: int) -> tuple:
    """前後の有効な検出位置を時刻で線形補間する"""
    source = tuple(estimates)
    filled = {}
    for track_id in _track_ids(source):
        for left, right in _anchor_pairs(source, track_id):
            if _is_fillable(source, left, right, max_gap):
                for i in range(left + 1, right):
                    filled[i] = _linear_point(source[left], source[right], source[i], track_id)
    return tuple(filled.get(i, e) for i, e in enumerate(source))


def _to_point(estimate, origin):
    return TrajectoryPoint(estimate.frame, estimate.t - origin, estimate.x, estimate.y)


def _fit_one(estimates, track_id, max_gap, inlier_px):
    """1軌道分の差し替えを {添字: Estimate} で返す。当てはめられない軌道はそのまま残す"""
    indices = [
        i for i, e in enumerate(estimates) if e.status == DETECTED and e.track_id == track_id
    ]
    if len(indices) < MIN_POINTS_FOR_FIT:
        return {}
    origin = estimates[indices[0]].t
    points = [_to_point(estimates[i], origin) for i in indices]
    try:
        fit = ransac_fit(points, inlier_threshold=inlier_px)
    except ValueError:
        return {}

    inlier_frames = {p.frame for p in fit.inliers}
    kept = [i for i in indices if estimates[i].frame in inlier_frames]
    dropped = {
        i: Estimate(estimates[i].frame, estimates[i].t, MISSING)
        for i in indices
        if estimates[i].frame not in inlier_frames
    }
    view = tuple(dropped.get(i, e) for i, e in enumerate(estimates))
    changes = dict(dropped)
    for left, right in zip(kept, kept[1:]):
        if _is_fillable(view, left, right, max_gap):
            for i in range(left + 1, right):
                x, y = position_at(fit, view[i].t - origin)
                changes[i] = replace(view[i], status=INTERPOLATED, x=x, y=y, track_id=track_id)
    return changes


def fit_fill(estimates, max_gap: int, inlier_px: float) -> tuple:
    """既存の trajectory_fitter で軌道ごとに2次式を当て、外れた検出を除いて欠損を埋める"""
    source = tuple(estimates)
    changes = {}
    for track_id in _track_ids(source):
        changes.update(_fit_one(source, track_id, max_gap, inlier_px))
    return tuple(changes.get(i, e) for i, e in enumerate(source))
