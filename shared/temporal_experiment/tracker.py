"""単一の蓋を追う追跡ループ。

候補ごとに仮の軌道を持ち、動いている軌道の中からフレームごとに1点を出力する。
誤検出に引きずられて本物を見失わないよう、確定前の軌道を並行して保持する。

静止物（壁のロゴなど）は動画全体で何度も検出されるので、直近の観測の移動幅で
「動いている点」だけを出力の対象にする。動いた点の直前 motion_window - 1 点は
遡って出力するので、最大でその分だけ未来の観測を見ている。
"""

from dataclasses import dataclass, replace
from typing import Optional

from .config import TrackerConfig
from .schema import DETECTED, MISSING, PREDICTED, Candidate, Estimate, TrackEvent, TrackingResult


@dataclass(frozen=True)
class TrackPoint:
    frame: int
    t: float
    x: float
    y: float
    moving: bool
    candidate: Optional[Candidate] = None  # None は予測点


@dataclass(frozen=True)
class Track:
    track_id: int
    state: object
    points: tuple
    hits: int = 1
    misses: int = 0
    confirmed: bool = False  # 一度でも動いた軌道
    moving: bool = False  # 直近の観測で動いている


def _is_moving(points, config) -> bool:
    observed = [p for p in points if p.candidate is not None][-config.motion_window :]
    if len(observed) < config.min_hits:
        return False
    xs = [p.x for p in observed]
    ys = [p.y for p in observed]
    return max(max(xs) - min(xs), max(ys) - min(ys)) >= config.min_track_extent_px


def _best_match(track, motion, candidates):
    """ゲートを通る最も近い候補を返す。無ければ最も近い候補と、その距離を返す"""
    scored = [(motion.score(track.state, c.cx, c.cy), c) for c in candidates]
    if not scored:
        return None, None, False
    accepted = [item for item in scored if item[0][1]]
    (distance, passed), candidate = min(accepted or scored, key=lambda item: item[0][0])
    return candidate, distance, passed


def _hit(track, motion, candidate, record, config) -> Track:
    state = motion.correct(track.state, candidate.cx, candidate.cy, record.t)
    point = TrackPoint(record.frame, record.t, candidate.cx, candidate.cy, False, candidate)
    moving = _is_moving(track.points + (point,), config)
    points = track.points + (replace(point, moving=moving),)
    return replace(
        track,
        state=state,
        points=points,
        hits=track.hits + 1,
        misses=0,
        confirmed=track.confirmed or moving,
        moving=moving,
    )


def _miss(track, motion, record, config) -> Track:
    misses = track.misses + 1
    if not track.confirmed or misses > config.max_missed_frames:
        return replace(track, misses=misses)
    x, y = motion.position(track.state)
    point = TrackPoint(record.frame, record.t, x, y, track.moving)
    return replace(track, points=track.points + (point,), misses=misses)


def _is_alive(track, config) -> bool:
    limit = config.max_missed_frames if track.confirmed else config.tentative_max_misses
    return track.misses <= limit


def _priority(track) -> tuple:
    """動いている軌道が先に候補を取る。静止物の長い履歴に本物を奪われないため"""
    return (not track.moving, not track.confirmed, -track.hits)


def _step(tracks, record, motion, config):
    """1フレーム進める。優先度の順に候補を取り、残りの候補で仮の軌道を作る"""
    available = list(record.candidates)
    updated, events = [], []
    for track in sorted(tracks, key=_priority):
        advanced = replace(track, state=motion.advance(track.state, record.t))
        candidate, distance, accepted = _best_match(advanced, motion, available)
        if accepted:
            available.remove(candidate)
            updated.append(_hit(advanced, motion, candidate, record, config))
            continue
        if candidate is not None and advanced.moving:
            events.append(TrackEvent(record.frame, "gate_rejected", track.track_id, distance))
        updated.append(_miss(advanced, motion, record, config))
    return updated, available, events


def _new_track(track_id, candidate, record, motion) -> Track:
    point = TrackPoint(record.frame, record.t, candidate.cx, candidate.cy, False, candidate)
    return Track(track_id, motion.start(candidate.cx, candidate.cy, record.t), (point,))


def _emitted_points(track, config, emits_prediction) -> tuple:
    """出力する点。観測は動いた点とその直前の点、予測は動いている間の点だけを残す"""
    observed = [p for p in track.points if p.candidate is not None]
    keep = set()
    for index, point in enumerate(observed):
        if point.moving:
            first = max(0, index - config.motion_window + 1)
            keep.update(p.frame for p in observed[first : index + 1])
    predicted = [p for p in track.points if p.candidate is None and p.moving]
    return tuple(p for p in observed if p.frame in keep) + (
        tuple(predicted) if emits_prediction else ()
    )


def _to_estimate(point, track_id) -> Estimate:
    if point.candidate is None:
        return Estimate(point.frame, point.t, PREDICTED, point.x, point.y, track_id=track_id)
    candidate = point.candidate
    return Estimate(
        point.frame, point.t, DETECTED, point.x, point.y, candidate.conf, candidate.box, track_id
    )


def _emit(records, finished, config, emits_prediction) -> tuple:
    """フレームごとに、実検出を優先し、次に動いた観測の多い軌道の点を出す。

    軌道の良さは最後まで見て決めるので、この選択はオフラインの処理になる。
    """
    chosen = {}
    for track in finished:
        quality = sum(p.moving for p in track.points if p.candidate is not None)
        for point in _emitted_points(track, config, emits_prediction):
            rank = (point.candidate is not None, quality)
            if point.frame not in chosen or rank > chosen[point.frame][0]:
                chosen[point.frame] = (rank, _to_estimate(point, track.track_id))
    return tuple(
        chosen[r.frame][1] if r.frame in chosen else Estimate(r.frame, r.t, MISSING)
        for r in records
    )


def _skip_frames(tracks, skipped, config) -> tuple:
    """入力に無いフレームも欠損として数え、上限を超えた軌道を (続行, 終了) に分ける"""
    aged = [replace(t, misses=t.misses + skipped) for t in tracks]
    alive = [t for t in aged if _is_alive(t, config)]
    return alive, [t for t in aged if not _is_alive(t, config) and t.confirmed]


def run_tracker(records, motion, config: TrackerConfig) -> TrackingResult:
    active, finished, events = [], [], []
    next_id = 0
    previous_frame = None
    for record in records:
        if previous_frame is not None and record.frame > previous_frame + 1:
            active, expired = _skip_frames(active, record.frame - previous_frame - 1, config)
            finished.extend(expired)
        previous_frame = record.frame
        updated, leftovers, step_events = _step(active, record, motion, config)
        events.extend(step_events)
        spawned = [
            _new_track(next_id + i, candidate, record, motion)
            for i, candidate in enumerate(leftovers)
        ]
        next_id += len(spawned)
        for track in updated:
            if not _is_alive(track, config) and track.confirmed:
                events.append(TrackEvent(record.frame, "track_lost", track.track_id))
        finished.extend(t for t in updated if not _is_alive(t, config) and t.confirmed)
        active = [t for t in updated if _is_alive(t, config)] + spawned
    finished.extend(t for t in active if t.confirmed)
    emitted = _emit(records, finished, config, motion.emits_prediction)
    return TrackingResult(emitted, tuple(events))
