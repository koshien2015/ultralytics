"""イベント検出。

方針:
- 手動指定は常に自動推定より優先する。
- Pose の動きだけでリリースを断定しない（手動 or キャップ追跡のみ）。
- 最大肘屈曲は1フレームの最小値ではなく、平滑化後系列の局所最小として求める。
- 探索区間は「足接地の少し前〜リリース」に制限できる。
"""

from __future__ import annotations

import numpy as np

from pitching.config import PitchConfig
from pitching.events.release_source import ReleaseSource
from pitching.models import EVENT_SOURCE_PRIORITY, EventName, EventSource, PitchEvent


def resolve_events(
    config: PitchConfig,
    frame_indices: np.ndarray,
    elbow_angles: np.ndarray,
    release_source: ReleaseSource | None = None,
) -> dict[EventName, PitchEvent]:
    """すべてのイベントを決定して返す。frame_index は元動画のフレーム番号。"""
    events: dict[EventName, PitchEvent] = {}

    events[EventName.PITCH_START] = PitchEvent(
        EventName.PITCH_START,
        int(frame_indices[0]) if frame_indices.size else None,
        EventSource.MANUAL,
        "解析区間の先頭",
    )
    events[EventName.PITCH_END] = PitchEvent(
        EventName.PITCH_END,
        int(frame_indices[-1]) if frame_indices.size else None,
        EventSource.MANUAL,
        "解析区間の末尾",
    )

    events[EventName.FOOT_CONTACT] = _foot_contact(config)
    events[EventName.RELEASE] = _release(config, release_source)

    contact_position = _position_of(frame_indices, events[EventName.FOOT_CONTACT].frame_index)
    release_position = _position_of(frame_indices, events[EventName.RELEASE].frame_index)

    search_start, search_end = search_window(
        length=int(elbow_angles.size),
        contact_position=contact_position,
        release_position=release_position,
        margin=config.event_detection.search_margin_frames,
    )

    flexion_position = detect_max_flexion(
        elbow_angles, search_start, search_end, config.event_detection.local_minimum_order
    )
    events[EventName.MAX_ELBOW_FLEXION] = PitchEvent(
        EventName.MAX_ELBOW_FLEXION,
        _frame_of(frame_indices, flexion_position),
        EventSource.POSE_HEURISTIC,
        f"平滑化後の肘角度の局所最小（探索区間 {search_start}〜{search_end}）",
    )

    extension_position = detect_extension_start(
        elbow_angles,
        flexion_position,
        config.event_detection.extension_min_frames,
        config.event_detection.extension_min_delta_deg,
    )
    events[EventName.EXTENSION_START] = PitchEvent(
        EventName.EXTENSION_START,
        _frame_of(frame_indices, extension_position),
        EventSource.POSE_HEURISTIC,
        f"最大屈曲後に肘角度の増加が{config.event_detection.extension_min_frames}フレーム継続した地点",
    )
    return events


def search_window(
    length: int, contact_position: int | None, release_position: int | None, margin: int
) -> tuple[int, int]:
    """最大肘屈曲の探索区間（閉区間の添字）。"""
    start = 0 if contact_position is None else max(0, contact_position - margin)
    end = length - 1 if release_position is None else min(length - 1, release_position)
    if start >= end:
        return 0, max(0, length - 1)
    return start, end


def detect_max_flexion(
    angles: np.ndarray, start: int, end: int, order: int
) -> int | None:
    """区間内で最も深い局所最小を返す。局所最小が無ければ区間内の最小値。"""
    if angles.size == 0 or start > end:
        return None

    minima = [
        position
        for position in range(start, end + 1)
        if _is_local_minimum(angles, position, order, start, end)
    ]
    if minima:
        return min(minima, key=lambda position: angles[position])

    window = angles[start : end + 1]
    if not np.any(np.isfinite(window)):
        return None
    return start + int(np.nanargmin(window))


def detect_extension_start(
    angles: np.ndarray, flexion_position: int | None, min_frames: int, min_delta_deg: float
) -> int | None:
    """最大屈曲後、肘角度の増加が min_frames 連続した最初の地点を返す。"""
    if flexion_position is None or angles.size == 0:
        return None

    streak = 0
    streak_start: int | None = None
    for position in range(flexion_position + 1, angles.size):
        previous, current = angles[position - 1], angles[position]
        if not (np.isfinite(previous) and np.isfinite(current)):
            streak = 0
            streak_start = None
            continue
        if current - previous >= min_delta_deg:
            if streak_start is None:
                streak_start = position - 1
            streak += 1
            if streak >= min_frames:
                return streak_start
        else:
            streak = 0
            streak_start = None
    return None


def _is_local_minimum(angles: np.ndarray, position: int, order: int, start: int, end: int) -> bool:
    center = angles[position]
    if not np.isfinite(center):
        return False
    neighbours_lower = 0
    for offset in range(1, order + 1):
        for neighbour in (position - offset, position + offset):
            if neighbour < start or neighbour > end:
                continue
            value = angles[neighbour]
            if not np.isfinite(value):
                continue
            if value < center:
                return False
            if value > center:
                neighbours_lower += 1
    # 平坦な区間を全部局所最小にしないため、少なくとも片側は明確に高いこと
    return neighbours_lower > 0


def _foot_contact(config: PitchConfig) -> PitchEvent:
    frame = config.events.stride_foot_contact_frame
    if frame is not None:
        return PitchEvent(EventName.FOOT_CONTACT, frame, EventSource.MANUAL, "設定ファイルで指定")
    return PitchEvent(
        EventName.FOOT_CONTACT,
        None,
        EventSource.MANUAL,
        "未指定。自動検出は行わない（誤検出しやすいため）",
    )


def _release(config: PitchConfig, release_source: ReleaseSource | None) -> PitchEvent:
    """手動指定 > キャップ追跡。Pose だけからは決めない。"""
    candidates: list[tuple[int, EventSource, str]] = []

    manual_frame = config.events.release_frame
    if manual_frame is not None:
        candidates.append((manual_frame, EventSource.MANUAL, "設定ファイルで指定"))

    if release_source is not None:
        candidate = release_source.detect()
        if candidate is not None:
            candidates.append((candidate.frame_index, candidate.source, candidate.note))

    if not candidates:
        return PitchEvent(
            EventName.RELEASE,
            None,
            EventSource.MANUAL,
            "未指定。Poseの動きだけからリリースは決めない",
        )

    frame, source, note = max(candidates, key=lambda item: EVENT_SOURCE_PRIORITY[item[1]])
    return PitchEvent(EventName.RELEASE, frame, source, note)


def _position_of(frame_indices: np.ndarray, frame: int | None) -> int | None:
    if frame is None:
        return None
    matches = np.flatnonzero(frame_indices == frame)
    return int(matches[0]) if matches.size else None


def _frame_of(frame_indices: np.ndarray, position: int | None) -> int | None:
    if position is None or not (0 <= position < frame_indices.size):
        return None
    return int(frame_indices[position])
