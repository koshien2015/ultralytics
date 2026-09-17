"""時系列から特定フレームの値を安全に取り出す小道具。

欠損（NaN）と範囲外を区別せず、どちらも None にする。
「値が無い」ことを None で一貫して表すため。
"""

from __future__ import annotations

import numpy as np


def value_at(values: np.ndarray, position: int | None) -> float | None:
    if position is None or not (0 <= position < values.size):
        return None
    value = values[position]
    return None if not np.isfinite(value) else float(value)


def frame_at(frame_indices: np.ndarray, position: int | None) -> int | None:
    if position is None or not (0 <= position < frame_indices.size):
        return None
    return int(frame_indices[position])


def difference(later: float | None, earlier: float | None) -> float | None:
    if later is None or earlier is None:
        return None
    return later - earlier


def duration(timestamps: np.ndarray, start: int | None, end: int | None) -> float | None:
    if start is None or end is None:
        return None
    if not (0 <= start < timestamps.size and 0 <= end < timestamps.size):
        return None
    return float(timestamps[end] - timestamps[start])
