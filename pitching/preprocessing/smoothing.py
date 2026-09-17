"""Savitzky–Golay フィルタによる平滑化。

対称なフィルタなので位相遅れがなく、イベントのタイミングをずらさない
（project.md「平滑化によってイベントタイミングが大きくずれないように」）。

欠損をまたいで平滑化すると存在しない動きを作るため、
有効値が連続する区間ごとに掛ける。区間が窓より短ければ平滑化しない。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from pitching.preprocessing.tracks import KeypointTracks


def normalize_window(window_length: int, segment_length: int, polyorder: int) -> int | None:
    """区間長に収まる奇数の窓長を返す。平滑化できないなら None。"""
    window = min(window_length, segment_length)
    if window % 2 == 0:
        window -= 1
    if window <= polyorder or window < 3:
        return None
    return window


def smooth_values(values: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """1次元系列を平滑化する。NaN 区間は NaN のまま、長さは不変。"""
    array = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(array)
    for start, end in _valid_runs(valid):
        segment = array[start : end + 1]
        window = normalize_window(window_length, segment.size, polyorder)
        if window is None:
            continue
        array[start : end + 1] = savgol_filter(segment, window, polyorder)
    return array


def smooth_tracks(tracks: KeypointTracks, window_length: int, polyorder: int) -> KeypointTracks:
    """全キーポイントの x / y を平滑化した新しい tracks を返す。"""
    smoothed = tracks.copy()
    for name, array in smoothed.xy.items():
        array[:, 0] = smooth_values(array[:, 0], window_length, polyorder)
        array[:, 1] = smooth_values(array[:, 1], window_length, polyorder)
    return smoothed


def _valid_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for position, flag in enumerate(mask):
        if flag and start is None:
            start = position
        elif not flag and start is not None:
            runs.append((start, position - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs
