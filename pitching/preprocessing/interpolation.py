"""短い欠損だけを線形補間する。

長い欠損を埋めると、実際には観測していない動きを作り出してしまうので埋めない。
時系列の先頭・末尾の欠損は外挿になるため、長さに関わらず埋めない。
"""

from __future__ import annotations

import numpy as np

from pitching.preprocessing.confidence_filter import _true_runs
from pitching.preprocessing.tracks import KeypointTracks


def interpolate_short_gaps(
    tracks: KeypointTracks, max_gap_frames: int
) -> tuple[KeypointTracks, dict[str, list[dict]]]:
    """max_gap_frames 以下の欠損区間だけを線形補間する。

    Returns:
        (補間後の tracks, キーポイント名 → 補間した区間のリスト)
    """
    if max_gap_frames < 0:
        raise ValueError(f"max_gap_frames は 0 以上: {max_gap_frames}")

    filled = tracks.copy()
    report: dict[str, list[dict]] = {}

    for name, array in filled.xy.items():
        missing = ~np.isfinite(array[:, 0]) | ~np.isfinite(array[:, 1])
        for start, end in _true_runs(missing):
            length = end - start + 1
            if length > max_gap_frames:
                continue
            if start == 0 or end == filled.length - 1:
                # 外挿はしない
                continue
            _linear_fill(array, start, end)
            report.setdefault(name, []).append(
                {
                    "start_frame": int(filled.frame_indices[start]),
                    "end_frame": int(filled.frame_indices[end]),
                    "frames": length,
                }
            )

    return filled, report


def _linear_fill(array: np.ndarray, start: int, end: int) -> None:
    """array[start:end+1] を前後の有効値で線形補間する（in-place、copy 済み配列に対して）。"""
    before = array[start - 1]
    after = array[end + 1]
    steps = end - start + 2
    for offset in range(1, steps):
        ratio = offset / steps
        array[start - 1 + offset] = before + (after - before) * ratio
