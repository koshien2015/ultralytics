"""信頼度が閾値未満のキーポイントを欠損値にする。

落とした値は座標だけを NaN にし、信頼度は残す。
「低信頼度で落とした」と「そもそも検出されなかった」を出力で区別するため。
"""

from __future__ import annotations

import numpy as np

from pitching.preprocessing.tracks import KeypointTracks


def apply_threshold(tracks: KeypointTracks, threshold: float) -> KeypointTracks:
    """信頼度が threshold 未満のフレームの座標を NaN にした新しい tracks を返す。"""
    filtered = tracks.copy()
    for name, array in filtered.xy.items():
        low = filtered.confidence[name] < threshold
        array[low] = np.nan
    return filtered


def missing_intervals(tracks: KeypointTracks, name: str) -> list[tuple[int, int]]:
    """欠損が連続する区間を [start_position, end_position] の閉区間で返す。"""
    array = tracks.xy.get(name)
    if array is None:
        return []
    missing = ~np.isfinite(array[:, 0]) | ~np.isfinite(array[:, 1])
    return _true_runs(missing)


def low_confidence_report(tracks: KeypointTracks, threshold: float) -> dict[str, list[dict]]:
    """キーポイントごとの低信頼度区間。フレーム番号は元動画の番号で返す。"""
    report: dict[str, list[dict]] = {}
    for name in tracks.names:
        runs = _true_runs(tracks.confidence[name] < threshold)
        if not runs:
            continue
        report[name] = [
            {
                "start_frame": int(tracks.frame_indices[start]),
                "end_frame": int(tracks.frame_indices[end]),
                "frames": end - start + 1,
                "mean_confidence": float(np.mean(tracks.confidence[name][start : end + 1])),
            }
            for start, end in runs
        ]
    return report


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True が連続する区間の (start, end) 閉区間リスト。"""
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
