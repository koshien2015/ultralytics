"""2投球の同期。

同じフレーム番号どうしを比べない。イベント基準で揃えるか、
足接地〜リリースを0〜100%に時間正規化して比べる。
元のフレーム番号と正規化後の進行率は両方保持する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitching.analysis import PitchAnalysis
from pitching.models import EventName

DEFAULT_GRID_POINTS = 101


@dataclass(frozen=True)
class NormalizedSeries:
    """時間正規化した系列。"""

    pitch_id: str
    key: str
    progress_percent: np.ndarray
    values: np.ndarray
    source_frames: np.ndarray
    covered: bool
    note: str = ""


def normalization_grid(points: int = DEFAULT_GRID_POINTS) -> np.ndarray:
    """0〜100% の等間隔グリッド。"""
    return np.linspace(0.0, 100.0, points)


def time_normalize(
    analysis: PitchAnalysis, key: str, grid: np.ndarray | None = None
) -> NormalizedSeries:
    """足接地=0%、リリース=100% として系列をグリッド上に載せ替える。"""
    target_grid = normalization_grid() if grid is None else np.asarray(grid, dtype=float)
    values = analysis.series[key]
    progress = analysis.progress_percent

    usable = np.isfinite(progress) & np.isfinite(values)
    inside = usable & (progress >= 0.0) & (progress <= 100.0)
    if np.count_nonzero(inside) < 2:
        return NormalizedSeries(
            pitch_id=analysis.pitch_id,
            key=key,
            progress_percent=target_grid,
            values=np.full(target_grid.shape, np.nan),
            source_frames=np.full(target_grid.shape, -1, dtype=int),
            covered=False,
            note="足接地とリリースが揃っていないため時間正規化できない",
        )

    source_progress = progress[inside]
    source_values = values[inside]
    order = np.argsort(source_progress)
    interpolated = np.interp(
        target_grid, source_progress[order], source_values[order], left=np.nan, right=np.nan
    )

    frames = analysis.frame_indices[inside][order]
    nearest = np.searchsorted(source_progress[order], target_grid).clip(0, frames.size - 1)
    return NormalizedSeries(
        pitch_id=analysis.pitch_id,
        key=key,
        progress_percent=target_grid,
        values=interpolated,
        source_frames=frames[nearest],
        covered=True,
    )


def event_offset(
    first: PitchAnalysis, second: PitchAnalysis, anchor: EventName
) -> int | None:
    """anchor イベントを揃えるために second をずらすフレーム数。"""
    first_position = first.position_of_event(anchor)
    second_position = second.position_of_event(anchor)
    if first_position is None or second_position is None:
        return None
    return first_position - second_position


def align_by_event(
    first: PitchAnalysis, second: PitchAnalysis, key: str, anchor: EventName
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """anchor を 0 とした相対フレームで2投球の系列を返す。

    Returns:
        (相対フレーム, first の値, second の値)。anchor が無ければ None。
    """
    first_position = first.position_of_event(anchor)
    second_position = second.position_of_event(anchor)
    if first_position is None or second_position is None:
        return None

    first_values = first.series[key]
    second_values = second.series[key]
    low = -min(first_position, second_position)
    high = min(first_values.size - first_position, second_values.size - second_position) - 1
    if high < low:
        return None

    offsets = np.arange(low, high + 1)
    return (
        offsets,
        first_values[first_position + offsets],
        second_values[second_position + offsets],
    )
