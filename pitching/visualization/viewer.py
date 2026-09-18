"""棒人間ビューア用のデータを組み立てる。

HTML の雛形は viewer_template.py にある。ここは数値の用意だけを担当する。

表示のための正規化:
- 原点は基準イベント（足接地、無ければ先頭）の股関節中点
- 単位は身体サイズ（既定は両肩間距離の中央値）
- X は打者方向を正、Y は上方向を正（画像座標のYは下向きなので反転する）

これで、撮影距離も左右の向きも違う2投球を同じ土俵に載せられる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pitching.analysis import PitchAnalysis
from pitching.metrics.geometry import facing_sign, midpoint
from pitching.models import EventName
from pitching.visualization.skeleton import SKELETON_EDGES

# 数値パネルに出す系列
PANEL_SERIES = {
    "elbow_angle_deg": "肘角度",
    "forearm_angle_deg": "前腕角度",
    "trunk_lean_deg": "体幹傾き",
    "lead_knee_angle_deg": "前脚膝角度",
    "elbow_extension_velocity_deg_per_sec": "肘伸展角速度",
}

# 進行率でそろえるときの分割数（0〜100% を 1% 刻み）
NORMALIZED_SAMPLES = 101


class ViewerError(RuntimeError):
    """ビューアを作れないときに送出する。"""


@dataclass(frozen=True)
class DisplayFrame:
    """表示用に正規化した1フレーム。"""

    frame_index: int
    timestamp_sec: float
    progress_percent: float | None
    points: dict[str, tuple[float, float]]


def build_payload(analyses: list[PitchAnalysis]) -> dict:
    """ビューアに埋め込む JSON を組み立てる。"""
    if not analyses:
        raise ViewerError("表示する投球がありません")
    if len(analyses) > 2:
        raise ViewerError("比較できるのは2投球までです")

    return {
        "edges": [list(edge) for edge in SKELETON_EDGES],
        "panel_series": PANEL_SERIES,
        "normalized_samples": NORMALIZED_SAMPLES,
        "pitches": [_pitch_payload(analysis) for analysis in analyses],
    }


def _pitch_payload(analysis: PitchAnalysis) -> dict:
    config = analysis.config
    sign = facing_sign(config.batter_direction.value)
    origin, scale = _reference_frame(analysis)

    frames = [
        _display_frame(analysis, position, origin, scale, sign)
        for position in range(analysis.frame_indices.size)
    ]

    return {
        "pitch_id": config.pitch_id,
        "label": config.result.label,
        "description": config.result.description,
        "fps": config.fps,
        "throwing_hand": config.throwing_hand.value,
        "throwing_side": config.side_prefix(throwing=True),
        "lead_side": config.side_prefix(throwing=False),
        "batter_direction": config.batter_direction.value,
        "scale_px": scale,
        "scale_mode": config.metrics.normalization,
        "events": {
            name.value: {"frame": event.frame_index, "source": event.source.value}
            for name, event in analysis.events.items()
        },
        "frames": [
            {
                "f": frame.frame_index,
                "t": round(frame.timestamp_sec, 4),
                "p": frame.progress_percent,
                "k": {
                    name: [round(x, 4), round(y, 4)] for name, (x, y) in frame.points.items()
                },
            }
            for frame in frames
        ],
        "normalized": _normalized_frames(analysis, frames),
        "series": {
            key: [_json_number(value) for value in analysis.series[key]]
            for key in PANEL_SERIES
            if key in analysis.series
        },
    }


def _reference_frame(analysis: PitchAnalysis) -> tuple[tuple[float, float], float]:
    """原点（股関節中点）と長さの基準を決める。

    原点は足接地フレーム。無ければ股関節が見える最初のフレーム。
    毎フレームの股関節中点を原点にすると、身体の移動そのものが消えてしまう。
    """
    scale = analysis.quality.get("body_scale", {}).get("scale_px")
    if not scale or not math.isfinite(scale) or scale <= 0:
        raise ViewerError("身体サイズを推定できないため、表示用の正規化ができません")

    anchor = analysis.position_of_event(EventName.FOOT_CONTACT)
    positions = [anchor] if anchor is not None else []
    positions += list(range(analysis.frame_indices.size))

    for position in positions:
        if position is None:
            continue
        origin = _hip_center(analysis, position)
        if origin is not None:
            return origin, float(scale)

    raise ViewerError("股関節が検出できたフレームがありません")


def _hip_center(analysis: PitchAnalysis, position: int) -> tuple[float, float] | None:
    frame = analysis.smoothed_series.frames[position]
    return midpoint(frame.get("left_hip"), frame.get("right_hip"))


def _display_frame(
    analysis: PitchAnalysis,
    position: int,
    origin: tuple[float, float],
    scale: float,
    sign: int,
) -> DisplayFrame:
    frame = analysis.smoothed_series.frames[position]
    points: dict[str, tuple[float, float]] = {}
    for name, keypoint in frame.keypoints.items():
        if not keypoint.is_valid:
            continue
        points[name] = (
            (keypoint.x - origin[0]) * sign / scale,
            -(keypoint.y - origin[1]) / scale,
        )

    progress = analysis.progress_percent[position]
    return DisplayFrame(
        frame_index=int(analysis.frame_indices[position]),
        timestamp_sec=float(analysis.timestamps[position]),
        progress_percent=None if not np.isfinite(progress) else round(float(progress), 3),
        points=points,
    )


def _normalized_frames(analysis: PitchAnalysis, frames: list[DisplayFrame]) -> list[dict] | None:
    """足接地=0%、リリース=100% に引き伸ばした座標列。

    2投球を同じ進行率で並べて見るために使う。区間が決まらなければ None。
    """
    progress = np.array(
        [np.nan if frame.progress_percent is None else frame.progress_percent for frame in frames]
    )
    inside = np.isfinite(progress)
    if np.count_nonzero(inside) < 2:
        return None

    grid = np.linspace(0.0, 100.0, NORMALIZED_SAMPLES)
    source_progress = progress[inside]
    order = np.argsort(source_progress)
    names = sorted({name for frame in frames for name in frame.points})

    resampled: dict[str, list[list[float] | None]] = {}
    for name in names:
        xs = np.array([frame.points.get(name, (np.nan, np.nan))[0] for frame in frames])[inside]
        ys = np.array([frame.points.get(name, (np.nan, np.nan))[1] for frame in frames])[inside]
        valid = np.isfinite(xs[order]) & np.isfinite(ys[order])
        if np.count_nonzero(valid) < 2:
            resampled[name] = [None] * NORMALIZED_SAMPLES
            continue
        sample_progress = source_progress[order][valid]
        x_values = np.interp(grid, sample_progress, xs[order][valid], left=np.nan, right=np.nan)
        y_values = np.interp(grid, sample_progress, ys[order][valid], left=np.nan, right=np.nan)
        resampled[name] = [
            None if not (np.isfinite(x) and np.isfinite(y)) else [round(float(x), 4), round(float(y), 4)]
            for x, y in zip(x_values, y_values)
        ]

    return [
        {
            "p": round(float(grid[index]), 2),
            "k": {
                name: values[index]
                for name, values in resampled.items()
                if values[index] is not None
            },
        }
        for index in range(NORMALIZED_SAMPLES)
    ]


def _json_number(value) -> float | None:
    return None if not np.isfinite(value) else round(float(value), 4)


def render_html(analyses: list[PitchAnalysis], title: str | None = None) -> str:
    """ビューアの HTML を組み立てて返す。"""
    import html
    import json

    from pitching.visualization.viewer_template import TEMPLATE

    payload = build_payload(analyses)
    heading = title or " vs ".join(analysis.pitch_id for analysis in analyses)
    meta = " / ".join(
        f"{analysis.pitch_id}［{analysis.config.result.label}］"
        f"{analysis.frame_indices.size}フレーム "
        f"{analysis.config.fps:g}fps "
        f"{analysis.config.throwing_hand.value}投げ "
        f"打者{analysis.config.batter_direction.value}"
        for analysis in analyses
    )
    return (
        TEMPLATE.replace("__TITLE__", html.escape(heading))
        .replace("__META__", html.escape(meta))
        # file:// では fetch が使えないので、データはそのまま埋め込む
        .replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    )


def write_viewer(
    analyses: list[PitchAnalysis], output_path: str | Path, title: str | None = None
) -> Path:
    """ビューアの HTML を書き出す。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(analyses, title), encoding="utf-8")
    return path
