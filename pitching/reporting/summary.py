"""PitchAnalysis から、投球単位の要約（metrics.json の中身）を組み立てる。

観測値と、その解釈のための注意書きを分けて入れる。
解釈や良否の判定はここでは行わない。
"""

from __future__ import annotations

import math

from pitching.analysis import PitchAnalysis

INTERPRETATION_NOTES = [
    "すべて2D画像座標に基づく観測値。奥行きや肩の内外旋は推定していない。",
    "ピクセル量は身体サイズで正規化した相対値。実世界の長さではない。",
    "1球対1球の比較は観測された差であり、因果関係や一般則ではない。",
]


def build_summary(analysis: PitchAnalysis) -> dict:
    """投球単位の要約特徴量。"""
    config = analysis.config
    return {
        "pitch_id": config.pitch_id,
        "result": {"label": config.result.label, "description": config.result.description},
        "capture": {
            "video_path": config.video_path,
            "fps": config.fps,
            "throwing_hand": config.throwing_hand.value,
            "batter_direction": config.batter_direction.value,
            "start_frame": config.start_frame,
            "end_frame": config.end_frame,
            "frames_analyzed": int(analysis.frame_indices.size),
        },
        "events": {
            name.value: {
                "frame": event.frame_index,
                "source": event.source.value,
                "note": event.note,
                "progress_percent": _progress_at(analysis, event.frame_index),
            }
            for name, event in analysis.events.items()
        },
        "settings": {
            "preprocessing": config.preprocessing.model_dump(),
            "metrics": config.metrics.model_dump(),
            "event_detection": config.event_detection.model_dump(),
        },
        "features": _clean(analysis.features),
        "quality": _clean(analysis.quality),
        "notes": INTERPRETATION_NOTES,
    }


def _progress_at(analysis: PitchAnalysis, frame: int | None) -> float | None:
    if frame is None:
        return None
    position = analysis.position_of_event_frame(frame)
    if position is None:
        return None
    value = analysis.progress_percent[position]
    return None if not math.isfinite(value) else float(value)


def _clean(value):
    """JSON にできない値（NaN / numpy 型）を落とす。"""
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if hasattr(value, "item") and hasattr(value, "dtype"):
        return _clean(value.item())
    return value


def config_from_summary(summary: dict) -> "PitchConfig":
    """metrics.json から PitchConfig を復元する。

    compare で比較グラフを描くとき、元の設定ファイルを再指定させないため。
    """
    from pitching.config import PitchConfig

    capture = summary.get("capture", {})
    events = summary.get("events", {})
    settings = summary.get("settings", {})
    return PitchConfig.model_validate(
        {
            "pitch_id": summary.get("pitch_id", ""),
            "video_path": capture.get("video_path", ""),
            "throwing_hand": capture.get("throwing_hand", "right"),
            "batter_direction": capture.get("batter_direction", "left"),
            "fps": capture.get("fps", 60.0),
            "start_frame": capture.get("start_frame", 0),
            "end_frame": capture.get("end_frame"),
            "events": {
                "stride_foot_contact_frame": events.get("foot_contact", {}).get("frame"),
                "release_frame": events.get("release", {}).get("frame"),
            },
            "result": summary.get("result", {"label": "", "description": ""}),
            **{key: value for key, value in settings.items() if value},
        }
    )
