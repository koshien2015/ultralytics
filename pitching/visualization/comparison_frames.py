"""良い球と悪い球の、同じイベントのフレームを左右に並べた比較画像。

フレーム番号ではなくイベントで対応付ける（同じフレーム番号どうしは比べない）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pitching.analysis import PitchAnalysis
from pitching.models import EventName
from pitching.visualization.video_overlay import OverlayError, draw_pose_skeleton

COMPARED_EVENTS = (
    EventName.FOOT_CONTACT,
    EventName.MAX_ELBOW_FLEXION,
    EventName.RELEASE,
)
LABEL_COLOR = (255, 255, 255)


def write_comparison_frames(
    first: PitchAnalysis,
    second: PitchAnalysis,
    output_dir: str | Path,
    draw_skeleton: bool = True,
) -> list[Path]:
    """イベントごとに左右並びの画像を書き出す。動画が無いイベントは飛ばす。"""
    import cv2

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for event_name in COMPARED_EVENTS:
        left = _grab(cv2, first, event_name, draw_skeleton)
        right = _grab(cv2, second, event_name, draw_skeleton)
        if left is None or right is None:
            continue
        path = directory / f"compare_{event_name.value}.png"
        cv2.imwrite(str(path), _side_by_side(cv2, left, right))
        written.append(path)
    return written


def _grab(cv2, analysis: PitchAnalysis, event_name: EventName, draw: bool):
    """該当イベントのフレーム画像を取り出し、注記を入れて返す。"""
    event = analysis.events.get(event_name)
    if event is None or event.frame_index is None:
        return None

    video_path = Path(analysis.config.video_path)
    if not video_path.is_file():
        return None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OverlayError(f"動画を開けません: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, event.frame_index)
        ok, image = capture.read()
    finally:
        capture.release()

    if not ok:
        return None

    if draw:
        position = analysis.position_of_event_frame(event.frame_index)
        pose_frame = (
            analysis.smoothed_series.frames[position] if position is not None else None
        )
        if pose_frame is not None:
            draw_pose_skeleton(
                cv2,
                image,
                pose_frame,
                analysis.config.side_prefix(throwing=True),
                analysis.config.preprocessing.confidence_threshold,
            )

    caption = (
        f"{analysis.config.display_name} "
        f"{event_name.value} f{event.frame_index} ({event.source.value})"
    )
    cv2.putText(
        image, caption, (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, LABEL_COLOR, 2, cv2.LINE_AA,
    )
    return image


def _side_by_side(cv2, left, right):
    """高さを揃えて横に連結する。"""
    height = min(left.shape[0], right.shape[0])
    resized = [
        cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height))
        for image in (left, right)
    ]
    return np.hstack(resized)
