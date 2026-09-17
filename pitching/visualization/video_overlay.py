"""元動画に解析結果を重ねた動画を出力する。

このモジュールと comparison_frames だけが OpenCV に依存する。
数値は解析済みの PitchAnalysis から読むだけで、ここでは何も計算しない。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pitching.analysis import PitchAnalysis
from pitching.models import EventName
from pitching.visualization.skeleton import SKELETON_EDGES, throwing_arm_edges

SKELETON_COLOR = (200, 200, 200)
THROWING_ARM_COLOR = (0, 200, 255)
WARNING_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)


class OverlayError(RuntimeError):
    """解析動画の書き出しに失敗したときに送出する。"""


def render_overlay_video(
    analysis: PitchAnalysis,
    video_path: str | Path,
    output_path: str | Path,
    use_smoothed: bool = True,
) -> Path:
    """解析区間について、骨格と数値を重ねた動画を書き出す。"""
    import cv2

    source = Path(video_path)
    if not source.is_file():
        raise OverlayError(f"動画が見つかりません: {source}")

    series = analysis.smoothed_series if use_smoothed else analysis.raw_series
    frames_by_index = {frame.frame_index: frame for frame in series.frames}
    threshold = analysis.config.preprocessing.confidence_threshold
    throwing_side = analysis.config.side_prefix(throwing=True)
    event_labels = _event_labels(analysis)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OverlayError(f"動画を開けません: {source}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    try:
        start = int(analysis.frame_indices[0])
        end = int(analysis.frame_indices[-1])
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)

        for frame_index in range(start, end + 1):
            ok, image = capture.read()
            if not ok:
                break
            if writer is None:
                writer = _open_writer(cv2, output, image, analysis.config.fps)

            pose_frame = frames_by_index.get(frame_index)
            if pose_frame is not None:
                draw_pose_skeleton(cv2, image, pose_frame, throwing_side, threshold)
            _draw_text(cv2, image, analysis, frame_index, pose_frame, threshold, event_labels)
            writer.write(image)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if writer is None:
        raise OverlayError("フレームを1枚も読み込めませんでした")
    return output


def _open_writer(cv2, path: Path, image, fps: float):
    height, width = image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise OverlayError(f"出力動画を作成できません: {path}")
    return writer


def draw_pose_skeleton(cv2, image, pose_frame, throwing_side: str, threshold: float) -> None:
    """骨格を描く。投球腕は太く色を変えて強調する。"""
    highlighted = set(throwing_arm_edges(throwing_side))
    for first, second in SKELETON_EDGES:
        start = pose_frame.get(first)
        end = pose_frame.get(second)
        if start is None or end is None:
            continue
        if min(start.confidence, end.confidence) < threshold:
            continue
        color = THROWING_ARM_COLOR if (first, second) in highlighted else SKELETON_COLOR
        thickness = 3 if (first, second) in highlighted else 1
        cv2.line(
            image,
            (int(round(start.x)), int(round(start.y))),
            (int(round(end.x)), int(round(end.y))),
            color,
            thickness,
            cv2.LINE_AA,
        )

    for name in (f"{throwing_side}_shoulder", f"{throwing_side}_elbow", f"{throwing_side}_wrist"):
        keypoint = pose_frame.get(name)
        if keypoint is None:
            continue
        cv2.circle(
            image, (int(round(keypoint.x)), int(round(keypoint.y))), 5, THROWING_ARM_COLOR, -1
        )


def _draw_text(cv2, image, analysis, frame_index, pose_frame, threshold, event_labels) -> None:
    position = analysis.position_of_event_frame(frame_index)
    lines = [f"frame {frame_index}  {analysis.pitch_id}"]

    if position is not None:
        lines.append(f"elbow  {_format(analysis.series['elbow_angle_deg'][position])} deg")
        lines.append(f"forearm {_format(analysis.series['forearm_angle_deg'][position])} deg")
        progress = analysis.progress_percent[position]
        if np.isfinite(progress):
            lines.append(f"progress {progress:6.1f} %")

    label = event_labels.get(frame_index)
    if label:
        lines.append(f"EVENT: {label}")

    for order, text in enumerate(lines):
        cv2.putText(
            image, text, (12, 30 + order * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2, cv2.LINE_AA,
        )

    warnings = _low_confidence_names(pose_frame, analysis, threshold)
    if warnings:
        cv2.putText(
            image, f"LOW CONF: {', '.join(warnings)}",
            (12, 30 + len(lines) * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, WARNING_COLOR, 2, cv2.LINE_AA,
        )


def _low_confidence_names(pose_frame, analysis, threshold: float) -> list[str]:
    """投球腕のキーポイントのうち、信頼度が閾値未満のもの。"""
    if pose_frame is None:
        return ["no pose"]
    side = analysis.config.side_prefix(throwing=True)
    names = (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
    return [
        name.replace(f"{side}_", "")
        for name in names
        if pose_frame.confidence_of(name) < threshold
    ]


def _event_labels(analysis: PitchAnalysis) -> dict[int, str]:
    labels: dict[int, str] = {}
    for name in (
        EventName.FOOT_CONTACT,
        EventName.MAX_ELBOW_FLEXION,
        EventName.EXTENSION_START,
        EventName.RELEASE,
    ):
        event = analysis.events.get(name)
        if event is None or event.frame_index is None:
            continue
        labels[event.frame_index] = f"{name.value} ({event.source.value})"
    return labels


def _format(value: float) -> str:
    return "  n/a" if not np.isfinite(value) else f"{value:6.1f}"
