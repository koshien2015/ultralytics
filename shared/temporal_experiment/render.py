"""比較動画。同じ映像に手法ごとの結果を描いて横に並べる。

推定位置を実検出と誤認しないよう、状態ごとに色と線種を変える。
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .schema import DETECTED, INTERPOLATED, PREDICTED

TRAIL_FRAMES = 30
BASE_WIDTH = 640  # この幅のときに読める大きさを基準に、マークと文字を拡大する
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class MarkStyle:
    color: tuple  # BGR
    solid: bool
    label: str


STYLES = {
    DETECTED: MarkStyle((80, 220, 80), True, "detected"),
    INTERPOLATED: MarkStyle((40, 140, 255), False, "interpolated"),
    PREDICTED: MarkStyle((255, 220, 60), False, "predicted"),
}


def _scale(image) -> float:
    return image.shape[1] / BASE_WIDTH


def _weight(image) -> int:
    return max(1, round(1.5 * _scale(image)))


def _dashed_rect(image, box, color) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    edges = (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1)))
    for (ax, ay), (bx, by) in edges:
        length = max(abs(bx - ax), abs(by - ay), 1)
        for start in range(0, length, 8):
            end = min(start + 4, length)
            first = (ax + (bx - ax) * start // length, ay + (by - ay) * start // length)
            second = (ax + (bx - ax) * end // length, ay + (by - ay) * end // length)
            cv2.line(image, first, second, color, _weight(image), cv2.LINE_AA)


def _draw_trail(image, trail) -> None:
    placed = [e for e in trail if e.has_position]
    for left, right in zip(placed, placed[1:]):
        if right.frame - left.frame > 1:
            continue
        cv2.line(image, (int(left.x), int(left.y)), (int(right.x), int(right.y)),
                 STYLES[right.status].color, _weight(image), cv2.LINE_AA)


def _draw_mark(image, estimate) -> None:
    style = STYLES[estimate.status]
    center = (int(estimate.x), int(estimate.y))
    weight = _weight(image)
    radius = max(3, round(6 * _scale(image)))
    cv2.circle(image, center, radius, style.color, -1 if style.solid else weight, cv2.LINE_AA)
    if estimate.box is not None:
        if style.solid:
            x1, y1, x2, y2 = (int(v) for v in estimate.box)
            cv2.rectangle(image, (x1, y1), (x2, y2), style.color, weight, cv2.LINE_AA)
        else:
            _dashed_rect(image, estimate.box, style.color)
    text = style.label if estimate.conf is None else f"{style.label} {estimate.conf:.2f}"
    offset = radius + 4
    cv2.putText(image, text, (center[0] + offset, center[1] - offset), FONT, 0.5 * _scale(image),
                style.color, weight, cv2.LINE_AA)


def draw_panel(frame, title, trail, frame_number, draw_trail=True) -> np.ndarray:
    """trail は現在フレームまでの推定列。元のフレームは書き換えない"""
    image = frame.copy()
    if draw_trail:
        _draw_trail(image, trail)
    current = trail[-1] if trail else None
    if current is not None and current.frame == frame_number and current.has_position:
        _draw_mark(image, current)
    scale = _scale(image)
    cv2.putText(image, f"{title}  frame {frame_number}", (8, round(24 * scale)), FONT, 0.6 * scale,
                (255, 255, 255), _weight(image), cv2.LINE_AA)
    return image


def _resize(image, width) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def render_comparison(video: Path, out: Path, estimates: dict, start: int, end: int,
                      panel_width: int, draw_trail: bool = True) -> Path:
    """estimates は {表示名: Estimate の並び}。[start, end) を横並びの動画にする"""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"動画を開けません: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    by_frame = {name: {e.frame: e for e in values} for name, values in estimates.items()}
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = None
    for frame_number in range(start, end):
        ok, frame = capture.read()
        if not ok:
            break
        panels = []
        for name, lookup in by_frame.items():
            first = max(start, frame_number - TRAIL_FRAMES)
            trail = tuple(lookup[f] for f in range(first, frame_number + 1) if f in lookup)
            panel = draw_panel(frame, name, trail, frame_number, draw_trail)
            panels.append(_resize(panel, panel_width))
        row = cv2.hconcat(panels)
        if writer is None:
            size = (row.shape[1], row.shape[0])
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        writer.write(row)
    capture.release()
    if writer is None:
        raise RuntimeError(f"指定の範囲にフレームがありません: {start}〜{end}")
    writer.release()
    return Path(out)
