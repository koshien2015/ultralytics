from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np

from pitching.domain.entities.frame import FrameMeta


class VideoReader:
    """cv2.VideoCapture のシンプルなラッパ。"""

    def __init__(self, path: Path) -> None:
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

    @property
    def fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def size(self) -> Tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def __iter__(self) -> Iterator[Tuple[FrameMeta, np.ndarray]]:
        fps = self.fps
        w, h = self.size
        frame_index = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield FrameMeta(
                frame_index=frame_index,
                timestamp=frame_index / fps if fps > 0 else 0.0,
                width=w,
                height=h,
            ), frame
            frame_index += 1

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *_) -> None:
        self.release()
