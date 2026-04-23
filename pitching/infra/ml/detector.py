from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np

from pitching.domain.entities.detection import DiffDetection, YoloDetection


class ObjectDetector(Protocol):
    """フレーム画像から物体を検出する interface。"""

    def detect(self, frame: np.ndarray, frame_index: int) -> Tuple[YoloDetection, ...]:
        ...


class FrameDiffDetectorProtocol(Protocol):
    """フレームペアから差分ベースの物体検出を行う interface。"""

    def detect_on_pair(
        self,
        prev: np.ndarray,
        curr: np.ndarray,
        next_frame: np.ndarray,
        frame_index: int,
    ) -> Tuple[DiffDetection, ...]:
        ...
