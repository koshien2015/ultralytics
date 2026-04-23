from __future__ import annotations

from typing import Protocol, Tuple

import numpy as np

from pitching.domain.entities.pose import PoseFrame


class PoseEstimator(Protocol):
    """姿勢推定の interface。将来 MediaPipe / YOLO-Pose などで実装する。"""

    def estimate(self, frame: np.ndarray, frame_index: int) -> Tuple[PoseFrame, ...]:
        ...
