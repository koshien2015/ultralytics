from __future__ import annotations

from typing import Tuple

import numpy as np

from pitching.domain.entities.pose import PoseFrame


class NullPoseEstimator:
    """
    姿勢推定の stub 実装。常に空 tuple を返す。

    TODO: 実装を差し替える場合はここを MediaPipe または YOLOv8-Pose 実装に置き換える。
          PoseEstimator Protocol に準拠していれば pipeline の変更は不要。
    """

    def estimate(self, frame: np.ndarray, frame_index: int) -> Tuple[PoseFrame, ...]:
        return ()
