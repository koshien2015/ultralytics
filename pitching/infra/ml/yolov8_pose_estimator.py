from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from pitching.domain.entities.pose import Keypoint, PoseFrame

logger = logging.getLogger(__name__)

_COCO_KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


class UltralyticsYoloPoseEstimator:
    """YOLOv8-Pose による姿勢推定の実装。COCO 17 keypoint を返す。"""

    def __init__(
        self,
        model_path: str | Path,
        min_keypoint_confidence: float = 0.3,
    ) -> None:
        from ultralytics import YOLO
        self._model = YOLO(str(model_path))
        self._min_keypoint_confidence = min_keypoint_confidence
        logger.info("YOLOv8-Pose model loaded: %s", model_path)

    def estimate(self, frame: np.ndarray, frame_index: int) -> Tuple[PoseFrame, ...]:
        results = self._model(frame, verbose=False)
        pose_frames: List[PoseFrame] = []

        for result in results:
            kp_data = result.keypoints.data  # shape: (N, 17, 3) — x, y, conf
            for person_id in range(len(kp_data)):
                keypoints = tuple(
                    Keypoint(
                        name=_COCO_KEYPOINT_NAMES[k],
                        x=float(kp_data[person_id, k, 0]),
                        y=float(kp_data[person_id, k, 1]),
                        confidence=float(kp_data[person_id, k, 2]),
                    )
                    for k in range(17)
                )
                pose_frames.append(PoseFrame(
                    frame_index=frame_index,
                    person_id=person_id,
                    keypoints=keypoints,
                ))

        return tuple(pose_frames)
