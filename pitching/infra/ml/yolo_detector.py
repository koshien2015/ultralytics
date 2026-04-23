from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from pitching.domain.entities.bbox import BBox
from pitching.domain.entities.detection import YoloDetection

logger = logging.getLogger(__name__)


class UltralyticsYoloDetector:
    """ultralytics YOLO を使った物体検出の実装。"""

    def __init__(self, model_path: str | Path, target_classes: List[int] | None = None) -> None:
        from ultralytics import YOLO
        self._model = YOLO(str(model_path))
        self._target_classes = set(target_classes) if target_classes else None
        logger.info("YOLO model loaded: %s", model_path)

    def detect(self, frame: np.ndarray, frame_index: int) -> Tuple[YoloDetection, ...]:
        results = self._model(frame, verbose=False)
        detections: List[YoloDetection] = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if self._target_classes and class_id not in self._target_classes:
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0])
                detections.append(YoloDetection(
                    frame_index=frame_index,
                    class_id=class_id,
                    class_name=result.names.get(class_id, str(class_id)),
                    bbox=BBox(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                ))

        return tuple(detections)
