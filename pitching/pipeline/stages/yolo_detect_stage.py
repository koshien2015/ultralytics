from __future__ import annotations

import dataclasses
import logging
from typing import List

from pitching.domain.entities.detection import YoloDetection
from pitching.infra.ml.yolo_detector import UltralyticsYoloDetector
from pitching.infra.video.reader import VideoReader
from pitching.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class YoloDetectStage:
    """強調動画に対して YOLO 検出を実行し、全クラスの検出結果を artifacts に保存する。"""

    name = "yolo_detect"

    def __init__(self, detector: UltralyticsYoloDetector) -> None:
        self._detector = detector

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.enhanced_video_path is None or not ctx.enhanced_video_path.exists():
            raise RuntimeError("enhanced_video_path が未設定です。FrameDiffStage を先に実行してください。")

        detections: List[YoloDetection] = []

        with VideoReader(ctx.enhanced_video_path) as reader:
            for meta, frame in reader:
                dets = self._detector.detect(frame, meta.frame_index)
                detections.extend(dets)
                if meta.frame_index % 30 == 0:
                    logger.debug("YoloDetectStage: frame %d processed", meta.frame_index)

        logger.info("YoloDetectStage: %d detections total", len(detections))

        new_artifacts = dataclasses.replace(
            ctx.artifacts, yolo_detections=tuple(detections)
        )
        return dataclasses.replace(ctx, artifacts=new_artifacts)
