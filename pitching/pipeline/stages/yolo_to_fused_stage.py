from __future__ import annotations

import dataclasses
import logging
from typing import List

from pitching.domain.entities.detection import DetectionSource, FusedDetection
from pitching.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

CLASS_BALL = 0


class YoloToFusedStage:
    """
    Phase 4 用の暫定ステージ。
    YOLO 検出のボールクラスをそのまま FusedDetection(YOLO_PRIMARY) に変換する。
    Phase 5 の FusionStage に置き換えられる。
    """

    name = "yolo_to_fused"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        fused: List[FusedDetection] = []
        for det in ctx.artifacts.yolo_detections:
            if det.class_id != CLASS_BALL:
                continue
            fused.append(FusedDetection(
                frame_index=det.frame_index,
                class_id=CLASS_BALL,
                bbox=det.bbox,
                center_x=det.bbox.center_x,
                center_y=det.bbox.center_y,
                confidence=det.confidence,
                source=DetectionSource.FUSED_YOLO_PRIMARY,
            ))

        logger.info("YoloToFusedStage: %d fused detections", len(fused))
        new_artifacts = dataclasses.replace(ctx.artifacts, fused_detections=tuple(fused))
        return dataclasses.replace(ctx, artifacts=new_artifacts)
