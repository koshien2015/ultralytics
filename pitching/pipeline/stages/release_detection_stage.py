from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from typing import Dict, List

from pitching.domain.entities.pitch import ReleaseEvent
from pitching.domain.services.release_detector import detect_release, resolve_pitcher_state
from pitching.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class ReleaseDetectionStage:
    """
    YOLO 検出のクラス遷移（motion→release）からリリースイベントを検出する。
    """

    name = "release_detection"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        by_frame: Dict[int, List[int]] = defaultdict(list)
        for det in ctx.artifacts.yolo_detections:
            by_frame[det.frame_index].append(det.class_id)

        releases: List[ReleaseEvent] = []
        prev_state = None
        pitch_id = 1

        for frame_index in range(ctx.total_frames):
            class_ids = tuple(by_frame.get(frame_index, []))
            curr_state = resolve_pitcher_state(class_ids)

            event = detect_release(
                prev_state=prev_state,
                curr_state=curr_state,
                pitch_id=pitch_id,
                frame_index=frame_index,
                fps=ctx.fps,
            )
            if event:
                releases.append(event)
                logger.info("ReleaseDetectionStage: release detected at frame %d", frame_index)
                pitch_id += 1

            prev_state = curr_state

        logger.info("ReleaseDetectionStage: %d releases", len(releases))
        new_artifacts = dataclasses.replace(ctx.artifacts, release_events=tuple(releases))
        return dataclasses.replace(ctx, artifacts=new_artifacts)
