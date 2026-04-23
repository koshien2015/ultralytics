from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

from pitching.domain.entities.detection import FusedDetection
from pitching.domain.entities.track import Track
from pitching.infra.ml.tracker import NearestNeighborTracker
from pitching.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class TrackingStage:
    """FusedDetection のフレーム列をトラッカーに流し、Track を生成する。"""

    name = "tracking"

    def __init__(self, tracker: NearestNeighborTracker) -> None:
        self._tracker = tracker

    def run(self, ctx: PipelineContext) -> PipelineContext:
        # フレームごとに FusedDetection をグループ化
        by_frame: Dict[int, List[FusedDetection]] = defaultdict(list)
        for det in ctx.artifacts.fused_detections:
            by_frame[det.frame_index].append(det)

        final_tracks: Tuple[Track, ...] = ()
        for frame_index in range(ctx.total_frames):
            dets = tuple(by_frame.get(frame_index, []))
            final_tracks = self._tracker.update(dets, frame_index)

        logger.info("TrackingStage: %d tracks", len(final_tracks))
        new_artifacts = dataclasses.replace(ctx.artifacts, tracks=final_tracks)
        return dataclasses.replace(ctx, artifacts=new_artifacts)
