from __future__ import annotations

import dataclasses
import logging
from typing import Dict, List

from pitching.domain.entities.batter_metrics import BatterSwingMetrics, SwingFrameMetrics
from pitching.domain.entities.pose_role import BATTER_SWING_CLASS_IDS, PoseRole
from pitching.domain.services.batter_analyzer import aggregate_swing, analyze_swing_frame
from pitching.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class BatterAnalysisStage:
    """打者の役割付き PoseFrame からスイングごとのメトリクスを計算する。"""

    name = "batter_analysis"

    def __init__(
        self,
        batting_hand: str = "right",
        min_keypoint_confidence: float = 0.3,
    ) -> None:
        self._batting_hand = batting_hand
        self._min_conf = min_keypoint_confidence

    def run(self, ctx: PipelineContext) -> PipelineContext:
        batter_swing_frames = [
            rf for rf in ctx.artifacts.role_assigned_pose_frames
            if rf.role == PoseRole.BATTER and rf.is_swinging
        ]

        # リリースイベントと対応付け（最近接のリリースに帰属）
        swing_metrics = self._group_by_pitch(batter_swing_frames, ctx)

        logger.info("BatterAnalysisStage: %d swing metrics computed", len(swing_metrics))
        new_artifacts = dataclasses.replace(ctx.artifacts, batter_metrics=tuple(swing_metrics))
        return dataclasses.replace(ctx, artifacts=new_artifacts)

    def _group_by_pitch(self, swing_rfs, ctx: PipelineContext) -> List[BatterSwingMetrics]:
        if not ctx.artifacts.release_events:
            return []

        # フレームを最近接リリースに割り当て
        pitch_buckets: Dict[int, List[SwingFrameMetrics]] = {
            r.pitch_id: [] for r in ctx.artifacts.release_events
        }
        release_frames = [(r.pitch_id, r.release_frame) for r in ctx.artifacts.release_events]

        for rf in swing_rfs:
            closest_pitch_id = min(
                release_frames,
                key=lambda pr: abs(rf.frame_index - pr[1]),
            )[0]
            frame_metric = analyze_swing_frame(rf, self._batting_hand, self._min_conf)
            pitch_buckets[closest_pitch_id].append(frame_metric)

        results = []
        for pitch_id, frames in pitch_buckets.items():
            results.append(aggregate_swing(pitch_id, tuple(frames)))
        return results
