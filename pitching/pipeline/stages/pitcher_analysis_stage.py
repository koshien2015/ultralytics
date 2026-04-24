from __future__ import annotations

import dataclasses
import logging
from typing import List

from pitching.domain.entities.pitcher_metrics import PitcherFrameMetrics, PitcherPitchMetrics
from pitching.domain.entities.pose_role import PITCHER_CLASS_IDS, PoseRole
from pitching.domain.services.pitcher_analyzer import aggregate_pitcher_pitch, analyze_pitcher_frame
from pitching.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class PitcherAnalysisStage:
    """投手の役割付き PoseFrame から投球ごとのメトリクスを計算する。"""

    name = "pitcher_analysis"

    def __init__(
        self,
        throwing_hand: str = "right",
        min_keypoint_confidence: float = 0.3,
        motion_window_frames: int = 60,
    ) -> None:
        self._throwing_hand = throwing_hand
        self._min_conf = min_keypoint_confidence
        self._window = motion_window_frames

    def run(self, ctx: PipelineContext) -> PipelineContext:
        pitcher_frames = [
            rf for rf in ctx.artifacts.role_assigned_pose_frames
            if rf.role == PoseRole.PITCHER
        ]

        pitch_metrics: List[PitcherPitchMetrics] = []
        for release in ctx.artifacts.release_events:
            # リリースフレームを中心に前後 window フレームを取得
            start = release.release_frame - self._window
            end = release.release_frame + 10
            motion_rfs = [rf for rf in pitcher_frames if start <= rf.frame_index <= end]

            frame_metrics = tuple(
                analyze_pitcher_frame(rf, self._throwing_hand, self._min_conf)
                for rf in motion_rfs
            )
            pitch_metrics.append(
                aggregate_pitcher_pitch(release.pitch_id, release.release_frame, frame_metrics)
            )

        logger.info("PitcherAnalysisStage: %d pitch metrics computed", len(pitch_metrics))
        new_artifacts = dataclasses.replace(ctx.artifacts, pitcher_metrics=tuple(pitch_metrics))
        return dataclasses.replace(ctx, artifacts=new_artifacts)
