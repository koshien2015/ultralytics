from __future__ import annotations

import dataclasses
import logging
from typing import List

from pitching.domain.entities.pitch import Pitch
from pitching.domain.services.trajectory_builder import build_pitch
from pitching.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class PitchAnalysisStage:
    """リリースイベント・FusedDetection・StrikeZoneSeries から Pitch を構築する。"""

    name = "pitch_analysis"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.artifacts.release_events:
            logger.info("PitchAnalysisStage: no releases found, skipping")
            return ctx

        if ctx.artifacts.strike_zone_series is None:
            logger.warning("PitchAnalysisStage: no strike zone series, skipping")
            return ctx

        pitches: List[Pitch] = []
        releases = sorted(ctx.artifacts.release_events, key=lambda r: r.release_frame)

        for i, release in enumerate(releases):
            # 次のリリース前までのボール検出を使う
            next_release_frame = (
                releases[i + 1].release_frame if i + 1 < len(releases) else ctx.total_frames
            )
            ball_dets = tuple(
                d for d in ctx.artifacts.fused_detections
                if release.release_frame <= d.frame_index < next_release_frame
            )
            pitch = build_pitch(
                release=release,
                ball_detections=ball_dets,
                strike_zone_series=ctx.artifacts.strike_zone_series,
                fps=ctx.fps,
            )
            pitches.append(pitch)

        logger.info("PitchAnalysisStage: %d pitches built", len(pitches))
        new_artifacts = dataclasses.replace(ctx.artifacts, pitches=tuple(pitches))
        return dataclasses.replace(ctx, artifacts=new_artifacts)
