from __future__ import annotations

import dataclasses
import logging
from typing import List

import numpy as np

from pitching.domain.entities.pose import PoseFrame
from pitching.infra.ml.pose_estimator import PoseEstimator
from pitching.infra.video.reader import VideoReader
from pitching.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class PoseEstimationStage:
    """
    姿勢推定ステージ。
    NullPoseEstimator が渡された場合は空 tuple を返すだけ。
    将来 MediaPipe / YOLOv8-Pose の実装を差し込める差し込み口。
    """

    name = "pose_estimation"

    def __init__(self, estimator: PoseEstimator) -> None:
        self._estimator = estimator

    def run(self, ctx: PipelineContext) -> PipelineContext:
        pose_frames: List[PoseFrame] = []

        with VideoReader(ctx.video_path) as reader:
            for meta, frame in reader:
                results = self._estimator.estimate(frame, meta.frame_index)
                pose_frames.extend(results)

        logger.info("PoseEstimationStage: %d pose frames", len(pose_frames))
        new_artifacts = dataclasses.replace(ctx.artifacts, pose_frames=tuple(pose_frames))
        return dataclasses.replace(ctx, artifacts=new_artifacts)
