from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from pitching.domain.entities.detection import DiffDetection
from pitching.infra.ml.diff_detector import FrameDiffDetector
from pitching.infra.video.reader import VideoReader
from pitching.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class FrameDiffStage:
    """
    連続3フレームの差分ANDでボール候補を検出し、
    YOLO用の強調動画も生成する。
    tennis.py の責務を分離したステージ。
    """

    name = "frame_diff"

    def __init__(self, detector: FrameDiffDetector) -> None:
        self._detector = detector

    def run(self, ctx: PipelineContext) -> PipelineContext:
        enhanced_path = ctx.output_dir / f"{ctx.base_name}_enhance.mp4"
        ctx.output_dir.mkdir(parents=True, exist_ok=True)

        diff_dets: List[DiffDetection] = []
        w, h = ctx.video_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(enhanced_path), fourcc, ctx.fps, (w, h))

        with VideoReader(ctx.video_path) as reader:
            frames = list(reader)  # (meta, frame) のリスト

        logger.info("FrameDiffStage: processing %d frames", len(frames))

        for i, (meta, curr) in enumerate(frames):
            prev = frames[i - 1][1] if i > 0 else curr
            nxt = frames[i + 1][1] if i < len(frames) - 1 else curr

            dets = self._detector.detect_on_pair(prev, curr, nxt, meta.frame_index)
            diff_dets.extend(dets)

            enhanced = self._detector.make_enhanced_frame(prev, curr, nxt)
            writer.write(enhanced)

        writer.release()
        logger.info("FrameDiffStage: enhanced video saved to %s", enhanced_path)

        new_artifacts = dataclasses.replace(
            ctx.artifacts, diff_detections=tuple(diff_dets)
        )
        return dataclasses.replace(
            ctx,
            artifacts=new_artifacts,
            enhanced_video_path=enhanced_path,
        )
