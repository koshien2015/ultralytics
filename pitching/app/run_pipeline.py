from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from pitching.config.loader import load_config
from pitching.config.schema import PipelineConfig
from pitching.domain.entities.bbox import BBox
from pitching.domain.entities.detection import (
    DetectionSource,
    DiffDetection,
    FusedDetection,
    YoloDetection,
)
from pitching.domain.entities.pitch import Pitch, PitchTrajectoryPoint, ReleaseEvent
from pitching.domain.entities.pose import PoseFrame
from pitching.domain.entities.strike_zone import StrikeZone, StrikeZoneSeries
from pitching.domain.entities.track import Track, TrackPoint
from pitching.infra.ml.diff_detector import FrameDiffDetector
from pitching.infra.ml.null_pose_estimator import NullPoseEstimator
from pitching.infra.ml.tracker import NearestNeighborTracker
from pitching.infra.ml.yolo_detector import UltralyticsYoloDetector
from pitching.infra.storage.checkpoint import JsonCheckpointStore
from pitching.infra.video.reader import VideoReader
from pitching.pipeline.context import PipelineArtifacts, PipelineContext
from pitching.pipeline.runner import PipelineRunner
from pitching.pipeline.stages.frame_diff_stage import FrameDiffStage
from pitching.pipeline.stages.fusion_stage import FusionStage
from pitching.pipeline.stages.pitch_analysis_stage import PitchAnalysisStage
from pitching.pipeline.stages.pose_estimation_stage import PoseEstimationStage
from pitching.pipeline.stages.release_detection_stage import ReleaseDetectionStage
from pitching.pipeline.stages.rendering_stage import RenderingStage
from pitching.pipeline.stages.strike_zone_stage import StrikeZoneStage
from pitching.pipeline.stages.tracking_stage import TrackingStage
from pitching.pipeline.stages.yolo_detect_stage import YoloDetectStage

logger = logging.getLogger(__name__)

# checkpoint の JSON 復元に必要な全型の registry
ARTIFACT_REGISTRY: dict = {
    "BBox": BBox,
    "DetectionSource": DetectionSource,
    "DiffDetection": DiffDetection,
    "FusedDetection": FusedDetection,
    "PipelineArtifacts": PipelineArtifacts,
    "Pitch": Pitch,
    "PitchTrajectoryPoint": PitchTrajectoryPoint,
    "PoseFrame": PoseFrame,
    "ReleaseEvent": ReleaseEvent,
    "StrikeZone": StrikeZone,
    "StrikeZoneSeries": StrikeZoneSeries,
    "Track": Track,
    "TrackPoint": TrackPoint,
    "YoloDetection": YoloDetection,
}


def build_stages(cfg: PipelineConfig) -> list:
    """設定から全ステージをインスタンス化して返す（composition root）。"""
    diff_detector = FrameDiffDetector(
        threshold=cfg.frame_diff.threshold,
        gamma=cfg.frame_diff.gamma,
        dilation_size=cfg.frame_diff.dilation_size,
        erosion_kernel_size=cfg.frame_diff.erosion_kernel_size,
        area_min=cfg.frame_diff.area_min,
        area_max=cfg.frame_diff.area_max,
    )
    yolo_detector = UltralyticsYoloDetector(
        model_path=cfg.yolo.model_path,
        target_classes=None,  # 全クラスを検出し、後段でフィルタ
    )
    tracker = NearestNeighborTracker(
        max_trajectory_length=cfg.tracking.max_trajectory_length,
        fade_frames=cfg.tracking.fade_frames,
        max_match_distance_px=cfg.tracking.max_match_distance_px,
    )
    pose_estimator = NullPoseEstimator()

    return [
        FrameDiffStage(diff_detector),
        YoloDetectStage(yolo_detector),
        FusionStage(cfg.fusion, cfg.frame_diff, cfg.yolo.min_confidence),
        PoseEstimationStage(pose_estimator),
        TrackingStage(tracker),
        StrikeZoneStage(cfg.strike_zone.width_px, cfg.strike_zone.fixed_center_x),
        ReleaseDetectionStage(),
        PitchAnalysisStage(),
        RenderingStage(cfg.rendering),
    ]


def run(
    video_path: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
    resume_from: Optional[str] = None,
    stop_after: Optional[str] = None,
    enable_checkpoint: bool = True,
) -> None:
    """パイプラインを実行するエントリポイント。"""
    cfg = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 動画メタデータを取得
    with VideoReader(video_path) as reader:
        fps = reader.fps
        total_frames = reader.total_frames
        video_size = reader.size

    logger.info("Input: %s  fps=%.1f  frames=%d  size=%s",
                video_path, fps, total_frames, video_size)

    ctx = PipelineContext(
        config=cfg,
        video_path=video_path,
        output_dir=output_dir,
        fps=fps,
        total_frames=total_frames,
        video_size=video_size,
    )

    store: Optional[JsonCheckpointStore] = None
    if enable_checkpoint:
        store = JsonCheckpointStore(output_dir, ARTIFACT_REGISTRY)

    stages = build_stages(cfg)
    runner = PipelineRunner(
        stages=stages,
        checkpoint_store=store,
        resume_from=resume_from,
        stop_after=stop_after,
    )
    runner.run(ctx)
    logger.info("Pipeline finished. Output: %s", output_dir)
