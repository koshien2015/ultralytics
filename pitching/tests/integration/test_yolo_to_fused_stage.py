import dataclasses
from pathlib import Path

import pytest

from pitching.config.schema import PipelineConfig
from pitching.domain.entities.bbox import BBox
from pitching.domain.entities.detection import DetectionSource, YoloDetection
from pitching.pipeline.context import PipelineArtifacts, PipelineContext
from pitching.pipeline.stages.yolo_to_fused_stage import YoloToFusedStage


def make_ctx(yolo_dets, tmp_path):
    artifacts = PipelineArtifacts(yolo_detections=tuple(yolo_dets))
    return PipelineContext(
        config=PipelineConfig(),
        video_path=tmp_path / "v.mp4",
        output_dir=tmp_path / "out",
        fps=30.0,
        total_frames=10,
        video_size=(100, 100),
        artifacts=artifacts,
    )


def make_yolo(frame_index, class_id, conf=0.9):
    bbox = BBox(10.0, 10.0, 30.0, 30.0)
    return YoloDetection(
        frame_index=frame_index,
        class_id=class_id,
        class_name="test",
        bbox=bbox,
        confidence=conf,
    )


def test_ball_detections_converted(tmp_path):
    dets = [make_yolo(0, class_id=0), make_yolo(1, class_id=0)]
    ctx = make_ctx(dets, tmp_path)
    result = YoloToFusedStage().run(ctx)
    assert len(result.artifacts.fused_detections) == 2
    assert all(f.source == DetectionSource.FUSED_YOLO_PRIMARY
               for f in result.artifacts.fused_detections)


def test_non_ball_classes_excluded(tmp_path):
    dets = [make_yolo(0, class_id=1), make_yolo(0, class_id=2), make_yolo(0, class_id=0)]
    ctx = make_ctx(dets, tmp_path)
    result = YoloToFusedStage().run(ctx)
    assert len(result.artifacts.fused_detections) == 1


def test_empty_yolo_produces_empty_fused(tmp_path):
    ctx = make_ctx([], tmp_path)
    result = YoloToFusedStage().run(ctx)
    assert result.artifacts.fused_detections == ()
