import dataclasses
from pathlib import Path

import pytest

from pitching.config.schema import FusionConfig, FrameDiffConfig, PipelineConfig
from pitching.domain.entities.bbox import BBox
from pitching.domain.entities.detection import (
    DetectionSource,
    DiffDetection,
    YoloDetection,
)
from pitching.pipeline.context import PipelineArtifacts, PipelineContext
from pitching.pipeline.stages.fusion_stage import FusionStage

FUSION_CFG = FusionConfig(max_jump_px=80.0, min_direction_cos=0.3)
DIFF_CFG = FrameDiffConfig(area_min=10.0, area_max=5000.0)
YOLO_MIN_CONF = 0.3


def make_stage() -> FusionStage:
    return FusionStage(FUSION_CFG, DIFF_CFG, YOLO_MIN_CONF)


def make_ctx(yolo_dets, diff_dets, tmp_path, total_frames=10):
    artifacts = PipelineArtifacts(
        yolo_detections=tuple(yolo_dets),
        diff_detections=tuple(diff_dets),
    )
    return PipelineContext(
        config=PipelineConfig(),
        video_path=tmp_path / "v.mp4",
        output_dir=tmp_path / "out",
        fps=30.0,
        total_frames=total_frames,
        video_size=(100, 100),
        artifacts=artifacts,
    )


def make_yolo_ball(frame_index, cx=50.0, cy=50.0, conf=0.9):
    bbox = BBox(cx - 5, cy - 5, cx + 5, cy + 5)
    return YoloDetection(
        frame_index=frame_index, class_id=0,
        class_name="ball", bbox=bbox, confidence=conf,
    )


def make_diff(frame_index, cx=50.0, cy=50.0, area=100.0):
    return DiffDetection(frame_index=frame_index, center_x=cx, center_y=cy, area=area)


# --- YOLO 優先 ---

def test_yolo_detection_used_when_confident(tmp_path):
    ctx = make_ctx([make_yolo_ball(0, conf=0.9)], [], tmp_path)
    result = make_stage().run(ctx)
    assert len(result.artifacts.fused_detections) == 1
    assert result.artifacts.fused_detections[0].source == DetectionSource.FUSED_YOLO_PRIMARY


def test_low_conf_yolo_falls_back_to_diff(tmp_path):
    yolo = [make_yolo_ball(0, conf=0.1)]  # 閾値以下
    diff = [make_diff(0)]
    ctx = make_ctx(yolo, diff, tmp_path)
    result = make_stage().run(ctx)
    assert result.artifacts.fused_detections[0].source == DetectionSource.FUSED_DIFF_FALLBACK


# --- 差分補完 ---

def test_diff_fills_gap_where_yolo_misses(tmp_path):
    # フレーム 0 は YOLO、フレーム 1 は YOLO なし→差分で補完
    yolo = [make_yolo_ball(0, cx=50.0, cy=50.0)]
    diff = [make_diff(1, cx=55.0, cy=50.0)]  # 直前から 5px → 補完される
    ctx = make_ctx(yolo, diff, tmp_path, total_frames=2)
    result = make_stage().run(ctx)
    sources = {f.source for f in result.artifacts.fused_detections}
    assert DetectionSource.FUSED_YOLO_PRIMARY in sources
    assert DetectionSource.FUSED_DIFF_FALLBACK in sources


def test_diff_too_far_from_previous_rejected(tmp_path):
    yolo = [make_yolo_ball(0, cx=50.0, cy=50.0)]
    diff = [make_diff(1, cx=500.0, cy=500.0)]  # 遠すぎて除外
    ctx = make_ctx(yolo, diff, tmp_path, total_frames=2)
    result = make_stage().run(ctx)
    fused = result.artifacts.fused_detections
    # フレーム 1 の補完が除外されているので YOLO のフレーム 0 だけ
    assert len(fused) == 1
    assert fused[0].frame_index == 0


def test_diff_inside_body_bbox_rejected(tmp_path):
    # 人物クラス (class_id=2) の bbox 内にある差分は除外
    body_det = YoloDetection(
        frame_index=1, class_id=2, class_name="batter",
        bbox=BBox(40.0, 40.0, 60.0, 60.0), confidence=0.95,
    )
    yolo = [make_yolo_ball(0, cx=50.0, cy=50.0), body_det]
    diff = [make_diff(1, cx=50.0, cy=50.0)]  # 人物 bbox のど真ん中
    ctx = make_ctx(yolo, diff, tmp_path, total_frames=2)
    result = make_stage().run(ctx)
    # フレーム 1 の差分補完が除外されているはず
    frame1_fused = [f for f in result.artifacts.fused_detections if f.frame_index == 1]
    assert len(frame1_fused) == 0


def test_no_detections_produces_empty(tmp_path):
    ctx = make_ctx([], [], tmp_path)
    result = make_stage().run(ctx)
    assert result.artifacts.fused_detections == ()


# --- source が軌跡 JSON まで伝播するか ---

def test_diff_fallback_source_propagated(tmp_path):
    yolo = [make_yolo_ball(0, cx=50.0)]
    diff = [make_diff(1, cx=52.0)]
    ctx = make_ctx(yolo, diff, tmp_path, total_frames=2)
    result = make_stage().run(ctx)
    diff_fused = [f for f in result.artifacts.fused_detections
                  if f.source == DetectionSource.FUSED_DIFF_FALLBACK]
    assert len(diff_fused) == 1
    assert diff_fused[0].frame_index == 1
