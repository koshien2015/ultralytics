import pytest
from pitching.domain.entities.bbox import BBox
from pitching.domain.entities.detection import (
    DetectionSource,
    DiffDetection,
    FusedDetection,
    YoloDetection,
)
from pitching.domain.services.fusion_policy import fuse_frame

DEFAULTS = dict(
    yolo_min_conf=0.3,
    diff_area_min=10.0,
    diff_area_max=5000.0,
    max_jump_px=80.0,
    min_direction_cos=0.3,
)


def make_yolo(conf=0.8, class_id=0, cx=100.0, cy=100.0) -> YoloDetection:
    bbox = BBox(cx - 10, cy - 10, cx + 10, cy + 10)
    return YoloDetection(frame_index=0, class_id=class_id, class_name="ball", bbox=bbox, confidence=conf)


def make_diff(cx=100.0, cy=100.0, area=50.0) -> DiffDetection:
    return DiffDetection(frame_index=0, center_x=cx, center_y=cy, area=area)


def test_yolo_primary_when_confident():
    result = fuse_frame(0, (make_yolo(conf=0.9),), (), (), (), **DEFAULTS)
    assert result is not None
    assert result.source == DetectionSource.FUSED_YOLO_PRIMARY


def test_yolo_below_conf_falls_to_diff():
    result = fuse_frame(0, (make_yolo(conf=0.1),), (make_diff(),), (), (), **DEFAULTS)
    assert result is not None
    assert result.source == DetectionSource.FUSED_DIFF_FALLBACK


def test_diff_area_too_small_excluded():
    result = fuse_frame(0, (), (make_diff(area=1.0),), (), (), **DEFAULTS)
    assert result is None


def test_diff_area_too_large_excluded():
    result = fuse_frame(0, (), (make_diff(area=99999.0),), (), (), **DEFAULTS)
    assert result is None


def test_diff_inside_body_bbox_excluded():
    body = BBox(80.0, 80.0, 120.0, 120.0)
    result = fuse_frame(0, (), (make_diff(cx=100.0, cy=100.0),), (), (body,), **DEFAULTS)
    assert result is None


def test_diff_too_far_from_recent_excluded():
    last = FusedDetection(
        frame_index=0, class_id=0, bbox=None,
        center_x=0.0, center_y=0.0, confidence=0.0,
        source=DetectionSource.FUSED_YOLO_PRIMARY,
    )
    result = fuse_frame(1, (), (make_diff(cx=500.0, cy=500.0),), (last,), (), **DEFAULTS)
    assert result is None


def test_no_detection_returns_none():
    result = fuse_frame(0, (), (), (), (), **DEFAULTS)
    assert result is None
