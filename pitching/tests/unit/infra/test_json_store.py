import pytest
from pathlib import Path

from pitching.domain.entities.bbox import BBox
from pitching.domain.entities.detection import (
    DetectionSource,
    FusedDetection,
    YoloDetection,
)
from pitching.domain.entities.pitch import Pitch, PitchTrajectoryPoint, ReleaseEvent
from pitching.infra.storage.json_store import load_json, save_json

REGISTRY = {
    "BBox": BBox,
    "YoloDetection": YoloDetection,
    "FusedDetection": FusedDetection,
    "DetectionSource": DetectionSource,
    "Pitch": Pitch,
    "PitchTrajectoryPoint": PitchTrajectoryPoint,
    "ReleaseEvent": ReleaseEvent,
}


def test_roundtrip_fused_detection(tmp_path):
    original = FusedDetection(
        frame_index=5,
        class_id=0,
        bbox=BBox(10.0, 20.0, 30.0, 40.0),
        center_x=20.0,
        center_y=30.0,
        confidence=0.85,
        source=DetectionSource.FUSED_YOLO_PRIMARY,
    )
    path = tmp_path / "det.json"
    save_json(original, path)
    restored = load_json(path, FusedDetection, REGISTRY)

    assert restored == original


def test_roundtrip_fused_detection_no_bbox(tmp_path):
    original = FusedDetection(
        frame_index=10,
        class_id=0,
        bbox=None,
        center_x=55.0,
        center_y=77.0,
        confidence=0.0,
        source=DetectionSource.FUSED_DIFF_FALLBACK,
    )
    path = tmp_path / "det_no_bbox.json"
    save_json(original, path)
    restored = load_json(path, FusedDetection, REGISTRY)

    assert restored == original


def test_roundtrip_pitch(tmp_path):
    release = ReleaseEvent(pitch_id=1, release_frame=30, release_time_sec=1.0)
    point = PitchTrajectoryPoint(
        frame_index=35,
        elapsed_time_sec=0.167,
        x_norm=0.1,
        y_norm=0.6,
        z=0.167,
        source=DetectionSource.FUSED_YOLO_PRIMARY,
    )
    original = Pitch(pitch_id=1, release=release, trajectory=(point,), is_strike=True)

    path = tmp_path / "pitch.json"
    save_json(original, path)
    restored = load_json(path, Pitch, REGISTRY)

    assert restored == original
    assert restored.trajectory[0].source == DetectionSource.FUSED_YOLO_PRIMARY


def test_schema_version_mismatch_raises(tmp_path):
    import json
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 99, "data": {}}))

    with pytest.raises(ValueError, match="Schema version mismatch"):
        load_json(path, FusedDetection, REGISTRY)
