import pytest
from pathlib import Path

from pitching.domain.entities.bbox import BBox
from pitching.domain.entities.detection import DetectionSource, FusedDetection
from pitching.infra.storage.checkpoint import JsonCheckpointStore

REGISTRY = {
    "BBox": BBox,
    "FusedDetection": FusedDetection,
    "DetectionSource": DetectionSource,
}

SAMPLE = FusedDetection(
    frame_index=1,
    class_id=0,
    bbox=None,
    center_x=100.0,
    center_y=200.0,
    confidence=0.0,
    source=DetectionSource.FUSED_DIFF_FALLBACK,
)


def test_save_and_load(tmp_path):
    store = JsonCheckpointStore(tmp_path, REGISTRY)
    store.save("fusion", SAMPLE)
    restored = store.load("fusion", FusedDetection)
    assert restored == SAMPLE


def test_exists_before_and_after(tmp_path):
    store = JsonCheckpointStore(tmp_path, REGISTRY)
    assert not store.exists("fusion")
    store.save("fusion", SAMPLE)
    assert store.exists("fusion")


def test_load_missing_raises(tmp_path):
    store = JsonCheckpointStore(tmp_path, REGISTRY)
    with pytest.raises(FileNotFoundError):
        store.load("nonexistent", FusedDetection)


def test_overwrite_checkpoint(tmp_path):
    store = JsonCheckpointStore(tmp_path, REGISTRY)
    store.save("fusion", SAMPLE)

    updated = FusedDetection(
        frame_index=99,
        class_id=0,
        bbox=None,
        center_x=999.0,
        center_y=999.0,
        confidence=0.0,
        source=DetectionSource.FUSED_YOLO_PRIMARY,
    )
    store.save("fusion", updated)
    restored = store.load("fusion", FusedDetection)
    assert restored.frame_index == 99
