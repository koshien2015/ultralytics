import pytest

from pitching.domain.entities.detection import DetectionSource, FusedDetection
from pitching.infra.ml.tracker import NearestNeighborTracker


def make_det(frame_index: int, cx: float, cy: float) -> FusedDetection:
    return FusedDetection(
        frame_index=frame_index,
        class_id=0,
        bbox=None,
        center_x=cx,
        center_y=cy,
        confidence=0.9,
        source=DetectionSource.FUSED_YOLO_PRIMARY,
    )


def test_new_detection_creates_track():
    tracker = NearestNeighborTracker()
    det = make_det(0, 100.0, 200.0)
    tracks = tracker.update((det,), frame_index=0)
    assert len(tracks) == 1
    assert tracks[0].points[0].x == pytest.approx(100.0)


def test_consecutive_detections_extend_track():
    tracker = NearestNeighborTracker()
    tracker.update((make_det(0, 100.0, 200.0),), frame_index=0)
    tracker.update((make_det(1, 110.0, 200.0),), frame_index=1)
    tracks = tracker.update((make_det(2, 120.0, 200.0),), frame_index=2)
    assert len(tracks) == 1
    assert len(tracks[0].points) == 3


def test_distant_detection_creates_new_track():
    tracker = NearestNeighborTracker(max_match_distance_px=50.0)
    tracker.update((make_det(0, 100.0, 100.0),), frame_index=0)
    tracks = tracker.update((make_det(1, 500.0, 500.0),), frame_index=1)
    # 前のトラック + 新しいトラック
    assert len(tracks) == 2


def test_track_expires_after_fade_frames():
    tracker = NearestNeighborTracker(fade_frames=2)
    tracker.update((make_det(0, 100.0, 100.0),), frame_index=0)
    tracker.update((), frame_index=1)
    tracker.update((), frame_index=2)
    tracks = tracker.update((), frame_index=3)  # fade_frames=2 を超えて削除
    assert len(tracks) == 0


def test_max_trajectory_length():
    tracker = NearestNeighborTracker(max_trajectory_length=3)
    for i in range(10):
        tracker.update((make_det(i, float(i * 5), 100.0),), frame_index=i)
    tracks = tracker.update((), frame_index=10)
    assert len(tracks[0].points) <= 3
