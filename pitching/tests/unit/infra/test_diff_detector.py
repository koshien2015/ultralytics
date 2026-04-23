import numpy as np
import pytest

from pitching.infra.ml.diff_detector import FrameDiffDetector


def make_frame(h: int = 100, w: int = 100) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_frame_with_blob(cx: int, cy: int, r: int = 5, h: int = 100, w: int = 100) -> np.ndarray:
    frame = make_frame(h, w)
    import cv2
    cv2.circle(frame, (cx, cy), r, (255, 255, 255), -1)
    return frame


def test_no_motion_returns_empty():
    detector = FrameDiffDetector(area_min=1.0, area_max=99999.0)
    blank = make_frame()
    result = detector.detect_on_pair(blank, blank, blank, frame_index=0)
    assert result == ()


def test_moving_blob_detected():
    detector = FrameDiffDetector(threshold=30, area_min=1.0, area_max=99999.0)
    prev = make_frame_with_blob(30, 50)
    curr = make_frame_with_blob(50, 50)
    nxt = make_frame_with_blob(70, 50)
    result = detector.detect_on_pair(prev, curr, nxt, frame_index=5)
    assert len(result) > 0
    assert result[0].frame_index == 5


def test_area_filter_excludes_small():
    detector = FrameDiffDetector(threshold=30, area_min=10000.0, area_max=99999.0)
    prev = make_frame_with_blob(30, 50, r=3)
    curr = make_frame_with_blob(50, 50, r=3)
    nxt = make_frame_with_blob(70, 50, r=3)
    result = detector.detect_on_pair(prev, curr, nxt, frame_index=0)
    assert result == ()


def test_make_enhanced_frame_shape():
    detector = FrameDiffDetector()
    frame = make_frame(80, 120)
    enhanced = detector.make_enhanced_frame(frame, frame, frame)
    assert enhanced.shape == frame.shape
