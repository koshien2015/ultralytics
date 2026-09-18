"""ultralytics アダプタのうち、YOLO を必要としない部分のテスト。

推論そのものは実素材とGPUが要るので検証しない（shared/tests/test_pose.py と同じ方針）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pitching.adapters.ultralytics_adapter import (
    ExtractionError,
    _select_pitcher,
    _to_pose_frame,
    extract_pose_series,
)

KEYPOINT_NAMES = ("nose", "left_shoulder", "right_shoulder")


@dataclass
class StubPerson:
    track_id: int | None
    keypoints: np.ndarray
    scores: np.ndarray


class StubPoseModule:
    """shared/pose.py の代わり。keypoint_bbox だけ使う。"""

    KEYPOINT_NAMES = KEYPOINT_NAMES

    @staticmethod
    def keypoint_bbox(keypoints, scores, min_score):
        confident = np.asarray(scores) >= min_score
        if not confident.any():
            return None
        points = np.asarray(keypoints)[confident]
        return (
            float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max()),
        )


def make_person(track_id, size, score=0.9):
    keypoints = np.array([[0.0, 0.0], [size, 0.0], [size, size]])
    return StubPerson(track_id, keypoints, np.full(3, score))


def test_pitcher_selection_is_delegated_to_shared_pose():
    """人物の選び方は shared/pose.py に1つだけ置く（track.py の書き出しと揃えるため）。"""
    sentinel = (make_person(7, 50.0), "role_bbox")

    class StubWithSelector(StubPoseModule):
        @staticmethod
        def select_person(result, role, min_score):
            assert role == "pitcher"
            return sentinel

    assert _select_pitcher({"persons": [], "roles": {}}, StubWithSelector, 0.3) is sentinel


def test_missing_person_becomes_missing_keypoints():
    frame = _to_pose_frame(None, KEYPOINT_NAMES, 42, 60.0)

    assert frame.frame_index == 42
    assert frame.timestamp_sec == pytest.approx(42 / 60.0)
    assert all(frame.get(name) is None for name in KEYPOINT_NAMES)


def test_person_is_converted_to_keypoints():
    frame = _to_pose_frame(make_person(1, 20.0), KEYPOINT_NAMES, 10, 60.0)

    shoulder = frame.get("left_shoulder")
    assert (shoulder.x, shoulder.y) == pytest.approx((20.0, 0.0))
    assert frame.confidence_of("nose") == pytest.approx(0.9)


def test_missing_video_is_reported(pitch_config):
    config = pitch_config.model_copy(deep=True)
    config.video_path = "does_not_exist.mp4"

    with pytest.raises(ExtractionError):
        extract_pose_series(config)
