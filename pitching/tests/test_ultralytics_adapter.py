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


def test_role_box_selection_wins():
    """役割bboxで投手が分かるなら、大きさに関係なくそれを採る。"""
    small = make_person(1, 10.0)
    large = make_person(2, 100.0)
    result = {"persons": [small, large], "roles": {1: "pitcher", 2: "catcher"}}

    person, mode = _select_pitcher(result, StubPoseModule, 0.3)

    assert person is small
    assert mode == "role_bbox"


def test_largest_bbox_is_the_fallback():
    small = make_person(1, 10.0)
    large = make_person(2, 100.0)
    result = {"persons": [small, large], "roles": {}}

    person, mode = _select_pitcher(result, StubPoseModule, 0.3)

    assert person is large
    assert mode == "largest_bbox"


def test_no_person_is_reported():
    person, mode = _select_pitcher({"persons": [], "roles": {}}, StubPoseModule, 0.3)

    assert person is None
    assert mode == "none"


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
