"""
UltralyticsYoloPoseEstimator のユニットテスト。
torch / ultralytics が未インストールの環境でも動くよう sys.modules をモックする。
"""
from __future__ import annotations

import sys
from typing import Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ultralytics と torch を事前にモック（import 時の依存解決を回避）
_mock_ultralytics = MagicMock()
sys.modules.setdefault("ultralytics", _mock_ultralytics)
sys.modules.setdefault("torch", MagicMock())

from pitching.domain.entities.pose import Keypoint, PoseFrame  # noqa: E402

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def make_fake_keypoints(n_persons: int = 1, conf: float = 0.9) -> np.ndarray:
    data = np.zeros((n_persons, 17, 3), dtype=np.float32)
    for p in range(n_persons):
        for k in range(17):
            data[p, k] = [float(k * 10 + p), float(k * 5 + p), conf]
    return data


def make_mock_model(n_persons: int = 1, conf: float = 0.9) -> MagicMock:
    mock_result = MagicMock()
    mock_result.keypoints.data = make_fake_keypoints(n_persons, conf)
    mock_model = MagicMock()
    mock_model.return_value = [mock_result]
    return mock_model


@pytest.fixture
def estimator_factory():
    """モデルロードをスキップして estimator を生成するファクトリ。"""
    from pitching.infra.ml.yolov8_pose_estimator import UltralyticsYoloPoseEstimator

    def _make(model=None, min_conf=0.3):
        est = UltralyticsYoloPoseEstimator.__new__(UltralyticsYoloPoseEstimator)
        est._model = model or make_mock_model()
        est._min_keypoint_confidence = min_conf
        return est

    return _make


# ---------------------------------------------------------------------------

def test_returns_pose_frames_for_each_person(estimator_factory):
    est = estimator_factory(model=make_mock_model(n_persons=2))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = est.estimate(frame, frame_index=5)

    assert len(result) == 2
    assert all(isinstance(pf, PoseFrame) for pf in result)
    assert result[0].frame_index == 5
    assert result[1].frame_index == 5
    assert result[0].person_id == 0
    assert result[1].person_id == 1


def test_each_pose_frame_has_17_keypoints(estimator_factory):
    est = estimator_factory()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = est.estimate(frame, frame_index=0)

    assert len(result[0].keypoints) == 17
    assert [kp.name for kp in result[0].keypoints] == COCO_KEYPOINTS


def test_keypoint_coordinates_and_confidence(estimator_factory):
    est = estimator_factory(model=make_mock_model(n_persons=1, conf=0.85))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = est.estimate(frame, frame_index=0)

    nose = result[0].keypoints[0]
    assert nose.name == "nose"
    assert nose.x == pytest.approx(0.0)
    assert nose.y == pytest.approx(0.0)
    assert nose.confidence == pytest.approx(0.85)

    right_wrist = result[0].keypoints[10]
    assert right_wrist.name == "right_wrist"
    assert right_wrist.x == pytest.approx(100.0)
    assert right_wrist.y == pytest.approx(50.0)


def test_empty_frame_returns_empty_tuple(estimator_factory):
    mock_result = MagicMock()
    mock_result.keypoints.data = np.zeros((0, 17, 3), dtype=np.float32)
    mock_model = MagicMock()
    mock_model.return_value = [mock_result]

    est = estimator_factory(model=mock_model)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = est.estimate(frame, frame_index=0)

    assert result == ()


def test_result_is_frozen(estimator_factory):
    est = estimator_factory()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = est.estimate(frame, frame_index=0)

    with pytest.raises((AttributeError, TypeError)):
        result[0].frame_index = 999  # type: ignore


def test_low_confidence_keypoints_still_included(estimator_factory):
    """全キーポイントは常に返す（フィルタは呼び出し側の責務）。"""
    est = estimator_factory(model=make_mock_model(conf=0.1), min_conf=0.5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = est.estimate(frame, frame_index=0)

    assert len(result[0].keypoints) == 17
