import numpy as np

from pitching.infra.ml.null_pose_estimator import NullPoseEstimator
from pitching.infra.ml.pose_estimator import PoseEstimator


def test_returns_empty_tuple():
    estimator = NullPoseEstimator()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = estimator.estimate(frame, frame_index=0)
    assert result == ()


def test_conforms_to_protocol():
    # Protocol への適合を静的チェックの代わりに動作確認
    estimator: PoseEstimator = NullPoseEstimator()  # type: ignore[assignment]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert estimator.estimate(frame, 0) == ()
