"""
Phase 7: PoseEstimator の差し込み口が正しく機能することを検証。

- NullPoseEstimator は空 tuple を返す（stub）
- カスタム実装を差し込んでも PoseEstimationStage / Pipeline を変更不要
- Protocol 準拠の実装であれば型チェックなしに差し替え可能
"""
import dataclasses
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pitching.config.schema import PipelineConfig
from pitching.domain.entities.frame import FrameMeta
from pitching.domain.entities.pose import Keypoint, PoseFrame
from pitching.infra.ml.null_pose_estimator import NullPoseEstimator
from pitching.pipeline.context import PipelineArtifacts, PipelineContext
from pitching.pipeline.stages.pose_estimation_stage import PoseEstimationStage


# --- テスト用のカスタム PoseEstimator ---

class FakePoseEstimator:
    """
    MediaPipe / YOLOv8-Pose の代わりとなるフェイク実装。
    PoseEstimator Protocol に準拠する最小実装。
    """

    def estimate(self, frame: np.ndarray, frame_index: int) -> Tuple[PoseFrame, ...]:
        return (
            PoseFrame(
                frame_index=frame_index,
                person_id=0,
                keypoints=(
                    Keypoint(name="right_wrist", x=100.0, y=200.0, confidence=0.9),
                    Keypoint(name="right_elbow", x=120.0, y=180.0, confidence=0.85),
                ),
            ),
        )


# --- ヘルパー ---

def make_ctx(tmp_path: Path, video_path: Path) -> PipelineContext:
    return PipelineContext(
        config=PipelineConfig(),
        video_path=video_path,
        output_dir=tmp_path / "out",
        fps=30.0,
        total_frames=3,
        video_size=(100, 100),
    )


def fake_video_frames(n: int = 3):
    """VideoReader.__iter__ を差し替える疑似フレーム列。"""
    for i in range(n):
        meta = FrameMeta(frame_index=i, timestamp=i / 30.0, width=100, height=100)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        yield meta, frame


# --- NullPoseEstimator のテスト ---

def test_null_estimator_produces_empty_pose_frames(tmp_path):
    """NullPoseEstimator（stub）を使うと pose_frames が空のまま。"""
    video_path = tmp_path / "v.mp4"
    ctx = make_ctx(tmp_path, video_path)
    stage = PoseEstimationStage(NullPoseEstimator())

    with patch("pitching.pipeline.stages.pose_estimation_stage.VideoReader") as mock_reader:
        mock_reader.return_value.__enter__.return_value = fake_video_frames(3)
        result = stage.run(ctx)

    assert result.artifacts.pose_frames == ()


# --- カスタム実装への差し替えテスト ---

def test_custom_estimator_swapped_without_pipeline_change(tmp_path):
    """
    FakePoseEstimator を差し込んでも PoseEstimationStage 自体は変更不要。
    フレームごとに PoseFrame が artifacts に積まれることを確認。
    """
    video_path = tmp_path / "v.mp4"
    ctx = make_ctx(tmp_path, video_path)
    stage = PoseEstimationStage(FakePoseEstimator())

    with patch("pitching.pipeline.stages.pose_estimation_stage.VideoReader") as mock_reader:
        mock_reader.return_value.__enter__.return_value = fake_video_frames(3)
        result = stage.run(ctx)

    assert len(result.artifacts.pose_frames) == 3
    assert result.artifacts.pose_frames[0].frame_index == 0
    assert result.artifacts.pose_frames[2].frame_index == 2


def test_keypoints_correctly_stored(tmp_path):
    """差し込んだ実装のキーポイントが artifacts に正しく保存される。"""
    video_path = tmp_path / "v.mp4"
    ctx = make_ctx(tmp_path, video_path)
    stage = PoseEstimationStage(FakePoseEstimator())

    with patch("pitching.pipeline.stages.pose_estimation_stage.VideoReader") as mock_reader:
        mock_reader.return_value.__enter__.return_value = fake_video_frames(1)
        result = stage.run(ctx)

    pose = result.artifacts.pose_frames[0]
    assert len(pose.keypoints) == 2
    wrist = next(k for k in pose.keypoints if k.name == "right_wrist")
    assert wrist.x == pytest.approx(100.0)
    assert wrist.confidence == pytest.approx(0.9)


def test_null_and_custom_produce_different_results(tmp_path):
    """NullPoseEstimator と FakePoseEstimator の結果が異なることで差し替えが有効と確認。"""
    video_path = tmp_path / "v.mp4"
    ctx = make_ctx(tmp_path, video_path)

    with patch("pitching.pipeline.stages.pose_estimation_stage.VideoReader") as mock_reader:
        mock_reader.return_value.__enter__.return_value = fake_video_frames(2)
        null_result = PoseEstimationStage(NullPoseEstimator()).run(ctx)

    with patch("pitching.pipeline.stages.pose_estimation_stage.VideoReader") as mock_reader:
        mock_reader.return_value.__enter__.return_value = fake_video_frames(2)
        fake_result = PoseEstimationStage(FakePoseEstimator()).run(ctx)

    assert null_result.artifacts.pose_frames == ()
    assert len(fake_result.artifacts.pose_frames) == 2
