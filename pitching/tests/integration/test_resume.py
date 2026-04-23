"""
PipelineRunner の resume / stop_after 機能の integration テスト。
実際の動画や ML モデルは使わず、ダミーステージで検証する。
"""
import dataclasses
from pathlib import Path

import pytest

from pitching.config.schema import PipelineConfig
from pitching.domain.entities.bbox import BBox
from pitching.domain.entities.detection import DetectionSource, FusedDetection
from pitching.infra.storage.checkpoint import JsonCheckpointStore
from pitching.pipeline.context import PipelineArtifacts, PipelineContext
from pitching.pipeline.runner import PipelineRunner

# checkpoint の JSON 復元に必要な型
REGISTRY = {
    "BBox": BBox,
    "DetectionSource": DetectionSource,
    "FusedDetection": FusedDetection,
    "PipelineArtifacts": PipelineArtifacts,
}


def make_ctx(tmp_path: Path) -> PipelineContext:
    return PipelineContext(
        config=PipelineConfig(),
        video_path=tmp_path / "v.mp4",
        output_dir=tmp_path / "out",
        fps=30.0,
        total_frames=10,
        video_size=(100, 100),
    )


class TaggingStage:
    """実行されると artifacts に自分の名前を記録するダミーステージ。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.run_count = 0

    def run(self, ctx: PipelineContext) -> PipelineContext:
        self.run_count += 1
        # ダミーの FusedDetection に stage 名をエンコード
        det = FusedDetection(
            frame_index=self.run_count,
            class_id=0,
            bbox=None,
            center_x=float(self.run_count),
            center_y=0.0,
            confidence=0.0,
            source=DetectionSource.FUSED_YOLO_PRIMARY,
        )
        new_artifacts = dataclasses.replace(
            ctx.artifacts,
            fused_detections=ctx.artifacts.fused_detections + (det,),
        )
        return dataclasses.replace(ctx, artifacts=new_artifacts)


# --- stop_after ---

def test_stop_after_saves_checkpoint_and_halts(tmp_path):
    store = JsonCheckpointStore(tmp_path / "out", REGISTRY)
    s1, s2, s3 = TaggingStage("s1"), TaggingStage("s2"), TaggingStage("s3")

    runner = PipelineRunner([s1, s2, s3], checkpoint_store=store, stop_after="s2")
    runner.run(make_ctx(tmp_path))

    assert s1.run_count == 1
    assert s2.run_count == 1
    assert s3.run_count == 0              # 停止済み

    assert store.exists("s1")
    assert store.exists("s2")
    assert not store.exists("s3")         # 実行されていないのでなし


# --- resume_from ---

def test_resume_from_skips_earlier_stages(tmp_path):
    store = JsonCheckpointStore(tmp_path / "out", REGISTRY)

    # 1st run: s1 と s2 のチェックポイントを保存
    s1a, s2a, s3a = TaggingStage("s1"), TaggingStage("s2"), TaggingStage("s3")
    PipelineRunner([s1a, s2a, s3a], checkpoint_store=store).run(make_ctx(tmp_path))

    # 2nd run: s2 から再実行
    s1b, s2b, s3b = TaggingStage("s1"), TaggingStage("s2"), TaggingStage("s3")
    PipelineRunner(
        [s1b, s2b, s3b], checkpoint_store=store, resume_from="s2"
    ).run(make_ctx(tmp_path))

    assert s1b.run_count == 0   # スキップ
    assert s2b.run_count == 1   # 再実行
    assert s3b.run_count == 1   # 再実行


def test_resume_loads_previous_checkpoint_artifacts(tmp_path):
    """resume 時に前ステージのチェックポイントから artifacts が復元されるか検証。"""
    store = JsonCheckpointStore(tmp_path / "out", REGISTRY)

    # 1st run: s1 → s2 と実行してチェックポイントを保存
    s1a, s2a = TaggingStage("s1"), TaggingStage("s2")
    PipelineRunner([s1a, s2a], checkpoint_store=store).run(make_ctx(tmp_path))

    # s1 のチェックポイントに 1 件の FusedDetection が保存されているはず
    loaded = store.load("s1", PipelineArtifacts)
    assert len(loaded.fused_detections) == 1

    # 2nd run: s2 から再実行 → s1 のチェックポイントが artifacts にロードされる
    s1b, s2b = TaggingStage("s1"), TaggingStage("s2")
    final = PipelineRunner(
        [s1b, s2b], checkpoint_store=store, resume_from="s2"
    ).run(make_ctx(tmp_path))

    # s2 が実行され、s1 のチェックポイント (1件) に s2 の追加 (1件) で計 2 件
    assert len(final.artifacts.fused_detections) == 2


def test_stop_after_then_resume(tmp_path):
    """stop_after で途中停止 → resume_from で続きを実行。"""
    store = JsonCheckpointStore(tmp_path / "out", REGISTRY)
    s1, s2, s3 = TaggingStage("s1"), TaggingStage("s2"), TaggingStage("s3")

    # 1st: s2 で停止
    PipelineRunner([s1, s2, s3], checkpoint_store=store, stop_after="s2").run(make_ctx(tmp_path))
    assert s3.run_count == 0

    # 2nd: s3 から再実行
    s1b, s2b, s3b = TaggingStage("s1"), TaggingStage("s2"), TaggingStage("s3")
    PipelineRunner([s1b, s2b, s3b], checkpoint_store=store, resume_from="s3").run(make_ctx(tmp_path))

    assert s1b.run_count == 0
    assert s2b.run_count == 0
    assert s3b.run_count == 1
