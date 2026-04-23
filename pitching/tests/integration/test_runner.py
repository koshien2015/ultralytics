import dataclasses
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from pitching.config.schema import PipelineConfig
from pitching.infra.storage.checkpoint import JsonCheckpointStore
from pitching.pipeline.context import PipelineArtifacts, PipelineContext
from pitching.pipeline.runner import PipelineRunner


def make_ctx(tmp_path: Path) -> PipelineContext:
    return PipelineContext(
        config=PipelineConfig(),
        video_path=tmp_path / "input.mp4",
        output_dir=tmp_path / "out",
        fps=30.0,
        total_frames=10,
        video_size=(100, 100),
    )


class CountingStage:
    """実行回数を記録するダミーステージ。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.run_count = 0

    def run(self, ctx: PipelineContext) -> PipelineContext:
        self.run_count += 1
        return ctx


def test_all_stages_run_in_order(tmp_path):
    ctx = make_ctx(tmp_path)
    s1, s2, s3 = CountingStage("s1"), CountingStage("s2"), CountingStage("s3")
    runner = PipelineRunner([s1, s2, s3])
    runner.run(ctx)
    assert s1.run_count == 1
    assert s2.run_count == 1
    assert s3.run_count == 1


def test_stop_after_skips_remaining(tmp_path):
    ctx = make_ctx(tmp_path)
    s1, s2, s3 = CountingStage("s1"), CountingStage("s2"), CountingStage("s3")
    runner = PipelineRunner([s1, s2, s3], stop_after="s2")
    runner.run(ctx)
    assert s1.run_count == 1
    assert s2.run_count == 1
    assert s3.run_count == 0


def test_no_checkpoint_store_runs_all(tmp_path):
    ctx = make_ctx(tmp_path)
    stages = [CountingStage(f"s{i}") for i in range(5)]
    runner = PipelineRunner(stages)
    runner.run(ctx)
    assert all(s.run_count == 1 for s in stages)
