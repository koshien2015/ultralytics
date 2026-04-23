from __future__ import annotations

from typing import Protocol

from pitching.pipeline.context import PipelineContext


class Stage(Protocol):
    """パイプラインステージの interface。各ステージは入力 ctx を受け取り更新した ctx を返す。"""

    name: str

    def run(self, ctx: PipelineContext) -> PipelineContext:
        ...
