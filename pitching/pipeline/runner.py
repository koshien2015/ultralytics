from __future__ import annotations

import dataclasses
import logging
from typing import List, Optional

from pitching.infra.storage.checkpoint import JsonCheckpointStore
from pitching.pipeline.context import PipelineArtifacts, PipelineContext
from pitching.pipeline.stage import Stage

logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    ステージリストを順番に実行する。
    checkpoint_store があれば各ステージ完了後に artifacts を保存し、
    resume_from が指定されたステージ以前はロードしてスキップする。
    """

    def __init__(
        self,
        stages: List[Stage],
        checkpoint_store: Optional[JsonCheckpointStore] = None,
        resume_from: Optional[str] = None,
        stop_after: Optional[str] = None,
    ) -> None:
        self._stages = stages
        self._store = checkpoint_store
        self._resume_from = resume_from
        self._stop_after = stop_after

    def run(self, ctx: PipelineContext) -> PipelineContext:
        resuming = self._resume_from is not None

        for stage in self._stages:
            if resuming:
                if self._store and self._store.exists(stage.name):
                    logger.info("[%s] checkpoint found, loading and skipping", stage.name)
                    artifacts = self._store.load(stage.name, PipelineArtifacts)
                    ctx = dataclasses.replace(ctx, artifacts=artifacts)
                else:
                    logger.info("[%s] no checkpoint, running stage", stage.name)
                    resuming = False

                if stage.name == self._resume_from:
                    resuming = False
                    logger.info("[%s] resume point reached, executing from next stage", stage.name)
                    continue

                continue

            logger.info("[%s] running", stage.name)
            ctx = stage.run(ctx)

            if self._store:
                self._store.save(stage.name, ctx.artifacts)
                logger.info("[%s] checkpoint saved", stage.name)

            if self._stop_after and stage.name == self._stop_after:
                logger.info("Stopping after stage: %s", stage.name)
                break

        return ctx
