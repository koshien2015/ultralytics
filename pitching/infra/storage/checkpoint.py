from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pitching.infra.storage.json_store import load_json, save_json

logger = logging.getLogger(__name__)


class JsonCheckpointStore:
    """ステージ名をキーに中間成果物を JSON で保存・ロードする。"""

    def __init__(self, output_dir: Path, registry: dict[str, type]) -> None:
        self._dir = output_dir / "_checkpoints"
        self._registry = registry

    def _path(self, stage_name: str) -> Path:
        return self._dir / f"{stage_name}.json"

    def exists(self, stage_name: str) -> bool:
        return self._path(stage_name).exists()

    def save(self, stage_name: str, data: Any) -> Path:
        path = self._path(stage_name)
        save_json(data, path, self._registry)
        logger.info("Checkpoint saved: %s", path)
        return path

    def load(self, stage_name: str, root_type: type) -> Any:
        path = self._path(stage_name)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        data = load_json(path, root_type, self._registry)
        logger.info("Checkpoint loaded: %s", path)
        return data
