from __future__ import annotations

from pathlib import Path

import yaml

from pitching.config.schema import PipelineConfig


def load_config(path: Path | None = None) -> PipelineConfig:
    """
    YAML ファイルから PipelineConfig を読み込む。
    path が None の場合はデフォルト値のみで構成する。
    """
    if path is None:
        return PipelineConfig()

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return PipelineConfig.model_validate(raw)
