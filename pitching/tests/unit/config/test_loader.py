import textwrap
from pathlib import Path

import pytest

from pitching.config.loader import load_config
from pitching.config.schema import PipelineConfig


def test_load_defaults_when_no_path():
    cfg = load_config(None)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.yolo.min_confidence == 0.3
    assert cfg.fusion.max_jump_px == 80.0


def test_load_from_yaml(tmp_path):
    yaml_content = textwrap.dedent("""\
        yolo:
          min_confidence: 0.5
        fusion:
          max_jump_px: 120.0
    """)
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    cfg = load_config(config_file)
    assert cfg.yolo.min_confidence == pytest.approx(0.5)
    assert cfg.fusion.max_jump_px == pytest.approx(120.0)
    # 未指定項目はデフォルト値
    assert cfg.tracking.fade_frames == 60


def test_load_default_yaml():
    default_path = Path(__file__).parents[3] / "config" / "defaults" / "default.yaml"
    cfg = load_config(default_path)
    assert cfg.yolo.model_path == "shared/yolo8m_20251109.pt"
    assert cfg.rendering.glow_thickness == 15


def test_invalid_confidence_raises(tmp_path):
    yaml_content = "yolo:\n  min_confidence: 1.5\n"
    config_file = tmp_path / "bad.yaml"
    config_file.write_text(yaml_content)

    with pytest.raises(Exception):
        load_config(config_file)
