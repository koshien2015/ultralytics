"""tools/*.py の「先頭の設定だけ書き換えて引数なしで実行する」使い方のテスト。

コマンドを毎回打たずに済ませるための入口なので、設定が引数に正しく
組み立てられること、未設定のときに理由が出ることを固定する。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def load(name: str):
    """ファイル名にハイフンが入っていて import 文では読めないので直接読む。"""
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRunAnalysis:
    def test_settings_become_run_arguments(self):
        module = load("run-analysis")
        module.PITCHES = [
            {"pose": "/tmp/a_pose.json", "release": 152, "contact": 140, "label": "1球目"},
            {"pose": "/tmp/b_pose.json", "release": 160},
        ]
        module.OUTPUT_DIR = "/tmp/out"
        module.THROWING_HAND = "left"
        module.BATTER_DIRECTION = "right"

        arguments = module.build_arguments()

        assert arguments[:6] == ["--output", "/tmp/out", "--hand", "left", "--batter", "right"]
        # イベントは直前の --pose に属する
        assert arguments.index("--pose") < arguments.index("--release")
        assert "152" in arguments and "160" in arguments
        assert "1球目" in arguments

    def test_relative_paths_are_resolved_from_the_script(self):
        module = load("run-analysis")

        resolved = module.resolve("../../output/x")

        assert Path(resolved).is_absolute()

    def test_absolute_paths_are_left_alone(self):
        module = load("run-analysis")

        assert module.resolve("/tmp/a.json") == "/tmp/a.json"

    def test_pitches_without_a_pose_file_are_skipped(self):
        module = load("run-analysis")
        module.PITCHES = [{"pose": "/tmp/a_pose.json"}, {"pose": None}]

        assert module.build_arguments().count("--pose") == 1


class TestMakeViewer:
    def test_settings_become_viewer_arguments(self):
        module = load("make-viewer")
        module.PITCHES = [
            {"pose": "/tmp/a_pose.json", "release": 152},
            {"pose": "/tmp/b_pose.json", "release": 160},
        ]
        module.OUTPUT = "/tmp/v.html"

        arguments = module.arguments_from_settings()

        assert arguments[:2] == ["/tmp/a_pose.json", "/tmp/b_pose.json"]
        assert "--output" in arguments and "/tmp/v.html" in arguments
        # 順に対応するので、投球の数だけ値が並ぶ
        assert arguments.count("--release") == 2

    def test_empty_settings_explain_what_to_do(self):
        module = load("make-viewer")
        module.PITCHES = []

        with pytest.raises(SystemExit) as error:
            module.arguments_from_settings()

        assert "PITCHES" in str(error.value)


class TestExtractPose:
    def test_settings_are_used_as_defaults(self):
        module = load("extract-pose")
        module.VIDEO = "/tmp/a.mov"
        module.START_FRAME = 100

        # 設定を既定値にした parser を作り直す（main は推論まで走るので呼ばない）
        source = (TOOLS / "extract-pose.py").read_text(encoding="utf-8")

        assert 'nargs="?", default=VIDEO' in source
        assert "default=START_FRAME" in source
        assert "VIDEO = None" in source

    def test_output_defaults_to_the_video_name(self):
        source = (TOOLS / "extract-pose.py").read_text(encoding="utf-8")

        assert '_pose.json' in source, "出力名の既定が {動画名}_pose.json でない"
