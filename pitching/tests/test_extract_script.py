"""tools/extract-pose.py（Docker 用の単体台本）のテスト。

推論は実素材とGPUが要るので対象外。書き出す JSON がこのパッケージで
読み戻せること、投手の選び方、依存が軽いままであることを見る。
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pitching.adapters.json_adapter import load_pose_json

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "extract-pose.py"


def load_script():
    """ファイル名にハイフンが入っていて import 文では読めないので、直接読む。"""
    spec = importlib.util.spec_from_file_location("extract_pose", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class StubPerson:
    track_id: int | None
    keypoints: np.ndarray
    scores: np.ndarray


class StubPoseModule:
    KEYPOINT_NAMES = ("nose", "left_shoulder", "right_shoulder")

    @staticmethod
    def keypoint_bbox(keypoints, scores, min_score):
        confident = np.asarray(scores) >= min_score
        if not confident.any():
            return None
        points = np.asarray(keypoints)[confident]
        return (
            float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max()),
        )


def make_person(track_id, size, score=0.9):
    return StubPerson(track_id, np.array([[0.0, 0.0], [size, 0.0], [size, size]]), np.full(3, score))


def imported_modules() -> set[str]:
    """台本が import しているモジュールの最上位名を集める。"""
    import ast

    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_script_avoids_heavy_dependencies():
    """推論環境（素の ultralytics コンテナ）で動くよう、重い依存を持ち込まない。"""
    heavy = {"pydantic", "matplotlib", "scipy", "pitching", "yaml", "pandas"}

    assert imported_modules() & heavy == set()


def test_script_only_needs_ultralytics_side_packages():
    """標準ライブラリ以外は cv2 / ultralytics / shared の pose だけ。"""
    import sys

    outside = {
        name for name in imported_modules()
        if name not in sys.stdlib_module_names and name != "__future__"
    }

    assert outside <= {"cv2", "ultralytics", "pose"}


def test_pitcher_selection_is_delegated_to_shared_pose():
    """選び方は shared/pose.py の select_person に任せる。"""
    module = load_script()
    sentinel = (make_person(7, 50.0), "role_bbox")

    class StubWithSelector(StubPoseModule):
        @staticmethod
        def select_person(result, role, min_score):
            return sentinel

    assert module.select_pitcher({}, StubWithSelector, 0.3) is sentinel


def test_missing_person_becomes_null_keypoints():
    module = load_script()

    payload = module.frame_payload(None, StubPoseModule.KEYPOINT_NAMES, 42, 60.0)

    assert payload["frame_index"] == 42
    assert payload["timestamp_sec"] == pytest.approx(0.7, abs=1e-3)
    assert payload["keypoints"]["nose"] == [None, None, 0.0]


def test_output_is_readable_by_the_package(tmp_path):
    """台本が書いた JSON を load_pose_json がそのまま読めること。"""
    module = load_script()
    names = StubPoseModule.KEYPOINT_NAMES
    payload = {
        "schema_version": module.SCHEMA_VERSION,
        "meta": {
            "pitch_id": "throw_01",
            "fps": 60.0,
            "video_path": "input/a.mov",
            "adapter": "ultralytics(extract-pose.py)",
            "notes": ["person_selection=['role_bbox']"],
        },
        "frames": [
            module.frame_payload(make_person(1, 20.0), names, index, 60.0)
            for index in range(100, 105)
        ],
    }
    path = tmp_path / "a.pose.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    series = load_pose_json(path)

    assert len(series) == 5
    assert series.pitch_id == "throw_01"
    assert series.fps == 60.0
    assert series.frames[0].frame_index == 100
    assert series.frames[0].get("left_shoulder") is not None
