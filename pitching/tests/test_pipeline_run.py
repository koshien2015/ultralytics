"""run（extract → analyze → compare → viewer）のテスト。

pose.json を先に置いておけば抽出は飛ばされるので、YOLO 無しで全段を通せる。
"""

from __future__ import annotations

import json

import pytest

from pitching.adapters.json_adapter import save_pose_json
from pitching.cli import main
from pitching.config import PitchConfig
from pitching.pipeline import RunOptions, run
from pitching.tests.support import make_series

ANGLES = [
    175.0, 170.0, 160.0, 148.0, 135.0, 120.0, 108.0, 98.0, 90.0, 85.0,
    82.0, 88.0, 100.0, 118.0, 138.0, 158.0, 172.0, 176.0, 177.0, 177.0,
]


def place_pose(output_dir, pitch_id: str):
    """run が使い回す pose.json を先に置く。"""
    series = make_series(ANGLES, start=100)
    series.pitch_id = pitch_id
    return save_pose_json(series, output_dir / pitch_id / "pose.json")


def config_for(pitch_id: str, release_frame: int, label: str = "") -> PitchConfig:
    return PitchConfig.model_validate(
        {
            "pitch_id": pitch_id,
            "throwing_hand": "right",
            "batter_direction": "left",
            "fps": 60,
            "start_frame": 100,
            "end_frame": 119,
            "events": {"stride_foot_contact_frame": 104, "release_frame": release_frame},
            "result": {"label": label},
            "preprocessing": {"smoothing_window": 5},
        }
    )


def test_run_produces_everything_for_two_pitches(tmp_path):
    for pitch_id in ("first", "second"):
        place_pose(tmp_path, pitch_id)

    result = run(
        [config_for("first", 116), config_for("second", 113)],
        RunOptions(output_dir=tmp_path, charts=False),
    )

    assert len(result.analyses) == 2
    assert (tmp_path / "first" / "metrics.json").is_file()
    assert (tmp_path / "second" / "metrics.json").is_file()
    assert (tmp_path / "comparison" / "comparison.json").is_file()
    assert result.outputs["viewer"].is_file()


def test_run_reuses_existing_keypoints(tmp_path):
    """既に pose.json があれば推論しない（YOLO を呼ばずに完走する）。"""
    place_pose(tmp_path, "first")

    result = run([config_for("first", 116)], RunOptions(output_dir=tmp_path, charts=False))

    assert any("既存のキーポイントを使用" in message for message in result.messages)


def test_run_accepts_a_single_pitch(tmp_path):
    place_pose(tmp_path, "solo")

    result = run([config_for("solo", 116)], RunOptions(output_dir=tmp_path, charts=False))

    assert "comparison" not in result.outputs
    assert result.outputs["viewer"].is_file()


def test_run_rejects_three_pitches(tmp_path):
    with pytest.raises(ValueError):
        run(
            [config_for(name, 116) for name in ("a", "b", "c")],
            RunOptions(output_dir=tmp_path, charts=False),
        )


def test_run_aligns_fps_with_the_keypoints(tmp_path):
    """設定の fps がキーポイント側とずれていたら、動画側に合わせる。"""
    place_pose(tmp_path, "first")
    config = config_for("first", 116)
    config.fps = 30.0

    result = run([config], RunOptions(output_dir=tmp_path, charts=False))

    assert any("fps を 30 → 60" in message for message in result.messages)
    assert result.analyses[0].config.fps == 60.0


def test_cli_run_groups_options_per_video(tmp_path):
    """--release は直前の --video に付く。"""
    for pitch_id in ("first", "second"):
        place_pose(tmp_path, pitch_id)

    code = main([
        "run", "--output", str(tmp_path),
        "--video", "first.mp4", "--id", "first", "--release", "116", "--contact", "104",
        "--start", "100", "--end", "119", "--label", "1球目",
        "--video", "second.mp4", "--id", "second", "--release", "113", "--contact", "104",
        "--start", "100", "--end", "119", "--label", "2球目",
        "--no-charts",
    ])

    assert code == 0
    first = json.loads((tmp_path / "first" / "metrics.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "second" / "metrics.json").read_text(encoding="utf-8"))
    assert first["events"]["release"]["frame"] == 116
    assert second["events"]["release"]["frame"] == 113
    assert first["result"]["label"] == "1球目"


def test_cli_run_uses_video_name_as_default_id(tmp_path):
    place_pose(tmp_path, "throw_01")

    code = main([
        "run", "--output", str(tmp_path),
        "--video", str(tmp_path / "throw_01.mp4"),
        "--release", "116", "--contact", "104", "--start", "100", "--end", "119",
        "--no-charts", "--no-viewer",
    ])

    assert code == 0
    assert (tmp_path / "throw_01" / "metrics.json").is_file()


def test_cli_run_rejects_options_before_video(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["run", "--output", str(tmp_path), "--release", "116", "--video", "a.mp4"])

    assert "--video より後" in capsys.readouterr().err
