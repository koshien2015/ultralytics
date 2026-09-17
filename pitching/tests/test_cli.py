"""CLI のテスト。動画を使わない analyze / compare 経路を通す。"""

from __future__ import annotations

import json

import pytest
import yaml

from pitching.adapters.json_adapter import save_pose_json
from pitching.cli import main
from pitching.tests.support import make_series

GOOD_ANGLES = [
    175.0, 170.0, 160.0, 148.0, 135.0, 120.0, 108.0, 98.0, 90.0, 85.0,
    82.0, 88.0, 100.0, 118.0, 138.0, 158.0, 172.0, 176.0, 177.0, 177.0,
]


def write_inputs(directory, pitch_id: str, release_frame: int, label: str):
    """pose.json と設定 YAML を用意する。"""
    series = make_series(GOOD_ANGLES, start=100)
    pose_path = save_pose_json(series, directory / "pose.json")
    config = {
        "pitch_id": pitch_id,
        "throwing_hand": "right",
        "batter_direction": "left",
        "fps": 60,
        "start_frame": 100,
        "end_frame": 119,
        "events": {"stride_foot_contact_frame": 104, "release_frame": release_frame},
        "result": {"label": label, "description": "合成データ"},
        "preprocessing": {"smoothing_window": 5},
    }
    config_path = directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return pose_path, config_path


def test_analyze_and_compare(tmp_path, capsys):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad"
    good_pose, good_config = write_inputs(good_dir, "sample_good", 116, "good")
    bad_pose, bad_config = write_inputs(bad_dir, "sample_bad", 113, "bad")

    assert main(["analyze", "--pose-data", str(good_pose), "--config", str(good_config),
                 "--output", str(good_dir), "--no-charts"]) == 0
    assert main(["analyze", "--pose-data", str(bad_pose), "--config", str(bad_config),
                 "--output", str(bad_dir), "--no-charts"]) == 0
    assert main(["compare", "--pitch-a", str(good_dir / "metrics.json"),
                 "--pitch-b", str(bad_dir / "metrics.json"),
                 "--output", str(tmp_path / "comparison"), "--no-charts"]) == 0

    report = json.loads((tmp_path / "comparison" / "comparison.json").read_text(encoding="utf-8"))
    assert report["pitch_a"]["pitch_id"] == "sample_good"
    # 早いリリースの方が肘が伸びていない
    assert report["difference"]["elbow_angle_at_release_diff"] > 0


def test_compare_draws_charts_from_saved_pose(tmp_path):
    good_dir = tmp_path / "good"
    bad_dir = tmp_path / "bad"
    good_pose, good_config = write_inputs(good_dir, "sample_good", 116, "good")
    bad_pose, bad_config = write_inputs(bad_dir, "sample_bad", 113, "bad")
    main(["analyze", "--pose-data", str(good_pose), "--config", str(good_config),
          "--output", str(good_dir), "--no-charts"])
    main(["analyze", "--pose-data", str(bad_pose), "--config", str(bad_config),
          "--output", str(bad_dir), "--no-charts"])

    assert main(["compare", "--pitch-a", str(good_dir / "metrics.json"),
                 "--pitch-b", str(bad_dir / "metrics.json"),
                 "--output", str(tmp_path / "comparison")]) == 0

    charts = list((tmp_path / "comparison" / "charts").glob("*.png"))
    assert charts


def test_missing_config_returns_error_code(tmp_path, capsys):
    result = main(["analyze", "--pose-data", str(tmp_path / "pose.json"),
                   "--config", str(tmp_path / "missing.yaml"), "--output", str(tmp_path)])

    assert result == 1
    assert "エラー" in capsys.readouterr().err


def test_analyze_warns_about_unresolved_events(tmp_path, capsys):
    pose_path, config_path = write_inputs(tmp_path, "sample", 116, "good")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["events"]["release_frame"] = None
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    assert main(["analyze", "--pose-data", str(pose_path), "--config", str(config_path),
                 "--output", str(tmp_path), "--no-charts"]) == 0
    assert "未確定のイベント: release" in capsys.readouterr().out
