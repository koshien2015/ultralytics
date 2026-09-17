"""解析パイプライン・出力・比較の結合テスト（動画なし）。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pitching.analysis import analyze_pitch
from pitching.comparison.comparator import compare_summaries
from pitching.models import EventName, Keypoint, PoseFrame
from pitching.reporting.summary import build_summary, config_from_summary
from pitching.reporting.writers import write_analysis, write_comparison
from pitching.tests.support import make_series


def test_analysis_extracts_elbow_features(pitch_series, pitch_config, pitch_angles):
    analysis = analyze_pitch(pitch_series, pitch_config)
    elbow = analysis.features["elbow"]

    assert elbow["minimum_elbow_angle_deg"] == pytest.approx(min(pitch_angles), abs=2.0)
    assert elbow["elbow_angle_at_release_deg"] == pytest.approx(pitch_angles[16], abs=2.0)
    assert elbow["elbow_extension_range_deg"] > 0
    # 最大屈曲(10) → リリース(16) は 6 フレーム = 0.1 秒
    assert elbow["flexion_to_release_time_sec"] == pytest.approx(0.1, abs=0.02)
    assert elbow["peak_extension_velocity_deg_per_sec"] > 0


def test_analysis_reports_events_with_sources(pitch_series, pitch_config):
    analysis = analyze_pitch(pitch_series, pitch_config)

    assert analysis.events[EventName.RELEASE].source.value == "manual"
    assert analysis.events[EventName.MAX_ELBOW_FLEXION].source.value == "pose_heuristic"
    assert analysis.events[EventName.MAX_ELBOW_FLEXION].frame_index == 10
    assert analysis.events[EventName.EXTENSION_START].frame_index is not None


def test_low_confidence_frames_are_excluded_from_metrics(pitch_series, pitch_config):
    """信頼度不足の値は解析対象から外れ、角度が NaN になる。"""
    frames = list(pitch_series.frames)
    target = frames[12]
    frames[12] = PoseFrame(
        target.frame_index,
        target.timestamp_sec,
        {
            **target.keypoints,
            "right_wrist": Keypoint(target.keypoints["right_wrist"].x, target.keypoints["right_wrist"].y, 0.1),
        },
    )
    pitch_series.frames = frames

    config = pitch_config.model_copy(deep=True)
    config.preprocessing.max_gap_frames = 0
    analysis = analyze_pitch(pitch_series, config)

    position = analysis.position_of_event_frame(12)
    assert np.isnan(analysis.series["elbow_angle_deg"][position])
    assert "right_wrist" in analysis.quality["low_confidence_intervals"]


def test_short_gaps_are_recovered_by_interpolation(pitch_series, pitch_config):
    frames = list(pitch_series.frames)
    target = frames[12]
    frames[12] = PoseFrame(
        target.frame_index,
        target.timestamp_sec,
        {**target.keypoints, "right_wrist": Keypoint(float("nan"), float("nan"), 0.1)},
    )
    pitch_series.frames = frames

    analysis = analyze_pitch(pitch_series, pitch_config)

    position = analysis.position_of_event_frame(12)
    assert np.isfinite(analysis.series["elbow_angle_deg"][position])
    assert "right_wrist" in analysis.quality["interpolated_gaps"]


def test_missing_keypoints_are_handled_safely(pitch_config):
    """脚のキーポイントが全く無くても落ちない。該当特徴量だけ None になる。"""
    series = make_series([175.0 - index * 4 for index in range(20)])
    series.frames = [
        PoseFrame(
            frame.frame_index,
            frame.timestamp_sec,
            {
                name: keypoint
                for name, keypoint in frame.keypoints.items()
                if "ankle" not in name and "knee" not in name
            },
        )
        for frame in series.frames
    ]

    analysis = analyze_pitch(series, pitch_config)

    assert analysis.features["lower_body"]["lead_knee_angle_at_release_deg"] is None
    assert analysis.features["elbow"]["elbow_angle_at_release_deg"] is not None


def test_release_features_are_unavailable_without_release(pitch_series, pitch_config):
    config = pitch_config.model_copy(deep=True)
    config.events.release_frame = None

    analysis = analyze_pitch(pitch_series, config)

    assert analysis.features["release_position"]["available"] is False
    assert "release" in analysis.quality["unresolved_events"]


def test_analysis_outputs_are_written(tmp_path, pitch_series, pitch_config):
    analysis = analyze_pitch(pitch_series, pitch_config)

    written = write_analysis(analysis, tmp_path)

    for path in written.values():
        assert path.is_file()
    summary = json.loads(written["metrics_json"].read_text(encoding="utf-8"))
    assert summary["pitch_id"] == "synthetic_good"
    assert summary["events"]["release"]["source"] == "manual"
    # NaN を書かない（JSON として読み戻せる）
    assert "NaN" not in written["metrics_json"].read_text(encoding="utf-8")


def test_summary_round_trips_into_config(pitch_series, pitch_config):
    analysis = analyze_pitch(pitch_series, pitch_config)
    summary = build_summary(analysis)

    restored = config_from_summary(summary)

    assert restored.pitch_id == pitch_config.pitch_id
    assert restored.events.release_frame == pitch_config.events.release_frame
    assert restored.preprocessing.smoothing_window == pitch_config.preprocessing.smoothing_window


def test_comparison_reports_differences_and_values(tmp_path, pitch_series, pitch_config, pitch_angles):
    good = analyze_pitch(pitch_series, pitch_config)

    bad_config = pitch_config.model_copy(deep=True)
    bad_config.pitch_id = "synthetic_bad"
    bad_config.result.label = "bad"
    # 伸展途中の早いリリース
    bad_config.events.release_frame = 13
    bad = analyze_pitch(make_series(pitch_angles), bad_config)

    report = compare_summaries(build_summary(good), build_summary(bad))

    difference = report["difference"]["elbow_angle_at_release_diff"]
    assert difference is not None and difference > 0
    observation = report["observations"]["elbow_angle_at_release_diff"]
    assert observation["pitch_a"] is not None and observation["pitch_b"] is not None
    assert report["event_alignment"]["release"]["usable_as_anchor"] is True

    written = write_comparison(report, tmp_path)
    assert written["comparison_json"].is_file()
    assert written["comparison_csv"].is_file()


def test_comparison_marks_unavailable_items(pitch_series, pitch_config):
    without_release = pitch_config.model_copy(deep=True)
    without_release.events.release_frame = None

    good = build_summary(analyze_pitch(pitch_series, pitch_config))
    unknown = build_summary(analyze_pitch(pitch_series, without_release))

    report = compare_summaries(good, unknown)

    assert "elbow_angle_at_release_diff" in report["unavailable"]
    assert report["difference"]["elbow_angle_at_release_diff"] is None


def test_charts_are_written(tmp_path, pitch_series, pitch_config):
    from pitching.visualization import charts

    analysis = analyze_pitch(pitch_series, pitch_config)

    written = charts.write_pitch_charts(analysis, tmp_path)

    assert written
    assert all(path.is_file() for path in written)
