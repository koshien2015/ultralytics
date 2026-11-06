"""保存形式・正解データ・設定のテスト。"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temporal_experiment.clips import load_clips, select_truths, unlabelled_frames
from temporal_experiment.config import ExperimentConfig, load_config
from temporal_experiment.pipeline import run_methods
from temporal_experiment.schema import Candidate, FrameRecord, GroundTruth
from temporal_experiment.storage import (
    load_detections,
    load_truth_csv,
    load_truth_yolo,
    save_detections,
)


def sample_records():
    return (
        FrameRecord(0, 0.0, (Candidate(1.0, 2.0, 3.0, 4.0, 0.9), Candidate(5, 6, 7, 8, 0.2)), 12.5),
        FrameRecord(1, 0.033, (), 11.0),
    )


def test_detections_round_trip_keeps_every_candidate(tmp_path):
    path = tmp_path / "det.json"
    meta = {"video_id": "a", "fps": 30.0, "width": 1280, "height": 720}
    save_detections(path, meta, sample_records())
    loaded_meta, records = load_detections(path)
    assert loaded_meta == meta
    assert records == sample_records()


def test_saved_detections_record_presence_and_center(tmp_path):
    path = tmp_path / "det.json"
    save_detections(path, {"video_id": "a"}, sample_records())
    frames = json.loads(path.read_text(encoding="utf-8"))["frames"]
    assert frames[0]["detected"] is True
    assert frames[1]["detected"] is False
    assert frames[0]["candidates"][0]["cx"] == pytest.approx(2.0)


def test_truth_csv_reads_annotator_format(tmp_path):
    path = tmp_path / "gt.csv"
    path.write_text(
        "video_path,frame_id,x,y,visible\nv.mp4,3,10.5,20.0,1\nv.mp4,4,-1,-1,0\n", encoding="utf-8"
    )
    assert load_truth_csv(path) == (
        GroundTruth(3, 10.5, 20.0, True),
        GroundTruth(4, -1.0, -1.0, False),
    )


def test_truth_yolo_converts_box_center_to_pixels(tmp_path):
    (tmp_path / "vid_000012.txt").write_text("3 0.1 0.1 0.1 0.1\n0 0.5 0.25 0.01 0.01\n")
    assert load_truth_yolo(tmp_path, "vid", 1000, 400, cap_class=0) == (
        GroundTruth(12, 500.0, 100.0, True),
    )


def test_truth_yolo_treats_label_without_cap_as_invisible(tmp_path):
    (tmp_path / "vid_000012.txt").write_text("3 0.1 0.1 0.1 0.1\n")
    assert load_truth_yolo(tmp_path, "vid", 1000, 400, cap_class=0)[0].visible is False


def test_truth_yolo_ignores_other_videos(tmp_path):
    (tmp_path / "other_000012.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert load_truth_yolo(tmp_path, "vid", 1000, 400, cap_class=0) == ()


def test_truth_yolo_rejects_two_caps_in_one_frame(tmp_path):
    (tmp_path / "vid_000012.txt").write_text("0 0.5 0.5 0.1 0.1\n0 0.1 0.1 0.1 0.1\n")
    with pytest.raises(ValueError, match="vid_000012"):
        load_truth_yolo(tmp_path, "vid", 1000, 400, cap_class=0)


def write_clips(path):
    path.write_text(
        "clips:\n"
        "  - {id: a, start: 10, end: 12, split: tuning}\n"
        "  - {id: b, start: 20, end: 22, split: eval}\n",
        encoding="utf-8",
    )


def test_select_truths_never_mixes_tuning_and_eval(tmp_path):
    path = tmp_path / "clips.yaml"
    write_clips(path)
    truths = [GroundTruth(f, 0.0, 0.0) for f in (10, 11, 20, 21, 30)]
    assert [g.frame for g in select_truths(truths, load_clips(path), "eval")] == [20, 21]
    assert [g.frame for g in select_truths(truths, load_clips(path), "tuning")] == [10, 11]


def test_overlapping_tuning_and_eval_clips_are_rejected(tmp_path):
    path = tmp_path / "clips.yaml"
    path.write_text(
        "clips:\n"
        "  - {id: a, start: 10, end: 30, split: tuning}\n"
        "  - {id: b, start: 20, end: 40, split: eval}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重な"):
        load_clips(path)


def test_unknown_split_is_rejected(tmp_path):
    path = tmp_path / "clips.yaml"
    path.write_text("clips:\n  - {id: a, start: 1, end: 2, split: test}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split"):
        load_clips(path)


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("tracker:\n  max_speed: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不明"):
        load_config(path)


def test_distance_settings_scale_with_video_width():
    scaled = ExperimentConfig().scaled_for(1920)
    assert scaled.tracker.max_speed_px_s == pytest.approx(3750.0)
    assert scaled.evaluation.match_radius_px == pytest.approx(22.5)
    assert scaled.tracker.max_missed_frames == 5


def test_run_methods_returns_every_compared_method():
    records = tuple(
        FrameRecord(f, f / 30, (Candidate(10.0 * f, 0.0, 10.0 * f + 4, 4.0, 0.8),))
        for f in range(10)
    )
    assert set(run_methods(records, ExperimentConfig())) == {
        "yolo", "nearest", "linear", "kalman", "ransac",
    }


def test_run_methods_reports_post_processing_time():
    records = (FrameRecord(0, 0.0, ()),)
    result = run_methods(records, ExperimentConfig())
    assert result["kalman"].elapsed_ms >= 0.0


def test_clip_nested_inside_long_clip_cannot_hide_overlap(tmp_path):
    path = tmp_path / "clips.yaml"
    path.write_text(
        "clips:\n"
        "  - {id: a, start: 0, end: 100, split: tuning}\n"
        "  - {id: b, start: 10, end: 20, split: tuning}\n"
        "  - {id: c, start: 50, end: 60, split: eval}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重な"):
        load_clips(path)


def test_clip_with_reversed_range_is_rejected(tmp_path):
    path = tmp_path / "clips.yaml"
    path.write_text("clips:\n  - {id: a, start: 100, end: 0, split: eval}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="start"):
        load_clips(path)


def test_unlabelled_frames_in_split_are_listed(tmp_path):
    path = tmp_path / "clips.yaml"
    write_clips(path)
    truths = [GroundTruth(10, 0.0, 0.0), GroundTruth(20, 0.0, 0.0), GroundTruth(21, 0.0, 0.0)]
    assert unlabelled_frames(truths, load_clips(path), "tuning") == [11]
    assert unlabelled_frames(truths, load_clips(path), "eval") == []
