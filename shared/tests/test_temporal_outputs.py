"""評価のまとめ・グラフ・比較動画のテスト。"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temporal_experiment.cli import main
from temporal_experiment.config import ExperimentConfig
from temporal_experiment.infer import is_complete_enhanced
from temporal_experiment.pipeline import run_methods
from temporal_experiment.plots import frame_errors, save_all_plots
from temporal_experiment.render import STYLES, draw_panel, render_comparison
from temporal_experiment.report import load_estimates, markdown_table, save_estimates, summarize
from temporal_experiment.schema import (
    DETECTED,
    INTERPOLATED,
    MISSING,
    PREDICTED,
    Candidate,
    Estimate,
    FrameRecord,
    GroundTruth,
    TrackEvent,
)
from temporal_experiment.storage import save_detections


def box(x, y, conf=0.8):
    return Candidate(x - 5, y - 5, x + 5, y + 5, conf)


def synthetic_records(n=20, drop=(10, 11)):
    return tuple(
        FrameRecord(f, f / 30, () if f in drop else (box(20.0 + 8 * f, 40.0),), 5.0)
        for f in range(n)
    )


def truths(n=20):
    return tuple(GroundTruth(f, 20.0 + 8 * f, 40.0) for f in range(n))


def summary():
    records = synthetic_records()
    return summarize(records, run_methods(records, ExperimentConfig()), truths(), ExperimentConfig())


def test_summary_separates_recall_from_method_coverage():
    result = summary()
    assert result["detector_recall"] == pytest.approx(0.9)
    assert result["methods"]["yolo"]["coverage"] == pytest.approx(0.9)
    assert result["methods"]["linear"]["coverage"] == pytest.approx(1.0)


def test_summary_reports_inference_and_post_processing_time():
    result = summary()
    assert result["mean_infer_ms"] == pytest.approx(5.0)
    assert result["methods"]["kalman"]["post_ms"] >= 0.0


def test_markdown_table_marks_undefined_values():
    table = markdown_table({"detector_recall": None, "methods": {
        "yolo": {"coverage": None, "correct_coverage": None, "mae_output": None, "mae_all": None,
                 "p95_output": None, "p95_all": None, "max_missing_run": 0, "wrong_outputs": 0,
                 "phantom_outputs": 0, "predicted_tail": 0, "post_ms": 1.0}}})
    assert "| yolo | - |" in table


def test_estimates_round_trip(tmp_path):
    path = tmp_path / "estimates.json"
    runs = run_methods(synthetic_records(), ExperimentConfig())
    save_estimates(path, runs)
    loaded = load_estimates(path)
    assert loaded["linear"] == runs["linear"].estimates


def test_frame_errors_leave_missing_frames_undefined():
    estimates = (Estimate(0, 0.0, DETECTED, 23.0, 44.0), Estimate(1, 0.1, MISSING))
    errors = frame_errors(estimates, (GroundTruth(0, 20.0, 40.0), GroundTruth(1, 20.0, 40.0)))
    assert errors[0] == pytest.approx(5.0)
    assert np.isnan(errors[1])


def test_plots_are_written_for_every_required_chart(tmp_path):
    records = synthetic_records()
    runs = run_methods(records, ExperimentConfig())
    paths = save_all_plots(tmp_path, runs, truths(), summarize(records, runs, truths(), ExperimentConfig()))
    assert {p.name for p in paths} == {
        "coverage.png", "error.png", "error_timeline.png", "status_strip.png",
    }
    assert all(p.stat().st_size > 0 for p in paths)


def test_plots_survive_methods_without_any_output(tmp_path):
    records = tuple(FrameRecord(f, f / 30, (), 5.0) for f in range(5))
    runs = run_methods(records, ExperimentConfig())
    truth = tuple(GroundTruth(f, 1.0, 1.0) for f in range(5))
    paths = save_all_plots(tmp_path, runs, truth, summarize(records, runs, truth, ExperimentConfig()))
    assert len(paths) == 4


def test_estimated_position_never_looks_like_a_detection():
    assert STYLES[DETECTED].color != STYLES[INTERPOLATED].color
    assert STYLES[DETECTED].color != STYLES[PREDICTED].color
    assert STYLES[INTERPOLATED].color != STYLES[PREDICTED].color
    assert STYLES[DETECTED].solid and not STYLES[PREDICTED].solid


def test_panel_does_not_modify_source_frame():
    frame = np.zeros((80, 120, 3), np.uint8)
    drawn = draw_panel(frame, "yolo", (Estimate(0, 0.0, DETECTED, 40.0, 40.0, 0.9, (35, 35, 45, 45)),), 0)
    assert frame.sum() == 0
    assert drawn.sum() > 0


def test_panel_writes_confidence_only_for_detections():
    frame = np.zeros((80, 200, 3), np.uint8)
    detected = Estimate(0, 0.0, DETECTED, 40.0, 40.0, 0.9, (35, 35, 45, 45))
    predicted = Estimate(0, 0.0, PREDICTED, 40.0, 40.0)
    assert draw_panel(frame, "a", (detected,), 0).sum() > draw_panel(frame, "a", (predicted,), 0).sum()


def write_video(path, n=6, size=(160, 96)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for _ in range(n):
        writer.write(np.zeros((size[1], size[0], 3), np.uint8))
    writer.release()


def test_comparison_video_has_one_panel_per_method(tmp_path):
    video = tmp_path / "in.mp4"
    write_video(video)
    runs = run_methods(synthetic_records(6, drop=()), ExperimentConfig())
    out = tmp_path / "out.mp4"
    render_comparison(video, out, {k: runs[k].estimates for k in ("yolo", "linear", "kalman")}, 0, 6, 160)
    capture = cv2.VideoCapture(str(out))
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 480
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 6


def test_cli_track_and_evaluate_work_without_reinference(tmp_path):
    run_dir = tmp_path / "vid"
    run_dir.mkdir()
    meta = {"video_id": "vid", "fps": 30.0, "width": 1280, "height": 720}
    save_detections(run_dir / "detections.json", meta, synthetic_records())
    truth = tmp_path / "gt.csv"
    truth.write_text(
        "video_path,frame_id,x,y,visible\n"
        + "".join(f"v,{f},{20 + 8 * f},40,1\n" for f in range(20)),
        encoding="utf-8",
    )
    clips = tmp_path / "clips.yaml"
    clips.write_text("clips:\n  - {id: a, start: 0, end: 20, split: eval}\n", encoding="utf-8")

    assert main(["--runs", str(tmp_path), "track", "vid"]) == 0
    assert main(["--runs", str(tmp_path), "evaluate", "vid", "--truth", str(truth),
                 "--clips", str(clips), "--split", "eval"]) == 0
    result = json.loads((run_dir / "eval" / "metrics.json").read_text(encoding="utf-8"))
    assert result["split"] == "eval"
    assert (run_dir / "eval" / "coverage.png").exists()


def test_cli_refuses_evaluation_without_labelled_frames(tmp_path):
    run_dir = tmp_path / "vid"
    run_dir.mkdir()
    save_detections(run_dir / "detections.json", {"width": 1280}, synthetic_records())
    truth = tmp_path / "gt.csv"
    truth.write_text("video_path,frame_id,x,y,visible\n", encoding="utf-8")
    clips = tmp_path / "clips.yaml"
    clips.write_text("clips:\n  - {id: a, start: 0, end: 20, split: eval}\n", encoding="utf-8")
    main(["--runs", str(tmp_path), "track", "vid"])
    with pytest.raises(SystemExit, match="正解"):
        main(["--runs", str(tmp_path), "evaluate", "vid", "--truth", str(truth),
              "--clips", str(clips), "--split", "eval"])


def test_marks_grow_with_frame_width():
    def painted(width):
        frame = np.zeros((width // 2, width, 3), np.uint8)
        x = width / 2
        estimate = Estimate(0, 0.0, PREDICTED, x, width / 4)
        drawn = draw_panel(frame, "", (estimate,), 0)
        return np.count_nonzero(drawn.sum(axis=2)) / (width * width / 2)

    assert painted(1920) >= painted(640) * 0.6


def test_summary_counts_events_only_inside_evaluated_frames():
    records = synthetic_records(40, drop=())
    runs = run_methods(records, ExperimentConfig())
    event = TrackEvent(30, "gate_rejected", 0)
    patched = {**runs, "kalman": replace(runs["kalman"], events=(event,))}
    inside = summarize(records, patched, truths(40), ExperimentConfig())
    outside = summarize(records, patched, truths(10), ExperimentConfig())
    assert inside["methods"]["kalman"]["gate_rejected"] == 1
    assert outside["methods"]["kalman"]["gate_rejected"] == 0


def test_table_says_which_frames_percentile_and_time_cover():
    table = markdown_table(summary())
    assert "P95(出力のみ)" in table
    assert "P95(全フレーム)" in table
    assert "後処理ms(動画全体)" in table


def test_incomplete_enhanced_video_is_detected(tmp_path):
    source = tmp_path / "in.mp4"
    write_video(source, n=6)
    short = tmp_path / "short.mp4"
    write_video(short, n=3)
    complete = tmp_path / "complete.mp4"
    write_video(complete, n=5)
    assert not is_complete_enhanced(short, source)
    assert is_complete_enhanced(complete, source)


def test_cli_refuses_yolo_labels_with_missing_frames(tmp_path):
    run_dir = tmp_path / "vid"
    run_dir.mkdir()
    save_detections(run_dir / "detections.json", {"width": 1000, "height": 500}, synthetic_records())
    labels = tmp_path / "labels"
    labels.mkdir()
    for frame in (0, 1, 3):  # prelabel.py は何も検出しないフレームに .txt を作らない
        (labels / f"vid_{frame:06d}.txt").write_text("0 0.5 0.5 0.01 0.01\n", encoding="utf-8")
    clips = tmp_path / "clips.yaml"
    clips.write_text("clips:\n  - {id: a, start: 0, end: 4, split: eval}\n", encoding="utf-8")
    main(["--runs", str(tmp_path), "track", "vid"])
    with pytest.raises(SystemExit, match=r"\[2\]"):
        main(["--runs", str(tmp_path), "evaluate", "vid", "--truth-yolo", str(labels),
              "--clips", str(clips), "--split", "eval"])
