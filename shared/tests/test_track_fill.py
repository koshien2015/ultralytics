"""track.py の線形補完のテスト。"""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

SHARED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHARED))

from temporal_experiment.report import load_estimates
from temporal_experiment.schema import DETECTED, INTERPOLATED, MISSING, Candidate, FrameRecord
from temporal_experiment.track_fill import export_linear_fill, frame_record, linear_track


def fake_box(x1, y1, x2, y2, conf, cls):
    return SimpleNamespace(xyxy=[(x1, y1, x2, y2)], conf=[conf], cls=[cls])


def cap_at(x, y=40.0):
    return Candidate(x - 5, y - 5, x + 5, y + 5, 0.8)


def moving_records(n=20, drop=(), step=8.0):
    return tuple(
        FrameRecord(f, f / 30, () if f in drop else (cap_at(20.0 + step * f),))
        for f in range(n)
    )


def write_video(path, n=20, size=(160, 96)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    for _ in range(n):
        writer.write(np.zeros((size[1], size[0], 3), np.uint8))
    writer.release()


def test_frame_record_keeps_only_target_classes():
    result = SimpleNamespace(boxes=[fake_box(10, 20, 30, 40, 0.9, 0), fake_box(1, 2, 3, 4, 0.8, 1)])

    record = frame_record(7, 0.25, [result], [0])

    assert (record.frame, record.t) == (7, 0.25)
    assert record.candidates == (Candidate(10.0, 20.0, 30.0, 40.0, 0.9),)


def test_frame_record_keeps_every_class_when_no_target_is_set():
    result = SimpleNamespace(boxes=[fake_box(10, 20, 30, 40, 0.9, 0), fake_box(1, 2, 3, 4, 0.8, 1)])

    assert len(frame_record(7, 0.25, [result], []).candidates) == 2


def test_frame_skipped_by_prefilter_is_kept_as_an_empty_record():
    assert frame_record(7, 0.25, [], [0]).candidates == ()


def test_short_gap_is_interpolated_between_detections():
    run = linear_track(moving_records(drop=(10, 11)), width=1280)

    by_frame = {e.frame: e for e in run.estimates}
    assert by_frame[9].status == DETECTED
    assert by_frame[10].status == INTERPOLATED
    assert by_frame[10].x == 100.0


def test_gap_over_the_limit_is_left_missing():
    run = linear_track(moving_records(drop=range(10, 17)), width=1280)

    assert {e.status for e in run.estimates if 10 <= e.frame < 17} == {MISSING}


def test_tracker_limits_scale_with_video_width():
    records = moving_records(step=120.0)  # 3600px/s。幅1280の上限 2500px/s を超える

    narrow = linear_track(records, width=1280)
    wide = linear_track(records, width=2560)

    assert not any(e.has_position for e in narrow.estimates)
    assert sum(e.status == DETECTED for e in wide.estimates) == 20


def export_blank_video(tmp_path):
    video = tmp_path / "in_detected.mp4"
    write_video(video, n=20, size=(320, 192))
    return export_linear_fill(moving_records(drop=(10, 11)), video, tmp_path, "in", 320)


def read_gray(video_path, frame):
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    _, image = capture.read()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(int)


def test_export_writes_estimates_and_keeps_the_video_size(tmp_path):
    json_path, video_path = export_blank_video(tmp_path)

    assert json_path == tmp_path / "in_filled.json"
    assert sum(e.status == INTERPOLATED for e in load_estimates(json_path)["linear"]) == 2
    capture = cv2.VideoCapture(str(video_path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320


def test_interpolated_frame_gets_only_a_box_outline(tmp_path):
    _, video_path = export_blank_video(tmp_path)

    gray = read_gray(video_path, 10)  # 補間の位置は x=100, y=40、bbox は 95〜105 と 35〜45

    assert gray[33:48, 93:108].max() > 40  # bbox の線がある
    assert gray[38:43, 98:103].max() < 30  # 円で塗らない
    outside = gray.copy()
    outside[30:50, 90:110] = 0
    assert outside.max() < 30  # 文字もフレーム番号も足さない


def test_detected_frame_is_left_as_it_was(tmp_path):
    _, video_path = export_blank_video(tmp_path)

    assert read_gray(video_path, 9).max() < 30


def test_track_extends_the_detected_video():
    tree = ast.parse((SHARED / "track.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "export_linear_fill"
    ]

    assert [c.args[1].id for c in calls] == ["output_path"]


def test_track_keeps_linear_fill_off_by_default():
    tree = ast.parse((SHARED / "track.py").read_text(encoding="utf-8"))
    flags = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "ENABLE_LINEAR_FILL"
    ]

    assert flags == [False]
