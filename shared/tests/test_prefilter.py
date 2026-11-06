"""prefilter のテスト（YOLO / torch 不要）。"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prefilter
from prefilter import ActivityProfile, InferenceGate, PrefilterConfig, PrefilterError


# --- 合成動画のフィクスチャ ---------------------------------------------------

WIDTH, HEIGHT, FPS = 320, 180, 30.0


def _make_video(path, pitches=3, interval=180, lead=45, burst=12, seed=3):
    """静止背景に、一定間隔で短い動きのバーストが入るだけの動画。"""
    rng = np.random.default_rng(seed)
    background = np.full((HEIGHT, WIDTH, 3), 70, np.uint8)
    cv2.rectangle(background, (0, 120), (WIDTH, HEIGHT), (100, 100, 100), -1)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    starts = [lead + interval * i for i in range(pitches)]
    total = lead + interval * pitches
    try:
        for number in range(total):
            frame = np.clip(background + rng.normal(0, 1.0, background.shape), 0, 255).astype(np.uint8)
            for start in starts:
                offset = number - start
                if 0 <= offset < burst:
                    cv2.rectangle(frame, (200 + offset * 4, 40), (230 + offset * 4, 100), (240, 240, 240), -1)
            writer.write(frame)
    finally:
        writer.release()
    return tuple(starts), total


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    path = tmp_path_factory.mktemp("prefilter") / "synth.mp4"
    starts, total = _make_video(path)
    return path, starts, total


# --- scan_activity ------------------------------------------------------------

def test_scan_activity_returns_a_profile(synth):
    path, _starts, total = synth
    profile = prefilter.scan_activity(str(path))
    assert profile.fps == pytest.approx(FPS, abs=0.1)
    assert profile.activity.size == total - 2  # 3フレーム差分で両端が落ちる
    assert profile.frames[0] == 1  # マスクは中央フレームに対応する


def test_scan_activity_rejects_a_missing_file(tmp_path):
    with pytest.raises(PrefilterError):
        prefilter.scan_activity(str(tmp_path / "nope.mp4"))


def test_roi_limits_where_activity_is_measured(synth):
    """動きの無い領域にROIを置けば活動量はゼロになる。"""
    path, _starts, _total = synth
    empty = prefilter.scan_activity(str(path), PrefilterConfig(roi=(0.0, 0.7, 0.3, 0.3)))
    full = prefilter.scan_activity(str(path))
    assert empty.activity.max() == 0
    assert full.activity.max() > 0


# --- detect_windows -----------------------------------------------------------

def _profile(activity, fps=FPS, total=None):
    values = np.array(activity, dtype=np.float64)
    return ActivityProfile(
        frames=np.arange(values.size, dtype=np.int64),
        activity=values,
        fps=fps,
        total_frames=total if total is not None else values.size,
    )


def test_detect_windows_finds_each_burst(synth):
    """投球間隔(6秒)がマージン幅(3.5秒)より広いので、区間は投球ごとに分かれる。"""
    path, starts, _total = synth
    profile = prefilter.scan_activity(str(path))
    windows = prefilter.detect_windows(profile, PrefilterConfig(min_event_interval_sec=1.0))
    assert len(windows) == len(starts)
    for start, (begin, end) in zip(starts, windows):
        assert begin <= start < end


def test_detect_windows_applies_margins():
    config = PrefilterConfig(pre_margin_sec=1.0, post_margin_sec=2.0, smooth_window_sec=0.05)
    activity = [0.0] * 60 + [500.0] * 10 + [0.0] * 200
    windows = prefilter.detect_windows(_profile(activity), config)
    assert len(windows) == 1
    begin, end = windows[0]
    assert begin == pytest.approx(60 - 30, abs=3)
    assert end == pytest.approx(60 + 60, abs=3)


def test_detect_windows_merges_overlapping_spans():
    """近接した2つのバーストは1つの区間にまとまる。"""
    activity = [0.0] * 60 + [500.0] * 5 + [0.0] * 20 + [500.0] * 5 + [0.0] * 200
    windows = prefilter.detect_windows(_profile(activity), PrefilterConfig(smooth_window_sec=0.05))
    assert len(windows) == 1


def test_detect_windows_keeps_distant_bursts_separate():
    activity = [0.0] * 60 + [500.0] * 5 + [0.0] * 300 + [500.0] * 5 + [0.0] * 60
    windows = prefilter.detect_windows(_profile(activity), PrefilterConfig(smooth_window_sec=0.05))
    assert len(windows) == 2


def test_detect_windows_on_a_flat_profile():
    assert prefilter.detect_windows(_profile([0.0] * 200)) == ()


def test_detect_windows_on_an_empty_profile():
    assert prefilter.detect_windows(_profile([])) == ()


def test_windows_never_leave_the_video():
    activity = [500.0] * 10 + [0.0] * 50
    begin, end = prefilter.detect_windows(_profile(activity, total=60), PrefilterConfig(smooth_window_sec=0.05))[0]
    assert begin >= 0
    assert end <= 60


def test_windows_frame_count():
    assert prefilter.windows_frame_count(((0, 10), (20, 25))) == 15


# --- InferenceGate ------------------------------------------------------------

def test_gate_skips_outside_windows():
    gate = InferenceGate(((100, 130),), PrefilterConfig(search_stride=1))
    assert not gate.should_infer(50)
    assert gate.should_infer(100)
    assert not gate.should_infer(130)  # 終端は含まない


def test_gate_thins_while_searching():
    gate = InferenceGate(None, PrefilterConfig(search_stride=5))
    decisions = [gate.should_infer(n) for n in range(10)]
    assert decisions == [True, False, False, False, False, True, False, False, False, False]


def test_gate_switches_to_dense_after_a_detection():
    """検出したら間引きをやめて連続で推論する。"""
    gate = InferenceGate(None, PrefilterConfig(search_stride=5, dense_frames=3))
    assert gate.should_infer(0)
    gate.note_detection(0, True)
    assert [gate.should_infer(n) for n in (1, 2, 3)] == [True, True, True]
    assert not gate.should_infer(4)  # dense_frames を過ぎたら間引きに戻る


def test_gate_stays_sparse_when_nothing_is_found():
    gate = InferenceGate(None, PrefilterConfig(search_stride=3))
    gate.should_infer(0)
    gate.note_detection(0, False)
    assert not gate.should_infer(1)


def test_gate_without_windows_covers_everything():
    gate = InferenceGate(None, PrefilterConfig(search_stride=1))
    assert all(gate.should_infer(n) for n in range(20))


def test_gate_counts_and_summarises():
    gate = InferenceGate(None, PrefilterConfig(search_stride=4))
    for n in range(20):
        gate.should_infer(n)
    assert gate.inferred == 5
    assert gate.skipped == 15
    assert "25.0%" in gate.summary


def test_gate_rejects_bad_stride():
    with pytest.raises(ValueError):
        InferenceGate(None, PrefilterConfig(search_stride=0))


# --- 統合 ---------------------------------------------------------------------

def test_build_gate_cuts_the_frames_to_infer(synth):
    """事前検出＋間引きで、推論フレーム数が実際に減る。"""
    path, _starts, total = synth
    gate, windows = prefilter.build_gate(str(path), PrefilterConfig(search_stride=5, min_event_interval_sec=1.0))
    assert windows
    for number in range(total):
        gate.should_infer(number)
    assert gate.inferred < total * 0.5


def test_build_gate_without_windows_disables_prescan(synth):
    path, _starts, _total = synth
    gate, windows = prefilter.build_gate(str(path), use_windows=False)
    assert windows == ()
    assert gate.in_window(999999)
