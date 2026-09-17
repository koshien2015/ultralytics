"""前処理（信頼度フィルタ・補間・平滑化）の単体テスト。"""

from __future__ import annotations

import numpy as np
import pytest

from pitching.models import Keypoint, PoseFrame, PoseSeries
from pitching.preprocessing import confidence_filter, interpolation, smoothing
from pitching.preprocessing.tracks import from_series
from pitching.tests.support import make_series


def build_tracks(values: list[tuple[float, float, float]]):
    """(x, y, confidence) の列から1点だけの tracks を作る。"""
    frames = [
        PoseFrame(index, index / 60.0, {"right_wrist": Keypoint(x, y, confidence)})
        for index, (x, y, confidence) in enumerate(values)
    ]
    return from_series(PoseSeries(frames=frames, fps=60.0))


def test_low_confidence_keypoints_become_missing():
    tracks = build_tracks([(10.0, 10.0, 0.9), (20.0, 20.0, 0.2), (30.0, 30.0, 0.9)])
    filtered = confidence_filter.apply_threshold(tracks, 0.5)

    assert filtered.point("right_wrist", 0) == (10.0, 10.0)
    assert filtered.point("right_wrist", 1) is None
    # 元の tracks は壊さない
    assert tracks.point("right_wrist", 1) == (20.0, 20.0)


def test_low_confidence_interval_is_reported():
    tracks = build_tracks([(10.0, 10.0, 0.9), (20.0, 20.0, 0.2), (30.0, 30.0, 0.1)])
    report = confidence_filter.low_confidence_report(tracks, 0.5)

    assert report["right_wrist"][0]["start_frame"] == 1
    assert report["right_wrist"][0]["end_frame"] == 2
    assert report["right_wrist"][0]["frames"] == 2


def test_only_short_gaps_are_interpolated():
    values = [(0.0, 0.0, 0.9), (np.nan, np.nan, 0.1), (20.0, 20.0, 0.9)]
    values += [(np.nan, np.nan, 0.1)] * 4 + [(70.0, 70.0, 0.9)]
    tracks = build_tracks(values)

    filled, report = interpolation.interpolate_short_gaps(tracks, max_gap_frames=2)

    # 1フレームの欠損は埋まる
    assert filled.point("right_wrist", 1) == pytest.approx((10.0, 10.0))
    # 4フレームの欠損は埋めない
    assert all(filled.point("right_wrist", position) is None for position in range(3, 7))
    assert len(report["right_wrist"]) == 1


def test_edge_gaps_are_not_extrapolated():
    tracks = build_tracks([(np.nan, np.nan, 0.1), (10.0, 10.0, 0.9), (np.nan, np.nan, 0.1)])
    filled, report = interpolation.interpolate_short_gaps(tracks, max_gap_frames=5)

    assert filled.point("right_wrist", 0) is None
    assert filled.point("right_wrist", 2) is None
    assert report == {}


def test_smoothing_keeps_frame_count():
    series = make_series([170.0, 160.0, 140.0, 120.0, 110.0, 130.0, 150.0])
    tracks = from_series(series)

    smoothed = smoothing.smooth_tracks(tracks, window_length=5, polyorder=2)

    assert smoothed.length == tracks.length
    assert smoothed.xy["right_wrist"].shape == tracks.xy["right_wrist"].shape


def test_smoothing_keeps_missing_frames_missing():
    values = [(float(index), float(index), 0.9) for index in range(10)]
    values[5] = (np.nan, np.nan, 0.1)
    tracks = build_tracks(values)

    smoothed = smoothing.smooth_tracks(tracks, window_length=5, polyorder=2)

    assert smoothed.point("right_wrist", 5) is None
    assert smoothed.length == 10


def test_smoothing_does_not_shift_peak_timing(pitch_angles):
    """平滑化でイベントのタイミングがずれない（対称フィルタなので位相遅れが無い）。"""
    angles = np.array(pitch_angles)
    noisy = angles + np.array([0.6, -0.6] * (angles.size // 2))

    smoothed = smoothing.smooth_values(noisy, window_length=5, polyorder=2)

    assert int(np.argmin(smoothed)) == int(np.argmin(angles))


def test_short_segment_is_left_unsmoothed():
    tracks = build_tracks([(1.0, 1.0, 0.9), (5.0, 5.0, 0.9)])
    smoothed = smoothing.smooth_tracks(tracks, window_length=9, polyorder=2)

    assert smoothed.point("right_wrist", 0) == (1.0, 1.0)
