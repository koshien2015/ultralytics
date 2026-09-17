"""2投球の同期（時間正規化・イベント基準）のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from pitching.analysis import analyze_pitch
from pitching.comparison.alignment import align_by_event, event_offset, time_normalize
from pitching.models import EventName


def test_progress_is_zero_at_foot_contact_and_hundred_at_release(pitch_series, pitch_config):
    analysis = analyze_pitch(pitch_series, pitch_config)

    contact = analysis.position_of_event(EventName.FOOT_CONTACT)
    release = analysis.position_of_event(EventName.RELEASE)

    assert analysis.progress_percent[contact] == pytest.approx(0.0)
    assert analysis.progress_percent[release] == pytest.approx(100.0)


def test_time_normalized_series_is_on_a_common_grid(pitch_series, pitch_config):
    analysis = analyze_pitch(pitch_series, pitch_config)

    normalized = time_normalize(analysis, "elbow_angle_deg")

    assert normalized.covered
    assert normalized.progress_percent[0] == pytest.approx(0.0)
    assert normalized.progress_percent[-1] == pytest.approx(100.0)
    assert normalized.values.size == normalized.progress_percent.size
    assert np.isfinite(normalized.values).all()


def test_time_normalization_keeps_source_frames(pitch_series, pitch_config):
    """正規化後も元のフレーム番号を辿れる。"""
    analysis = analyze_pitch(pitch_series, pitch_config)

    normalized = time_normalize(analysis, "elbow_angle_deg")

    assert normalized.source_frames[0] == pitch_config.events.stride_foot_contact_frame
    assert normalized.source_frames[-1] == pitch_config.events.release_frame


def test_time_normalization_reports_when_events_are_missing(pitch_series, pitch_config):
    config = pitch_config.model_copy(deep=True)
    config.events.release_frame = None

    analysis = analyze_pitch(pitch_series, config)
    normalized = time_normalize(analysis, "elbow_angle_deg")

    assert not normalized.covered
    assert np.isnan(normalized.values).all()


def test_two_pitches_are_compared_on_the_same_scale(pitch_config):
    """長さの違う2投球でも、正規化後は同じ軸で比べられる。"""
    from pitching.tests.support import make_series

    slow_angles = [175.0 - index * 5 for index in range(19)] + [85.0]
    fast_config = pitch_config.model_copy(deep=True)
    fast_config.end_frame = 11
    fast_config.events.stride_foot_contact_frame = 2
    fast_config.events.release_frame = 10

    slow = analyze_pitch(make_series(slow_angles), pitch_config)
    fast = analyze_pitch(make_series(slow_angles[:12]), fast_config)

    slow_normalized = time_normalize(slow, "elbow_angle_deg")
    fast_normalized = time_normalize(fast, "elbow_angle_deg")

    assert slow_normalized.values.size == fast_normalized.values.size


def test_event_offset_and_alignment(pitch_series, pitch_config):
    other_config = pitch_config.model_copy(deep=True)
    other_config.pitch_id = "synthetic_bad"

    first = analyze_pitch(pitch_series, pitch_config)
    second = analyze_pitch(pitch_series, other_config)

    assert event_offset(first, second, EventName.RELEASE) == 0

    aligned = align_by_event(first, second, "elbow_angle_deg", EventName.RELEASE)
    assert aligned is not None
    offsets, first_values, second_values = aligned
    assert offsets[np.argmin(np.abs(offsets))] == 0
    assert np.allclose(first_values, second_values, equal_nan=True)
