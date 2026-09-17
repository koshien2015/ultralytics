"""イベント検出の単体テスト。模擬角度系列だけで検証する。"""

from __future__ import annotations

import numpy as np
import pytest

from pitching.config import PitchConfig
from pitching.events.detector import (
    detect_extension_start,
    detect_max_flexion,
    resolve_events,
    search_window,
)
from pitching.events.release_source import CapSeparationRelease, CapTrack, ReleaseCandidate
from pitching.models import EventName, EventSource


def test_max_flexion_is_local_minimum(pitch_angles):
    angles = np.array(pitch_angles)
    position = detect_max_flexion(angles, 0, angles.size - 1, order=2)

    assert position == int(np.argmin(angles))


def test_max_flexion_ignores_single_frame_noise():
    """1フレームだけ落ち込んだノイズではなく、谷になっている位置を採る。"""
    angles = np.array([170.0, 40.0, 170.0, 150.0, 120.0, 100.0, 95.0, 110.0, 140.0, 170.0])
    # ノイズより後ろだけを探索区間にすれば、本来の谷が採れる
    position = detect_max_flexion(angles, 2, angles.size - 1, order=2)

    assert position == 6


def test_search_window_is_limited_by_events():
    start, end = search_window(length=50, contact_position=20, release_position=35, margin=5)

    assert (start, end) == (15, 35)


def test_search_window_falls_back_to_whole_series():
    start, end = search_window(length=50, contact_position=None, release_position=None, margin=5)

    assert (start, end) == (0, 49)


def test_extension_start_requires_sustained_increase():
    angles = np.array([100.0, 95.0, 90.0, 90.2, 89.9, 92.0, 96.0, 102.0, 110.0])
    flexion = 2

    position = detect_extension_start(angles, flexion, min_frames=3, min_delta_deg=0.5)

    # 90.2 → 89.9 で途切れるので、92.0 から続く増加の開始点（添字4）を返す
    assert position == 4


def test_extension_start_returns_none_without_flexion():
    angles = np.array([100.0, 110.0, 120.0])

    assert detect_extension_start(angles, None, 2, 0.5) is None


def test_missing_values_do_not_break_detection():
    angles = np.array([170.0, np.nan, 120.0, np.nan, 90.0, 100.0, 130.0, np.nan])

    position = detect_max_flexion(angles, 0, angles.size - 1, order=1)

    assert position == 4


def test_all_missing_returns_none():
    angles = np.full(5, np.nan)

    assert detect_max_flexion(angles, 0, 4, order=1) is None


def _config(release_frame: int | None) -> PitchConfig:
    return PitchConfig.model_validate(
        {
            "pitch_id": "t",
            "throwing_hand": "right",
            "events": {"stride_foot_contact_frame": 4, "release_frame": release_frame},
        }
    )


class _StubReleaseSource:
    def __init__(self, frame: int, source: EventSource) -> None:
        self._candidate = ReleaseCandidate(frame, source, "stub")

    def detect(self) -> ReleaseCandidate:
        return self._candidate


def test_manual_event_beats_automatic_estimate(pitch_angles):
    """手動指定は自動推定より優先される。"""
    frame_indices = np.arange(len(pitch_angles))
    events = resolve_events(
        _config(release_frame=16),
        frame_indices,
        np.array(pitch_angles),
        _StubReleaseSource(11, EventSource.CAP_TRACKING),
    )

    release = events[EventName.RELEASE]
    assert release.frame_index == 16
    assert release.source is EventSource.MANUAL


def test_cap_tracking_is_used_when_manual_is_absent(pitch_angles):
    events = resolve_events(
        _config(release_frame=None),
        np.arange(len(pitch_angles)),
        np.array(pitch_angles),
        _StubReleaseSource(11, EventSource.CAP_TRACKING),
    )

    release = events[EventName.RELEASE]
    assert release.frame_index == 11
    assert release.source is EventSource.CAP_TRACKING


def test_release_is_never_guessed_from_pose_alone(pitch_angles):
    """Pose の動きだけからリリースを決めない。"""
    events = resolve_events(
        _config(release_frame=None), np.arange(len(pitch_angles)), np.array(pitch_angles), None
    )

    assert events[EventName.RELEASE].frame_index is None


def test_cap_separation_detects_first_separated_frame():
    wrist = {index: (100.0, 100.0) for index in range(5)}
    caps = [
        CapTrack(0, 100.0, 100.0),
        CapTrack(1, 102.0, 100.0),
        CapTrack(2, 140.0, 100.0),
        CapTrack(3, 180.0, 100.0),
    ]
    detector = CapSeparationRelease(
        cap_tracks=caps, wrist_positions=wrist, body_scale_px=100.0, separation_ratio=0.25
    )

    candidate = detector.detect()

    assert candidate is not None
    assert candidate.frame_index == 2
    assert candidate.source is EventSource.CAP_TRACKING


def test_cap_separation_needs_body_scale():
    detector = CapSeparationRelease(cap_tracks=[], wrist_positions={}, body_scale_px=None)

    assert detector.detect() is None
