from pitching.domain.services.release_detector import detect_release, resolve_pitcher_state


def test_resolve_motion():
    assert resolve_pitcher_state((0, 1, 2)) == "motion"


def test_resolve_release():
    assert resolve_pitcher_state((0, 5)) == "release"


def test_resolve_release_takes_priority():
    assert resolve_pitcher_state((1, 5)) == "release"


def test_resolve_none():
    assert resolve_pitcher_state((0, 2)) is None


def test_detect_release_on_transition():
    event = detect_release("motion", "release", pitch_id=1, frame_index=30, fps=30.0)
    assert event is not None
    assert event.pitch_id == 1
    assert event.release_frame == 30
    assert event.release_time_sec == pytest.approx(1.0)


def test_detect_release_no_transition():
    assert detect_release("motion", "motion", 1, 30, 30.0) is None
    assert detect_release(None, "release", 1, 30, 30.0) is None
    assert detect_release("release", "release", 1, 30, 30.0) is None


import pytest
