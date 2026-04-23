import pytest
from pitching.domain.entities.strike_zone import StrikeZone
from pitching.domain.services.normalizer import normalize_position

ZONE = StrikeZone(
    frame_index=0,
    left=50.0, right=150.0,
    top=100.0, bottom=300.0,
    center_x=100.0, center_y=200.0,
    zone_width=100.0,
)


def test_center_normalizes_to_zero_x():
    x, y = normalize_position(100.0, 200.0, ZONE, camera_angle_deg=90.0)
    assert x == pytest.approx(0.0, abs=1e-6)


def test_top_of_zone_normalizes_to_one_y():
    _, y = normalize_position(100.0, 100.0, ZONE, camera_angle_deg=90.0)
    assert y == pytest.approx(1.0)


def test_bottom_of_zone_normalizes_to_zero_y():
    _, y = normalize_position(100.0, 300.0, ZONE, camera_angle_deg=90.0)
    assert y == pytest.approx(0.0)
