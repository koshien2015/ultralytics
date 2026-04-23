import pytest
from pitching.domain.entities.bbox import BBox
from pitching.domain.services.strike_zone_estimator import estimate_strike_zone


def test_zone_height_from_batter():
    batter = BBox(x1=100, y1=200, x2=200, y2=600)  # 高さ400px
    zone, center_x = estimate_strike_zone(
        frame_index=0,
        batter_bbox=batter,
        catcher_center_x=150.0,
        locked_center_x=None,
        zone_width=100.0,
    )
    assert zone.top == pytest.approx(200 + 400 * 0.4)   # 360
    assert zone.bottom == pytest.approx(200 + 400 * 0.75)  # 500
    assert center_x == pytest.approx(150.0)


def test_locked_center_x_takes_priority():
    batter = BBox(x1=100, y1=200, x2=200, y2=600)
    zone, used = estimate_strike_zone(
        frame_index=1,
        batter_bbox=batter,
        catcher_center_x=999.0,  # 無視されるはず
        locked_center_x=150.0,
        zone_width=100.0,
    )
    assert zone.center_x == pytest.approx(150.0)
    assert used == pytest.approx(150.0)


def test_zone_width():
    batter = BBox(x1=100, y1=200, x2=200, y2=600)
    zone, _ = estimate_strike_zone(0, batter, 150.0, None, 80.0)
    assert zone.right - zone.left == pytest.approx(80.0)
