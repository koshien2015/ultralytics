import pytest
from pitching.domain.entities.bbox import BBox
from pitching.domain.services.camera_angle_estimator import estimate_camera_angle


def test_vertical_alignment_gives_90():
    # 投手と捕手が画面上で縦に並ぶ（dx=0）→ 90度
    pitcher = BBox(100, 100, 200, 300)
    catcher = BBox(100, 400, 200, 600)
    angle = estimate_camera_angle(pitcher, catcher)
    assert angle == pytest.approx(90.0, abs=1.0)


def test_clipped_between_0_and_90():
    pitcher = BBox(0, 0, 100, 100)
    catcher = BBox(0, 0, 100, 100)  # 同位置（dy=0）
    angle = estimate_camera_angle(pitcher, catcher)
    assert 0.0 <= angle <= 90.0
