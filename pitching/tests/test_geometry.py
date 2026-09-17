"""geometry の単体テスト。座標系の約束事を固定する。"""

from __future__ import annotations

import math

import pytest

from pitching.metrics.geometry import (
    angle_difference,
    classify_direction,
    direction_angle,
    facing_sign,
    joint_angle,
    line_angle,
)


def test_collinear_points_give_180_degrees():
    assert joint_angle((0.0, 0.0), (10.0, 0.0), (20.0, 0.0)) == pytest.approx(180.0)


def test_perpendicular_points_give_90_degrees():
    assert joint_angle((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)) == pytest.approx(90.0)


def test_mirroring_does_not_change_elbow_angle():
    """左右反転しても、なす角は変わらない（右投げ・左投げで同じ式が使える）。"""
    original = joint_angle((0.0, 0.0), (10.0, 5.0), (25.0, 30.0))
    mirrored = joint_angle((0.0, 0.0), (-10.0, 5.0), (-25.0, 30.0))
    assert original == pytest.approx(mirrored)


def test_vertical_flip_does_not_change_elbow_angle():
    original = joint_angle((0.0, 0.0), (10.0, 5.0), (25.0, 30.0))
    flipped = joint_angle((0.0, 0.0), (10.0, -5.0), (25.0, -30.0))
    assert original == pytest.approx(flipped)


def test_forearm_angle_sign_respects_image_y_axis():
    """画像のYは下向き。上に伸びた前腕は正の角度になる。"""
    # 打者が画像の右（sign=+1）。肘(0,0) → 手首(10,-10) は画面上では上方向。
    assert direction_angle((0.0, 0.0), (10.0, -10.0), 1) == pytest.approx(45.0)
    assert direction_angle((0.0, 0.0), (10.0, 10.0), 1) == pytest.approx(-45.0)


def test_forearm_angle_is_measured_toward_batter():
    """打者が画像の左（sign=-1）なら、左上向きが正になる。"""
    assert direction_angle((0.0, 0.0), (-10.0, -10.0), -1) == pytest.approx(45.0)
    assert direction_angle((0.0, 0.0), (10.0, -10.0), -1) == pytest.approx(135.0)


def test_classify_direction_uses_tolerance():
    assert classify_direction(30.0) == "upward"
    assert classify_direction(-30.0) == "downward"
    assert classify_direction(5.0) == "horizontal"
    assert classify_direction(None) is None


def test_line_angle_is_direction_free():
    assert line_angle((0.0, 0.0), (10.0, -10.0), 1) == pytest.approx(
        line_angle((10.0, -10.0), (0.0, 0.0), 1)
    )


def test_angle_difference_folds_into_quarter_turn():
    """80度と-80度の線は、鋭角側で見れば20度しか違わない（符号は回転の向き）。"""
    assert angle_difference(80.0, -80.0) == pytest.approx(-20.0)
    assert angle_difference(30.0, 10.0) == pytest.approx(20.0)


def test_missing_points_return_none():
    assert joint_angle(None, (1.0, 1.0), (2.0, 2.0)) is None
    assert joint_angle((math.nan, 0.0), (1.0, 1.0), (2.0, 2.0)) is None
    assert direction_angle((0.0, 0.0), None, 1) is None


def test_zero_length_vector_returns_none():
    """ゼロ除算を起こさない。"""
    assert joint_angle((1.0, 1.0), (1.0, 1.0), (2.0, 2.0)) is None


def test_facing_sign_rejects_unknown_value():
    with pytest.raises(ValueError):
        facing_sign("front")
