"""trajectory_fitter のユニットテスト

実データを模した合成軌跡（60fps・25フレーム・ノイズ・欠損・誤検出入り）で
フィット・補間・コース判定・球速推定を検証する。
"""

import random

import numpy as np
import pytest

from trajectory_fitter import (
    MIN_POINTS_FOR_FIT,
    TrajectoryPoint,
    analyze_pitch,
    estimate_speed_kmh,
    interpolate_gaps,
    position_at,
    ransac_fit,
    zone_from_position,
)

FPS = 60.0
NUM_FRAMES = 25  # リリース〜捕球


def make_true_trajectory(cx=(0.3, -1.2, 0.5), cy=(0.9, 0.2, -2.0)):
    """2次式に従う真の軌跡を生成する（正規化座標系）"""
    points = []
    for frame in range(NUM_FRAMES):
        t = frame / FPS
        x = cx[0] + cx[1] * t + cx[2] * t * t
        y = cy[0] + cy[1] * t + cy[2] * t * t
        points.append(TrajectoryPoint(frame=frame, t=t, x=x, y=y))
    return points


def add_noise(points, sigma=0.01, seed=1):
    rng = random.Random(seed)
    return [
        TrajectoryPoint(
            frame=p.frame, t=p.t,
            x=p.x + rng.gauss(0, sigma),
            y=p.y + rng.gauss(0, sigma),
        )
        for p in points
    ]


def drop_frames(points, drop_ratio=0.4, seed=2):
    """検出漏れを模して中間フレームを欠損させる（端点=リリース・捕球点は保持）"""
    rng = random.Random(seed)
    kept = [
        p for i, p in enumerate(points)
        if i in (0, len(points) - 1) or rng.random() > drop_ratio
    ]
    # フィット可能な点数は必ず残す
    assert len(kept) >= MIN_POINTS_FOR_FIT
    return kept


def add_outliers(points, num=3, seed=3):
    """誤検出（軌跡から大きく外れた点）を混入させる"""
    rng = random.Random(seed)
    outliers = [
        TrajectoryPoint(
            frame=p.frame, t=p.t,
            x=p.x + rng.choice([-1, 1]) * rng.uniform(1.0, 2.0),
            y=p.y + rng.choice([-1, 1]) * rng.uniform(1.0, 2.0),
        )
        for p in rng.sample(points, num)
    ]
    return sorted(points + outliers, key=lambda p: p.frame)


class TestRansacFit:
    def test_clean_data_recovers_coefficients(self):
        points = make_true_trajectory()
        fit = ransac_fit(points)
        assert fit.coeffs_x == pytest.approx((0.3, -1.2, 0.5), abs=1e-6)
        assert fit.coeffs_y == pytest.approx((0.9, 0.2, -2.0), abs=1e-6)
        assert len(fit.outliers) == 0

    def test_noisy_sparse_data_still_fits(self):
        points = drop_frames(add_noise(make_true_trajectory()))
        fit = ransac_fit(points)
        # ノイズσ=0.01に対して係数の定数項が近いこと
        assert fit.coeffs_x[0] == pytest.approx(0.3, abs=0.05)
        assert fit.coeffs_y[0] == pytest.approx(0.9, abs=0.05)
        assert fit.rmse < 0.05

    def test_outliers_are_rejected(self):
        clean = add_noise(make_true_trajectory())
        contaminated = add_outliers(clean, num=4)
        fit = ransac_fit(contaminated)
        assert len(fit.outliers) == 4
        # 誤検出込みでも係数が汚染されないこと
        assert fit.coeffs_x[0] == pytest.approx(0.3, abs=0.05)

    def test_too_few_points_raises(self):
        points = make_true_trajectory()[:MIN_POINTS_FOR_FIT - 1]
        with pytest.raises(ValueError, match="最低"):
            ransac_fit(points)

    def test_all_same_time_raises(self):
        points = [
            TrajectoryPoint(frame=i, t=0.0, x=0.0, y=0.0) for i in range(10)
        ]
        with pytest.raises(ValueError, match="同一時刻"):
            ransac_fit(points)

    def test_deterministic_with_same_seed(self):
        points = add_outliers(add_noise(make_true_trajectory()))
        fit1 = ransac_fit(points, seed=42)
        fit2 = ransac_fit(points, seed=42)
        assert fit1.coeffs_x == fit2.coeffs_x
        assert fit1.coeffs_y == fit2.coeffs_y


class TestInterpolateGaps:
    def test_fills_missing_frames(self):
        points = drop_frames(make_true_trajectory())
        fit = ransac_fit(points)
        completed = interpolate_gaps(fit, FPS)
        frames = [p.frame for p in completed]
        # 範囲内が連番で埋まっていること
        assert frames == list(range(min(frames), max(frames) + 1))

    def test_interpolated_positions_match_truth(self):
        truth = {p.frame: p for p in make_true_trajectory()}
        points = drop_frames(list(truth.values()))
        fit = ransac_fit(points)
        completed = interpolate_gaps(fit, FPS)
        for p in completed:
            assert p.x == pytest.approx(truth[p.frame].x, abs=1e-6)
            assert p.y == pytest.approx(truth[p.frame].y, abs=1e-6)

    def test_invalid_fps_raises(self):
        fit = ransac_fit(make_true_trajectory())
        with pytest.raises(ValueError, match="fps"):
            interpolate_gaps(fit, 0)


class TestZoneFromPosition:
    @pytest.mark.parametrize("x,y,expected", [
        (-0.4, 0.9, 1),   # 左上
        (0.0, 0.9, 2),    # 中上
        (0.4, 0.9, 3),    # 右上
        (-0.4, 0.5, 4),
        (0.0, 0.5, 5),    # ど真ん中
        (0.4, 0.5, 6),
        (-0.4, 0.1, 7),   # 左下
        (0.0, 0.1, 8),
        (0.4, 0.1, 9),    # 右下
    ])
    def test_nine_zones(self, x, y, expected):
        assert zone_from_position(x, y) == expected

    @pytest.mark.parametrize("x,y", [
        (-0.6, 0.5),  # 左外
        (0.6, 0.5),   # 右外
        (0.0, 1.1),   # 高め外
        (0.0, -0.1),  # 低め外
    ])
    def test_outside_is_ball(self, x, y):
        assert zone_from_position(x, y) == "ball"

    def test_boundary_is_strike(self):
        # ゾーン境界ぴったりはストライク扱い
        assert zone_from_position(0.5, 0.0) == 9
        assert zone_from_position(-0.5, 1.0) == 1


class TestEstimateSpeed:
    def test_speed_from_duration(self):
        # 25フレーム@60fps = 0.4秒で9.22m → 23.05m/s → 82.98km/h
        fit = ransac_fit(make_true_trajectory())
        expected = 9.22 / (24 / FPS) * 3.6
        assert estimate_speed_kmh(fit) == pytest.approx(expected, rel=1e-6)


class TestAnalyzePitch:
    def _to_json_format(self, points):
        return [
            {"frame": p.frame, "time": p.t, "x": p.x, "y": p.y} for p in points
        ]

    def test_full_pipeline(self):
        # 真の通過点（t=24/60）: x=0.3-1.2*0.4+0.5*0.16=-0.1, y=0.9-2.0*0.16=0.58
        truth = make_true_trajectory(cy=(0.9, 0.0, -2.0))
        points = add_outliers(drop_frames(add_noise(truth)), num=2)
        result = analyze_pitch(self._to_json_format(points), FPS)

        assert result["course_zone"] == 5  # (-0.1, 0.58)は中段中央
        assert result["course_x"] == pytest.approx(-0.1, abs=0.05)
        assert result["course_y"] == pytest.approx(0.58, abs=0.05)
        assert result["num_outliers"] == 2
        assert result["speed_kmh"] > 0
        # 補間で軌跡が実検出より密になっていること
        assert len(result["trajectory"]) >= result["num_inliers"]

    def test_ball_course(self):
        # 大きく外れるコース（x切片を右に外す）
        points = make_true_trajectory(cx=(1.0, 0.5, 0.0), cy=(0.5, 0.0, 0.0))
        result = analyze_pitch(self._to_json_format(points), FPS)
        assert result["course_zone"] == "ball"

    def test_insufficient_data_raises(self):
        points = make_true_trajectory()[:3]
        with pytest.raises(ValueError):
            analyze_pitch(self._to_json_format(points), FPS)
