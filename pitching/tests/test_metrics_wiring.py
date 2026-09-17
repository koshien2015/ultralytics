"""手で置いた座標で、特徴量の配線と符号を固定するテスト。

合成データ生成器（support.py）と解析側が同じ約束事を共有しているため、
生成器を通したテストだけでは符号の取り違えを検出できない。
ここでは期待値を手計算した座標を直接置く。
"""

from __future__ import annotations

import numpy as np
import pytest

from pitching.analysis import analyze_pitch
from pitching.metrics.torso import torso_series
from pitching.models import EventName, Keypoint, PoseFrame, PoseSeries
from pitching.tests.support import make_series

# 肘を頂点として直角になる配置（肩→肘ベクトルに垂直な前腕）
RIGHT_SHOULDER = (100.0, 100.0)
RIGHT_ELBOW = (140.0, 120.0)
RIGHT_WRIST_90DEG = (160.0, 80.0)
# 肩・肘・手首が一直線になる配置
LEFT_SHOULDER = (70.0, 102.0)
LEFT_ELBOW = (60.0, 130.0)
LEFT_WRIST_180DEG = (50.0, 158.0)


def make_arm_series(frames: int = 12) -> PoseSeries:
    """右腕が90度、左腕が180度の姿勢を並べた時系列。"""
    points = {
        "nose": (95.0, 70.0),
        "right_shoulder": RIGHT_SHOULDER,
        "left_shoulder": LEFT_SHOULDER,
        "right_elbow": RIGHT_ELBOW,
        "left_elbow": LEFT_ELBOW,
        "right_wrist": RIGHT_WRIST_90DEG,
        "left_wrist": LEFT_WRIST_180DEG,
        "right_hip": (98.0, 170.0),
        "left_hip": (72.0, 172.0),
        "right_knee": (100.0, 215.0),
        "left_knee": (68.0, 218.0),
        "right_ankle": (102.0, 260.0),
        "left_ankle": (60.0, 262.0),
    }
    return PoseSeries(
        frames=[
            PoseFrame(
                index,
                index / 60.0,
                {name: Keypoint(x, y, 0.9) for name, (x, y) in points.items()},
            )
            for index in range(frames)
        ],
        fps=60.0,
        pitch_id="wiring",
    )


def config_for(hand: str, pitch_config):
    config = pitch_config.model_copy(deep=True)
    config.throwing_hand = hand
    config.start_frame = 0
    config.end_frame = 11
    config.events.stride_foot_contact_frame = 2
    config.events.release_frame = 8
    return config


def test_throwing_hand_selects_the_right_arm(pitch_config):
    """右投げ設定なら右腕（90度）、左投げ設定なら左腕（180度）を見る。"""
    series = make_arm_series()

    right = analyze_pitch(series, config_for("right", pitch_config))
    left = analyze_pitch(series, config_for("left", pitch_config))

    assert right.features["elbow"]["elbow_angle_at_release_deg"] == pytest.approx(90.0, abs=0.5)
    assert left.features["elbow"]["elbow_angle_at_release_deg"] == pytest.approx(180.0, abs=0.5)


def test_lead_leg_is_opposite_to_throwing_hand(pitch_config):
    """踏み出し脚は投球腕の反対側。"""
    config = config_for("right", pitch_config)

    assert config.side_prefix(throwing=True) == "right"
    assert config.side_prefix(throwing=False) == "left"


def build_trunk_tracks(shoulder_x: float):
    """肩中点のX位置だけを変えた1フレームの tracks を作る。"""
    from pitching.preprocessing.tracks import from_series

    points = {
        "left_shoulder": (shoulder_x - 15.0, 100.0),
        "right_shoulder": (shoulder_x + 15.0, 100.0),
        "left_hip": (85.0, 170.0),
        "right_hip": (115.0, 170.0),
    }
    frame = PoseFrame(0, 0.0, {name: Keypoint(x, y, 0.9) for name, (x, y) in points.items()})
    return from_series(PoseSeries(frames=[frame], fps=60.0))


def test_trunk_lean_is_zero_when_upright():
    # 打者は画像の左（sign=-1）
    series = torso_series(build_trunk_tracks(shoulder_x=100.0), sign=-1)

    assert series["trunk_lean_deg"][0] == pytest.approx(0.0, abs=1e-6)


def test_trunk_lean_sign_distinguishes_the_two_directions():
    """打者側への傾きと反対側への傾きが、同じ値にならない。"""
    sign = -1  # 打者は画像の左
    toward_batter = torso_series(build_trunk_tracks(shoulder_x=80.0), sign)["trunk_lean_deg"][0]
    away_from_batter = torso_series(build_trunk_tracks(shoulder_x=120.0), sign)["trunk_lean_deg"][0]

    assert toward_batter > 0
    assert away_from_batter < 0
    assert toward_batter == pytest.approx(-away_from_batter, abs=1e-6)


def test_wrist_lead_measures_forward_position(pitch_config):
    """手首が肘を追い越すタイミングは、角度ではなく前後位置で測る。"""
    series = make_arm_series()
    # 打者は画像の左（sign=-1）。右手首(160) は右肘(140) より打者と反対側にある。
    analysis = analyze_pitch(series, config_for("right", pitch_config))

    assert analysis.series["wrist_lead_px"][0] == pytest.approx(-20.0)
    assert analysis.features["forearm"]["wrist_passes_elbow_frame_offset"] is None


def test_wrist_lead_is_positive_when_wrist_is_ahead(pitch_config):
    series = make_arm_series()
    config = config_for("right", pitch_config)
    config.batter_direction = "right"  # 打者を反対側にすると符号が反転する

    analysis = analyze_pitch(series, config)

    assert analysis.series["wrist_lead_px"][0] == pytest.approx(20.0)
    assert analysis.features["forearm"]["wrist_passes_elbow_frame_offset"] == 0


def test_progress_is_nan_outside_the_contact_to_release_span(pitch_config):
    """区間外に外挿した進行率は出さない（フレーム番号で辿る）。"""
    analysis = analyze_pitch(make_series([170.0 - index * 4 for index in range(20)]), pitch_config)

    contact = analysis.position_of_event(EventName.FOOT_CONTACT)
    release = analysis.position_of_event(EventName.RELEASE)

    assert np.isnan(analysis.progress_percent[contact - 1])
    assert np.isnan(analysis.progress_percent[release + 1])
    assert np.isfinite(analysis.progress_percent[contact : release + 1]).all()
