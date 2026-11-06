"""時系列追跡・補完のテスト（指示書 §8 の9ケース、YOLO / torch 不要）。

人工の座標データだけを使う。特に、補完できないときに架空の軌道を作らないことを確かめる。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temporal_experiment.config import KalmanConfig, TrackerConfig
from temporal_experiment.interpolate import fit_fill, linear_fill
from temporal_experiment.methods import baseline_estimates, run_kalman, run_nearest
from temporal_experiment.schema import (
    DETECTED,
    INTERPOLATED,
    MISSING,
    PREDICTED,
    Candidate,
    FrameRecord,
)

FPS = 30.0
CONFIG = TrackerConfig(max_speed_px_s=1200.0, max_missed_frames=5)
KALMAN = KalmanConfig()


def cand(x, y, conf=0.8, size=10.0):
    half = size / 2
    return Candidate(x - half, y - half, x + half, y + half, conf)


def line_xy(frame):
    """10px/フレームで右へ、4px/フレームで下へ動く蓋の位置"""
    return 100.0 + 10.0 * frame, 200.0 + 4.0 * frame


def records(n, drop=(), extra=None, fps=FPS, position=line_xy):
    extra = extra or {}
    result = []
    for frame in range(n):
        found = () if frame in drop else (cand(*position(frame)),)
        result.append(FrameRecord(frame, frame / fps, found + tuple(extra.get(frame, ()))))
    return tuple(result)


def statuses(estimates):
    return [e.status for e in estimates]


def run_all_position_methods(recs):
    nearest = run_nearest(recs, CONFIG).estimates
    return {
        "linear": linear_fill(nearest, CONFIG.max_missed_frames),
        "kalman": run_kalman(recs, CONFIG, KALMAN).estimates,
    }


# ケース1: すべてのフレームで検出できる
def test_all_frames_detected_stay_detected():
    recs = records(20)
    for name, estimates in run_all_position_methods(recs).items():
        assert statuses(estimates) == [DETECTED] * 20, name
        assert (estimates[7].x, estimates[7].y) == line_xy(7), name


# ケース2: 1フレームのみ欠損する
def test_single_gap_is_interpolated_linearly():
    estimates = run_all_position_methods(records(20, drop={10}))["linear"]
    assert estimates[10].status == INTERPOLATED
    assert (estimates[10].x, estimates[10].y) == pytest.approx(line_xy(10))


def test_single_gap_is_predicted_by_kalman():
    estimates = run_all_position_methods(records(20, drop={10}))["kalman"]
    assert estimates[10].status == PREDICTED
    assert (estimates[10].x, estimates[10].y) == pytest.approx(line_xy(10), abs=2.0)


def test_estimated_frames_have_no_confidence():
    for estimates in run_all_position_methods(records(20, drop={10})).values():
        assert estimates[10].conf is None


# ケース3: 3〜5フレーム連続で欠損する
@pytest.mark.parametrize("length", [3, 4, 5])
def test_consecutive_gap_within_limit_is_filled(length):
    drop = set(range(10, 10 + length))
    methods = run_all_position_methods(records(25, drop=drop))
    assert all(methods["linear"][f].status == INTERPOLATED for f in drop)
    assert all(methods["kalman"][f].status == PREDICTED for f in drop)


def test_gap_over_limit_is_not_bridged():
    drop = set(range(10, 18))  # 上限5を超える8フレーム
    methods = run_all_position_methods(records(30, drop=drop))
    assert all(methods["linear"][f].status == MISSING for f in drop)
    assert methods["kalman"][17].status == MISSING


def test_kalman_stops_predicting_at_limit():
    drop = set(range(10, 18))
    estimates = run_all_position_methods(records(30, drop=drop))["kalman"]
    predicted = [e.frame for e in estimates if e.status == PREDICTED]
    assert predicted == [10, 11, 12, 13, 14]


# ケース4: 動画の先頭・末尾で欠損する
def test_head_gap_is_never_filled():
    for name, estimates in run_all_position_methods(records(20, drop={0, 1, 2})).items():
        assert statuses(estimates)[:3] == [MISSING] * 3, name


def test_tail_gap_is_not_interpolated_without_future_observation():
    estimates = run_all_position_methods(records(20, drop={17, 18, 19}))["linear"]
    assert statuses(estimates)[17:] == [MISSING] * 3


def test_tail_gap_is_labelled_predicted_by_kalman():
    estimates = run_all_position_methods(records(20, drop={17, 18, 19}))["kalman"]
    assert statuses(estimates)[17:] == [PREDICTED] * 3


# ケース5: 誤検出が突然出現する
def test_sudden_false_detection_is_not_followed():
    far = {10: (cand(900.0, 50.0, conf=0.95),)}
    for name, estimates in run_all_position_methods(records(20, extra=far)).items():
        assert (estimates[10].x, estimates[10].y) == pytest.approx(line_xy(10)), name


def test_false_detection_during_gap_is_not_adopted():
    far = {10: (cand(900.0, 50.0, conf=0.95),)}
    for name, estimates in run_all_position_methods(records(20, drop={10}, extra=far)).items():
        assert estimates[10].status != DETECTED, name
        assert estimates[10].x == pytest.approx(line_xy(10)[0], abs=2.0), name


def test_isolated_false_detection_alone_gives_no_track():
    recs = tuple(
        FrameRecord(f, f / FPS, (cand(500.0, 500.0),) if f == 5 else ()) for f in range(15)
    )
    for name, estimates in run_all_position_methods(recs).items():
        assert set(statuses(estimates)) == {MISSING}, name


def test_static_false_detection_is_not_a_track():
    recs = tuple(FrameRecord(f, f / FPS, (cand(500.0, 500.0),)) for f in range(15))
    for name, estimates in run_all_position_methods(recs).items():
        assert set(statuses(estimates)) == {MISSING}, name


# ケース6: 1フレームに複数候補が存在する
def test_closest_candidate_wins_over_higher_confidence():
    x, y = line_xy(10)
    near_false = {10: (cand(x + 40.0, y + 40.0, conf=0.99),)}
    for name, estimates in run_all_position_methods(records(20, extra=near_false)).items():
        assert (estimates[10].x, estimates[10].y) == pytest.approx((x, y)), name


def test_baseline_takes_highest_confidence_candidate():
    x, y = line_xy(10)
    recs = records(20, extra={10: (cand(x + 40.0, y + 40.0, conf=0.99),)})
    assert baseline_estimates(recs)[10].x == pytest.approx(x + 40.0)


# ケース7: 蓋の移動方向が急変する
def turning_xy(frame):
    if frame <= 10:
        return 100.0 + 20.0 * frame, 300.0
    return 300.0 - 20.0 * (frame - 10), 300.0


def test_kalman_records_breakdown_on_sudden_turn():
    result = run_kalman(records(22, position=turning_xy), CONFIG, KALMAN)
    rejected = [e.frame for e in result.events if e.kind == "gate_rejected"]
    assert rejected and min(rejected) >= 11


def test_kalman_reacquires_after_sudden_turn():
    estimates = run_kalman(records(22, position=turning_xy), CONFIG, KALMAN).estimates
    assert estimates[-1].status == DETECTED
    assert (estimates[-1].x, estimates[-1].y) == pytest.approx(turning_xy(21))


# ケース8: フレームレートが異なる
def test_speed_gate_uses_elapsed_time():
    slow = records(12, fps=30.0)  # 約323px/s
    fast = records(12, fps=240.0)  # 約2585px/s（上限1200px/s超）
    assert DETECTED in statuses(run_nearest(slow, CONFIG).estimates)
    assert DETECTED not in statuses(run_nearest(fast, CONFIG).estimates)


def test_kalman_prediction_follows_timestamps_at_60fps():
    def half_step(frame):
        return 100.0 + 5.0 * frame, 200.0 + 2.0 * frame

    recs = records(30, drop={15}, fps=60.0, position=half_step)
    estimates = run_kalman(recs, CONFIG, KALMAN).estimates
    assert (estimates[15].x, estimates[15].y) == pytest.approx(half_step(15), abs=2.0)


def test_linear_interpolation_uses_timestamps_not_frame_count():
    recs = (
        FrameRecord(0, 0.00, (cand(0.0, 0.0),)),
        FrameRecord(1, 0.01, (cand(10.0, 0.0),)),
        FrameRecord(2, 0.02, ()),
        FrameRecord(3, 0.08, ()),
        FrameRecord(4, 0.10, (cand(100.0, 0.0),)),
    )
    estimates = linear_fill(run_nearest(recs, CONFIG).estimates, 5)
    assert estimates[3].x == pytest.approx(80.0)


# ケース9: 動画内に蓋が存在しない
def test_no_cap_in_video_gives_only_missing():
    recs = tuple(FrameRecord(f, f / FPS, ()) for f in range(10))
    assert set(statuses(baseline_estimates(recs))) == {MISSING}
    for name, estimates in run_all_position_methods(recs).items():
        assert set(statuses(estimates)) == {MISSING}, name


def test_empty_video_gives_no_estimates():
    assert baseline_estimates(()) == ()
    for estimates in run_all_position_methods(()).values():
        assert estimates == ()


# 別の投球をつながない
def test_two_separate_tracks_are_not_interpolated_together():
    def second(frame):
        return 900.0 - 10.0 * frame, 100.0 + 4.0 * frame

    recs = tuple(
        FrameRecord(f, f / FPS, (cand(*(line_xy(f) if f < 8 else second(f))),) if f not in (8, 9) else ())
        for f in range(16)
    )
    estimates = linear_fill(run_nearest(recs, CONFIG).estimates, 5)
    assert statuses(estimates)[8:10] == [MISSING, MISSING]


# 入力を書き換えない
def test_linear_fill_does_not_mutate_input():
    nearest = run_nearest(records(20, drop={10}), CONFIG).estimates
    before = tuple(nearest)
    linear_fill(nearest, 5)
    assert nearest == before


# RANSAC（既存の trajectory_fitter を使う比較手法）
def curve_xy(frame):
    return 100.0 + 10.0 * frame, 400.0 - 12.0 * frame + 0.3 * frame * frame


def test_fit_fill_fills_gap_along_curve():
    nearest = run_nearest(records(20, drop={10}, position=curve_xy), CONFIG).estimates
    estimates = fit_fill(nearest, 5, inlier_px=5.0)
    assert estimates[10].status == INTERPOLATED
    assert (estimates[10].x, estimates[10].y) == pytest.approx(curve_xy(10), abs=0.5)


def test_fit_fill_keeps_short_track_without_fitting():
    nearest = run_nearest(records(4), CONFIG).estimates
    assert fit_fill(nearest, 5, inlier_px=5.0) == nearest


def test_fit_fill_drops_point_off_the_curve():
    x, y = curve_xy(10)
    recs = records(20, drop={10}, position=curve_xy, extra={10: (cand(x + 30.0, y),)})
    nearest = run_nearest(recs, CONFIG).estimates
    estimates = fit_fill(nearest, 5, inlier_px=5.0)
    assert estimates[10].status == INTERPOLATED
    assert estimates[10].x == pytest.approx(x, abs=0.5)


# フレーム番号が飛んだ入力でも別の投球をつながない
def test_skipped_frames_count_as_misses():
    frames = list(range(10)) + list(range(100, 110))
    recs = tuple(FrameRecord(f, f / FPS, (cand(100.0 + 10.0 * (f % 100), 200.0),)) for f in frames)
    for name, result in (
        ("nearest", run_nearest(recs, CONFIG)),
        ("kalman", run_kalman(recs, CONFIG, KALMAN)),
    ):
        estimates = result.estimates
        assert estimates[0].track_id != estimates[-1].track_id, name


# 固定物（壁のロゴなど）に軌道を乗っ取られない
WIDE = TrackerConfig(max_speed_px_s=2500.0, max_missed_frames=5)
LOGO = (500.0, 300.0)


def test_track_does_not_jump_to_far_object_after_gap():
    def position(frame):
        return line_xy(frame) if frame < 8 else LOGO

    recs = records(25, drop={8, 9, 10}, position=position)
    estimates = linear_fill(run_nearest(recs, WIDE).estimates, 5)
    assert statuses(estimates)[8:11] == [MISSING] * 3


def passing_cap_records():
    """静止した誤検出のそばを蓋が通り過ぎる"""
    result = []
    for frame in range(60):
        found = []
        if frame != 40:
            found.append(cand(300.0, 300.0, conf=0.9))
        if 30 <= frame < 50:
            found.append(cand(310.0 + 20.0 * (frame - 40), 300.0))
        result.append(FrameRecord(frame, frame / FPS, tuple(found)))
    return tuple(result)


def test_static_history_is_not_emitted_when_cap_passes():
    recs = passing_cap_records()
    for name, result in (
        ("nearest", run_nearest(recs, WIDE)),
        ("kalman", run_kalman(recs, WIDE, KALMAN)),
    ):
        assert set(statuses(result.estimates[:30])) == {MISSING}, name


def test_cap_keeps_priority_over_static_object():
    recs = passing_cap_records()
    for name, result in (
        ("nearest", run_nearest(recs, WIDE)),
        ("kalman", run_kalman(recs, WIDE, KALMAN)),
    ):
        assert result.estimates[35].x == pytest.approx(210.0), name


def test_slowly_drifting_object_is_not_a_track():
    recs = tuple(
        FrameRecord(f, f / FPS, (cand(300.0 + f * 0.02, 300.0),)) for f in range(300)
    )
    assert set(statuses(run_nearest(recs, WIDE).estimates)) == {MISSING}


def test_longer_true_track_wins_over_short_false_track():
    false_track = {f: (cand(800.0 + 10.0 * f, 600.0, conf=0.9),) for f in range(8, 13)}
    recs = records(25, drop=set(range(10)) | {13}, extra=false_track)
    estimates = linear_fill(run_nearest(recs, WIDE).estimates, 5)
    assert estimates[11].x == pytest.approx(line_xy(11)[0])
    assert estimates[13].status == INTERPOLATED


def test_static_object_after_flight_gets_no_prediction():
    def position(frame):
        return line_xy(min(frame, 10))

    recs = records(25, drop=set(range(20, 25)), position=position)
    estimates = run_kalman(recs, WIDE, KALMAN).estimates
    assert PREDICTED not in statuses(estimates)[20:]


def test_nearest_tries_next_candidate_when_closest_fails_gate():
    recs = (
        FrameRecord(0, 0.0, (cand(100.0, 100.0),)),
        FrameRecord(1, 0.1, (cand(130.0, 100.0),)),
        FrameRecord(2, 0.2, (cand(161.0, 100.0), cand(150.0, 100.0))),
    )
    estimates = run_nearest(recs, TrackerConfig(max_speed_px_s=300.0)).estimates
    assert estimates[2].x == pytest.approx(150.0)


def test_cap_track_keeps_detection_when_static_object_vanishes():
    recs = passing_cap_records()
    for name, result in (
        ("nearest", run_nearest(recs, WIDE)),
        ("kalman", run_kalman(recs, WIDE, KALMAN)),
    ):
        assert result.estimates[40].track_id == result.estimates[39].track_id, name


def test_long_static_prefix_of_a_moving_track_is_not_emitted():
    def position(frame):
        return (300.0 + 20.0 * max(frame - 20, 0), 300.0)

    estimates = run_nearest(records(32, position=position), WIDE).estimates
    assert set(statuses(estimates[:15])) == {MISSING}
    assert estimates[25].status == DETECTED
