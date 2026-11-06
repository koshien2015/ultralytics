"""評価指標のテスト。出力しなかったフレームで精度が良く見えないことを確かめる。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temporal_experiment.metrics import MetricsConfig, detector_recall, evaluate
from temporal_experiment.schema import (
    DETECTED,
    INTERPOLATED,
    MISSING,
    PREDICTED,
    Candidate,
    Estimate,
    FrameRecord,
    GroundTruth,
)

CONFIG = MetricsConfig(match_radius_px=10.0, error_cap_px=100.0)


def gt(frame, x=100.0, y=100.0, visible=True):
    return GroundTruth(frame, x, y, visible)


def est(frame, status, x=100.0, y=100.0):
    if status == MISSING:
        return Estimate(frame, frame / 30, MISSING)
    return Estimate(frame, frame / 30, status, x, y)


def test_coverage_counts_output_over_visible_frames():
    truth = [gt(0), gt(1), gt(2), gt(3)]
    estimates = [est(0, DETECTED), est(1, INTERPOLATED), est(2, MISSING), est(3, PREDICTED)]
    assert evaluate(estimates, truth, CONFIG).coverage == pytest.approx(0.75)


def test_invisible_frames_are_not_in_coverage_denominator():
    truth = [gt(0), gt(1, visible=False)]
    estimates = [est(0, DETECTED), est(1, MISSING)]
    assert evaluate(estimates, truth, CONFIG).coverage == pytest.approx(1.0)


def test_output_errors_use_euclidean_distance():
    result = evaluate([est(0, DETECTED, 103.0, 104.0)], [gt(0)], CONFIG)
    assert result.mae_output == pytest.approx(5.0)
    assert result.rmse_output == pytest.approx(5.0)


def test_missing_frames_worsen_all_frame_error_not_output_error():
    truth = [gt(0), gt(1)]
    estimates = [est(0, DETECTED), est(1, MISSING)]
    result = evaluate(estimates, truth, CONFIG)
    assert result.mae_output == pytest.approx(0.0)
    assert result.mae_all == pytest.approx(50.0)


def test_all_frame_error_is_capped():
    result = evaluate([est(0, DETECTED, 1000.0, 100.0)], [gt(0)], CONFIG)
    assert result.mae_all == pytest.approx(100.0)
    assert result.mae_output == pytest.approx(900.0)


def test_percentiles_are_reported():
    truth = [gt(f) for f in range(11)]
    estimates = [est(f, DETECTED, 100.0 + f, 100.0) for f in range(11)]
    result = evaluate(estimates, truth, CONFIG)
    assert result.p90_output == pytest.approx(9.0)
    assert result.p95_output == pytest.approx(9.5)


def test_longest_missing_run_is_within_visible_frames():
    truth = [gt(f) for f in range(8)]
    statuses = [DETECTED, MISSING, MISSING, DETECTED, MISSING, MISSING, MISSING, DETECTED]
    estimates = [est(f, s) for f, s in enumerate(statuses)]
    assert evaluate(estimates, truth, CONFIG).max_missing_run == 3


def test_missing_run_does_not_span_unlabelled_frames():
    truth = [gt(0), gt(1), gt(50), gt(51)]
    estimates = [est(0, MISSING), est(1, MISSING), est(50, MISSING), est(51, MISSING)]
    assert evaluate(estimates, truth, CONFIG).max_missing_run == 2


def test_far_output_counts_as_wrong_association():
    truth = [gt(0), gt(1)]
    estimates = [est(0, DETECTED), est(1, PREDICTED, 150.0, 100.0)]
    assert evaluate(estimates, truth, CONFIG).wrong_outputs == 1


def test_output_on_invisible_frame_counts_as_phantom():
    truth = [gt(0, visible=False), gt(1, visible=False)]
    estimates = [est(0, PREDICTED), est(1, MISSING)]
    result = evaluate(estimates, truth, CONFIG)
    assert result.phantom_outputs == 1
    assert result.coverage is None


def test_status_breakdown_keeps_detected_and_estimated_apart():
    truth = [gt(0), gt(1), gt(2)]
    estimates = [
        est(0, DETECTED, 101.0, 100.0),
        est(1, INTERPOLATED, 104.0, 100.0),
        est(2, PREDICTED, 108.0, 100.0),
    ]
    result = evaluate(estimates, truth, CONFIG)
    assert result.status_counts == {DETECTED: 1, INTERPOLATED: 1, PREDICTED: 1, MISSING: 0}
    assert result.mae_by_status[INTERPOLATED] == pytest.approx(4.0)
    assert result.mae_by_status[DETECTED] == pytest.approx(1.0)


def test_frame_without_estimate_counts_as_missing():
    assert evaluate([], [gt(0)], CONFIG).coverage == pytest.approx(0.0)


def box(x, y, conf=0.5):
    return Candidate(x - 5, y - 5, x + 5, y + 5, conf)


def test_recall_accepts_any_candidate_near_truth():
    recs = [
        FrameRecord(0, 0.0, (box(500.0, 500.0, 0.9), box(103.0, 100.0, 0.2))),
        FrameRecord(1, 0.1, (box(500.0, 500.0, 0.9),)),
    ]
    assert detector_recall(recs, [gt(0), gt(1)], CONFIG) == pytest.approx(0.5)


def test_recall_ignores_invisible_frames():
    recs = [FrameRecord(0, 0.0, ()), FrameRecord(1, 0.1, (box(100.0, 100.0),))]
    assert detector_recall(recs, [gt(0, visible=False), gt(1)], CONFIG) == pytest.approx(1.0)


def test_all_frame_percentile_counts_missing_frames():
    truth = [gt(f) for f in range(10)]
    estimates = [est(0, DETECTED)] + [est(f, MISSING) for f in range(1, 10)]
    result = evaluate(estimates, truth, CONFIG)
    assert result.p95_output == pytest.approx(0.0)
    assert result.p95_all == pytest.approx(100.0)


def test_predictions_after_last_detection_are_counted_as_tail():
    def tracked(frame, status):
        return Estimate(frame, frame / 30, status, 100.0, 100.0, track_id=1)

    truth = [gt(f) for f in range(5)]
    estimates = [
        tracked(0, DETECTED), tracked(1, PREDICTED), tracked(2, DETECTED),
        tracked(3, PREDICTED), tracked(4, PREDICTED),
    ]
    result = evaluate(estimates, truth, CONFIG)
    assert result.predicted_bridged == 1
    assert result.predicted_tail == 2
