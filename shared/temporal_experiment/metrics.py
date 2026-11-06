"""評価指標。定義は docs/temporal-tracking-experiment.md の「評価指標の定義」と対応する。"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .schema import DETECTED, INTERPOLATED, MISSING, PREDICTED, STATUSES


@dataclass(frozen=True)
class MetricsConfig:
    match_radius_px: float
    error_cap_px: float


@dataclass(frozen=True)
class MethodMetrics:
    n_visible: int
    coverage: Optional[float]  # 位置を出力した割合（分母は蓋が見えているフレーム）
    correct_coverage: Optional[float]  # 正解の近くに位置を出力した割合
    mae_output: Optional[float]  # 位置を出力したフレームだけの誤差
    rmse_output: Optional[float]
    p90_output: Optional[float]
    p95_output: Optional[float]
    mae_all: Optional[float]  # 欠損と大外れを error_cap_px として全フレームで計算した誤差
    rmse_all: Optional[float]
    p95_all: Optional[float]
    max_missing_run: int
    wrong_outputs: int  # 正解から match_radius_px を超えて離れた出力
    phantom_outputs: int  # 蓋が写っていないフレームへの出力（架空軌道）
    status_counts: dict
    mae_by_status: dict
    predicted_bridged: int  # 同じ軌道の検出に挟まれた予測
    predicted_tail: int  # 軌道の最後の検出より後の予測。後の観測で裏付けられない


def _distance(estimate, truth) -> float:
    return float(np.hypot(estimate.x - truth.x, estimate.y - truth.y))


def _mean(values) -> Optional[float]:
    return float(np.mean(values)) if len(values) else None


def _rmse(values) -> Optional[float]:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def _percentile(values, q) -> Optional[float]:
    return float(np.percentile(values, q)) if len(values) else None


def _longest_missing_run(visible, by_frame) -> int:
    longest = run = 0
    previous = None
    for truth in visible:
        if previous is not None and truth.frame != previous + 1:
            run = 0
        estimate = by_frame.get(truth.frame)
        run = run + 1 if estimate is None or not estimate.has_position else 0
        longest = max(longest, run)
        previous = truth.frame
    return longest


def _split_predictions(estimates, frames) -> tuple:
    """frames に含まれる予測を、検出に挟まれたものと軌道の末尾のものに分けて数える"""
    last_detected = {}
    for estimate in estimates:
        if estimate.status == DETECTED and estimate.track_id is not None:
            last_detected[estimate.track_id] = max(
                estimate.frame, last_detected.get(estimate.track_id, estimate.frame)
            )
    predicted = [e for e in estimates if e.status == PREDICTED and e.frame in frames]
    bridged = sum(e.frame < last_detected.get(e.track_id, -1) for e in predicted)
    return bridged, len(predicted) - bridged


def evaluate(estimates, truths, config: MetricsConfig) -> MethodMetrics:
    by_frame = {e.frame: e for e in estimates}
    ordered = sorted(truths, key=lambda g: g.frame)
    visible = [g for g in ordered if g.visible]
    hidden = [g for g in ordered if not g.visible]

    pairs = [
        (by_frame[g.frame], g)
        for g in visible
        if g.frame in by_frame and by_frame[g.frame].has_position
    ]
    errors = [_distance(e, g) for e, g in pairs]
    capped = [min(error, config.error_cap_px) for error in errors]
    capped_all = capped + [config.error_cap_px] * (len(visible) - len(errors))

    correct = sum(error <= config.match_radius_px for error in errors)
    status_counts = {status: 0 for status in STATUSES}
    for truth in visible:
        estimate = by_frame.get(truth.frame)
        status_counts[estimate.status if estimate else MISSING] += 1

    mae_by_status = {
        status: _mean([_distance(e, g) for e, g in pairs if e.status == status])
        for status in (DETECTED, INTERPOLATED, PREDICTED)
    }
    n_visible = len(visible)
    bridged, tail = _split_predictions(estimates, {g.frame for g in ordered})
    return MethodMetrics(
        n_visible=n_visible,
        coverage=len(errors) / n_visible if n_visible else None,
        correct_coverage=correct / n_visible if n_visible else None,
        mae_output=_mean(errors),
        rmse_output=_rmse(errors),
        p90_output=_percentile(errors, 90),
        p95_output=_percentile(errors, 95),
        mae_all=_mean(capped_all),
        rmse_all=_rmse(capped_all),
        p95_all=_percentile(capped_all, 95),
        max_missing_run=_longest_missing_run(visible, by_frame),
        wrong_outputs=len(errors) - correct,
        phantom_outputs=sum(
            1 for g in hidden if g.frame in by_frame and by_frame[g.frame].has_position
        ),
        status_counts=status_counts,
        mae_by_status=mae_by_status,
        predicted_bridged=bridged,
        predicted_tail=tail,
    )


def detector_recall(records, truths, config: MetricsConfig) -> Optional[float]:
    """YOLO 単体の Recall: 正解の近くに候補が1つでもあったフレームの割合"""
    by_frame = {r.frame: r for r in records}
    visible = [g for g in truths if g.visible]
    if not visible:
        return None

    def found(truth) -> bool:
        record = by_frame.get(truth.frame)
        if record is None:
            return False
        return any(
            np.hypot(c.cx - truth.x, c.cy - truth.y) <= config.match_radius_px
            for c in record.candidates
        )

    return sum(found(g) for g in visible) / len(visible)
