"""同じ検出結果に全手法を適用する。"""

import time
from dataclasses import dataclass

from .config import ExperimentConfig
from .interpolate import fit_fill, linear_fill
from .methods import baseline_estimates, run_kalman, run_nearest

METHODS = ("yolo", "nearest", "linear", "kalman", "ransac")


@dataclass(frozen=True)
class MethodRun:
    estimates: tuple
    events: tuple
    elapsed_ms: float


def _timed(compute) -> tuple:
    begin = time.perf_counter()
    value = compute()
    return value, (time.perf_counter() - begin) * 1000


def run_methods(records, config: ExperimentConfig) -> dict:
    """config は動画の幅に換算済みのものを渡す。linear と ransac の時間は nearest を含む"""
    tracker = config.tracker
    gap = tracker.max_missed_frames

    yolo, yolo_ms = _timed(lambda: baseline_estimates(records))
    nearest, nearest_ms = _timed(lambda: run_nearest(records, tracker))
    kalman, kalman_ms = _timed(lambda: run_kalman(records, tracker, config.kalman))
    linear, linear_ms = _timed(lambda: linear_fill(nearest.estimates, gap))
    ransac, ransac_ms = _timed(lambda: fit_fill(nearest.estimates, gap, config.fit.inlier_px))

    return {
        "yolo": MethodRun(yolo, (), yolo_ms),
        "nearest": MethodRun(nearest.estimates, nearest.events, nearest_ms),
        "linear": MethodRun(linear, nearest.events, nearest_ms + linear_ms),
        "kalman": MethodRun(kalman.estimates, kalman.events, kalman_ms),
        "ransac": MethodRun(ransac, nearest.events, nearest_ms + ransac_ms),
    }
