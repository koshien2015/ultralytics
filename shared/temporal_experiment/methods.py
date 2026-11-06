"""比較する手法の入口。"""

from .config import KalmanConfig, TrackerConfig
from .motion import KalmanMotion, NearestMotion
from .schema import DETECTED, MISSING, Estimate, TrackingResult
from .tracker import run_tracker


def baseline_estimates(records) -> tuple:
    """YOLO 単体: 各フレームで信頼度が最も高い候補を採る。前後のフレームは見ない"""

    def pick(record):
        if not record.candidates:
            return Estimate(record.frame, record.t, MISSING)
        best = max(record.candidates, key=lambda c: c.conf)
        return Estimate(record.frame, record.t, DETECTED, best.cx, best.cy, best.conf, best.box)

    return tuple(pick(record) for record in records)


def run_nearest(records, config: TrackerConfig) -> TrackingResult:
    return run_tracker(records, NearestMotion(config), config)


def run_kalman(records, tracker: TrackerConfig, kalman: KalmanConfig) -> TrackingResult:
    return run_tracker(records, KalmanMotion(kalman), tracker)
