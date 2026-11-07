"""track.py から使う線形補完。

前後の検出を使うので、全フレームの処理が終わった後に実行する。track.py の既存の出力は変えず、
結果を <動画名>_filled.json と <動画名>_filled.mp4 に別に書き出す。
_filled.mp4 は _detected.mp4 の補間したフレームに bbox を足した動画。
"""

import time
from pathlib import Path

from .config import load_config
from .interpolate import linear_fill
from .methods import run_nearest
from .pipeline import MethodRun
from .render import render_filled
from .report import save_estimates
from .schema import Candidate, FrameRecord

DEFAULT_CONFIG = Path(__file__).parent / "experiment.yaml"


def frame_record(frame: int, t: float, results, target_classes) -> FrameRecord:
    """YOLO の結果を候補に変換する。

    前段フィルタが飛ばしたフレームも空のまま残す。残さないと、追跡がそのフレームを欠損として
    数えても補完の出力に含まれず、補完する欠損の上限を判定できない。
    """
    candidates = tuple(
        Candidate(*(float(v) for v in box.xyxy[0]), float(box.conf[0]))
        for result in results
        for box in result.boxes
        if not target_classes or int(box.cls[0]) in target_classes
    )
    return FrameRecord(frame, t, candidates)


def linear_track(records, width: int, config_path: Path = DEFAULT_CONFIG) -> MethodRun:
    """最近傍追跡の軌道を線形補間する。距離の設定は動画の幅に合わせて換算する"""
    tracker = load_config(config_path).scaled_for(width).tracker
    begin = time.perf_counter()
    nearest = run_nearest(records, tracker)
    estimates = linear_fill(nearest.estimates, tracker.max_missed_frames)
    return MethodRun(estimates, nearest.events, (time.perf_counter() - begin) * 1000)


def export_linear_fill(records, detected_video, out_dir, base_name: str, width: int) -> tuple:
    """補完の結果を JSON と動画に書き出し、(JSON のパス, 動画のパス) を返す"""
    run = linear_track(records, width)
    json_path = Path(out_dir) / f"{base_name}_filled.json"
    video_path = Path(out_dir) / f"{base_name}_filled.mp4"
    save_estimates(json_path, {"linear": run})
    render_filled(Path(detected_video), video_path, run.estimates)
    return json_path, video_path
