"""ベースライン推論。全フレームを1回だけ推論して候補をすべて保存する。

track.py と同じく tennis.py の差分強調動画に対して推論する。差分強調は次のフレームを使うため、
「前向き推定」のカルマンフィルタも入力の時点で1フレーム先を見ている点に注意する。
"""

import contextlib
import io
import platform
import time
from pathlib import Path

import cv2

import tennis

from .config import InferenceConfig
from .schema import Candidate, FrameRecord
from .storage import save_detections


def _class_id(model, name: str) -> int:
    for class_id, class_name in model.names.items():
        if class_name == name:
            return int(class_id)
    raise ValueError(f"モデルにクラス {name} がありません: {model.names}")


def _frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def is_complete_enhanced(enhanced: Path, video: Path) -> bool:
    """差分強調は最終フレームを作らないので、元動画より1枚少なければ完全"""
    return _frame_count(enhanced) == _frame_count(video) - 1


def _prepare_source(video: Path, out_dir: Path, preprocess: str) -> Path:
    if preprocess == "raw":
        return video
    if preprocess != "enhanced":
        raise ValueError(f"preprocess は enhanced か raw です: {preprocess}")
    enhanced = out_dir / f"{video.stem}_enhance.mp4"
    if not enhanced.exists() or not is_complete_enhanced(enhanced, video):
        with contextlib.redirect_stdout(io.StringIO()):  # tennis.run は毎フレーム print する
            tennis.run(str(video), enhance_video_path=str(enhanced))
    return enhanced


def _candidates(result, cap_id: int) -> tuple:
    return tuple(
        Candidate(*(float(v) for v in box.xyxy[0]), float(box.conf[0]))
        for box in result.boxes
        if int(box.cls[0]) == cap_id
    )


def _video_meta(video: Path, capture, config: InferenceConfig, weights: Path) -> dict:
    return {
        "video_id": video.stem,
        "video_path": str(video),
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "weights": weights.name,
        "imgsz": config.imgsz,
        "conf": config.conf,
        "preprocess": config.preprocess,
        "cap_class": config.cap_class,
        "host": platform.platform(),
    }


def run_inference(video: Path, out_dir: Path, weights: Path, config: InferenceConfig) -> Path:
    from ultralytics import YOLO  # 追跡・評価だけなら torch を読み込まない

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    cap_id = _class_id(model, config.cap_class)
    begin = time.perf_counter()
    source = cv2.VideoCapture(str(_prepare_source(video, out_dir, config.preprocess)))
    preprocess_s = time.perf_counter() - begin
    original = cv2.VideoCapture(str(video))
    if not source.isOpened() or not original.isOpened():
        raise RuntimeError(f"動画を開けません: {video}")

    meta = _video_meta(video, original, config, weights)
    records = []
    while True:
        ok, frame = source.read()
        ok_original, _ = original.read()
        if not ok or not ok_original:
            break
        timestamp = original.get(cv2.CAP_PROP_POS_MSEC) / 1000
        begin = time.perf_counter()
        result = model(frame, imgsz=config.imgsz, conf=config.conf, verbose=False)[0]
        elapsed = (time.perf_counter() - begin) * 1000
        records.append(FrameRecord(len(records), timestamp, _candidates(result, cap_id), elapsed))
    source.release()
    original.release()

    path = out_dir / "detections.json"
    summary = {"frames_inferred": len(records), "preprocess_s": preprocess_s}
    save_detections(path, {**meta, **summary}, records)
    return path
