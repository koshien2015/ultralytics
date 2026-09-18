"""動画から YOLO Pose を回して PoseSeries を作るアダプタ。

shared/ の資産（PoseEstimator / RoleTracker / KEYPOINT_NAMES）をそのまま使う。
shared/*.py はフラット import で書かれているので、ここで sys.path に足して
遅延 import する。shared/ 側は変更しない。

このモジュールだけが ultralytics / OpenCV / shared に依存する。
解析本体は JSON しか読まない。

重要: 解析には連続フレームが要るので、track.py の prefilter（推論の間引き）は
使わない。間引くと「推論していないフレーム」と「信頼度が低いフレーム」が
区別できなくなり、短い欠損だけを補間する前処理が誤動作する。
"""

from __future__ import annotations

import sys
from pathlib import Path

from pitching.config import PitchConfig
from pitching.models import Keypoint, PoseFrame, PoseSeries

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"


class ExtractionError(RuntimeError):
    """動画からの姿勢抽出に失敗したときに送出する。"""


def _import_shared():
    """shared/pose.py を遅延 import する。"""
    if not SHARED_DIR.is_dir():
        raise ExtractionError(f"shared ディレクトリが見つかりません: {SHARED_DIR}")
    if str(SHARED_DIR) not in sys.path:
        sys.path.insert(0, str(SHARED_DIR))
    try:
        import pose  # type: ignore

        return pose
    except ImportError as error:
        raise ExtractionError(
            f"shared/pose.py を読み込めません（ultralytics / torch 未導入の可能性）: {error}"
        ) from error


def extract_pose_series(config: PitchConfig, video_path: str | None = None) -> PoseSeries:
    """設定の区間について、1フレームずつ姿勢を推定して PoseSeries を返す。"""
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - 実行環境依存
        raise ExtractionError("opencv-python が必要です") from error

    pose = _import_shared()
    source = video_path or config.video_path
    if not source:
        raise ExtractionError("video_path が指定されていません")
    if not Path(source).is_file():
        raise ExtractionError(f"動画が見つかりません: {source}")

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ExtractionError(f"動画を開けません: {source}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or config.fps
        estimator = pose.PoseEstimator(
            model_path=config.extraction.pose_model_path,
            conf=config.extraction.conf,
            imgsz=config.extraction.imgsz,
            min_keypoint_score=config.extraction.min_keypoint_score,
            min_overlap=config.extraction.min_overlap,
            roles=("pitcher",),
        )
        estimator.reset_tracker()
        detector = _load_detector(config)

        frames, notes = _run(capture, cv2, pose, estimator, detector, config)
    finally:
        capture.release()

    return PoseSeries(
        frames=frames,
        fps=fps,
        pitch_id=config.pitch_id,
        video_path=source,
        adapter="ultralytics",
        notes=notes,
    )


def _run(capture, cv2, pose, estimator, detector, config: PitchConfig):
    """区間内の全フレームを推論して PoseFrame のリストを作る。"""
    end_frame = config.end_frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame)

    frames: list[PoseFrame] = []
    notes: list[str] = []
    selection_modes: set[str] = set()
    frame_index = config.start_frame

    while True:
        if end_frame is not None and frame_index > end_frame:
            break
        ok, image = capture.read()
        if not ok:
            break

        detections = detector(image) if detector is not None else []
        result = estimator.update(image, detections)
        person, mode = _select_pitcher(result, pose, config.extraction.min_keypoint_score)
        selection_modes.add(mode)

        frames.append(
            _to_pose_frame(person, pose.KEYPOINT_NAMES, frame_index, config.fps)
        )
        frame_index += 1

    if not frames:
        raise ExtractionError("区間内のフレームを1枚も読み込めませんでした")

    notes.append(f"person_selection={sorted(selection_modes)}")
    if "role_bbox" not in selection_modes:
        notes.append(
            "検出モデル未指定のため、投手を役割bboxで特定できていない"
            "（最大の骨格を投手とみなした）。誤った人物を拾っていないか確認すること。"
        )
    return frames, notes


def _select_pitcher(result: dict, pose, min_score: float):
    """姿勢推定結果から投手1人を選ぶ。

    選び方は shared/pose.py に置いてある（track.py の書き出しと同じ挙動にするため）。
    """
    return pose.select_person(result, "pitcher", min_score)


def _to_pose_frame(person, keypoint_names, frame_index: int, fps: float) -> PoseFrame:
    keypoints: dict[str, Keypoint] = {}
    for order, name in enumerate(keypoint_names):
        if person is None:
            keypoints[name] = Keypoint.missing()
            continue
        x, y = person.keypoints[order]
        keypoints[name] = Keypoint(float(x), float(y), float(person.scores[order]))
    return PoseFrame(
        frame_index=frame_index,
        timestamp_sec=frame_index / fps if fps > 0 else 0.0,
        keypoints=keypoints,
    )


def _load_detector(config: PitchConfig):
    """役割bbox用の検出モデル。未指定なら None。"""
    model_path = config.extraction.detection_model_path
    if not model_path:
        return None
    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover - 実行環境依存
        raise ExtractionError("ultralytics が必要です") from error

    model = YOLO(model_path)

    def detect(image):
        return model.predict(
            image, conf=config.extraction.conf, imgsz=config.extraction.imgsz, verbose=False
        )

    return detect
