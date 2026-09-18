#!/usr/bin/env python3
"""動画1本から YOLO Pose のキーポイント列（pose.json）を書き出す。

    python extract-pose.py input/a.mov -o a.pose.json --start 100 --end 180

Docker の推論環境で単体で走らせるための台本。**この1ファイルと shared/pose.py
以外に何も要らない**（pydantic も matplotlib も import しない）ので、
リポジトリ全体をマウントしなくても動く。

出力した pose.json をホストに持ち帰れば、解析とビューアは GPU 無しで動く:

    python -m pitching run --output out/ --pose a.pose.json --release 152 ...

shared/pose.py の置き場所は、既定ではこの台本から見た ../../shared を使う。
別の場所なら --shared-dir か環境変数 PITCHING_SHARED_DIR で指定する。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

SCHEMA_VERSION = 1


def load_pose_module(shared_dir: Path):
    """shared/pose.py を import する。"""
    if not (shared_dir / "pose.py").is_file():
        raise SystemExit(
            f"shared/pose.py が見つかりません: {shared_dir}\n"
            "--shared-dir か PITCHING_SHARED_DIR で場所を指定してください"
        )
    if str(shared_dir) not in sys.path:
        sys.path.insert(0, str(shared_dir))
    import pose  # type: ignore

    return pose


def select_pitcher(result: dict, pose, min_score: float):
    """姿勢推定結果から投手1人を選ぶ。

    選び方そのものは shared/pose.py にある。track.py の書き出しと
    同じ結果になるよう、ここでは呼ぶだけにする。
    """
    return pose.select_person(result, "pitcher", min_score)


def frame_payload(person, keypoint_names, frame_index: int, fps: float) -> dict:
    """1フレーム分の JSON を作る。欠損は null で書く。"""
    keypoints = {}
    for order, name in enumerate(keypoint_names):
        if person is None:
            keypoints[name] = [None, None, 0.0]
            continue
        x, y = person.keypoints[order]
        score = float(person.scores[order])
        keypoints[name] = [
            None if not math.isfinite(float(x)) else round(float(x), 3),
            None if not math.isfinite(float(y)) else round(float(y), 3),
            round(score, 4),
        ]
    return {
        "frame_index": frame_index,
        "timestamp_sec": round(frame_index / fps, 6) if fps > 0 else 0.0,
        "keypoints": keypoints,
    }


def extract(args) -> dict:
    """解析区間を1フレームずつ推論する。

    間引きはしない。関節角度の時系列には連続フレームが要るうえ、間引くと
    「推論していないフレーム」と「信頼度が低いフレーム」を区別できなくなる。
    """
    import cv2

    pose = load_pose_module(Path(args.shared_dir))

    video = Path(args.video)
    if not video.is_file():
        raise SystemExit(f"動画が見つかりません: {video}")

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"動画を開けません: {video}")

    fps = args.fps or float(capture.get(cv2.CAP_PROP_FPS)) or 60.0
    estimator = pose.PoseEstimator(
        model_path=args.pose_model,
        conf=args.conf,
        imgsz=args.imgsz,
        min_keypoint_score=args.min_keypoint_score,
        roles=("pitcher",),
    )
    estimator.reset_tracker()
    detector = build_detector(args)

    frames = []
    modes = set()
    frame_index = args.start
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    try:
        while args.end is None or frame_index <= args.end:
            ok, image = capture.read()
            if not ok:
                break
            detections = detector(image) if detector is not None else []
            person, mode = select_pitcher(
                estimator.update(image, detections), pose, args.min_keypoint_score
            )
            modes.add(mode)
            frames.append(frame_payload(person, pose.KEYPOINT_NAMES, frame_index, fps))
            frame_index += 1
            if args.progress and len(frames) % args.progress == 0:
                print(f"  {len(frames)} フレーム...", flush=True)
    finally:
        capture.release()

    if not frames:
        raise SystemExit("区間内のフレームを1枚も読み込めませんでした")

    notes = [f"person_selection={sorted(modes)}"]
    if "role_bbox" not in modes:
        notes.append(
            "検出モデル未指定のため投手を役割bboxで特定できていない"
            "（最大の骨格を投手とみなした）。別人を拾っていないか確認すること。"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "pitch_id": args.pitch_id or video.stem,
            "fps": fps,
            "video_path": str(video),
            "adapter": "ultralytics(extract-pose.py)",
            "notes": notes,
        },
        "frames": frames,
    }


def build_detector(args):
    """役割bbox用の検出モデル。未指定なら None。"""
    if not args.detection_model:
        return None
    from ultralytics import YOLO

    model = YOLO(args.detection_model)
    return lambda image: model.predict(
        image, conf=args.conf, imgsz=args.imgsz, verbose=False
    )


def main(argv: list[str] | None = None) -> int:
    default_shared = os.environ.get(
        "PITCHING_SHARED_DIR", str(Path(__file__).resolve().parents[2] / "shared")
    )
    parser = argparse.ArgumentParser(description="動画からキーポイント列を書き出す")
    parser.add_argument("video", help="入力動画")
    parser.add_argument("-o", "--output", required=True, help="出力する pose.json")
    parser.add_argument("--start", type=int, default=0, help="解析区間の開始フレーム")
    parser.add_argument("--end", type=int, help="解析区間の終了フレーム（省略で最後まで）")
    parser.add_argument("--fps", type=float, help="fps（省略時は動画から取る）")
    parser.add_argument("--pitch-id", help="投球名（既定は動画のファイル名）")
    parser.add_argument("--pose-model", default="yolo11x-pose.pt", help="Pose 重み")
    parser.add_argument(
        "--detection-model", help="11クラス検出モデル。投手の特定に使う（推奨）"
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--min-keypoint-score", type=float, default=0.3)
    parser.add_argument("--shared-dir", default=default_shared, help="shared/pose.py の場所")
    parser.add_argument(
        "--progress", type=int, default=50, help="何フレームごとに進捗を出すか（0で出さない）"
    )
    args = parser.parse_args(argv)

    payload = extract(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print(f"{len(payload['frames'])} フレームを書き出しました: {output}")
    for note in payload["meta"]["notes"]:
        print(f"  注意: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
