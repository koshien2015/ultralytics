#!/usr/bin/env python3
"""動画1本から YOLO Pose のキーポイント列（pose.json）を書き出す。

    python extract-pose.py                       # 下の「設定」に従う
    python extract-pose.py input/a.mov -o a_pose.json --start 100 --end 180

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
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 1

# ============================ 設定 ============================
# 毎回コマンドを打たずに済むよう、ここを書き換えて引数なしで実行してもよい
# （track.py と同じ流儀）。コマンドラインで指定した場合はそちらが優先される。
VIDEO = None                       # 入力動画。例: "/shared/input/a.mov"
OUTPUT = None                      # 出力先。None なら {動画名}_pose.json
START_FRAME = 0                    # 解析区間の開始フレーム
END_FRAME = None                   # 終了フレーム。None なら最後まで
FPS = None                         # None なら動画から取る
POSE_MODEL = "yolo11x-pose.pt"     # Pose 重み
DETECTION_MODEL = None             # 11クラス検出モデル。投手の特定に使う（推奨）
IMGSZ = 960
CONF = 0.5
MIN_KEYPOINT_SCORE = 0.3
# ==============================================================


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
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    # **シークの着地点を読み戻す。** 長尺 HEVC では要求フレームではなく
    # キーフレームへ着地することがある。要求値をそのまま先頭の番号にすると、
    # 全フレームが一定量ずれたまま、警告もなく出力される。骨格が数フレーム
    # ずれて重なるのは「それらしく見えて間違っている」最悪の壊れ方なので、
    # 実際に着地した位置を番号の基準にし、要求値との差も記録して持ち帰る。
    landing = capture.get(cv2.CAP_PROP_POS_FRAMES)
    seek_landing = int(landing) if landing is not None and landing >= 0 else args.start
    frame_index = seek_landing

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

    if seek_landing != args.start:
        notes.append(
            f"シークが要求フレーム {args.start} ではなく {seek_landing} へ着地した"
            f"（差 {seek_landing - args.start} フレーム）。キーフレーム境界の可能性。"
            "フレーム番号はこの着地点を基準にしている。"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "pitch_id": args.pitch_id or video.stem,
            "fps": fps,
            "video_path": str(video),
            "adapter": "ultralytics(extract-pose.py)",
            "notes": notes,
            # 封筒を組み立てるために持ち帰る。要求値と着地点の両方を残す。
            "requested_start_frame": args.start,
            "seek_landing_frame": seek_landing,
            "end_frame": args.end if args.end is not None else frames[-1]["frame_index"],
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


def wrap_in_envelope(payload: dict, args) -> dict:
    """1 球フォルダ用の封筒を被せる。

    **入力動画（クリップ）が唯一の基準。** セッションも metadata.json も見ない
    ——アプリ外で撮った動画には親セッションが無いし、自己申告と実体はずれる
    （実データで durationMs が 5ms、fps が 8 桁目で食い違っていた）。

    姿勢推定は GPU のある別の PC で走るので、出力した JSON は**ファイルとして
    別のマシンへ渡り、人の手で 1 球フォルダへ置かれる**。置き間違いを検出できる
    よう、入力クリップの指紋を封筒へ入れる（投球ごとにクリップが違うので、
    ここでは指紋が実際に区別できる）。

    --envelope を付けない従来の使い方では呼ばれない。既存の出力は変わらない。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analysis_envelope import build_envelope
    from video_probe import ProbeError, probe_video

    video = Path(args.video)
    try:
        info = probe_video(video)
    except ProbeError as error:
        raise SystemExit(f"封筒を作れません: {error}")

    meta = payload["meta"]

    # **封筒と中身が別々の時間軸を使えないようにする。**
    # 中身の timestamp_sec は cv2 の CAP_PROP_FPS 由来で、コンテナの公称値
    # （240.0 など）が返ることがある。封筒の時間軸はファイルの stts 由来。
    # 表示は目盛表で行うので中身の timestamp_sec は使わないが、
    # 食い違ったままの値を「事実」として保存はしない。
    if abs(float(meta["fps"]) - info["fps"]) > 1e-6:
        raise SystemExit(
            f"fps が食い違っています。cv2 が申告した値 {meta['fps']} に対し、"
            f"動画の stts から求めた平均は {info['fps']} です。\n"
            f"--fps {info['fps']} を指定し直してください"
            "（--envelope を付けた場合は既定でそうなります）。"
        )

    return build_envelope(
        analysis_id="pose-" + Path(args.pose_model).stem,
        analysis_type="pose",
        clip_path=video,
        pitch_id=meta["pitch_id"],
        # OpenCV 4.5 以降は既定でデコード時に回転を当てる。推測させず宣言する。
        rotation_applied=True,
        producer={
            "name": "extract-pose.py",
            "version": str(SCHEMA_VERSION),
            "model": args.pose_model,
            "detectionModel": args.detection_model,
            "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        data=payload,
    )


def main(argv: list[str] | None = None) -> int:
    default_shared = os.environ.get(
        "PITCHING_SHARED_DIR", str(Path(__file__).resolve().parents[2] / "shared")
    )
    parser = argparse.ArgumentParser(description="動画からキーポイント列を書き出す")
    parser.add_argument("video", nargs="?", default=VIDEO, help="入力動画")
    parser.add_argument("-o", "--output", default=OUTPUT, help="出力する pose.json")
    parser.add_argument("--start", type=int, default=START_FRAME, help="解析区間の開始フレーム")
    parser.add_argument("--end", type=int, default=END_FRAME,
                        help="解析区間の終了フレーム（省略で最後まで）")
    parser.add_argument("--fps", type=float, default=FPS, help="fps（省略時は動画から取る）")
    parser.add_argument("--pitch-id", help="投球名（既定は動画のファイル名）")
    parser.add_argument("--pose-model", default=POSE_MODEL, help="Pose 重み")
    parser.add_argument(
        "--detection-model", default=DETECTION_MODEL,
        help="11クラス検出モデル。投手の特定に使う（推奨）",
    )
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--min-keypoint-score", type=float, default=MIN_KEYPOINT_SCORE)
    parser.add_argument("--shared-dir", default=default_shared, help="shared/pose.py の場所")
    parser.add_argument(
        "--envelope", action="store_true",
        help="1 球フォルダ用の封筒付き JSON を書き出す"
             "（入力クリップの指紋・目盛表・座標系を添える）",
    )
    parser.add_argument(
        "--progress", type=int, default=50, help="何フレームごとに進捗を出すか（0で出さない）"
    )
    args = parser.parse_args(argv)

    # 封筒を出すなら、fps は動画の stts から求めた値を既定にする。
    # cv2 の CAP_PROP_FPS はコンテナの公称値（240.0 など）を返すことがあり、
    # 実測（239.488...）とずれる。指定を省いたときだけ埋めるので、
    # 明示した --fps は尊重する。
    if args.envelope and args.fps is None and args.video:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from video_probe import ProbeError, probe_video

        try:
            args.fps = probe_video(args.video)["fps"]
        except ProbeError as error:
            raise SystemExit(f"入力動画から時間情報を読めません: {error}")

    if not args.video:
        raise SystemExit(
            "入力動画が指定されていません。"
            "引数で渡すか、このファイル先頭の VIDEO を設定してください"
        )
    # 出力先を省いたら動画の隣に {動画名}_pose.json を作る
    if not args.output:
        args.output = str(Path(args.video).with_name(f"{Path(args.video).stem}_pose.json"))

    payload = extract(args)
    if args.envelope:
        payload = wrap_in_envelope(payload, args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    frames = payload["data"]["frames"] if "data" in payload else payload["frames"]
    print(f"{len(frames)} フレームを書き出しました: {output}")
    meta = payload["data"]["meta"] if "data" in payload else payload["meta"]
    for note in meta["notes"]:
        print(f"  注意: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
