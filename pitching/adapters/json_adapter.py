"""保存済みキーポイント列（JSON / CSV）の読み書き。

解析本体はこの形式しか知らない。動画から抽出したかどうかに依存しない。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from pitching.models import Keypoint, PoseFrame, PoseSeries

SCHEMA_VERSION = 1
CSV_COLUMNS = ("frame_index", "timestamp_sec", "keypoint", "x", "y", "confidence")


def save_pose_json(series: PoseSeries, path: str | Path) -> Path:
    """PoseSeries を JSON に保存する。NaN は null で書く。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "pitch_id": series.pitch_id,
            "fps": series.fps,
            "video_path": series.video_path,
            "adapter": series.adapter,
            "notes": series.notes,
        },
        "frames": [
            {
                "frame_index": frame.frame_index,
                "timestamp_sec": frame.timestamp_sec,
                "keypoints": {
                    name: [_json_number(kp.x), _json_number(kp.y), kp.confidence]
                    for name, kp in frame.keypoints.items()
                },
            }
            for frame in series.frames
        ],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path


def load_pose_json(path: str | Path) -> PoseSeries:
    """JSON から PoseSeries を読み込む。"""
    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"キーポイントJSONが見つかりません: {input_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"キーポイントJSONが不正です: {input_path}: {error}") from error

    if "frames" not in payload:
        raise ValueError(f"キーポイントJSONに frames がありません: {input_path}")

    meta = payload.get("meta", {})
    frames = [
        PoseFrame(
            frame_index=int(entry["frame_index"]),
            timestamp_sec=float(entry["timestamp_sec"]),
            keypoints={
                name: Keypoint(_number(values[0]), _number(values[1]), float(values[2]))
                for name, values in entry.get("keypoints", {}).items()
            },
        )
        for entry in payload["frames"]
    ]
    return PoseSeries(
        frames=sorted(frames, key=lambda frame: frame.frame_index),
        fps=float(meta.get("fps", 60.0)),
        pitch_id=str(meta.get("pitch_id", "")),
        video_path=str(meta.get("video_path", "")),
        adapter=str(meta.get("adapter", "json")),
        notes=list(meta.get("notes", [])),
    )


def save_pose_csv(series: PoseSeries, path: str | Path) -> Path:
    """1行1キーポイントの long 形式で保存する。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for frame in series.frames:
            for name, keypoint in frame.keypoints.items():
                writer.writerow(
                    [
                        frame.frame_index,
                        f"{frame.timestamp_sec:.6f}",
                        name,
                        _csv_number(keypoint.x),
                        _csv_number(keypoint.y),
                        f"{keypoint.confidence:.4f}",
                    ]
                )
    return output_path


def load_pose_csv(path: str | Path, fps: float, pitch_id: str = "") -> PoseSeries:
    """save_pose_csv が書いた形式を読み戻す。"""
    input_path = Path(path)
    grouped: dict[int, dict] = {}
    with input_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frame_index = int(row["frame_index"])
            entry = grouped.setdefault(
                frame_index, {"timestamp_sec": float(row["timestamp_sec"]), "keypoints": {}}
            )
            entry["keypoints"][row["keypoint"]] = Keypoint(
                _number(row["x"]), _number(row["y"]), float(row["confidence"])
            )

    frames = [
        PoseFrame(index, value["timestamp_sec"], value["keypoints"])
        for index, value in sorted(grouped.items())
    ]
    return PoseSeries(frames=frames, fps=fps, pitch_id=pitch_id, adapter="csv")


def _json_number(value: float):
    return None if not math.isfinite(value) else float(value)


def _csv_number(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.3f}"


def _number(value) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)
