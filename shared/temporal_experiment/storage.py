"""推論結果と正解データの読み書き。後続の処理を YOLO の再推論なしで実行するための保存形式。"""

import csv
import json
import re
from pathlib import Path

from .schema import Candidate, FrameRecord, GroundTruth


def _candidate_to_dict(candidate: Candidate) -> dict:
    return {
        "box": list(candidate.box),
        "cx": candidate.cx,
        "cy": candidate.cy,
        "conf": candidate.conf,
    }


def _record_to_dict(record: FrameRecord) -> dict:
    return {
        "frame": record.frame,
        "t": record.t,
        "detected": bool(record.candidates),
        "infer_ms": record.infer_ms,
        "candidates": [_candidate_to_dict(c) for c in record.candidates],
    }


def _record_from_dict(raw: dict) -> FrameRecord:
    candidates = tuple(Candidate(*c["box"], c["conf"]) for c in raw["candidates"])
    return FrameRecord(raw["frame"], raw["t"], candidates, raw["infer_ms"])


def save_detections(path: Path, meta: dict, records) -> None:
    payload = {"meta": meta, "frames": [_record_to_dict(r) for r in records]}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_detections(path: Path) -> tuple:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["meta"], tuple(_record_from_dict(raw) for raw in payload["frames"])


def load_truth_csv(path: Path) -> tuple:
    """TrackNet/src/annotate.py の形式（frame_id, x, y, visible）を読む"""
    with open(path, newline="", encoding="utf-8") as f:
        return tuple(
            GroundTruth(
                int(row["frame_id"]), float(row["x"]), float(row["y"]), int(row["visible"]) == 1
            )
            for row in csv.DictReader(f)
        )


def _truth_from_label(path: Path, frame: int, width: int, height: int, cap_class: int):
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    caps = [row for row in rows if int(row[0]) == cap_class]
    if len(caps) > 1:
        raise ValueError(f"1フレームに蓋が複数あります: {path.name}")
    if not caps:
        return GroundTruth(frame, -1.0, -1.0, False)
    return GroundTruth(frame, float(caps[0][1]) * width, float(caps[0][2]) * height, True)


def load_truth_yolo(label_dir: Path, video_stem: str, width: int, height: int, cap_class: int) -> tuple:
    """人が確認した YOLO 形式のラベルから中心座標を読む。

    蓋の行が無いラベルは「蓋が写っていないと確認した」として扱うので、
    未確認の下書き（prelabel.py の出力）を渡してはいけない。
    """
    pattern = re.compile(rf"^{re.escape(video_stem)}_(\d+)\.txt$")
    truths = []
    for path in sorted(Path(label_dir).glob("*.txt")):
        match = pattern.match(path.name)
        if match:
            truths.append(_truth_from_label(path, int(match[1]), width, height, cap_class))
    return tuple(truths)
