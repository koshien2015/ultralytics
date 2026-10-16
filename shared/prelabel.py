"""ラベリング用の下書き（YOLO形式 .txt）を作るスクリプト。

make_dataset.py が書き出したフレーム画像に学習済みモデルを掛け、
キャップの検出を1球ごとの軌跡で整理してから下書きラベルとして書き出す。
人は矩形をゼロから描かず、下書きを確認して直すだけにするのが狙い。

キャップの整理（trajectory_fitter の RANSAC をピクセル座標で使う）:
- 軌跡に乗らない検出（投手の腕など）は捨てる
- 1フレームに複数あれば軌跡に最も近い1個だけ残す（キャップは原則1個）
- 取りこぼしたフレームは軌跡から位置を、前後の検出から大きさを補間して足す
- 点が少なくて軌跡を引けない区間は、各フレームで最も信頼度の高い検出を残す
- 補間しない長い欠損と軌跡の前後数枚は、キャップの付け忘れ候補（missing_cap）として挙げる

キャップ以外のクラスは軌跡を使わず、信頼度で足切りしてそのまま書く。

既存の .txt は上書きしない（--overwrite を付けたときだけ上書きする）。
何も検出されなかったフレームにはファイルを作らない。空ファイルは
「背景と確認済み」の意味になってしまうため。

各矩形の出どころ（検出・補間・軌跡なし・除外・付け忘れ候補）は prelabel_report.csv に
確認の優先度順で書く。付け忘れ候補・軌跡なし・補間から先に見ると効率がよい。

人物クラスはほぼ全フレームに付くため、「ラベルの無い画像を捨てる」だけでは
キャップの写ったフレームを選べなくなる。キャップを見落としたフレームに人物だけの
ラベルが残ると「ここにキャップは無い」と学習してしまうので、付け忘れ候補は必ず確認する。

使用例:
    python prelabel.py datasets/frames --weights yolo26m-1280px-120epoch.pt --imgsz 1280
    python prelabel.py datasets/frames --weights best.pt --out datasets/frames_draft
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np

from trajectory_fitter import MIN_POINTS_FOR_FIT, TrajectoryPoint, position_at, ransac_fit

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REPORT_NAME = "prelabel_report.csv"
CLASSES_NAME = "classes.txt"
FRAME_NAME_PATTERN = re.compile(r"^(?P<stem>.+)_(?P<frame>\d{6})$")
LOW_CONF = 0.5  # これ未満の検出はレポートで確認の優先度を上げる
DUPLICATE_IOU = 0.9  # label_qa と同じ基準。同一クラスでこれを超える矩形は1つにする
# 軌跡からのずれの差がこれ以内の2重検出は同じキャップとみなし、信頼度で選ぶ
SAME_CAP_RESIDUAL_PX = 5.0

# 確認の優先度（小さいほど先に見る）。ここに無い detected は低信頼度→高信頼度の順、
# キャップ以外の重複（overlap）は NMS の取り残しなので最後
REVIEW_PRIORITY = {
    "missing_cap": 0, "unfit": 1, "interpolated": 2, "outlier": 3, "duplicate": 4, "static": 5,
}


@dataclass(frozen=True)
class Detection:
    """検出1件（座標はピクセル、cx/cy は中心）"""

    cls: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float


@dataclass(frozen=True)
class LabelBox:
    """書き出す矩形1件。source は detected / interpolated / unfit"""

    cls: int
    cx: float
    cy: float
    w: float
    h: float
    source: str
    conf: Optional[float]
    residual: Optional[float] = None  # 軌跡からのずれ（ピクセル）


@dataclass(frozen=True)
class Rejected:
    """書き出さなかった検出。reason は outlier（キャップのみ）/ duplicate"""

    frame: int
    detection: Detection
    reason: str
    residual: Optional[float] = None


@dataclass(frozen=True)
class MissingCap:
    """軌跡の途中や前後で、キャップのラベルが無いフレーム（cx/cy は軌跡からの予測）"""

    frame: int
    cx: float
    cy: float


@dataclass(frozen=True)
class PrelabelConfig:
    """既定値は 1080p・60fps の素材を想定した仮値。実素材で調整すること。"""

    cap_class: int = 0
    inlier_px: float = 20.0  # 軌跡からこれ以上ずれた検出は誤検出とみなす
    segment_gap: int = 15  # 検出がこれより長く途切れたら別の球とみなす
    max_fill_gap: int = 8  # これより長い欠損は補間しない
    other_conf: float = 0.5  # キャップ以外のクラスの足切り
    track_margin: int = 3  # 軌跡の前後このフレーム数までをキャップ付け忘れの確認対象にする
    static_extent_px: float = 30.0  # 検出の散らばりが縦横ともこれ未満の軌跡は止まった誤検出とみなす


@dataclass
class Summary:
    images: int = 0
    written: int = 0
    skipped_existing: int = 0
    empty: int = 0
    unparsed: int = 0
    cap_frames: int = 0
    sources: dict = field(default_factory=dict)


# 画像パスの列を受け取り、各画像について (幅, 高さ, 検出のリスト) を順に返す
Detector = Callable[[Sequence[Path]], Iterable[tuple]]


def parse_frame_name(path: Path) -> Optional[tuple]:
    """`{動画名}_{フレーム番号:06d}` を (動画名, フレーム番号) にする。合わなければ None"""
    match = FRAME_NAME_PATTERN.match(path.stem)
    if match is None:
        return None
    return match.group("stem"), int(match.group("frame"))


def group_frames(paths: Iterable[Path]) -> dict:
    """動画名ごとに [(フレーム番号, パス), ...] をフレーム順で返す（名前の合わない画像は除く）"""
    grouped: dict = {}
    for path in paths:
        parsed = parse_frame_name(path)
        if parsed is not None:
            stem, frame = parsed
            grouped.setdefault(stem, []).append((frame, path))
    return {stem: sorted(items) for stem, items in sorted(grouped.items())}


def split_segments(frames: Sequence[int], max_gap: int) -> list:
    """昇順のフレーム番号を、間隔が max_gap を超えるところで区切る"""
    segments: list = []
    for frame in frames:
        if segments and frame - segments[-1][-1] <= max_gap:
            segments[-1] = [*segments[-1], frame]
        else:
            segments.append([frame])
    return segments


def _best_per_frame(candidates: Mapping[int, Sequence[tuple]], key) -> tuple:
    """フレームごとに key 最小の候補を残し、残りを duplicate として返す"""
    kept: dict = {}
    rejected: list = []
    for frame, items in candidates.items():
        ordered = sorted(items, key=key)
        kept[frame] = ordered[0]
        rejected.extend(
            Rejected(frame, det, "duplicate", residual) for det, residual in ordered[1:]
        )
    return kept, rejected


def _unfit_segment(detections: Mapping[int, Sequence[Detection]], frames: Sequence[int]) -> tuple:
    candidates = {f: [(d, None) for d in detections[f]] for f in frames}
    kept, rejected = _best_per_frame(candidates, key=lambda item: -item[0].conf)
    boxes = {
        f: LabelBox(d.cls, d.cx, d.cy, d.w, d.h, "unfit", d.conf) for f, (d, _) in kept.items()
    }
    return boxes, rejected, []


def _on_track_key(items: Sequence[tuple]):
    """軌跡に最も近い候補と同程度に近いものの中では信頼度の高い方を先にする"""
    closest = min(residual for _, residual in items)

    def key(item):
        det, residual = item
        same_cap = residual <= closest + SAME_CAP_RESIDUAL_PX
        return (0, -det.conf) if same_cap else (1, residual)

    return key


def _fill_gaps(fit, origin: int, boxes: dict, available: set, max_gap: int) -> dict:
    """隣り合う採用フレームの間を、軌跡上の位置と前後の大きさの線形補間で埋める"""
    filled: dict = {}
    frames = sorted(boxes)
    for before, after in zip(frames, frames[1:]):
        if after - before - 1 > max_gap:
            continue
        a, b = boxes[before], boxes[after]
        for frame in range(before + 1, after):
            if frame not in available:
                continue
            ratio = (frame - before) / (after - before)
            x, y = position_at(fit, float(frame - origin))
            filled[frame] = LabelBox(
                a.cls, x, y,
                a.w + (b.w - a.w) * ratio, a.h + (b.h - a.h) * ratio,
                "interpolated", None,
            )
    return filled


def _residual(fit, origin: int, frame: int, det: Detection) -> float:
    x, y = position_at(fit, float(frame - origin))
    return float(np.hypot(det.cx - x, det.cy - y))


def _is_static(inliers: Sequence[tuple], extent_px: float) -> bool:
    """検出の散らばりが縦横とも extent_px 未満なら、止まっている物体とみなす"""
    xs = [d.cx for _, d, _ in inliers]
    ys = [d.cy for _, d, _ in inliers]
    return max(xs) - min(xs) < extent_px and max(ys) - min(ys) < extent_px


def _take_track(remaining: Sequence[tuple], origin: int, config: PrelabelConfig) -> tuple:
    """残りの点から軌跡を1本取り出す。

    フィットに乗った点でも、時間が segment_gap より離れた塊は別の球なので混ぜない。
    最も大きい塊だけで引き直す（2球の点を1本の2次式でつなぐことがあるため）。

    Returns:
        (fit, [(フレーム, Detection, 残差), ...] 採用点, [Rejected] 引き直しで外れた点, 使った添字)
        フィットできなければ ValueError
    """
    points = [TrajectoryPoint(frame=f, t=float(f - origin), x=d.cx, y=d.cy) for f, d in remaining]
    fit = ransac_fit(points, inlier_threshold=config.inlier_px)
    inlier_idx = [
        i for i, (f, d) in enumerate(remaining) if _residual(fit, origin, f, d) < config.inlier_px
    ]
    chunks = split_segments(sorted({remaining[i][0] for i in inlier_idx}), config.segment_gap)
    chunk = set(max(chunks, key=len)) if chunks else set()
    used = [i for i in inlier_idx if remaining[i][0] in chunk]
    if len(chunk) >= MIN_POINTS_FOR_FIT:
        try:
            fit = ransac_fit([points[i] for i in used], inlier_threshold=config.inlier_px)
        except ValueError:
            pass  # 引き直せなければ最初のフィットのまま

    scored = [(remaining[i][0], remaining[i][1], _residual(fit, origin, *remaining[i])) for i in used]
    inliers = [item for item in scored if item[2] < config.inlier_px]
    outliers = [Rejected(f, d, "outlier", r) for f, d, r in scored if r >= config.inlier_px]
    if len(chunk) < MIN_POINTS_FOR_FIT:
        # 軌跡と呼べるほど続いていない。誤検出として外し、次の軌跡を探す
        outliers = [*outliers, *(Rejected(f, d, "outlier", r) for f, d, r in inliers)]
        inliers = []
    return fit, inliers, outliers, used


def _find_tracks(pairs: Sequence[tuple], origin: int, config: PrelabelConfig) -> tuple:
    """RANSAC を繰り返して、区間内の軌跡をすべて取り出す。

    1回のフィットだけだと、動かない誤検出（地面のキャップなど）が最も多くの点を集めて
    投球を押しのけてしまう。採用した点を除いた残りに再度フィットし、
    止まっている軌跡は static として捨てる。

    Returns:
        ([(fit, [(フレーム, Detection, 残差), ...]), ...], [Rejected, ...])
    """
    remaining = list(pairs)
    tracks: list = []
    rejected: list = []
    while len({f for f, _ in remaining}) >= MIN_POINTS_FOR_FIT:
        try:
            fit, inliers, outliers, used = _take_track(remaining, origin, config)
        except ValueError:
            break
        if not used:
            break
        rejected = [*rejected, *outliers]
        if inliers and _is_static(inliers, config.static_extent_px):
            rejected = [*rejected, *(Rejected(f, d, "static", r) for f, d, r in inliers)]
        elif inliers:
            tracks = [*tracks, (fit, inliers)]
        used_set = set(used)
        remaining = [pair for i, pair in enumerate(remaining) if i not in used_set]

    leftovers = [
        Rejected(f, d, "outlier",
                 min((_residual(fit, origin, f, d) for fit, _ in tracks), default=None))
        for f, d in remaining
    ]
    return tracks, [*rejected, *leftovers]


def _track_labels(fit, inliers: Sequence[tuple], origin: int, available: set,
                  config: PrelabelConfig) -> tuple:
    """1本の軌跡から、フレームごとの矩形・重複・付け忘れ候補を作る"""
    candidates: dict = {}
    for frame, det, residual in inliers:
        candidates.setdefault(frame, []).append((det, residual))

    kept: dict = {}
    duplicates: list = []
    for frame, items in candidates.items():
        frame_kept, frame_dups = _best_per_frame({frame: items}, key=_on_track_key(items))
        kept = {**kept, **frame_kept}
        duplicates = [*duplicates, *frame_dups]
    boxes = {
        f: LabelBox(d.cls, d.cx, d.cy, d.w, d.h, "detected", d.conf, r)
        for f, (d, r) in kept.items()
    }
    labelled = {**boxes, **_fill_gaps(fit, origin, boxes, available, config.max_fill_gap)}
    missing = [
        MissingCap(f, *position_at(fit, float(f - origin)))
        for f in range(min(boxes) - config.track_margin, max(boxes) + config.track_margin + 1)
        if f in available and f not in labelled
    ]
    return labelled, duplicates, missing


def _merge_tracks(labelled: dict, other: dict) -> tuple:
    """2本の軌跡が同じフレームを取り合ったら信頼度の高い方を残す（補間は検出に負ける）"""
    merged = dict(labelled)
    dropped: list = []
    for frame, box in other.items():
        current = merged.get(frame)
        if current is not None and (current.conf or -1.0) >= (box.conf or -1.0):
            loser = box
        else:
            loser = current
            merged[frame] = box
        if loser is not None and loser.source == "detected":
            det = Detection(loser.cls, loser.cx, loser.cy, loser.w, loser.h, loser.conf)
            dropped.append(Rejected(frame, det, "duplicate", loser.residual))
    return merged, dropped


def _fit_segment(detections, frames, available, config) -> tuple:
    origin = frames[0]
    pairs = [(f, d) for f in frames for d in detections[f]]
    tracks, rejected = _find_tracks(pairs, origin, config)
    if not tracks:
        if any(r.reason == "static" for r in rejected):
            return {}, rejected, []
        return _unfit_segment(detections, frames)

    labelled: dict = {}
    missing: list = []
    for fit, inliers in tracks:
        track_boxes, duplicates, track_missing = _track_labels(fit, inliers, origin, available, config)
        labelled, dropped = _merge_tracks(labelled, track_boxes)
        rejected = [*rejected, *duplicates, *dropped]
        missing = [*missing, *track_missing]
    return labelled, rejected, missing


def refine_cap_track(
    detections: Mapping[int, Sequence[Detection]],
    available: set,
    config: PrelabelConfig,
) -> tuple:
    """1本の動画のキャップ検出を軌跡で整理する。

    Args:
        detections: {フレーム番号: [キャップの Detection, ...]}
        available: 画像が存在するフレーム番号（補間はここにだけ足す）

    Returns:
        ({フレーム番号: LabelBox}, [Rejected, ...], [MissingCap, ...])
    """
    frames = sorted(f for f, dets in detections.items() if dets)
    boxes: dict = {}
    rejected: list = []
    missing: list = []
    for segment in split_segments(frames, config.segment_gap):
        if len(segment) < MIN_POINTS_FOR_FIT:
            seg_boxes, seg_rejected, seg_missing = _unfit_segment(detections, segment)
        else:
            seg_boxes, seg_rejected, seg_missing = _fit_segment(
                detections, segment, available, config
            )
        boxes = {**boxes, **seg_boxes}
        rejected = [*rejected, *seg_rejected]
        missing = [*missing, *seg_missing]
    # 前後の余白が隣の球のラベルと重なったものは除く
    return boxes, rejected, [m for m in missing if m.frame not in boxes]


def _clip(box: LabelBox, width: int, height: int) -> tuple:
    x1 = min(max(box.cx - box.w / 2, 0.0), width)
    x2 = min(max(box.cx + box.w / 2, 0.0), width)
    y1 = min(max(box.cy - box.h / 2, 0.0), height)
    y2 = min(max(box.cy + box.h / 2, 0.0), height)
    return x1, y1, x2, y2


def _inside(box: LabelBox, width: int, height: int) -> bool:
    """画像内に面積が残るか（完全に外れた矩形は label_qa が不正とみなす）"""
    x1, y1, x2, y2 = _clip(box, width, height)
    return x2 > x1 and y2 > y1


def to_yolo_line(box: LabelBox, width: int, height: int) -> str:
    """ピクセル座標の矩形を YOLO 形式の1行にする（画像内に収まるよう切り詰める）"""
    x1, y1, x2, y2 = _clip(box, width, height)
    return (
        f"{box.cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
    )


def _review_priority(row: dict) -> tuple:
    priority = REVIEW_PRIORITY.get(row["source"])
    if row["source"] == "overlap":
        priority = len(REVIEW_PRIORITY) + 2
    elif priority is None:
        conf = float(row["conf"]) if row["conf"] else 1.0
        priority = len(REVIEW_PRIORITY) + (0 if conf < LOW_CONF else 1)
    return priority, row["image"]


def _report_row(path, frame, cls, cx, cy, w, h, source, conf, residual, status) -> dict:
    return {
        "image": path.name, "frame": frame, "cls": cls, "source": source,
        "conf": "" if conf is None else f"{conf:.3f}",
        "residual_px": "" if residual is None else f"{residual:.1f}",
        "cx": f"{cx:.1f}", "cy": f"{cy:.1f}",
        "w": "" if w is None else f"{w:.1f}", "h": "" if h is None else f"{h:.1f}",
        "status": status,
    }


def _write_report(out_dir: Path, rows: list) -> None:
    fields = ["image", "frame", "cls", "source", "conf", "residual_px",
              "cx", "cy", "w", "h", "status"]
    with open(out_dir / REPORT_NAME, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=_review_priority))


def _write_classes(out_dir: Path, class_names: Optional[Sequence[str]]) -> None:
    path = out_dir / CLASSES_NAME
    if class_names and not path.exists():
        path.write_text("".join(f"{name}\n" for name in class_names), encoding="utf-8")


def _detect_all(image_dir: Path, detector: Detector, summary: Summary) -> dict:
    """{動画名: {フレーム番号: (パス, 幅, 高さ, 検出)}} を返す"""
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    grouped = group_frames(images)
    paths = [path for items in grouped.values() for _, path in items]
    summary.images = len(paths)
    summary.unparsed = len(images) - len(paths)

    results = dict(zip(paths, detector(paths)))
    if len(results) != len(paths):
        raise RuntimeError(f"検出器の結果が {len(results)} 件しかありません（画像 {len(paths)} 枚）")
    return {
        stem: {frame: (path, *results[path]) for frame, path in items}
        for stem, items in grouped.items()
    }


def _iou(a: Detection, b: Detection) -> float:
    ix = max(0.0, min(a.cx + a.w / 2, b.cx + b.w / 2) - max(a.cx - a.w / 2, b.cx - b.w / 2))
    iy = max(0.0, min(a.cy + a.h / 2, b.cy + b.h / 2) - max(a.cy - a.h / 2, b.cy - b.h / 2))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def _boxes_for_frame(dets, cap_box, config) -> tuple:
    """書き出す矩形と、重複として落としたキャップ以外の検出を返す"""
    candidates = sorted(
        (d for d in dets if d.cls != config.cap_class and d.conf >= config.other_conf),
        key=lambda d: -d.conf,
    )
    kept: list = []
    dropped: list = []
    for det in candidates:
        if any(k.cls == det.cls and _iou(k, det) > DUPLICATE_IOU for k in kept):
            dropped.append(det)
        else:
            kept.append(det)
    others = [LabelBox(d.cls, d.cx, d.cy, d.w, d.h, "detected", d.conf) for d in kept]
    return ([cap_box] if cap_box else []) + others, dropped


def _missing_cap_status(label_path: Path, cap_class: int) -> Optional[str]:
    """付け忘れ候補の状態。人が既にキャップを付けたフレームなら None（挙げない）"""
    if not label_path.exists():
        return "no_cap_label"
    lines = label_path.read_text(encoding="utf-8").splitlines()
    if any(line.split()[:1] == [str(cap_class)] for line in lines):
        return None
    return "existing_label_without_cap"


def prelabel_directory(
    image_dir: Path,
    out_dir: Path,
    detector: Detector,
    config: PrelabelConfig,
    overwrite: bool = False,
    class_names: Optional[Sequence[str]] = None,
) -> Summary:
    """画像ディレクトリ全体に下書きラベルを書き、集計を返す"""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = Summary()
    rows: list = []

    for stem, frames in _detect_all(image_dir, detector, summary).items():
        cap_dets = {
            f: [d for d in dets if d.cls == config.cap_class]
            for f, (_, _, _, dets) in frames.items()
        }
        cap_boxes, rejected, missing = refine_cap_track(cap_dets, set(frames), config)

        for frame, (path, width, height, dets) in frames.items():
            boxes, dropped = _boxes_for_frame(dets, cap_boxes.get(frame), config)
            boxes = [b for b in boxes if _inside(b, width, height)]
            rejected = [*rejected, *(Rejected(frame, d, "overlap") for d in dropped)]
            if frame in cap_boxes:
                summary.cap_frames += 1
            if not boxes:
                summary.empty += 1
                continue
            label_path = out_dir / f"{path.stem}.txt"
            if label_path.exists() and not overwrite:
                status = "skipped_existing"
                summary.skipped_existing += 1
            else:
                lines = [to_yolo_line(b, width, height) for b in boxes]
                label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                status = "written"
                summary.written += 1
            for b in boxes:
                summary.sources[b.source] = summary.sources.get(b.source, 0) + 1
                rows.append(_report_row(path, frame, b.cls, b.cx, b.cy, b.w, b.h,
                                        b.source, b.conf, b.residual, status))

        for r in rejected:
            path = frames[r.frame][0]
            d = r.detection
            summary.sources[r.reason] = summary.sources.get(r.reason, 0) + 1
            rows.append(_report_row(path, r.frame, d.cls, d.cx, d.cy, d.w, d.h,
                                    r.reason, d.conf, r.residual, "not_written"))

        for m in missing:
            path, width, height, _ = frames[m.frame]
            if not (0 <= m.cx < width and 0 <= m.cy < height):
                continue  # 画面の外へ抜けた後のフレーム
            status = _missing_cap_status(out_dir / f"{path.stem}.txt", config.cap_class)
            if status is None:
                continue
            summary.sources["missing_cap"] = summary.sources.get("missing_cap", 0) + 1
            rows.append(_report_row(path, m.frame, config.cap_class, m.cx, m.cy,
                                    None, None, "missing_cap", None, None, status))

    _write_report(out_dir, rows)
    _write_classes(out_dir, class_names)
    return summary


# --- YOLO との接続（ここから下だけが ultralytics を使う） ------------------------


def load_dataset_names(data_yaml: Path) -> list:
    """data.yaml の names を ID 順のリストにする"""
    import yaml

    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def build_class_map(model_names: Mapping[int, str], dataset_names: Sequence[str]) -> tuple:
    """モデルのクラスIDをデータセットのクラスIDへ名前で対応付ける。

    Returns:
        ({モデルID: データセットID}, [データセットに無いモデルのクラス名])
    """
    index = {name: i for i, name in enumerate(dataset_names)}
    mapping = {mid: index[name] for mid, name in model_names.items() if name in index}
    unknown = [name for mid, name in model_names.items() if name not in index]
    return mapping, unknown


def make_yolo_detector(model, imgsz: int, conf: float, class_map: Mapping[int, int],
                       batch: int) -> Detector:
    def detect(paths: Sequence[Path]) -> Iterator[tuple]:
        for start in range(0, len(paths), batch):
            chunk = [str(p) for p in paths[start:start + batch]]
            for result in model.predict(chunk, imgsz=imgsz, conf=conf, verbose=False):
                height, width = result.orig_shape
                dets = [
                    Detection(class_map[int(box.cls[0])], *(float(v) for v in box.xywh[0]),
                              float(box.conf[0]))
                    for box in result.boxes if int(box.cls[0]) in class_map
                ]
                yield width, height, dets

    return detect


def main() -> None:
    defaults = PrelabelConfig()
    parser = argparse.ArgumentParser(description="ラベリング用の下書き（YOLO形式）を作る")
    parser.add_argument("images", type=Path, help="make_dataset.py が書き出した画像のディレクトリ")
    parser.add_argument("--weights", type=Path, required=True, help="学習済みモデル（.pt）")
    parser.add_argument("--out", type=Path, help="ラベルの出力先（省略時は画像と同じ場所）")
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data.yaml"),
                        help="クラス名の対応に使う data.yaml")
    parser.add_argument("--imgsz", type=int, default=1280, help="推論解像度（モデルの学習解像度に合わせる）")
    parser.add_argument("--conf", type=float, default=0.15, help="検出の信頼度閾値（track.py と同じ）")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--inlier-px", type=float, default=defaults.inlier_px)
    parser.add_argument("--segment-gap", type=int, default=defaults.segment_gap)
    parser.add_argument("--max-fill-gap", type=int, default=defaults.max_fill_gap)
    parser.add_argument("--other-conf", type=float, default=defaults.other_conf)
    parser.add_argument("--track-margin", type=int, default=defaults.track_margin)
    parser.add_argument("--static-extent-px", type=float, default=defaults.static_extent_px)
    parser.add_argument("--overwrite", action="store_true", help="既存の .txt を上書きする")
    args = parser.parse_args()

    if not args.images.is_dir():
        raise SystemExit(f"画像ディレクトリがありません: {args.images}")

    from ultralytics import YOLO

    dataset_names = load_dataset_names(args.data)
    model = YOLO(str(args.weights))
    class_map, unknown = build_class_map(model.names, dataset_names)
    if "cap" not in dataset_names or "cap" not in model.names.values():
        raise SystemExit(f"モデルか data.yaml に cap クラスがありません: model={model.names}")
    if unknown:
        print(f"⚠ data.yaml に無いクラスは書き出しません: {unknown}")
    missing = [n for n in dataset_names if n not in model.names.values()]
    if missing:
        print(f"⚠ モデルが出さないクラス（手作業でのラベリングが必要）: {missing}")

    config = PrelabelConfig(
        cap_class=dataset_names.index("cap"),
        inlier_px=args.inlier_px,
        segment_gap=args.segment_gap,
        max_fill_gap=args.max_fill_gap,
        other_conf=args.other_conf,
        track_margin=args.track_margin,
        static_extent_px=args.static_extent_px,
    )
    detector = make_yolo_detector(model, args.imgsz, args.conf, class_map, args.batch)
    out_dir = args.out or args.images
    summary = prelabel_directory(args.images, out_dir, detector, config,
                                 overwrite=args.overwrite, class_names=dataset_names)

    print(f"\n画像 {summary.images}枚: 書き出し {summary.written} / 既存のため未変更 "
          f"{summary.skipped_existing} / 検出なし {summary.empty}")
    if summary.unparsed:
        print(f"⚠ 名前が「動画名_6桁フレーム番号」でない画像 {summary.unparsed}枚は対象外")
    print(f"キャップの下書きがあるフレーム: {summary.cap_frames}枚")
    print(f"内訳: {summary.sources}")
    print(f"確認リスト: {out_dir / REPORT_NAME}（付け忘れ候補・軌跡なし・補間から先に見てください）")


if __name__ == "__main__":
    main()
