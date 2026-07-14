"""YOLOラベルの品質チェックスクリプト

ラベリングし直し作業の品質確認用。データセットを走査して以下を検出する:

- 画像とラベルの対応漏れ（ラベルなし画像 / 画像なしラベル）
- 空のラベルファイル
- 座標が[0,1]範囲外・書式不正の行
- キャップ(class 0)のbboxサイズ異常（小さすぎ=誤クリック疑い / 大きすぎ=別物体疑い）
- 同一クラスの重複bbox（IoU > 0.9）
- 1フレームにキャップが2個以上（キャップ野球では原則1個）
- クラス別の件数・キャップbboxのサイズ分布

使用例:
    python label_qa.py datasets/train datasets/valid
    python label_qa.py datasets/train --img-width 1920 --img-height 1080
"""

import argparse
import statistics
from pathlib import Path

CAP_CLASS_ID = 0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# キャップの妥当なピクセルサイズ範囲（1080p想定。ブラーで伸びるため上限は緩め）
DEFAULT_MIN_CAP_PX = 4
DEFAULT_MAX_CAP_PX = 150
DUPLICATE_IOU_THRESHOLD = 0.9


def parse_label_line(line):
    """YOLO形式の1行をパースする。不正なら None を返す"""
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        cls = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
    except ValueError:
        return None
    return {"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h}


def is_out_of_range(box):
    """正規化座標が[0,1]を外れているか"""
    values = (box["cx"], box["cy"], box["w"], box["h"])
    return any(v < 0.0 or v > 1.0 for v in values) or box["w"] <= 0 or box["h"] <= 0


def iou(a, b):
    """正規化xywh同士のIoU"""
    ax1, ay1 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2
    ax2, ay2 = a["cx"] + a["w"] / 2, a["cy"] + a["h"] / 2
    bx1, by1 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2
    bx2, by2 = b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def check_label_file(label_path, img_width, img_height, min_cap_px, max_cap_px):
    """1ラベルファイルを検査し、問題リストとキャップbboxのpxサイズを返す"""
    issues = []
    cap_sizes_px = []

    try:
        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as e:
        return [f"読み込み失敗: {e}"], []

    if not lines:
        return ["空のラベルファイル"], []

    boxes = []
    for i, line in enumerate(lines, start=1):
        box = parse_label_line(line)
        if box is None:
            issues.append(f"{i}行目: 書式不正 ({line!r})")
            continue
        if is_out_of_range(box):
            issues.append(f"{i}行目: 座標が[0,1]範囲外または幅高さが0以下")
            continue
        boxes.append(box)

    cap_boxes = [b for b in boxes if b["cls"] == CAP_CLASS_ID]

    for box in cap_boxes:
        w_px = box["w"] * img_width
        h_px = box["h"] * img_height
        cap_sizes_px.append((w_px, h_px))
        longest = max(w_px, h_px)
        if longest < min_cap_px:
            issues.append(
                f"キャップbboxが小さすぎ ({w_px:.0f}x{h_px:.0f}px): 誤クリックの疑い"
            )
        elif longest > max_cap_px:
            issues.append(
                f"キャップbboxが大きすぎ ({w_px:.0f}x{h_px:.0f}px): 別物体の疑い"
            )

    if len(cap_boxes) > 1:
        issues.append(f"キャップが{len(cap_boxes)}個ラベルされている（原則1個）")

    # 同一クラスの重複bbox
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i]["cls"] == boxes[j]["cls"] and iou(boxes[i], boxes[j]) > DUPLICATE_IOU_THRESHOLD:
                issues.append(
                    f"class {boxes[i]['cls']} に重複bbox（IoU > {DUPLICATE_IOU_THRESHOLD}）"
                )

    return issues, cap_sizes_px


def collect_split(split_dir):
    """images/labels ペアの構成を返す（どちらかが直下にある場合にも対応）"""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.is_dir():
        images_dir = split_dir
    if not labels_dir.is_dir():
        labels_dir = split_dir

    images = {
        p.stem: p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    } if images_dir.is_dir() else {}
    labels = {
        p.stem: p for p in labels_dir.iterdir() if p.suffix == ".txt"
    } if labels_dir.is_dir() else {}

    return images, labels


def report_split(split_dir, img_width, img_height, min_cap_px, max_cap_px):
    """1スプリット分の検査を実行し、(問題件数, キャップサイズ一覧, クラス件数) を返す"""
    images, labels = collect_split(split_dir)

    print(f"\n{'=' * 60}")
    print(f"検査対象: {split_dir}  (画像 {len(images)} / ラベル {len(labels)})")
    print("=" * 60)

    missing_labels = sorted(set(images) - set(labels))
    orphan_labels = sorted(set(labels) - set(images))
    if missing_labels:
        print(f"\n⚠ ラベルなし画像: {len(missing_labels)}件")
        for stem in missing_labels[:10]:
            print(f"    {images[stem].name}")
        if len(missing_labels) > 10:
            print(f"    ... 他{len(missing_labels) - 10}件")
    if orphan_labels:
        print(f"\n⚠ 画像なしラベル: {len(orphan_labels)}件")
        for stem in orphan_labels[:10]:
            print(f"    {labels[stem].name}")

    total_issues = len(missing_labels) + len(orphan_labels)
    all_cap_sizes = []
    class_counts = {}
    files_with_issues = 0

    for stem in sorted(labels):
        label_path = labels[stem]
        issues, cap_sizes = check_label_file(
            label_path, img_width, img_height, min_cap_px, max_cap_px
        )
        all_cap_sizes.extend(cap_sizes)

        for line in label_path.read_text(encoding="utf-8").splitlines():
            box = parse_label_line(line.strip()) if line.strip() else None
            if box:
                class_counts[box["cls"]] = class_counts.get(box["cls"], 0) + 1

        if issues:
            files_with_issues += 1
            total_issues += len(issues)
            print(f"\n  {label_path.name}:")
            for issue in issues:
                print(f"    - {issue}")

    print(f"\n--- サマリ ({split_dir.name}) ---")
    print(f"問題のあるファイル: {files_with_issues} / {len(labels)}")
    print(f"クラス別アノテーション数: "
          f"{dict(sorted(class_counts.items())) if class_counts else 'なし'}")

    if all_cap_sizes:
        longest_sides = [max(w, h) for w, h in all_cap_sizes]
        print(f"キャップbbox (class {CAP_CLASS_ID}): {len(all_cap_sizes)}個")
        print(f"  長辺px: 最小 {min(longest_sides):.0f} / "
              f"中央値 {statistics.median(longest_sides):.0f} / "
              f"最大 {max(longest_sides):.0f}")
    else:
        print(f"⚠ キャップ (class {CAP_CLASS_ID}) のアノテーションが1つもありません")

    return total_issues


def main():
    parser = argparse.ArgumentParser(description="YOLOラベルの品質チェック")
    parser.add_argument(
        "splits", nargs="+", type=Path,
        help="検査するディレクトリ（train / valid など。images+labels構成を想定）",
    )
    parser.add_argument("--img-width", type=int, default=1920, help="画像の幅px")
    parser.add_argument("--img-height", type=int, default=1080, help="画像の高さpx")
    parser.add_argument("--min-cap-px", type=int, default=DEFAULT_MIN_CAP_PX,
                        help="キャップbbox長辺の下限px（これ未満は異常として報告）")
    parser.add_argument("--max-cap-px", type=int, default=DEFAULT_MAX_CAP_PX,
                        help="キャップbbox長辺の上限px（これ超過は異常として報告）")
    args = parser.parse_args()

    total = 0
    for split_dir in args.splits:
        if not split_dir.is_dir():
            print(f"⚠ ディレクトリが見つかりません: {split_dir}")
            total += 1
            continue
        total += report_split(
            split_dir, args.img_width, args.img_height,
            args.min_cap_px, args.max_cap_px,
        )

    print(f"\n{'=' * 60}")
    if total == 0:
        print("✅ 問題は見つかりませんでした")
    else:
        print(f"⚠ 合計 {total} 件の問題が見つかりました")
    raise SystemExit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
