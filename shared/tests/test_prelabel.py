"""prelabel のテスト（YOLO / torch 不要）。

検出器は差し替え可能なので、合成した検出結果を返す偽物を使う。
"""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prelabel
from prelabel import Detection, PrelabelConfig

WIDTH, HEIGHT = 1920, 1080
CAP = 0
PITCHER = 1


def parabola(frame):
    """1球分の軌跡（ピクセル座標）。frame 100 で投げて右下へ曲がり落ちる。"""
    t = frame - 100
    return 900.0 + 12.0 * t, 300.0 + 4.0 * t + 0.5 * t * t


def cap_at(frame, conf=0.6, size=12.0, dx=0.0, dy=0.0):
    x, y = parabola(frame)
    return Detection(cls=CAP, cx=x + dx, cy=y + dy, w=size, h=size, conf=conf)


# --- ファイル名の解釈 -----------------------------------------------------------


def test_parse_frame_name_reads_stem_and_frame():
    assert prelabel.parse_frame_name(Path("game_01_000123.jpg")) == ("game_01", 123)


def test_parse_frame_name_rejects_other_names():
    assert prelabel.parse_frame_name(Path("cover.jpg")) is None
    assert prelabel.parse_frame_name(Path("game_123.jpg")) is None


def test_group_frames_sorts_by_frame_and_skips_unknown_names(tmp_path):
    paths = [tmp_path / n for n in ("b_000002.jpg", "a_000010.jpg", "b_000001.jpg", "x.jpg")]
    grouped = prelabel.group_frames(paths)
    assert list(grouped) == ["a", "b"]
    assert [frame for frame, _ in grouped["b"]] == [1, 2]


# --- 区間分割 ------------------------------------------------------------------


def test_split_segments_breaks_on_large_gaps():
    frames = [1, 2, 3, 10, 11, 40]
    assert prelabel.split_segments(frames, max_gap=7) == [[1, 2, 3, 10, 11], [40]]
    assert prelabel.split_segments(frames, max_gap=5) == [[1, 2, 3], [10, 11], [40]]


def test_split_segments_empty():
    assert prelabel.split_segments([], max_gap=5) == []


# --- キャップ軌跡の整理 -----------------------------------------------------------


def run_refine(detections, available=None, config=None):
    available = set(range(90, 130)) if available is None else available
    boxes, rejected, _ = prelabel.refine_cap_track(detections, available, config or PrelabelConfig())
    return boxes, rejected


def test_refine_keeps_inliers_and_rejects_outlier():
    dets = {f: [cap_at(f)] for f in range(100, 112)}
    dets[105] = [cap_at(105), cap_at(105, dx=300, conf=0.9)]  # 投手の腕などの誤検出
    boxes, rejected = run_refine(dets)

    assert sorted(boxes) == list(range(100, 112))
    assert all(b.source == "detected" for b in boxes.values())
    assert boxes[105].cx == pytest.approx(parabola(105)[0])
    assert [(r.frame, r.reason) for r in rejected] == [(105, "outlier")]


def test_refine_interpolates_missing_frames_with_neighbour_size():
    dets = {f: [cap_at(f, size=10.0 if f < 105 else 14.0)] for f in range(100, 112) if f not in (105, 106)}
    boxes, _ = run_refine(dets)

    for frame in (105, 106):
        box = boxes[frame]
        assert box.source == "interpolated"
        assert box.conf is None
        x, y = parabola(frame)
        assert box.cx == pytest.approx(x, abs=1e-6)
        assert box.cy == pytest.approx(y, abs=1e-6)
    assert 10.0 < boxes[105].w < boxes[106].w < 14.0


def test_refine_does_not_fill_frames_without_images():
    # stride 付きで書き出したデータセットは間のフレームの画像が無い
    dets = {f: [cap_at(f)] for f in range(100, 112) if f not in (105, 106)}
    available = set(range(90, 130)) - {106}
    boxes, _ = run_refine(dets, available=available)
    assert 105 in boxes
    assert 106 not in boxes


def test_refine_does_not_fill_long_gaps():
    dets = {f: [cap_at(f)] for f in list(range(100, 104)) + list(range(110, 114))}
    boxes, _ = run_refine(dets, config=PrelabelConfig(max_fill_gap=3, segment_gap=10))
    assert all(f not in boxes for f in range(104, 110))


def test_refine_keeps_one_cap_per_frame_closest_to_curve():
    dets = {f: [cap_at(f)] for f in range(100, 112)}
    dets[104] = [cap_at(104, dx=8.0, conf=0.9), cap_at(104, dx=1.0, conf=0.3)]
    boxes, rejected = run_refine(dets)

    assert boxes[104].conf == pytest.approx(0.3)
    assert [(r.frame, r.reason) for r in rejected] == [(104, "duplicate")]


def test_refine_short_segment_is_kept_as_unfit():
    dets = {f: [cap_at(f, conf=0.2 + 0.1 * i), cap_at(f, dx=50, conf=0.1)] for i, f in enumerate(range(100, 103))}
    boxes, rejected = run_refine(dets)

    assert sorted(boxes) == [100, 101, 102]
    assert all(b.source == "unfit" for b in boxes.values())
    assert boxes[102].conf == pytest.approx(0.4)  # 最も信頼度の高い検出を残す
    assert {r.reason for r in rejected} == {"duplicate"}


def test_refine_handles_each_segment_separately():
    first = {f: [cap_at(f)] for f in range(100, 110)}
    second = {f + 200: [cap_at(f, dy=-100)] for f in range(100, 110)}
    boxes, rejected = run_refine({**first, **second}, available=set(range(90, 320)))

    assert len(boxes) == 20
    assert rejected == []


# --- YOLO 形式の出力 ------------------------------------------------------------


def test_to_yolo_line_normalises_and_clips():
    box = prelabel.LabelBox(cls=CAP, cx=1915.0, cy=540.0, w=20.0, h=10.0, source="detected", conf=0.5)
    line = prelabel.to_yolo_line(box, WIDTH, HEIGHT)
    cls, cx, cy, w, h = line.split()
    assert cls == "0"
    assert float(cx) + float(w) / 2 <= 1.0 + 1e-9
    assert float(cy) == pytest.approx(0.5)
    assert len(line.split()) == 5  # label_qa は5列以外を不正とみなす


# --- ディレクトリ単位の実行 --------------------------------------------------------


def make_images(directory, stem, frames):
    directory.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        (directory / f"{stem}_{frame:06d}.jpg").write_bytes(b"")


def fake_detector(per_frame):
    """{(stem, frame): [Detection, ...]} を返す偽の検出器"""

    def detect(paths):
        for path in paths:
            stem, frame = prelabel.parse_frame_name(path)
            yield WIDTH, HEIGHT, per_frame.get((stem, frame), [])

    return detect


def read_report(out_dir):
    with open(out_dir / prelabel.REPORT_NAME, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_prelabel_directory_writes_labels_and_report(tmp_path):
    images = tmp_path / "frames"
    make_images(images, "pitch", range(95, 115))
    per_frame = {("pitch", f): [cap_at(f)] for f in range(100, 112) if f != 106}
    per_frame[("pitch", 108)] = [cap_at(108), Detection(PITCHER, 500, 500, 100, 300, 0.8)]

    summary = prelabel.prelabel_directory(images, images, fake_detector(per_frame), PrelabelConfig())

    assert summary.written == 12
    assert not (images / "pitch_000095.txt").exists()  # 何も無いフレームはファイルを作らない
    lines = (images / "pitch_000108.txt").read_text().splitlines()
    assert sorted(line.split()[0] for line in lines) == ["0", "1"]
    assert (images / "pitch_000106.txt").exists()

    report = read_report(images)
    sources = {(r["image"], r["source"]) for r in report}
    assert ("pitch_000106.jpg", "interpolated") in sources
    # 付け忘れ候補（軌跡の前後）の次に、補間した矩形が来る
    order = [r["source"] for r in report]
    assert order[:6] == ["missing_cap"] * 6  # 97-99 と 112-114
    assert order[6] == "interpolated"


def test_prelabel_directory_never_overwrites_existing_labels(tmp_path):
    images = tmp_path / "frames"
    make_images(images, "pitch", range(100, 112))
    (images / "pitch_000103.txt").write_text("0 0.5 0.5 0.01 0.01\n")
    per_frame = {("pitch", f): [cap_at(f)] for f in range(100, 112)}

    summary = prelabel.prelabel_directory(images, images, fake_detector(per_frame), PrelabelConfig())

    assert (images / "pitch_000103.txt").read_text() == "0 0.5 0.5 0.01 0.01\n"
    assert summary.skipped_existing == 1
    statuses = {r["image"]: r["status"] for r in read_report(images)}
    assert statuses["pitch_000103.jpg"] == "skipped_existing"


def test_prelabel_directory_overwrite_flag(tmp_path):
    images = tmp_path / "frames"
    make_images(images, "pitch", range(100, 112))
    (images / "pitch_000103.txt").write_text("stale\n")
    per_frame = {("pitch", f): [cap_at(f)] for f in range(100, 112)}

    prelabel.prelabel_directory(
        images, images, fake_detector(per_frame), PrelabelConfig(), overwrite=True
    )

    assert (images / "pitch_000103.txt").read_text().startswith("0 ")


def test_prelabel_directory_filters_low_conf_other_classes(tmp_path):
    images = tmp_path / "frames"
    make_images(images, "pitch", [100])
    per_frame = {("pitch", 100): [Detection(PITCHER, 500, 500, 100, 300, 0.3)]}

    summary = prelabel.prelabel_directory(
        images, tmp_path / "labels", fake_detector(per_frame), PrelabelConfig(other_conf=0.5)
    )

    assert summary.written == 0


def test_prelabel_directory_writes_classes_file_once(tmp_path):
    images = tmp_path / "frames"
    make_images(images, "pitch", [100])
    out = tmp_path / "labels"

    prelabel.prelabel_directory(
        images, out, fake_detector({}), PrelabelConfig(), class_names=["cap", "pitcher_motion"]
    )
    assert (out / "classes.txt").read_text() == "cap\npitcher_motion\n"

    (out / "classes.txt").write_text("edited\n")
    prelabel.prelabel_directory(
        images, out, fake_detector({}), PrelabelConfig(), class_names=["cap"]
    )
    assert (out / "classes.txt").read_text() == "edited\n"


# --- クラスIDの対応付け ----------------------------------------------------------


def test_build_class_map_matches_by_name():
    model_names = {0: "cap", 1: "pitcher", 2: "catcher"}
    dataset_names = ["cap", "pitcher_motion", "batter_stance", "umpire", "catcher"]
    mapping, unknown = prelabel.build_class_map(model_names, dataset_names)
    assert mapping == {0: 0, 2: 4}
    assert unknown == ["pitcher"]


def test_prelabel_directory_drops_duplicate_other_boxes(tmp_path):
    # label_qa は同一クラスで IoU > 0.9 の矩形を重複として扱う
    images = tmp_path / "frames"
    make_images(images, "pitch", [100])
    per_frame = {("pitch", 100): [
        Detection(PITCHER, 500, 500, 100, 300, 0.6),
        Detection(PITCHER, 501, 500, 100, 300, 0.9),
        Detection(PITCHER, 900, 500, 100, 300, 0.7),  # 離れた別人物は残す
    ]}

    prelabel.prelabel_directory(images, images, fake_detector(per_frame), PrelabelConfig())

    lines = (images / "pitch_000100.txt").read_text().splitlines()
    assert len(lines) == 2
    statuses = sorted((r["source"], r["conf"]) for r in read_report(images))
    assert ("overlap", "0.600") in statuses


def test_refine_prefers_confident_box_when_duplicates_overlap():
    # 同じキャップを2重に検出したときは、軌跡からのずれの差は誤差なので信頼度で選ぶ
    dets = {f: [cap_at(f)] for f in range(100, 112)}
    dets[104] = [cap_at(104, dx=2.0, conf=0.9), cap_at(104, dx=0.5, conf=0.3)]
    boxes, rejected = run_refine(dets)

    assert boxes[104].conf == pytest.approx(0.9)
    assert [(r.frame, r.reason) for r in rejected] == [(104, "duplicate")]


def test_refine_reports_frames_where_cap_is_missing():
    # 補間しない長い欠損と軌跡の前後は、キャップの付け忘れが起きやすい
    dets = {f: [cap_at(f)] for f in list(range(100, 104)) + list(range(110, 114))}
    config = PrelabelConfig(max_fill_gap=3, segment_gap=10, track_margin=2)
    available = set(range(90, 130)) - {108}
    _, _, missing = prelabel.refine_cap_track(dets, available, config)

    assert [m.frame for m in missing] == [98, 99, 104, 105, 106, 107, 109, 114, 115]
    x, y = parabola(105)
    assert missing[3].cx == pytest.approx(x, abs=1e-6)
    assert missing[3].cy == pytest.approx(y, abs=1e-6)


def test_prelabel_directory_lists_missing_caps_first_and_overlaps_last(tmp_path):
    images = tmp_path / "frames"
    make_images(images, "pitch", range(95, 115))
    per_frame = {("pitch", f): [cap_at(f, conf=0.3)] for f in range(100, 110)}
    per_frame[("pitch", 96)] = [
        Detection(PITCHER, 500, 500, 100, 300, 0.9),
        Detection(PITCHER, 501, 500, 100, 300, 0.8),
    ]

    summary = prelabel.prelabel_directory(
        images, images, fake_detector(per_frame), PrelabelConfig(track_margin=2)
    )

    report = read_report(images)
    assert report[0]["source"] == "missing_cap"
    assert report[0]["status"] == "no_cap_label"
    assert report[-1]["source"] == "overlap"
    assert summary.cap_frames == 10
    assert summary.sources["missing_cap"] == 4  # 98, 99, 110, 111


# --- レビューで見つかった取りこぼし ------------------------------------------------


def test_refine_static_false_positive_does_not_take_over_pitches():
    # 地面に落ちたキャップなど、動かない誤検出が続くと全投球が1区間につながる
    dets = {f: [Detection(CAP, 200.0, 900.0, 12.0, 12.0, 0.3)] for f in range(90, 330)}
    for start in (100, 250):
        for f in range(start, start + 15):
            dets[f] = [*dets[f], cap_at(f - start + 100, conf=0.8)]
    boxes, rejected = run_refine(dets, available=set(range(90, 330)))

    pitch_frames = [f for s in (100, 250) for f in range(s, s + 15)]
    assert all(boxes[f].conf == pytest.approx(0.8) for f in pitch_frames)
    assert all(f in pitch_frames for f in boxes)
    assert {r.reason for r in rejected} == {"static"}


def test_missing_cap_skips_frames_already_labelled_with_cap(tmp_path):
    images = tmp_path / "frames"
    make_images(images, "pitch", range(95, 115))
    (images / "pitch_000098.txt").write_text("0 0.5 0.5 0.01 0.01\n")  # 人がキャップを付けた
    (images / "pitch_000099.txt").write_text("3 0.5 0.5 0.1 0.2\n")  # 人物だけ
    per_frame = {("pitch", f): [cap_at(f)] for f in range(100, 110)}

    prelabel.prelabel_directory(
        images, images, fake_detector(per_frame), PrelabelConfig(track_margin=2)
    )

    missing = {r["image"]: r["status"] for r in read_report(images) if r["source"] == "missing_cap"}
    assert "pitch_000098.jpg" not in missing
    assert missing["pitch_000099.jpg"] == "existing_label_without_cap"
    assert missing["pitch_000110.jpg"] == "no_cap_label"


def test_missing_cap_outside_image_is_not_listed(tmp_path):
    images = tmp_path / "frames"
    make_images(images, "pitch", range(95, 115))
    # 右へ速く抜けていく軌跡。前後の予測位置は画像の外になる
    per_frame = {
        ("pitch", f): [Detection(CAP, 1500.0 + 60.0 * (f - 100), 500.0, 12, 12, 0.8)]
        for f in range(100, 107)
    }

    prelabel.prelabel_directory(
        images, images, fake_detector(per_frame), PrelabelConfig(track_margin=3)
    )

    missing = [int(r["frame"]) for r in read_report(images) if r["source"] == "missing_cap"]
    assert missing == [97, 98, 99]  # 107 以降は x > 1920


def test_box_entirely_outside_image_is_not_written(tmp_path):
    images = tmp_path / "frames"
    make_images(images, "pitch", [100])
    per_frame = {("pitch", 100): [Detection(PITCHER, 2100.0, 500.0, 100, 300, 0.9)]}

    summary = prelabel.prelabel_directory(images, images, fake_detector(per_frame), PrelabelConfig())

    assert summary.written == 0
    assert not (images / "pitch_000100.txt").exists()
