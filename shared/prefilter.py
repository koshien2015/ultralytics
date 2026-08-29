"""YOLO推論の前段フィルタ。

track.py は全フレームに YOLO を掛けているが、キャップが写っているのは
投球の 0.3〜0.5 秒だけで、90分の試合映像なら全体の数%しかない。
このモジュールは推論するフレームを2段階で絞る。

1. 事前検出（`scan_activity` → `detect_windows`）
   古典CVの3フレーム差分で投手ROIの活動量を測り、投球イベントの区間を出す。
   実時間の10倍以上の速度で全編を走査できる。

2. 間引き＋追従（`InferenceGate`）
   区間内でも、未検出のうちは N フレームに1回だけ推論する。
   キャップを見つけたらそこから連続で推論して飛翔を追い切る。

どちらも OpenCV と NumPy だけで動く（ultralytics / torch には依存しない）。

単体実行すると、指定動画で何フレームまで削れるかを見積もれる:
    python prefilter.py game.mp4 --roi 0.55 0.20 0.30 0.45
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Iterator, Sequence

import cv2
import numpy as np

Rect = tuple[float, float, float, float]


class PrefilterError(RuntimeError):
    """前段フィルタの処理に失敗したときに送出する。"""


@dataclass(frozen=True)
class PrefilterConfig:
    """既定値は 60fps の試合映像を想定した仮値。実素材で調整すること。"""

    roi: Rect | None = None
    """投手ROI（相対座標 x, y, w, h）。None なら全画面の活動量を使う。"""
    resize_long_edge: int = 480
    """活動量計測時の処理解像度。小さいほど速い。"""
    diff_threshold: int = 12
    morph_kernel: int = 3
    smooth_window_sec: float = 0.3
    threshold_factor: float = 2.0
    """閾値 = 活動量の中央値 × この係数。"""
    min_activity_px: int = 20
    """閾値の絶対下限。動きの少ない映像で中央値が0に潰れる対策。"""
    min_event_interval_sec: float = 1.5
    """これより短い間隔のイベントは同じ投球として統合する。"""
    pre_margin_sec: float = 1.0
    post_margin_sec: float = 2.5
    """イベント時刻の前後にとる余裕。区間はこの幅で作られる。"""
    search_stride: int = 5
    """未検出のあいだ何フレームに1回推論するか。1で間引きなし。"""
    dense_frames: int = 30
    """検出後に連続で推論するフレーム数。飛翔時間より長くとる。"""


@dataclass(frozen=True)
class ActivityProfile:
    """活動量の時系列。frames は元動画のフレーム番号。"""

    frames: np.ndarray
    activity: np.ndarray
    fps: float
    total_frames: int


def _to_gray(frame: np.ndarray, long_edge: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    current = max(gray.shape[0], gray.shape[1])
    if current <= long_edge:
        return gray
    scale = long_edge / current
    size = (max(1, int(round(gray.shape[1] * scale))), max(1, int(round(gray.shape[0] * scale))))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def _roi_pixels(roi: Rect | None, width: int, height: int) -> tuple[int, int, int, int]:
    if roi is None:
        return (0, 0, width, height)
    x = min(int(round(roi[0] * width)), width - 1)
    y = min(int(round(roi[1] * height)), height - 1)
    return (x, y, max(1, min(int(round(roi[2] * width)), width - x)),
            max(1, min(int(round(roi[3] * height)), height - y)))


def scan_activity(video_path: str, config: PrefilterConfig = PrefilterConfig()) -> ActivityProfile:
    """3フレーム差分でROI内の前景ピクセル数を時系列化する。

    返るマスクは中央フレームに対応するので、活動量は1フレーム遅れて記録される。
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        raise PrefilterError(f"動画を開けませんでした: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise PrefilterError(f"fpsを取得できませんでした: {video_path}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_kernel, config.morph_kernel))

        numbers: list[int] = []
        values: list[float] = []
        window: list[np.ndarray] = []
        roi_px: tuple[int, int, int, int] | None = None
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = _to_gray(frame, config.resize_long_edge)
            if roi_px is None:
                roi_px = _roi_pixels(config.roi, gray.shape[1], gray.shape[0])
            window.append(gray)
            if len(window) >= 3:
                mask = _three_frame_mask(window[-3], window[-2], window[-1], config.diff_threshold, kernel)
                x, y, w, h = roi_px
                numbers.append(index - 1)  # マスクは中央フレームに対応する
                values.append(float(cv2.countNonZero(mask[y : y + h, x : x + w])))
                window.pop(0)
            index += 1
    finally:
        capture.release()

    return ActivityProfile(
        frames=np.array(numbers, dtype=np.int64),
        activity=np.array(values, dtype=np.float64),
        fps=fps,
        total_frames=total if total > 0 else index,
    )


def _three_frame_mask(
    previous: np.ndarray, current: np.ndarray, following: np.ndarray, threshold: int, kernel: np.ndarray
) -> np.ndarray:
    """t-1,t,t+1 の2つの差分のAND。照明変動に強い。"""
    _, back = cv2.threshold(cv2.absdiff(current, previous), threshold, 255, cv2.THRESH_BINARY)
    _, forward = cv2.threshold(cv2.absdiff(following, current), threshold, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(back, forward)
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[: values.size]


def _runs_above(flags: np.ndarray) -> list[tuple[int, int]]:
    if flags.size == 0:
        return []
    padded = np.concatenate([[False], flags, [False]])
    changes = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1).tolist(), (np.flatnonzero(changes == -1) - 1).tolist()))


def detect_windows(
    profile: ActivityProfile, config: PrefilterConfig = PrefilterConfig()
) -> tuple[tuple[int, int], ...]:
    """活動量プロファイルから、推論すべきフレーム区間 [start, end) を作る。

    区間は前後マージンを付けたうえで、重なるものを結合して返す。
    """
    if profile.activity.size == 0:
        return ()
    window = max(1, int(round(config.smooth_window_sec * profile.fps)))
    smoothed = _smooth(profile.activity, window)
    threshold = max(
        float(np.median(smoothed)) * config.threshold_factor, float(config.min_activity_px)
    )
    starts = [int(profile.frames[begin]) for begin, _end in _runs_above(smoothed > threshold)]
    merged_starts = _merge_close(starts, config.min_event_interval_sec, profile.fps)

    pre = int(round(config.pre_margin_sec * profile.fps))
    post = int(round(config.post_margin_sec * profile.fps))
    spans = [(max(0, s - pre), min(profile.total_frames, s + post)) for s in merged_starts]
    return _merge_overlapping(spans)


def _merge_close(starts: Sequence[int], min_interval_sec: float, fps: float) -> list[int]:
    """近すぎるイベントを1つにまとめる。"""
    gap = min_interval_sec * fps
    merged: list[int] = []
    for start in starts:
        if merged and start - merged[-1] < gap:
            continue
        merged.append(start)
    return merged


def _merge_overlapping(spans: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """重なる区間を結合する。"""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def windows_frame_count(windows: Sequence[tuple[int, int]]) -> int:
    """区間に含まれるフレーム数の合計。"""
    return sum(end - start for start, end in windows)


class InferenceGate:
    """このフレームで推論すべきかを判定する。

    区間の外は常にスキップ。区間内でも、キャップ未検出のあいだは
    `search_stride` フレームに1回だけ推論し、検出したら `dense_frames`
    のあいだ連続で推論して飛翔を追い切る。
    """

    def __init__(
        self,
        windows: Sequence[tuple[int, int]] | None,
        config: PrefilterConfig = PrefilterConfig(),
    ) -> None:
        if config.search_stride < 1:
            raise ValueError("search_stride は1以上である必要があります")
        self._windows = tuple(windows) if windows is not None else None
        self._config = config
        self._dense_until = -1
        self.inferred = 0
        self.skipped = 0

    def in_window(self, frame_number: int) -> bool:
        """区間内か（区間指定が無ければ常に True）。"""
        if self._windows is None:
            return True
        return any(start <= frame_number < end for start, end in self._windows)

    def should_infer(self, frame_number: int) -> bool:
        """このフレームに推論を掛けるべきか。呼ぶたびに集計する。"""
        decision = self._decide(frame_number)
        if decision:
            self.inferred += 1
        else:
            self.skipped += 1
        return decision

    def _decide(self, frame_number: int) -> bool:
        if not self.in_window(frame_number):
            return False
        if frame_number <= self._dense_until:
            return True
        return frame_number % self._config.search_stride == 0

    def note_detection(self, frame_number: int, found: bool) -> None:
        """推論結果を伝える。検出できていれば以降しばらく密に推論する。"""
        if found:
            self._dense_until = frame_number + self._config.dense_frames

    @property
    def summary(self) -> str:
        total = self.inferred + self.skipped
        if total == 0:
            return "推論対象なし"
        rate = self.inferred / total
        return f"推論 {self.inferred}/{total} フレーム ({rate:.1%})、削減 {1 - rate:.1%}"


def build_gate(
    video_path: str, config: PrefilterConfig = PrefilterConfig(), *, use_windows: bool = True
) -> tuple[InferenceGate, tuple[tuple[int, int], ...]]:
    """動画を事前走査してゲートを作る。"""
    if not use_windows:
        return InferenceGate(None, config), ()
    windows = detect_windows(scan_activity(video_path, config), config)
    return InferenceGate(windows, config), windows


def _parse_args(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="推論フレーム数の削減量を見積もる")
    parser.add_argument("video")
    parser.add_argument("--roi", nargs=4, type=float, metavar=("X", "Y", "W", "H"),
                        help="投手ROI（相対座標）。省略時は全画面")
    parser.add_argument("--threshold-factor", type=float, default=PrefilterConfig.threshold_factor)
    parser.add_argument("--min-event-interval", type=float, default=PrefilterConfig.min_event_interval_sec)
    parser.add_argument("--search-stride", type=int, default=PrefilterConfig.search_stride)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import time

    args = _parse_args(argv)
    config = PrefilterConfig(
        roi=tuple(args.roi) if args.roi else None,
        threshold_factor=args.threshold_factor,
        min_event_interval_sec=args.min_event_interval,
        search_stride=args.search_stride,
    )
    started = time.perf_counter()
    profile = scan_activity(args.video, config)
    windows = detect_windows(profile, config)
    elapsed = time.perf_counter() - started

    total = profile.total_frames
    in_windows = windows_frame_count(windows)
    print(f"{args.video}: {total} フレーム / {total / profile.fps:.1f} 秒")
    print(f"事前走査 {elapsed:.1f} 秒 ({total / elapsed:.0f} fps = 実時間の {total / elapsed / profile.fps:.1f} 倍速)")
    print(f"検出区間 {len(windows)} 件、{in_windows} フレーム ({in_windows / total:.1%})")
    print(f"間引き(stride={config.search_stride})を併用した場合の推論フレーム数の目安: "
          f"{in_windows // config.search_stride}〜{in_windows} ({in_windows / config.search_stride / total:.1%}〜{in_windows / total:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
