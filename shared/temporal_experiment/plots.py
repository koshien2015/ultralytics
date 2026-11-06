"""スライドに貼る評価グラフ（静止画）。

配色は手法・状態ごとに固定し、順位で変えない。色だけに頼らないよう軸や凡例に名前を出す。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .schema import DETECTED, INTERPOLATED, MISSING, PREDICTED

METHOD_LABELS = {
    "yolo": "YOLO単体",
    "nearest": "最近傍追跡",
    "linear": "線形補間",
    "kalman": "カルマン",
    "ransac": "RANSAC曲線",
}
METHOD_COLORS = {
    "yolo": "#2a78d6",
    "nearest": "#eb6834",
    "linear": "#1baf7a",
    "kalman": "#eda100",
    "ransac": "#e87ba4",
}
STATUS_LABELS = {
    DETECTED: "検出",
    INTERPOLATED: "補間",
    PREDICTED: "予測",
    MISSING: "欠損",
}
STATUS_COLORS = {
    DETECTED: "#2a78d6",
    INTERPOLATED: "#eb6834",
    PREDICTED: "#1baf7a",
    MISSING: "#e4e3df",
}
JAPANESE_FONTS = ("Hiragino Sans", "Noto Sans CJK JP", "IPAexGothic", "Yu Gothic")
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def _available_font() -> str:
    """環境にある日本語フォントを選ぶ。無ければ既定に任せる（文字は豆腐になる）"""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return next((name for name in JAPANESE_FONTS if name in installed), "DejaVu Sans")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": _available_font(),
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": TEXT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def frame_errors(estimates, truths) -> np.ndarray:
    """正解ごとの中心位置誤差。位置を出力しなかったフレームは NaN"""
    by_frame = {e.frame: e for e in estimates}

    def error(truth):
        estimate = by_frame.get(truth.frame)
        if estimate is None or not estimate.has_position:
            return np.nan
        return float(np.hypot(estimate.x - truth.x, estimate.y - truth.y))

    return np.array([error(g) for g in truths], dtype=float)


def _plot_coverage(out_dir, summary) -> Path:
    names = list(summary["methods"])[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 1.4))
    for row, name in enumerate(names):
        values = summary["methods"][name]
        visible = max(values["n_visible"], 1)
        left = 0.0
        for status in (DETECTED, INTERPOLATED, PREDICTED):
            share = values["status_counts"][status] / visible
            ax.barh(row, share, left=left, height=0.55, color=STATUS_COLORS[status],
                    edgecolor=SURFACE, linewidth=2)
            left += share
        correct = values["correct_coverage"]
        note = "" if correct is None else f"（正しい位置 {correct:.1%}）"
        ax.text(left + 0.01, row, f"{left:.1%}{note}", va="center", color=TEXT, fontsize=10)
    ax.set_yticks(range(len(names)), [METHOD_LABELS[n] for n in names])
    ax.set_xlim(0, 1.4)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_title("位置推定カバー率（蓋が見えているフレームのうち位置を出力した割合）", loc="left", fontsize=12)
    ax.legend(
        handles=[Patch(color=STATUS_COLORS[s], label=STATUS_LABELS[s])
                 for s in (DETECTED, INTERPOLATED, PREDICTED)],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False,
    )
    return _save(fig, out_dir / "coverage.png")


def _plot_error(out_dir, summary) -> Path:
    panels = (
        ("mae_output", "MAE（出力フレームのみ）"),
        ("p95_output", "P95（出力フレームのみ）"),
        ("mae_all", "MAE（全フレーム・欠損に上限誤差）"),
    )
    names = list(summary["methods"])[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.6 * len(names) + 1.2), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        values = [summary["methods"][n][key] for n in names]
        known = [v for v in values if v is not None]
        top = max(known, default=1.0) or 1.0
        for row, (name, value) in enumerate(zip(names, values)):
            if value is None:
                ax.text(0, row, "出力なし", va="center", color=MUTED, fontsize=9)
                continue
            ax.barh(row, value, height=0.55, color=METHOD_COLORS[name])
            ax.text(value + top * 0.02, row, f"{value:.1f}", va="center", fontsize=10)
        ax.set_xlim(0, top * 1.2)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("px")
    axes[0].set_yticks(range(len(names)), [METHOD_LABELS[n] for n in names])
    return _save(fig, out_dir / "error.png")


def _visible_sorted(truths):
    return sorted((g for g in truths if g.visible), key=lambda g: g.frame)


def _plot_timeline(out_dir, runs, truths) -> Path:
    visible = _visible_sorted(truths)
    frames = np.array([g.frame for g in visible])
    breaks = np.flatnonzero(np.diff(frames) > 1) + 1 if len(frames) else np.array([], dtype=int)
    positions = np.arange(len(frames))
    fig, ax = plt.subplots(figsize=(12, 4))
    for name in ("yolo", "linear", "kalman", "ransac"):
        errors = frame_errors(runs[name].estimates, visible)
        ax.plot(positions, errors, color=METHOD_COLORS[name], linewidth=2,
                marker="o", markersize=3, label=METHOD_LABELS[name])
    for edge in breaks:
        ax.axvline(edge - 0.5, color=GRID, linewidth=1)
    ax.set_xlabel("正解のあるフレーム（縦線は投球の区切り。途切れは位置を出力していないフレーム）")
    ax.set_ylabel("中心位置誤差 (px)")
    ax.set_title("フレームごとの中心位置誤差", loc="left", fontsize=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    return _save(fig, out_dir / "error_timeline.png")


def _plot_status_strip(out_dir, runs, truths) -> Path:
    visible = _visible_sorted(truths)
    order = (DETECTED, INTERPOLATED, PREDICTED, MISSING)
    names = [n for n in ("yolo", "linear", "kalman", "ransac")]
    grid = np.full((len(names), max(len(visible), 1)), order.index(MISSING))
    for row, name in enumerate(names):
        by_frame = {e.frame: e.status for e in runs[name].estimates}
        for col, truth in enumerate(visible):
            grid[row, col] = order.index(by_frame.get(truth.frame, MISSING))
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(names) + 1.2))
    ax.imshow(grid, aspect="auto", interpolation="nearest", vmin=0, vmax=3,
              cmap=ListedColormap([STATUS_COLORS[s] for s in order]))
    ax.set_yticks(range(len(names)), [METHOD_LABELS[n] for n in names])
    ax.set_xlabel("蓋が見えているフレーム")
    ax.set_title("欠損フレームの補完状況", loc="left", fontsize=12)
    ax.legend(
        handles=[Patch(color=STATUS_COLORS[s], label=STATUS_LABELS[s]) for s in order],
        loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=4, frameon=False,
    )
    return _save(fig, out_dir / "status_strip.png")


def save_all_plots(out_dir: Path, runs: dict, truths, summary: dict) -> tuple:
    _style()
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    return (
        _plot_coverage(target, summary),
        _plot_error(target, summary),
        _plot_timeline(target, runs, truths),
        _plot_status_strip(target, runs, truths),
    )
