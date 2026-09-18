"""Matplotlib によるグラフ生成。

軸名や表題は英語にしている。日本語フォントが無い環境で豆腐（□）になり、
かえって読めなくなるため。

ただし投球名と覚え書きは利用者が付けるもので日本語が入りうるので、
日本語フォントが見つかれば使う。見つからなければ一度だけ知らせて、
以降の警告は黙らせる（同じ警告が1文字ごとに出て他が埋もれるため）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  バックエンド指定後に import する

from pitching.analysis import PitchAnalysis  # noqa: E402
from pitching.comparison.alignment import time_normalize  # noqa: E402
from pitching.models import EventName  # noqa: E402

# 日本語が出せるフォントの候補。先に見つかったものを使う。
JAPANESE_FONTS = (
    "Hiragino Sans", "Hiragino Maru Gothic Pro", "Noto Sans CJK JP", "Noto Sans JP",
    "IPAexGothic", "IPAGothic", "Yu Gothic", "Meiryo", "MS Gothic", "Arial Unicode MS",
)

def use_japanese_font() -> str | None:
    """日本語を出せるフォントを選ぶ。無ければ None を返し、警告を黙らせる。"""
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in JAPANESE_FONTS:
        if name in available:
            matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
            return name

    import warnings

    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    return None


_JAPANESE_FONT = use_japanese_font()

# 縦線で示すイベントと色
EVENT_STYLE = {
    EventName.FOOT_CONTACT: ("foot contact", "#2b8a3e"),
    EventName.MAX_ELBOW_FLEXION: ("max flexion", "#5f3dc4"),
    EventName.EXTENSION_START: ("extension start", "#e8590c"),
    EventName.RELEASE: ("release", "#c92a2a"),
}

# 1球分のグラフ（系列キー → (ファイル名, タイトル, 縦軸ラベル)）
SINGLE_PITCH_CHARTS = {
    "elbow_angle_deg": ("elbow_angle.png", "Elbow angle", "degrees"),
    "elbow_extension_velocity_deg_per_sec": (
        "elbow_extension_velocity.png",
        "Elbow extension velocity",
        "deg/sec",
    ),
    "forearm_angle_deg": ("forearm_angle.png", "Forearm angle (+up, 0=toward batter)", "degrees"),
    "trunk_lean_deg": ("trunk_angle.png", "Trunk lean (+toward batter)", "degrees"),
    "shoulder_hip_separation_deg": (
        "shoulder_hip_separation.png",
        "Shoulder-hip separation (apparent 2D)",
        "degrees",
    ),
    "lead_knee_angle_deg": ("lead_knee_angle.png", "Lead knee angle", "degrees"),
}


def write_pitch_charts(analysis: PitchAnalysis, output_dir: str | Path) -> list[Path]:
    """1投球分のグラフ一式を書き出す。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for key, (filename, title, ylabel) in SINGLE_PITCH_CHARTS.items():
        if key not in analysis.series:
            continue
        written.append(plot_series(analysis, key, directory / filename, title, ylabel))

    written.append(
        plot_normalized(
            [analysis],
            "elbow_angle_deg",
            directory / "elbow_angle_normalized.png",
            "Elbow angle (foot contact 0% - release 100%)",
            "degrees",
        )
    )
    return written


def write_comparison_charts(
    first: PitchAnalysis, second: PitchAnalysis, output_dir: str | Path
) -> list[Path]:
    """2投球を重ねた比較グラフを書き出す。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for key, (filename, title, ylabel) in SINGLE_PITCH_CHARTS.items():
        if key not in first.series or key not in second.series:
            continue
        written.append(
            plot_normalized(
                [first, second],
                key,
                directory / f"compare_{filename}",
                f"{title} (time normalized)",
                ylabel,
            )
        )
    return written


def plot_series(
    analysis: PitchAnalysis, key: str, path: Path, title: str, ylabel: str
) -> Path:
    """フレーム軸の系列グラフ。イベントを縦線で示す。"""
    figure, axes = plt.subplots(figsize=(10, 4.5))
    axes.plot(analysis.frame_indices, analysis.series[key], color="#1c7ed6", label="smoothed")

    raw_key = f"{key.removesuffix('_deg')}_raw_deg"
    if raw_key in analysis.series:
        axes.plot(
            analysis.frame_indices,
            analysis.series[raw_key],
            color="#adb5bd",
            linewidth=0.8,
            label="raw",
        )

    _draw_events(axes, analysis)
    axes.set_title(f"{title} - {analysis.pitch_id}")
    axes.set_xlabel("frame")
    axes.set_ylabel(ylabel)
    axes.grid(alpha=0.3)
    axes.legend(loc="best", fontsize="small")
    return _save(figure, path)


def plot_normalized(
    analyses: list[PitchAnalysis], key: str, path: Path, title: str, ylabel: str
) -> Path:
    """足接地〜リリースを0〜100%に正規化して重ねる。"""
    figure, axes = plt.subplots(figsize=(10, 4.5))
    colors = ("#1c7ed6", "#c92a2a", "#2b8a3e", "#e8590c")

    for order, analysis in enumerate(analyses):
        normalized = time_normalize(analysis, key)
        label = analysis.config.display_name
        if not normalized.covered:
            label = f"{label} [not aligned]"
        axes.plot(
            normalized.progress_percent,
            normalized.values,
            color=colors[order % len(colors)],
            label=label,
        )

    axes.axvline(0.0, color="#2b8a3e", linestyle="--", linewidth=1)
    axes.axvline(100.0, color="#c92a2a", linestyle="--", linewidth=1)
    axes.set_title(title)
    axes.set_xlabel("progress (%) : 0 = foot contact, 100 = release")
    axes.set_ylabel(ylabel)
    axes.grid(alpha=0.3)
    axes.legend(loc="best", fontsize="small")
    return _save(figure, path)


def _draw_events(axes, analysis: PitchAnalysis) -> None:
    for name, (label, color) in EVENT_STYLE.items():
        event = analysis.events.get(name)
        if event is None or event.frame_index is None:
            continue
        axes.axvline(event.frame_index, color=color, linestyle="--", linewidth=1)
        axes.annotate(
            f"{label} ({event.source.value})",
            xy=(event.frame_index, axes.get_ylim()[1]),
            xytext=(2, -10),
            textcoords="offset points",
            rotation=90,
            fontsize="x-small",
            color=color,
            va="top",
        )


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path
