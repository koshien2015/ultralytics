"""動画出力（解析動画・比較画像）のテスト。

実素材が無いので、無地のフレームだけの動画をその場で作って通す。
描画内容の見た目までは検証せず、落ちずにファイルが出ることを確かめる。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pitching.analysis import analyze_pitch  # noqa: E402
from pitching.visualization.comparison_frames import write_comparison_frames  # noqa: E402
from pitching.visualization.video_overlay import (  # noqa: E402
    OverlayError,
    render_overlay_video,
)


def make_video(path: Path, frames: int = 25, size: tuple[int, int] = (320, 320)) -> Path:
    """無地のフレームだけの動画を作る。"""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 60.0, size)
    assert writer.isOpened()
    for index in range(frames):
        image = np.full((size[1], size[0], 3), index * 5 % 255, dtype=np.uint8)
        writer.write(image)
    writer.release()
    return path


def test_overlay_video_is_written(tmp_path, pitch_series, pitch_config):
    video = make_video(tmp_path / "source.mp4")
    analysis = analyze_pitch(pitch_series, pitch_config)

    output = render_overlay_video(analysis, video, tmp_path / "overlay.mp4")

    assert output.is_file()
    assert output.stat().st_size > 0


def test_overlay_video_reports_missing_source(tmp_path, pitch_series, pitch_config):
    analysis = analyze_pitch(pitch_series, pitch_config)

    with pytest.raises(OverlayError):
        render_overlay_video(analysis, tmp_path / "missing.mp4", tmp_path / "overlay.mp4")


def test_comparison_frames_are_written(tmp_path, pitch_series, pitch_config):
    video = make_video(tmp_path / "source.mp4")

    good_config = pitch_config.model_copy(deep=True)
    good_config.video_path = str(video)
    bad_config = good_config.model_copy(deep=True)
    bad_config.pitch_id = "synthetic_bad"
    bad_config.result.label = "bad"
    bad_config.events.release_frame = 13

    good = analyze_pitch(pitch_series, good_config)
    bad = analyze_pitch(pitch_series, bad_config)

    written = write_comparison_frames(good, bad, tmp_path / "frames")

    # 足接地・最大屈曲・リリースの3枚
    assert len(written) == 3
    assert all(path.is_file() for path in written)


def test_comparison_frames_skip_when_video_is_missing(tmp_path, pitch_series, pitch_config):
    analysis = analyze_pitch(pitch_series, pitch_config)

    written = write_comparison_frames(analysis, analysis, tmp_path / "frames")

    assert written == []
