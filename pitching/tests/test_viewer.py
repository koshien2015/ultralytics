"""棒人間ビューアのテスト。

HTML の見た目は検証できないが、埋め込むデータの座標系・そろえ方・
ファイルが1枚で完結していることは検証できる。
"""

from __future__ import annotations

import json

import pytest

from pitching.analysis import analyze_pitch
from pitching.models import EventName
from pitching.tests.support import make_series
from pitching.visualization.viewer import (
    NORMALIZED_SAMPLES,
    ViewerError,
    build_payload,
    render_html,
    write_viewer,
)


def test_origin_is_the_hip_center_at_foot_contact(pitch_series, pitch_config):
    """基準イベントの股関節中点が原点になる。"""
    analysis = analyze_pitch(pitch_series, pitch_config)

    payload = build_payload([analysis])
    frames = payload["pitches"][0]["frames"]
    contact_frame = next(
        frame for frame in frames if frame["f"] == pitch_config.events.stride_foot_contact_frame
    )
    left_hip = contact_frame["k"]["left_hip"]
    right_hip = contact_frame["k"]["right_hip"]

    assert (left_hip[0] + right_hip[0]) / 2 == pytest.approx(0.0, abs=1e-6)
    assert (left_hip[1] + right_hip[1]) / 2 == pytest.approx(0.0, abs=1e-6)


def test_x_points_toward_the_batter_and_y_points_up(pitch_series, pitch_config):
    """打者方向が +X、上が +Y（画像座標のYは下向きなので反転している）。"""
    config = pitch_config.model_copy(deep=True)
    config.batter_direction = "left"
    analysis = analyze_pitch(pitch_series, config)

    frame = build_payload([analysis])["pitches"][0]["frames"][0]

    # support.py の合成データでは、鼻は股関節より上、左肩は右肩より画像の左＝打者側
    assert frame["k"]["nose"][1] > 0
    assert frame["k"]["left_shoulder"][0] > frame["k"]["right_shoulder"][0]


def test_flipping_batter_direction_flips_x(pitch_series, pitch_config):
    left = analyze_pitch(pitch_series, pitch_config)
    flipped = pitch_config.model_copy(deep=True)
    flipped.batter_direction = "right"
    right = analyze_pitch(pitch_series, flipped)

    payload = build_payload([left, right])
    left_nose = payload["pitches"][0]["frames"][0]["k"]["nose"]
    right_nose = payload["pitches"][1]["frames"][0]["k"]["nose"]

    assert left_nose[0] == pytest.approx(-right_nose[0])
    assert left_nose[1] == pytest.approx(right_nose[1])


def test_normalized_track_spans_zero_to_hundred(pitch_series, pitch_config):
    analysis = analyze_pitch(pitch_series, pitch_config)

    normalized = build_payload([analysis])["pitches"][0]["normalized"]

    assert len(normalized) == NORMALIZED_SAMPLES
    assert normalized[0]["p"] == pytest.approx(0.0)
    assert normalized[-1]["p"] == pytest.approx(100.0)
    assert "right_wrist" in normalized[50]["k"]


def test_normalized_track_is_absent_without_release(pitch_series, pitch_config):
    config = pitch_config.model_copy(deep=True)
    config.events.release_frame = None

    analysis = analyze_pitch(pitch_series, config)

    assert build_payload([analysis])["pitches"][0]["normalized"] is None


def test_two_pitches_share_one_payload(pitch_series, pitch_config, pitch_angles):
    bad_config = pitch_config.model_copy(deep=True)
    bad_config.pitch_id = "synthetic_bad"
    bad_config.events.release_frame = 13

    payload = build_payload(
        [analyze_pitch(pitch_series, pitch_config), analyze_pitch(make_series(pitch_angles), bad_config)]
    )

    assert [pitch["pitch_id"] for pitch in payload["pitches"]] == ["synthetic_good", "synthetic_bad"]
    assert payload["pitches"][0]["events"]["release"]["frame"] == 16
    assert payload["pitches"][1]["events"]["release"]["frame"] == 13


def test_more_than_two_pitches_is_rejected(pitch_series, pitch_config):
    analysis = analyze_pitch(pitch_series, pitch_config)

    with pytest.raises(ViewerError):
        build_payload([analysis, analysis, analysis])


def test_events_are_carried_with_their_source(pitch_series, pitch_config):
    analysis = analyze_pitch(pitch_series, pitch_config)

    events = build_payload([analysis])["pitches"][0]["events"]

    assert events["release"]["source"] == "manual"
    assert events[EventName.MAX_ELBOW_FLEXION.value]["source"] == "pose_heuristic"


def test_html_is_self_contained(tmp_path, pitch_series, pitch_config):
    """file:// で開くので、外部からの読み込みが無いこと。"""
    analysis = analyze_pitch(pitch_series, pitch_config)

    path = write_viewer([analysis], tmp_path / "viewer.html")
    html = path.read_text(encoding="utf-8")

    assert "const DATA = {" in html
    assert "fetch(" not in html
    assert "<script src" not in html
    assert "synthetic_good" in html


def test_html_escapes_the_title(pitch_series, pitch_config):
    analysis = analyze_pitch(pitch_series, pitch_config)

    html = render_html([analysis], title="<script>alert(1)</script>")

    assert "<title>&lt;script&gt;" in html


def test_embedded_json_is_valid(pitch_series, pitch_config):
    analysis = analyze_pitch(pitch_series, pitch_config)

    html = render_html([analysis])
    payload = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]

    assert json.loads(payload)["pitches"][0]["pitch_id"] == "synthetic_good"


def test_vectors_use_their_own_colors():
    """ベクトルは骨格と別の色で描く（線か矢印か見分けるため）。"""
    from pitching.visualization.viewer_template import TEMPLATE

    assert "const VECTOR_COLORS = ['#6EE7A0', '#C77DFF'];" in TEMPLATE
    assert "drawVectors(context, project, pitch, cursor, VECTOR_COLORS[index]);" in TEMPLATE


def test_skeleton_and_vector_colors_do_not_overlap():
    from pitching.visualization.viewer_template import TEMPLATE
    import re

    def palette(name: str) -> set[str]:
        line = re.search(rf"const {name} = \[(.*?)\];", TEMPLATE).group(1)
        return set(re.findall(r"#[0-9A-Fa-f]{6}", line))

    assert palette("COLORS") & palette("VECTOR_COLORS") == set()
