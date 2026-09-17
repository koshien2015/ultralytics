"""2投球の比較。

出すのは「観測された差」だけ。良否の判定も、原因の断定もしない。
差分と併せて、それぞれの実測値も残す。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

COMPARISON_DISCLAIMER = [
    "1球対1球の比較であり、統計的な結論ではない。",
    "差は観測値の差であって、球質の原因を示すものではない。",
    "どちらかのイベントが未確定の項目は None になる。",
]

# 比較する項目 → (要約内の位置, 表示名)
_FIELD_SOURCES: dict[str, tuple[tuple[str, str], str]] = {
    "elbow_angle_at_release_diff": (("elbow", "elbow_angle_at_release_deg"), "リリース時肘角度(度)"),
    "minimum_elbow_angle_diff": (("elbow", "minimum_elbow_angle_deg"), "最小肘角度(度)"),
    "elbow_extension_range_diff": (("elbow", "elbow_extension_range_deg"), "肘伸展量(度)"),
    "flexion_to_release_time_diff_sec": (
        ("elbow", "flexion_to_release_time_sec"),
        "最大屈曲→リリース時間(秒)",
    ),
    "peak_extension_velocity_diff": (
        ("elbow", "peak_extension_velocity_deg_per_sec"),
        "最大肘伸展角速度(度/秒)",
    ),
    "forearm_angle_at_release_diff": (
        ("forearm", "forearm_angle_at_release_deg"),
        "リリース時前腕角度(度)",
    ),
    "trunk_angle_at_release_diff": (("torso", "trunk_lean_at_release_deg"), "リリース時体幹傾き(度)"),
    "lead_knee_angle_at_release_diff": (
        ("lower_body", "lead_knee_angle_at_release_deg"),
        "リリース時前脚膝角度(度)",
    ),
    "normalized_release_x_diff": (
        ("release_position", "relative_to_shoulder_mid_x"),
        "正規化リリース位置X(肩中点基準)",
    ),
    "normalized_release_y_diff": (
        ("release_position", "relative_to_shoulder_mid_y"),
        "正規化リリース位置Y(肩中点基準)",
    ),
}


@dataclass(frozen=True)
class PitchComparison:
    """A − B の差。どちらかが欠測なら None。"""

    elbow_angle_at_release_diff: float | None
    minimum_elbow_angle_diff: float | None
    elbow_extension_range_diff: float | None
    flexion_to_release_time_diff_sec: float | None
    peak_extension_velocity_diff: float | None
    forearm_angle_at_release_diff: float | None
    trunk_angle_at_release_diff: float | None
    lead_knee_angle_at_release_diff: float | None
    normalized_release_x_diff: float | None
    normalized_release_y_diff: float | None


def compare_summaries(summary_a: dict, summary_b: dict) -> dict:
    """要約2件を比較し、差分・実測値・注意書きをまとめて返す。"""
    differences: dict[str, float | None] = {}
    observations: dict[str, dict] = {}

    for field_name, ((group, key), label) in _FIELD_SOURCES.items():
        value_a = _feature(summary_a, group, key)
        value_b = _feature(summary_b, group, key)
        difference = None if value_a is None or value_b is None else value_a - value_b
        differences[field_name] = difference
        observations[field_name] = {
            "label": label,
            "pitch_a": value_a,
            "pitch_b": value_b,
            "difference": difference,
        }

    comparison = PitchComparison(**differences)
    return {
        "pitch_a": _identity(summary_a),
        "pitch_b": _identity(summary_b),
        "difference": asdict(comparison),
        "observations": observations,
        "event_alignment": _event_alignment(summary_a, summary_b),
        "unavailable": [
            name for name, value in differences.items() if value is None
        ],
        "notes": COMPARISON_DISCLAIMER,
    }


def _identity(summary: dict) -> dict:
    return {
        "pitch_id": summary.get("pitch_id"),
        "label": summary.get("result", {}).get("label"),
        "description": summary.get("result", {}).get("description"),
    }


def _event_alignment(summary_a: dict, summary_b: dict) -> dict:
    """比較の前提として、どのイベントで揃えられるかを示す。"""
    alignment = {}
    events_a = summary_a.get("events", {})
    events_b = summary_b.get("events", {})
    for name in ("foot_contact", "max_elbow_flexion", "release"):
        frame_a = events_a.get(name, {}).get("frame")
        frame_b = events_b.get(name, {}).get("frame")
        alignment[name] = {
            "pitch_a_frame": frame_a,
            "pitch_a_source": events_a.get(name, {}).get("source"),
            "pitch_b_frame": frame_b,
            "pitch_b_source": events_b.get(name, {}).get("source"),
            "usable_as_anchor": frame_a is not None and frame_b is not None,
        }
    return alignment


def _feature(summary: dict, group: str, key: str) -> float | None:
    value = summary.get("features", {}).get(group, {}).get(key)
    return None if value is None else float(value)
