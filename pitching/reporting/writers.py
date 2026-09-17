"""解析結果をファイルに書き出す。

数値は JSON と CSV の両方で出す。CSV は表計算で開いて目視確認するため。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from pitching.adapters.json_adapter import save_pose_csv, save_pose_json
from pitching.analysis import PitchAnalysis
from pitching.reporting.summary import build_summary

# angles.csv に出す系列（キー → 列名）
_SERIES_COLUMNS = (
    "elbow_angle_deg",
    "elbow_angle_raw_deg",
    "elbow_extension_velocity_deg_per_sec",
    "forearm_angle_deg",
    "upper_arm_angle_deg",
    "trunk_lean_deg",
    "shoulder_line_deg",
    "hip_line_deg",
    "shoulder_hip_separation_deg",
    "lead_knee_angle_deg",
)


def write_analysis(analysis: PitchAnalysis, output_dir: str | Path) -> dict[str, Path]:
    """解析結果一式を output_dir に書き出し、作ったファイルを返す。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    summary = build_summary(analysis)
    written = {
        "metrics_json": _write_json(directory / "metrics.json", summary),
        "summary_csv": _write_summary_csv(directory / "summary.csv", summary),
        "events_csv": _write_events_csv(directory / "events.csv", summary),
        "angles_csv": _write_angles_csv(directory / "angles.csv", analysis),
        "pose_raw_json": save_pose_json(analysis.raw_series, directory / "pose_raw.json"),
        "pose_raw_csv": save_pose_csv(analysis.raw_series, directory / "pose_raw.csv"),
        "pose_smoothed_json": save_pose_json(
            analysis.smoothed_series, directory / "pose_smoothed.json"
        ),
        "pose_smoothed_csv": save_pose_csv(
            analysis.smoothed_series, directory / "pose_smoothed.csv"
        ),
    }
    return written


def write_comparison(report: dict, output_dir: str | Path) -> dict[str, Path]:
    """比較結果を書き出す。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    comparison_csv = directory / "comparison.csv"

    with comparison_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item", "label", "pitch_a", "pitch_b", "difference"])
        for name, observation in report["observations"].items():
            writer.writerow(
                [
                    name,
                    observation["label"],
                    _csv_value(observation["pitch_a"]),
                    _csv_value(observation["pitch_b"]),
                    _csv_value(observation["difference"]),
                ]
            )

    return {
        "comparison_json": _write_json(directory / "comparison.json", report),
        "comparison_csv": comparison_csv,
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_summary_csv(path: Path, summary: dict) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "key", "value"])
        for group, values in summary["features"].items():
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                writer.writerow([group, key, _csv_value(value)])
    return path


def _write_events_csv(path: Path, summary: dict) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event", "frame", "source", "progress_percent", "note"])
        for name, event in summary["events"].items():
            writer.writerow(
                [
                    name,
                    _csv_value(event["frame"]),
                    event["source"],
                    _csv_value(event["progress_percent"]),
                    event["note"],
                ]
            )
    return path


def _write_angles_csv(path: Path, analysis: PitchAnalysis) -> Path:
    columns = [name for name in _SERIES_COLUMNS if name in analysis.series]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_sec", "progress_percent", *columns])
        for position in range(analysis.frame_indices.size):
            writer.writerow(
                [
                    int(analysis.frame_indices[position]),
                    f"{analysis.timestamps[position]:.6f}",
                    _csv_value(analysis.progress_percent[position]),
                    *[_csv_value(analysis.series[name][position]) for name in columns],
                ]
            )
    return path


def _csv_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(float(value)) else f"{float(value):.4f}"
    return str(value)
