"""解析済みディレクトリ（analyze / extract の出力）を読み戻す。

metrics.json の設定と、隣のキーポイントJSONから PitchAnalysis を作り直す。
比較グラフやビューアで、元の設定ファイルを再指定させないため。
"""

from __future__ import annotations

import json
from pathlib import Path

from pitching.adapters.json_adapter import load_pose_json
from pitching.analysis import PitchAnalysis, analyze_pitch
from pitching.reporting.summary import config_from_summary

# 探す順。analyze が書いたものを優先し、無ければ extract が書いたものを使う。
POSE_FILENAMES = ("pose_raw.json", "pose.json")


def load_summary(path: str | Path) -> dict:
    """metrics.json を読む。"""
    summary_path = Path(path)
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"metrics.json が見つかりません: {summary_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"metrics.json が不正です: {summary_path}: {error}") from error


def find_pose_file(directory: str | Path) -> Path | None:
    for filename in POSE_FILENAMES:
        candidate = Path(directory) / filename
        if candidate.is_file():
            return candidate
    return None


def load_analysis(metrics_path: str | Path) -> PitchAnalysis | None:
    """metrics.json とその隣のキーポイントJSONから解析をやり直す。

    キーポイントJSONが無ければ None（数値の比較だけなら metrics.json で足りる）。
    """
    metrics = Path(metrics_path)
    pose_path = find_pose_file(metrics.parent)
    if pose_path is None:
        return None
    return analyze_pitch(load_pose_json(pose_path), config_from_summary(load_summary(metrics)))


def resolve_metrics_path(target: str | Path) -> Path:
    """ディレクトリを渡されたら、その中の metrics.json を指す。"""
    path = Path(target)
    return path / "metrics.json" if path.is_dir() else path
