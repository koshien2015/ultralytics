"""手法ごとの結果の保存と、評価のまとめ。"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .metrics import MetricsConfig, detector_recall, evaluate
from .schema import Estimate

TABLE_COLUMNS = (
    ("coverage", "カバー率"),
    ("correct_coverage", "正しいカバー率"),
    ("mae_output", "MAE(出力のみ)"),
    ("mae_all", "MAE(全フレーム)"),
    ("p95_output", "P95(出力のみ)"),
    ("p95_all", "P95(全フレーム)"),
    ("max_missing_run", "最大連続欠損"),
    ("wrong_outputs", "誤対応"),
    ("phantom_outputs", "架空出力"),
    ("predicted_tail", "末尾の予測"),
    ("post_ms", "後処理ms(動画全体)"),
)


def save_estimates(path: Path, runs: dict) -> None:
    payload = {
        name: {
            "elapsed_ms": run.elapsed_ms,
            "estimates": [asdict(e) for e in run.estimates],
            "events": [asdict(e) for e in run.events],
        }
        for name, run in runs.items()
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _estimate_from_dict(raw: dict) -> Estimate:
    box = tuple(raw["box"]) if raw["box"] is not None else None
    return Estimate(**{**raw, "box": box})


def load_estimates(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        name: tuple(_estimate_from_dict(raw) for raw in body["estimates"])
        for name, body in payload.items()
    }


def summarize(records, runs: dict, truths, config: ExperimentConfig) -> dict:
    """config は動画の幅に換算済みのものを渡す"""
    metrics_config = MetricsConfig(
        config.evaluation.match_radius_px, config.evaluation.error_cap_px
    )
    timings = [r.infer_ms for r in records if r.infer_ms is not None]
    frames = {g.frame for g in truths}

    def count(run, kind):
        return sum(e.kind == kind and e.frame in frames for e in run.events)

    return {
        "n_truth": len(truths),
        "detector_recall": detector_recall(records, truths, metrics_config),
        "mean_infer_ms": float(np.mean(timings)) if timings else None,
        "methods": {
            name: {
                **asdict(evaluate(run.estimates, truths, metrics_config)),
                "post_ms": run.elapsed_ms,
                "gate_rejected": count(run, "gate_rejected"),
                "track_lost": count(run, "track_lost"),
            }
            for name, run in runs.items()
        },
    }


def _cell(value) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def markdown_table(summary: dict) -> str:
    header = "| 手法 | " + " | ".join(label for _key, label in TABLE_COLUMNS) + " |"
    rule = "|---" * (len(TABLE_COLUMNS) + 1) + "|"
    rows = [
        f"| {name} | " + " | ".join(_cell(values[key]) for key, _label in TABLE_COLUMNS) + " |"
        for name, values in summary["methods"].items()
    ]
    return "\n".join([header, rule, *rows])
