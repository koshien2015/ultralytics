"""評価区間の定義。しきい値の調整（tuning）と評価（eval）を混ぜないための仕組み。"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import yaml

SPLITS = ("tuning", "eval", "reference")


@dataclass(frozen=True)
class Clip:
    clip_id: str
    start: int  # 含む
    end: int  # 含まない
    split: str


def _parse_clip(raw: dict) -> Clip:
    if raw["split"] not in SPLITS:
        raise ValueError(f"split は {SPLITS} のいずれかです: {raw}")
    if int(raw["start"]) >= int(raw["end"]):
        raise ValueError(f"start は end より小さい値にしてください: {raw}")
    return Clip(str(raw["id"]), int(raw["start"]), int(raw["end"]), raw["split"])


def _check_disjoint(clips) -> None:
    for left, right in combinations(clips, 2):
        overlaps = left.start < right.end and right.start < left.end
        if overlaps and left.split != right.split:
            raise ValueError(f"調整用と評価用の区間が重なっています: {left.clip_id}, {right.clip_id}")


def load_clips(path: Path) -> tuple:
    with open(path, encoding="utf-8") as f:
        clips = tuple(_parse_clip(raw) for raw in yaml.safe_load(f)["clips"])
    _check_disjoint(clips)
    return clips


def select_truths(truths, clips, split: str) -> tuple:
    ranges = [(c.start, c.end) for c in clips if c.split == split]
    return tuple(g for g in truths if any(start <= g.frame < end for start, end in ranges))


def unlabelled_frames(truths, clips, split: str) -> list:
    """区間内で正解の無いフレーム。検出抜けのフレームが正解から消えていないかを確かめる"""
    labelled = {g.frame for g in truths}
    wanted = {f for c in clips if c.split == split for f in range(c.start, c.end)}
    return sorted(wanted - labelled)
