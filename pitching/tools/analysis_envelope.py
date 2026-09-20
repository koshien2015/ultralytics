#!/usr/bin/env python3
"""解析結果の外側メタデータ（封筒）。**両リポジトリで同じ内容を持つ。**

解析器（姿勢・キャップ軌道）が出す JSON に「この結果はどのクリップの、
どの時間軸・座標系のものか」を持たせる。

**1 球 1 フォルダで自己完結させる方針（設計文書 §13）なので、封筒はセッションを知らない。**
アプリ外で撮った動画のように親セッションが無いクリップも、同じ形で扱える。
クリップが唯一の基準で、元動画のフレーム番号へ戻ることはない。

封筒に残る仕事は 2 つ:

1. **何から計算したかの記録** —— どのモデル・どの設定・いつ
2. **GPU の別 PC から戻ってきたファイルが、このフォルダのクリップに対して
   計算されたものかの照合** —— 投球ごとにクリップが違うので指紋で区別できる

**時刻の正本は `stts` の目盛表であって平均 fps ではない。**
実データ相当の値で、平均 fps で割ると最大 96ms（240fps の 23 枚分）ずれる。
骨格が 23 枚ずれて重なれば、それらしく見えたまま間違っている。

方針:
- 既存 JSON を巨大な単一スキーマへ移行させない。`data` に既存の中身をそのまま入れる
- 迷ったら結び付けない
- 不明な値を 0 や既定値で埋めない
- 壊れた 1 ファイルでビューア全体が作れなくならない
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# 封筒の版。読む側が理解できる上限でもある。
SCHEMA_VERSION = "2.0"

# フレーム番号の基準。**クリップ基準しかない。**
# 使われない枝を「念のため」残さない——呼び出しの無い分岐は、
# あとで間違った方が選ばれる経路になる。
FRAME_ORIGIN = "clip"


class EnvelopeProblem(Exception):
    """封筒そのものを読めない／使えない。呼ぶ側はこれを捕まえて他を続ける。"""


@dataclass(frozen=True)
class Problem:
    """結び付けを妨げる事実 1 件。

    fatal=True なら重ねない。False なら重ねるが、人間に見えるところへ出す。
    """

    reason: str
    fatal: bool


def build_envelope(*, analysis_id: str, analysis_type: str, clip_path: str | Path,
                   pitch_id: str, producer: dict, data: dict,
                   rotation_applied: bool = True) -> dict:
    """封筒を組み立てる。解析器側（GPU のある別 PC）から呼ぶ。

    クリップの事実は全て `probe_video()` から採る。`metadata.json` は見ない
    ——親セッションが無いこともあるし、自己申告と実体はずれる
    （実データで durationMs が 5ms、fps が 8 桁目で食い違っていた）。
    """
    from video_fingerprint import fingerprint_file
    from video_probe import probe_video, spec_for_fingerprint

    clip = Path(clip_path)
    info = probe_video(clip)

    return {
        "schemaVersion": SCHEMA_VERSION,
        "analysisId": analysis_id,
        "analysisType": analysis_type,
        "pitchId": pitch_id,
        "clip": {
            # basename だけ。解析は別 PC で走るので、あちらの絶対パスは意味を持たない。
            "file": clip.name,
            "fingerprint": fingerprint_file(clip, **spec_for_fingerprint(clip)),
            "frameCount": info["frame_count"],
            "durationMs": info["duration_ms"],
        },
        "timebase": {
            "unit": "frame",
            "frameOrigin": FRAME_ORIGIN,
            "timescale": info["timescale"],
            # **これが時刻の正本。** ランレングスのまま持つ。
            "tickTable": info["tick_table"],
            # 表示と要約のためだけの値。時刻の計算には使わない。
            "averageFps": info["fps"],
        },
        "coordinateSystem": {
            "type": "image_pixel",
            "origin": "top_left",
            "xDirection": "right",
            "yDirection": "down",
            "width": info["display_width"],
            "height": info["display_height"],
            "rotationDeg": info["rotation_deg"],
            # OpenCV 4.5 以降は既定でデコード時に回転を当てる。
            # 推測に依存させず、書いた側が宣言する。
            "rotationApplied": rotation_applied,
        },
        "producer": dict(producer),
        "data": data,
    }


def load_envelope(path: str | Path) -> dict:
    """封筒 JSON を読む。読めない理由は EnvelopeProblem で返す。

    呼ぶ側はこれを捕まえて、そのレイヤーだけ落として他を続ける。
    1 ファイルの破損でビューア全体が作れなくなるのは、指示書が禁じている。
    """
    target = Path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise EnvelopeProblem(f"解析ファイルが無い: {target}") from error
    except json.JSONDecodeError as error:
        raise EnvelopeProblem(f"解析ファイルが壊れている: {target}: {error}") from error
    except OSError as error:
        raise EnvelopeProblem(f"解析ファイルを読めない: {target}: {error}") from error

    if not isinstance(payload, dict):
        raise EnvelopeProblem(f"解析ファイルの中身がオブジェクトではない: {target}")
    return payload


def verify_against_clip(envelope: dict, clip_info: dict) -> list[Problem]:
    """この解析結果をこのフォルダのクリップへ重ねてよいかを調べる。

    `clip_info` は `probe_video(フォルダの clip.mov)` の結果。

    戻り値が空なら重ねてよい。fatal が 1 つでもあれば重ねない。
    fatal でない問題は重ねるが、必ず画面へ出す（黙って通さない）。
    """
    problems: list[Problem] = []

    if "schemaVersion" not in envelope:
        # 旧形式（pose JSON の schema_version: 1）。読めるが、何に対して
        # 計算されたものか言えない。「たぶん同じ」で重ねない。
        return [Problem(
            "封筒（schemaVersion）が無い古い形式のため、どのクリップの結果か確かめられない。"
            "解析をやり直すと結び付けられる。", fatal=True)]

    major = str(envelope["schemaVersion"]).split(".")[0]
    if major != SCHEMA_VERSION.split(".")[0]:
        return [Problem(
            f"封筒の版 {envelope['schemaVersion']} をこのビューアは読めない"
            f"（対応: {SCHEMA_VERSION}）。", fatal=True)]

    problems.extend(_clip_problems(envelope, clip_info))
    problems.extend(_timebase_problems(envelope, clip_info))
    problems.extend(_coordinate_problems(envelope, clip_info))
    return problems


def _clip_problems(envelope: dict, clip_info: dict) -> list[Problem]:
    """別 PC から戻ってきたファイルが、このフォルダのものか。"""
    from video_fingerprint import fingerprint_file
    from video_probe import spec_for_fingerprint

    declared = (envelope.get("clip") or {}).get("fingerprint")
    actual = clip_info.get("fingerprint")
    if actual is None:
        path = clip_info.get("path")
        if path is None:
            # 照合できないことを、照合できたように見せない。
            return [Problem("クリップの指紋を確かめられなかった（照合していない）。", fatal=False)]
        actual = fingerprint_file(path, **spec_for_fingerprint(path))

    if declared != actual:
        return [Problem(
            "クリップの指紋が一致しない。別の投球のフォルダへ置かれた解析結果か、"
            "クリップを作り直したあとの古い解析結果の可能性がある。重ねない。", fatal=True)]
    return []


def _timebase_problems(envelope: dict, clip_info: dict) -> list[Problem]:
    timebase = envelope.get("timebase") or {}
    problems: list[Problem] = []

    if timebase.get("frameOrigin") != FRAME_ORIGIN:
        problems.append(Problem(
            f"フレーム番号の基準が {timebase.get('frameOrigin')!r} になっている"
            f"（このビューアが読めるのは {FRAME_ORIGIN!r} だけ）。重ねない。", fatal=True))

    declared = timebase.get("tickTable")
    actual = clip_info.get("tick_table")
    if declared is not None and actual is not None:
        if [list(row) for row in declared] != [list(row) for row in actual]:
            problems.append(Problem(
                "時間軸（目盛表）がクリップと一致しない。クリップを作り直したあとに"
                "解析をやり直していない可能性がある。重ねない。", fatal=True))
    return problems


def _coordinate_problems(envelope: dict, clip_info: dict) -> list[Problem]:
    coordinates = envelope.get("coordinateSystem") or {}
    expected = (clip_info.get("display_width"), clip_info.get("display_height"))
    if (coordinates.get("width"), coordinates.get("height")) != expected:
        return [Problem(
            f"座標系の解像度がクリップと違う（解析: {coordinates.get('width')}x"
            f"{coordinates.get('height')} / クリップ: {expected[0]}x{expected[1]}）。重ねない。",
            fatal=True)]
    return []


def source_time_ms(envelope: dict, frame: int) -> float:
    """クリップのフレーム番号を、クリップ内の時刻（ms）へ直す。

    **目盛表を積み上げる。`frame / fps` では出さない。**
    平均 fps は区間ごとの差を潰すので、実データ相当の値で最大 96ms
    （240fps の 23 枚分）ずれる。
    """
    from video_probe import ProbeError, frame_to_ms

    timebase = envelope.get("timebase") or {}
    table = timebase.get("tickTable")
    timescale = timebase.get("timescale")
    if not table or not timescale:
        raise EnvelopeProblem("封筒に目盛表が無いため時刻へ直せない")

    info = {
        "tick_table": table,
        "timescale": timescale,
        "frame_count": sum(count for count, _ in table),
    }
    try:
        return frame_to_ms(info, frame)
    except ProbeError as error:
        raise EnvelopeProblem(str(error)) from error
