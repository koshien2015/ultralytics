"""姿勢推定の結果を、投球フォーム解析（pitching/）が読める JSON に書き出す。

track.py の `POSE_EXPORT` から使う。推論そのものはしない。渡された
`PoseEstimator.update` の結果から対象の人物を1人選んで溜めるだけ。

書き出す先は {動画名}_pose.json。手元に持ち帰って

    python -m pitching run --output out/ --pose 動画名_pose.json --release 152

とすれば、GPU 無しで解析・比較・ビューアまで作れる。

推論を掛けなかったフレームは記録しない。記録してしまうと、解析側から見て
「推論していない」のか「信頼度が低い」のかが区別できなくなり、短い欠損だけを
補間する前処理が誤動作する。連続していない区間はフレーム番号の飛びとして残る。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# pitching/adapters/json_adapter.py と合わせること
SCHEMA_VERSION = 1


def gate_windows(pose_windows, export_enabled: bool):
    """姿勢推定の推論を掛ける区間を決める。

    prefilter は「長い動画から投球区間を探す」前提で、活動量の中央値を基準に
    閾値を作る。全編が投球動作の切り出し済みクリップでは中央値そのものが高く、
    窓が1つも立たないことがある。その状態で書き出すと0フレームになり、
    「実行したのにファイルが無い」になる。

    書き出すつもりで実行しているときに限り、窓が空なら全フレームを対象にする
    （None は InferenceGate にとって「区間指定なし」の意味）。

    Returns:
        窓のタプル、または None（全フレーム対象）
    """
    if export_enabled and not pose_windows:
        return None
    return pose_windows


class PoseRecorder:
    """1本の動画分のキーポイントを溜めて JSON にする。"""

    def __init__(
        self,
        keypoint_names,
        fps: float,
        video_path: str,
        role: str = "pitcher",
        min_score: float = 0.3,
        pitch_id: str = "",
    ) -> None:
        self._names = tuple(keypoint_names)
        self._fps = float(fps) if fps and fps > 0 else 60.0
        self._video_path = str(video_path)
        self._role = role
        self._min_score = min_score
        self._pitch_id = pitch_id or Path(video_path).stem
        self._frames: list[dict] = []
        self._modes: set[str] = set()

    def __len__(self) -> int:
        return len(self._frames)

    def __bool__(self) -> bool:
        """記録が0件でも「recorder はある」と扱う。

        __len__ だけだと空の recorder が falsy になり、`if recorder:` が
        一度も通らないまま1フレームも記録されない（実際にそれで嵌った）。
        """
        return True

    @property
    def selection_modes(self) -> tuple[str, ...]:
        return tuple(sorted(self._modes))

    def record(self, frame_index: int, update_result: dict) -> str:
        """1フレーム分を溜める。選び方（role_bbox / largest_bbox / none）を返す。"""
        import pose  # 遅延 import。この台本単体でも読み込めるように。

        person, mode = pose.select_person(update_result, self._role, self._min_score)
        self._modes.add(mode)
        self._frames.append(self._frame_payload(person, frame_index))
        return mode

    def save(self, path: str | Path) -> Path:
        """溜めたものを書き出す。1フレームも無ければ書かない。"""
        output = Path(path)
        if not self._frames:
            raise ValueError("記録されたフレームがありません")

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self._payload(), ensure_ascii=False), encoding="utf-8"
        )
        return output

    def _payload(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "meta": {
                "pitch_id": self._pitch_id,
                "fps": self._fps,
                "video_path": self._video_path,
                "adapter": "ultralytics(track.py)",
                "notes": self._notes(),
            },
            "frames": self._frames,
        }

    def _notes(self) -> list[str]:
        notes = [f"person_selection={list(self.selection_modes)}"]
        if "role_bbox" not in self._modes:
            notes.append(
                "役割bboxで投手を特定できていない（最大の骨格を投手とみなした）。"
                "別人を拾っていないか確認すること。"
            )
        ranges = self.frame_ranges()
        notes.append(
            "推論した区間: " + ", ".join(f"{start}-{end}" for start, end in ranges)
        )
        if len(ranges) > 1:
            notes.append(
                "区間が分かれている。解析では1投球ぶんの区間を "
                "--start / --end で切り出すこと。"
            )
        return notes

    def frame_ranges(self) -> list[tuple[int, int]]:
        """連続して記録できたフレーム番号の区間。"""
        ranges: list[tuple[int, int]] = []
        for frame in self._frames:
            index = frame["frame_index"]
            if ranges and index == ranges[-1][1] + 1:
                ranges[-1] = (ranges[-1][0], index)
            else:
                ranges.append((index, index))
        return ranges

    def _frame_payload(self, person, frame_index: int) -> dict:
        keypoints = {}
        for order, name in enumerate(self._names):
            if person is None:
                keypoints[name] = [None, None, 0.0]
                continue
            x, y = person.keypoints[order]
            keypoints[name] = [
                _number(x), _number(y), round(float(person.scores[order]), 4)
            ]
        return {
            "frame_index": int(frame_index),
            "timestamp_sec": round(frame_index / self._fps, 6),
            "keypoints": keypoints,
        }


def _number(value) -> float | None:
    """欠損は null で書く。座標は3桁もあれば十分。"""
    number = float(value)
    return None if not math.isfinite(number) else round(number, 3)
