"""YOLO Pose による投球動作の骨格推定。

track.py の検出モデル（`data.yaml` の11クラス）は人物を役割付きで検出する
（pitcher_motion / batter_stance / catcher / umpire ...）。このモジュールは
公式 Pose 重みで取った骨格を、その役割bboxと突き合わせて「誰の骨格か」を
決める。Pose 側は人物をN人返すだけで誰が誰かは教えてくれないため。

役割bboxは毎フレーム来ない。prefilter の `search_stride` で検出自体が
間引かれるうえ、検出は tennis.py の強調フレームに対して行われる
（PIPELINE.md「静止クラス（打者・捕手・審判）はコントラストが0.6倍に落ちる」）。
そのため役割は **一度決めたらトラッカーIDで引き継ぐ**。

推論は元フレームに対して行う。強調フレームはキャップを目立たせるための
加工で、人物の姿勢推定には不利なため（PIPELINE.md の「人物は生フレーム」）。

純粋ロジック（役割マッピング・突き合わせ・描画）は ultralytics に依存しない。
`PoseEstimator` だけが遅延 import で YOLO を読む。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import cv2
import numpy as np

from prefilter import PrefilterConfig

Box = tuple[float, float, float, float]


class PoseError(RuntimeError):
    """姿勢推定の処理に失敗したときに送出する。"""


# COCO 17 キーポイント（公式 Pose 重みの出力順）
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

SKELETON_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),        # 肩と腕
    (5, 11), (6, 12), (11, 12),                      # 胴
    (11, 13), (13, 15), (12, 14), (14, 16),          # 脚
    (0, 1), (0, 2), (1, 3), (2, 4),                  # 頭部
)

# 検出クラスID → 役割。pitching_analysis.py の分類と揃えてある。
_DETECTION_ROLES = {
    1: "pitcher",   # pitcher_motion
    5: "pitcher",   # pitcher_release
    2: "batter",    # batter_stance
    6: "batter",    # batter_swing
    3: "umpire",
    4: "catcher",
    7: "catcher",   # catcher_stance
    8: "catcher",   # catcher_catch
    9: "catcher",   # catcher_throw
    10: "catcher",  # catcher_miss
}

ROLE_COLORS = {
    "pitcher": (0, 200, 255),   # BGR: オレンジ寄り
    "batter": (255, 200, 0),    # BGR: 水色寄り
    "catcher": (120, 255, 120),
    "umpire": (160, 160, 160),
}


@dataclass(frozen=True)
class PersonPose:
    """1人分の姿勢。keypoints は (17, 2) のピクセル座標、scores は (17,)。"""

    track_id: int | None
    keypoints: np.ndarray
    scores: np.ndarray


def role_of_detection(class_id: int) -> str | None:
    """検出クラスIDを役割名に変換する。人物でなければ None。"""
    return _DETECTION_ROLES.get(int(class_id))


def overlap_coefficient(a: Box, b: Box) -> float:
    """交差面積 ÷ 小さい方の面積。

    IoU ではなく包含係数を使う。Pose のキーポイント外接矩形と検出モデルの
    人物bboxは取り方が違って面積が揃わないため、IoU だと同一人物でも
    値が伸びない。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    if smaller <= 0:
        return 0.0
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    return (inter_w * inter_h) / smaller


def keypoint_bbox(
    keypoints: np.ndarray, scores: np.ndarray, min_score: float = 0.5
) -> Box | None:
    """信頼できるキーポイントだけの外接矩形。1点も無ければ None。"""
    confident = np.asarray(scores) >= min_score
    if not confident.any():
        return None
    points = np.asarray(keypoints)[confident]
    return (
        float(points[:, 0].min()), float(points[:, 1].min()),
        float(points[:, 0].max()), float(points[:, 1].max()),
    )


def build_role_boxes(detection_results: Iterable) -> dict[str, Box]:
    """検出結果から役割ごとのbboxを1つずつ拾う。

    同じ役割が複数出た場合は最初のものを採る（pitching_analysis.py の
    `detect_batter_catcher_pitcher` と同じ方針）。
    """
    boxes: dict[str, Box] = {}
    for result in detection_results:
        for box in result.boxes:
            role = role_of_detection(box.cls[0])
            if role is None or role in boxes:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            boxes[role] = (x1, y1, x2, y2)
    return boxes


class RoleTracker:
    """トラッカーIDに役割を貼り付け、以降のフレームへ引き継ぐ。

    役割bboxが来たフレームで突き合わせ、決まったIDは bbox が無い
    フレームでもその役割のまま扱う。

    投球が切り替わるときは `reset()` する。投球窓の境界だけを頼りにすると
    漏れる: 投球間隔が短いと `detect_windows` が隣接する窓をマージし、
    窓の外になるフレームが生じないため、リリース検出も併用すること。
    """

    def __init__(self, min_overlap: float = 0.5, min_score: float = 0.5) -> None:
        self._min_overlap = min_overlap
        self._min_score = min_score
        self._roles: dict[int, str] = {}

    def reset(self) -> None:
        """割り当てを忘れる。投球窓の切り替わりで呼ぶ。"""
        self._roles = {}

    def assign(
        self, persons: Sequence[PersonPose], role_boxes: dict[str, Box]
    ) -> dict[int, str]:
        """このフレームの {track_id: 役割} を返す。

        新規の突き合わせは重なりの大きい順に確定させ、1つの役割が
        複数人に付かないようにする。
        """
        candidates = []
        for person in persons:
            if person.track_id is None:
                continue
            person_box = keypoint_bbox(person.keypoints, person.scores, self._min_score)
            if person_box is None:
                continue
            for role, role_box in role_boxes.items():
                score = overlap_coefficient(person_box, role_box)
                if score >= self._min_overlap:
                    candidates.append((score, int(person.track_id), role))

        taken_roles = set()
        taken_ids = set()
        for _score, track_id, role in sorted(candidates, key=lambda c: -c[0]):
            if role in taken_roles or track_id in taken_ids:
                continue
            self._roles = {**self._roles, track_id: role}
            taken_roles.add(role)
            taken_ids.add(track_id)

        return {
            int(p.track_id): self._roles[int(p.track_id)]
            for p in persons
            if p.track_id is not None and int(p.track_id) in self._roles
        }


def select_person(
    update_result: dict,
    role: str = "pitcher",
    min_score: float = 0.5,
) -> tuple[PersonPose | None, str]:
    """`PoseEstimator.update` の結果から、対象の役割の人物を1人選ぶ。

    役割が決まっていればそれを採る。決まっていない（検出結果が無い・
    役割bboxと重ならない）場合は、最も大きい骨格で代用する。代用は
    別人を拾いうるので、どちらで選んだかを第2要素で返す。

    Returns:
        (人物 or None, "role_bbox" | "largest_bbox" | "none")
    """
    persons = update_result.get("persons", [])
    roles = update_result.get("roles", {})

    for person in persons:
        if person.track_id is not None and roles.get(int(person.track_id)) == role:
            return person, "role_bbox"

    best, best_area = None, 0.0
    for person in persons:
        box = keypoint_bbox(person.keypoints, person.scores, min_score)
        if box is None:
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > best_area:
            best, best_area = person, area
    return (best, "largest_bbox") if best is not None else (None, "none")


def pose_prefilter_config(
    base: PrefilterConfig,
    *,
    roi: tuple[float, float, float, float] | None = None,
    pre_margin_sec: float | None = None,
    post_margin_sec: float | None = None,
) -> PrefilterConfig:
    """姿勢推定用の窓設定を作る。

    キャップ用の窓は活動量の立ち上がり `s` の前後 (-1.0秒, +2.5秒) で、
    頭はすでにワインドアップを含んでいる。姿勢推定で削りたいのは
    キャップ飛翔を追うための後ろ側。また骨格は連続していないと動作解析に
    使えないので、間引き（search_stride）は無効にする。

    roi は「どのフレームを処理するか」を決めるだけで、どの人物を採るかには
    影響しない。投手ROIを指定しても打者の骨格は取れる。
    """
    overrides = {"search_stride": 1}
    if roi is not None:
        overrides["roi"] = roi
    if pre_margin_sec is not None:
        overrides["pre_margin_sec"] = pre_margin_sec
    if post_margin_sec is not None:
        overrides["post_margin_sec"] = post_margin_sec
    return replace(base, **overrides)


def draw_skeleton(
    frame: np.ndarray,
    person: PersonPose,
    color: tuple[int, int, int] = (0, 255, 255),
    min_score: float = 0.5,
    thickness: int = 2,
    joint_radius: int = 3,
    label: str | None = None,
) -> np.ndarray:
    """骨格を描いた新しいフレームを返す（入力は変更しない）。

    信頼度が min_score 未満のキーポイントと、その端点を含む辺は描かない。
    """
    canvas = frame.copy()
    keypoints = np.asarray(person.keypoints)
    scores = np.asarray(person.scores)
    visible = scores >= min_score

    for a, b in SKELETON_EDGES:
        if not (visible[a] and visible[b]):
            continue
        pt_a = (int(round(keypoints[a][0])), int(round(keypoints[a][1])))
        pt_b = (int(round(keypoints[b][0])), int(round(keypoints[b][1])))
        cv2.line(canvas, pt_a, pt_b, color, thickness, lineType=cv2.LINE_AA)

    for index in np.flatnonzero(visible):
        center = (int(round(keypoints[index][0])), int(round(keypoints[index][1])))
        cv2.circle(canvas, center, joint_radius, color, -1, lineType=cv2.LINE_AA)

    if label:
        box = keypoint_bbox(keypoints, scores, min_score)
        if box is not None:
            cv2.putText(canvas, label, (int(box[0]), int(box[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=cv2.LINE_AA)

    return canvas


class PoseEstimator:
    """Pose 重みのラッパ。PitchingAnalyzer と同じ update / draw の形に揃えてある。

    `model.track(persist=True)` でトラッカーIDを持続させるが、その状態は
    モデルオブジェクト側に残るので、投球窓が切り替わったら
    `reset_tracker()` を呼んでIDの持ち越しを断つ。
    """

    def __init__(
        self,
        model_path: str = "yolo11x-pose.pt",
        conf: float = 0.5,
        imgsz: int = 960,
        min_keypoint_score: float = 0.5,
        min_overlap: float = 0.5,
        roles: Sequence[str] = ("pitcher",),
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:  # pragma: no cover - 実行環境依存
            raise PoseError(
                "ultralytics が見つかりません。PIPELINE.md の手順で導入してください"
            ) from error

        self._model = YOLO(model_path)
        self._conf = conf
        self._imgsz = imgsz
        self._min_keypoint_score = min_keypoint_score
        self._roles = tuple(roles)
        self._tracker = RoleTracker(min_overlap=min_overlap, min_score=min_keypoint_score)
        self.frames_inferred = 0

    def reset_tracker(self) -> None:
        """役割の割り当てとトラッカー状態を捨てる。窓の切り替わりで呼ぶ。"""
        self._tracker.reset()
        # ultralytics はトラッカーをモデル側に持つので明示的に落とす
        predictor = getattr(self._model, "predictor", None)
        if predictor is not None and getattr(predictor, "trackers", None):
            for tracker in predictor.trackers:
                tracker.reset()

    def update(self, frame: np.ndarray, detection_results: Iterable) -> dict:
        """元フレームに姿勢推定を掛け、役割付きの骨格を返す。

        Args:
            frame: 元フレーム（強調フレームではない）
            detection_results: 同フレームの検出結果。役割bboxの供給元。
                推論が間引かれたフレームでは空で構わない（役割はIDで継続する）。

        Returns:
            dict: persons（PersonPose のリスト）, roles（{track_id: 役割}）
        """
        results = self._model.track(
            frame, conf=self._conf, imgsz=self._imgsz,
            persist=True, verbose=False,
        )
        self.frames_inferred += 1

        persons = _persons_from_results(results)
        role_boxes = build_role_boxes(detection_results)
        roles = self._tracker.assign(persons, role_boxes)
        return {"persons": persons, "roles": roles}

    def draw(self, frame: np.ndarray, update_result: dict, label_roles: bool = True) -> np.ndarray:
        """役割が判明していて、対象役割に含まれる人物だけ骨格を描く。"""
        canvas = frame
        roles = update_result["roles"]
        for person in update_result["persons"]:
            if person.track_id is None:
                continue
            role = roles.get(int(person.track_id))
            if role is None or role not in self._roles:
                continue
            canvas = draw_skeleton(
                canvas, person,
                color=ROLE_COLORS.get(role, (0, 255, 255)),
                min_score=self._min_keypoint_score,
                label=role if label_roles else None,
            )
        return canvas


def _persons_from_results(results) -> list[PersonPose]:
    """ultralytics の結果を PersonPose のリストに変換する。"""
    persons: list[PersonPose] = []
    for result in results:
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xy is None:
            continue
        coords = keypoints.xy.cpu().numpy()
        scores = (
            keypoints.conf.cpu().numpy()
            if getattr(keypoints, "conf", None) is not None
            else np.ones(coords.shape[:2])
        )
        ids = _track_ids(result, len(coords))
        for index in range(len(coords)):
            persons.append(PersonPose(
                track_id=ids[index],
                keypoints=coords[index],
                scores=scores[index],
            ))
    return persons


def _track_ids(result, count: int) -> list[int | None]:
    """トラッカーIDを取り出す。付いていないフレームでは None を並べる。"""
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "id", None) is None:
        return [None] * count
    return [int(value) for value in boxes.id.cpu().numpy().tolist()]
