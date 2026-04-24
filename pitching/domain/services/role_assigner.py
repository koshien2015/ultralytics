from __future__ import annotations

from typing import List, Tuple

from pitching.domain.entities.detection import YoloDetection
from pitching.domain.entities.pose import PoseFrame
from pitching.domain.entities.pose_role import (
    YOLO_CLASS_TO_ROLE,
    PoseRole,
    RoleAssignedPoseFrame,
)
from pitching.domain.services.keypoint_utils import bbox_from_keypoints, iou

# ボール (class 0) は人物ではないので除外
_PERSON_CLASS_IDS = set(YOLO_CLASS_TO_ROLE.keys())


def assign_roles(
    pose_frames: Tuple[PoseFrame, ...],
    yolo_detections: Tuple[YoloDetection, ...],
    min_keypoint_confidence: float = 0.3,
    min_iou_threshold: float = 0.1,
) -> Tuple[RoleAssignedPoseFrame, ...]:
    """
    PoseFrame ごとに YOLO bbox との IoU を計算し最大マッチで役割を付与する。
    IoU が閾値未満の場合は UNKNOWN。
    """
    # フレームごとに人物系 YOLO 検出をインデックス化
    yolo_by_frame: dict = {}
    for det in yolo_detections:
        if det.class_id not in _PERSON_CLASS_IDS:
            continue
        yolo_by_frame.setdefault(det.frame_index, []).append(det)

    results: List[RoleAssignedPoseFrame] = []
    for pf in pose_frames:
        pose_bbox = bbox_from_keypoints(pf, min_keypoint_confidence)
        candidates = yolo_by_frame.get(pf.frame_index, [])

        best_iou = 0.0
        best_det = None
        if pose_bbox is not None:
            for det in candidates:
                score = iou(pose_bbox, det.bbox)
                if score > best_iou:
                    best_iou = score
                    best_det = det

        if best_det is not None and best_iou >= min_iou_threshold:
            role = YOLO_CLASS_TO_ROLE.get(best_det.class_id, PoseRole.UNKNOWN)
            yolo_class_id = best_det.class_id
        else:
            role = PoseRole.UNKNOWN
            yolo_class_id = None

        results.append(RoleAssignedPoseFrame(
            pose_frame=pf,
            role=role,
            yolo_class_id=yolo_class_id,
            match_iou=best_iou,
        ))

    return tuple(results)
