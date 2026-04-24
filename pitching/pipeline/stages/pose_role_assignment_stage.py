from __future__ import annotations

import dataclasses
import logging

from pitching.domain.services.role_assigner import assign_roles
from pitching.pipeline.context import PipelineArtifacts, PipelineContext

logger = logging.getLogger(__name__)


class PoseRoleAssignmentStage:
    """PoseFrame に YOLO bbox との IoU マッチングで投手/打者/捕手ロールを付与する。"""

    name = "pose_role_assignment"

    def __init__(
        self,
        min_keypoint_confidence: float = 0.3,
        min_iou_threshold: float = 0.1,
    ) -> None:
        self._min_kp_conf = min_keypoint_confidence
        self._min_iou = min_iou_threshold

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assigned = assign_roles(
            pose_frames=ctx.artifacts.pose_frames,
            yolo_detections=ctx.artifacts.yolo_detections,
            min_keypoint_confidence=self._min_kp_conf,
            min_iou_threshold=self._min_iou,
        )

        from collections import Counter
        role_counts = Counter(r.role.value for r in assigned)
        logger.info("PoseRoleAssignmentStage: %d frames assigned %s", len(assigned), dict(role_counts))

        new_artifacts = dataclasses.replace(ctx.artifacts, role_assigned_pose_frames=assigned)
        return dataclasses.replace(ctx, artifacts=new_artifacts)
