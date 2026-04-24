from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pitching.domain.entities.pose import PoseFrame


class PoseRole(Enum):
    PITCHER = "pitcher"
    BATTER = "batter"
    CATCHER = "catcher"
    UMPIRE = "umpire"
    UNKNOWN = "unknown"


# YOLO クラス ID → PoseRole のマッピング
YOLO_CLASS_TO_ROLE: dict = {
    1: PoseRole.PITCHER,   # pitcher_motion
    2: PoseRole.BATTER,    # batter_stance
    3: PoseRole.UMPIRE,    # umpire
    4: PoseRole.CATCHER,   # catcher
    5: PoseRole.PITCHER,   # pitcher_release
    6: PoseRole.BATTER,    # batter_swing
    7: PoseRole.CATCHER,   # catcher_stance
    8: PoseRole.CATCHER,   # catcher_catch
    9: PoseRole.CATCHER,   # catcher_throw
    10: PoseRole.CATCHER,  # catcher_miss
}

# 打者の YOLO クラス（スイング判定に使用）
BATTER_SWING_CLASS_IDS = frozenset({6})       # batter_swing
BATTER_STANCE_CLASS_IDS = frozenset({2})      # batter_stance
PITCHER_CLASS_IDS = frozenset({1, 5})         # pitcher_motion, pitcher_release


@dataclass(frozen=True)
class RoleAssignedPoseFrame:
    pose_frame: PoseFrame
    role: PoseRole
    yolo_class_id: Optional[int]    # マッチした YOLO クラス（None = 未マッチ）
    match_iou: float                # bbox IoU スコア（未マッチは 0.0）

    @property
    def frame_index(self) -> int:
        return self.pose_frame.frame_index

    @property
    def is_swinging(self) -> bool:
        return self.yolo_class_id in BATTER_SWING_CLASS_IDS
