from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pitching.domain.entities.batter_metrics import BatterSwingMetrics, SwingFrameMetrics
from pitching.domain.entities.pose_role import BATTER_SWING_CLASS_IDS, RoleAssignedPoseFrame
from pitching.domain.services.keypoint_utils import (
    angle_between,
    find_keypoint,
    horizontal_angle,
)

_BATTER_SIDE: dict = {
    "right": {
        "wrist": "right_wrist",
        "hip_a": "left_hip",
        "hip_b": "right_hip",
        "front_knee": "left_knee",
        "front_ankle": "left_ankle",
        "front_hip": "left_hip",
    },
    "left": {
        "wrist": "left_wrist",
        "hip_a": "right_hip",
        "hip_b": "left_hip",
        "front_knee": "right_knee",
        "front_ankle": "right_ankle",
        "front_hip": "right_hip",
    },
}


def analyze_swing_frame(
    rf: RoleAssignedPoseFrame,
    batting_hand: str = "right",
    min_conf: float = 0.3,
) -> SwingFrameMetrics:
    pf = rf.pose_frame
    side = _BATTER_SIDE[batting_hand]

    def kp(name: str):
        return find_keypoint(pf, name, min_conf)

    wr = kp(side["wrist"])
    ha, hb = kp(side["hip_a"]), kp(side["hip_b"])
    ls, rs = kp("left_shoulder"), kp("right_shoulder")
    nose = kp("nose")
    fh = kp(side["front_hip"])
    fk = kp(side["front_knee"])
    fa = kp(side["front_ankle"])

    return SwingFrameMetrics(
        frame_index=pf.frame_index,
        wrist_x=wr.x if wr else None,
        wrist_y=wr.y if wr else None,
        hip_angle_deg=horizontal_angle(ha, hb) if (ha and hb) else None,
        shoulder_level_diff_px=(ls.y - rs.y) if (ls and rs) else None,
        head_x=nose.x if nose else None,
        head_y=nose.y if nose else None,
        front_knee_angle_deg=angle_between(fh, fk, fa) if (fh and fk and fa) else None,
    )


def aggregate_swing(
    pitch_id: int,
    swing_frames: Tuple[SwingFrameMetrics, ...],
) -> BatterSwingMetrics:
    if not swing_frames:
        return BatterSwingMetrics(
            pitch_id=pitch_id,
            swing_start_frame=0,
            swing_end_frame=0,
            wrist_path=(),
            hip_rotation_range_deg=None,
            avg_shoulder_level_diff_px=None,
            head_displacement_px=None,
            frames=(),
        )

    sorted_frames = tuple(sorted(swing_frames, key=lambda f: f.frame_index))

    wrist_path = tuple(
        (f.wrist_x, f.wrist_y)
        for f in sorted_frames
        if f.wrist_x is not None and f.wrist_y is not None
    )

    hip_angles = [f.hip_angle_deg for f in sorted_frames if f.hip_angle_deg is not None]
    hip_range = (max(hip_angles) - min(hip_angles)) if len(hip_angles) >= 2 else None

    shoulder_diffs = [abs(f.shoulder_level_diff_px) for f in sorted_frames if f.shoulder_level_diff_px is not None]
    avg_shoulder = sum(shoulder_diffs) / len(shoulder_diffs) if shoulder_diffs else None

    first_head = next(((f.head_x, f.head_y) for f in sorted_frames if f.head_x and f.head_y), None)
    last_head = next(((f.head_x, f.head_y) for f in reversed(sorted_frames) if f.head_x and f.head_y), None)
    head_disp = None
    if first_head and last_head:
        import math
        head_disp = math.hypot(last_head[0] - first_head[0], last_head[1] - first_head[1])

    return BatterSwingMetrics(
        pitch_id=pitch_id,
        swing_start_frame=sorted_frames[0].frame_index,
        swing_end_frame=sorted_frames[-1].frame_index,
        wrist_path=wrist_path,
        hip_rotation_range_deg=hip_range,
        avg_shoulder_level_diff_px=avg_shoulder,
        head_displacement_px=head_disp,
        frames=sorted_frames,
    )
