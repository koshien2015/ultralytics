from __future__ import annotations

from typing import List, Optional, Tuple

from pitching.domain.entities.pitcher_metrics import PitcherFrameMetrics, PitcherPitchMetrics
from pitching.domain.entities.pose_role import RoleAssignedPoseFrame
from pitching.domain.services.keypoint_utils import (
    angle_between,
    find_keypoint,
    horizontal_angle,
)

# 右投げ / 左投げで使うキーポイント名を切り替える
_THROW_SIDE: dict = {
    "right": {
        "shoulder": "right_shoulder",
        "elbow": "right_elbow",
        "wrist": "right_wrist",
        "hip_a": "left_hip",
        "hip_b": "right_hip",
        "knee": "right_knee",
        "ankle": "right_ankle",
        "opp_knee": "left_knee",
        "opp_ankle": "left_ankle",
    },
    "left": {
        "shoulder": "left_shoulder",
        "elbow": "left_elbow",
        "wrist": "left_wrist",
        "hip_a": "right_hip",
        "hip_b": "left_hip",
        "knee": "left_knee",
        "ankle": "left_ankle",
        "opp_knee": "right_knee",
        "opp_ankle": "right_ankle",
    },
}


def analyze_pitcher_frame(
    rf: RoleAssignedPoseFrame,
    throwing_hand: str = "right",
    min_conf: float = 0.3,
) -> PitcherFrameMetrics:
    pf = rf.pose_frame
    side = _THROW_SIDE[throwing_hand]

    def kp(name: str):
        return find_keypoint(pf, name, min_conf)

    # 肘角度（肩→肘→手首）
    sh, el, wr = kp(side["shoulder"]), kp(side["elbow"]), kp(side["wrist"])
    elbow_angle = angle_between(sh, el, wr) if (sh and el and wr) else None

    # 肩の傾き
    ls, rs = kp("left_shoulder"), kp("right_shoulder")
    shoulder_tilt = (ls.y - rs.y) if (ls and rs) else None

    # 腰の回転角
    ha, hb = kp(side["hip_a"]), kp(side["hip_b"])
    hip_angle = horizontal_angle(ha, hb) if (ha and hb) else None

    # 前膝の屈曲角（ストライド側の膝: 投げ手と逆側）
    opp_hip = kp(side["hip_a"])   # ストライド側の腰
    opp_knee = kp(side["opp_knee"])
    opp_ankle = kp(side["opp_ankle"])
    front_knee = angle_between(opp_hip, opp_knee, opp_ankle) if (opp_hip and opp_knee and opp_ankle) else None

    return PitcherFrameMetrics(
        frame_index=pf.frame_index,
        elbow_angle_deg=elbow_angle,
        shoulder_tilt_deg=shoulder_tilt,
        hip_angle_deg=hip_angle,
        front_knee_angle_deg=front_knee,
        wrist_x=wr.x if wr else None,
        wrist_y=wr.y if wr else None,
    )


def aggregate_pitcher_pitch(
    pitch_id: int,
    release_frame: int,
    frames: Tuple[PitcherFrameMetrics, ...],
) -> PitcherPitchMetrics:
    release = next((f for f in frames if f.frame_index == release_frame), None)
    if release is None and frames:
        release = min(frames, key=lambda f: abs(f.frame_index - release_frame))

    hip_angles = [f.hip_angle_deg for f in frames if f.hip_angle_deg is not None]
    hip_range = (max(hip_angles) - min(hip_angles)) if len(hip_angles) >= 2 else None

    return PitcherPitchMetrics(
        pitch_id=pitch_id,
        release_frame=release_frame,
        release_wrist_x=release.wrist_x if release else None,
        release_wrist_y=release.wrist_y if release else None,
        release_elbow_angle_deg=release.elbow_angle_deg if release else None,
        hip_rotation_range_deg=hip_range,
        frames=frames,
    )
