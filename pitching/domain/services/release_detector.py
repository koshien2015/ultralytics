from __future__ import annotations

from pitching.domain.entities.pitch import ReleaseEvent

# キャップ野球固有のクラスID
CLASS_PITCHER_MOTION = 1
CLASS_PITCHER_RELEASE = 5


def resolve_pitcher_state(class_ids_in_frame: tuple[int, ...]) -> str | None:
    """検出クラスIDのセットから投手の状態を返す。"""
    if CLASS_PITCHER_RELEASE in class_ids_in_frame:
        return "release"
    if CLASS_PITCHER_MOTION in class_ids_in_frame:
        return "motion"
    return None


def detect_release(
    prev_state: str | None,
    curr_state: str | None,
    pitch_id: int,
    frame_index: int,
    fps: float,
) -> ReleaseEvent | None:
    """
    motion → release の遷移を検出してReleaseEventを返す。
    遷移がない場合は None。
    """
    if prev_state == "motion" and curr_state == "release":
        return ReleaseEvent(
            pitch_id=pitch_id,
            release_frame=frame_index,
            release_time_sec=frame_index / fps,
        )
    return None
