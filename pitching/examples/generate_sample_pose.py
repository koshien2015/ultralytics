#!/usr/bin/env python3
"""実動画が無いときに一通り試すための、合成投球データを書き出す。

    python examples/generate_sample_pose.py output/sample

good / bad の2投球分の pose.json と config.yaml を作る。
本物の投球ではないので、数値の大小に意味は無い。動作確認と画面の確認用。

bad は「肘が伸びきる前にリリースした」想定で、リリースを3フレーム早めてある。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml  # noqa: E402

from pitching.adapters.json_adapter import save_pose_json  # noqa: E402
from pitching.models import Keypoint, PoseFrame, PoseSeries  # noqa: E402

FPS = 60.0
START_FRAME = 100
FRAMES = 40
FOOT_CONTACT_FRAME = START_FRAME + 12
RELEASE_FRAMES = {"good": START_FRAME + 30, "bad": START_FRAME + 27}

# 画面上のおおよその寸法（ピクセル）。打者は画像の左にいる想定。
HIP_WIDTH = 30.0
SHOULDER_WIDTH = 38.0
TORSO = 70.0
UPPER_ARM = 42.0
FOREARM = 45.0
THIGH = 52.0
SHIN = 50.0


def ease(value: float) -> float:
    """0〜1 を滑らかに立ち上げる。"""
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, value)))


def pose_at(progress: float) -> dict[str, tuple[float, float]]:
    """進行度 0〜1 の姿勢を作る。右投げ、打者は画像の左。"""
    stride = ease(progress) * 55.0          # 打者方向（左）への踏み出し
    lean = ease(progress) * 14.0            # 打者方向への体幹の倒れ
    rotation = ease(progress) * 12.0        # 肩の開き（見かけ）

    hip_x = 240.0 - stride * 0.45
    hip_y = 250.0
    shoulder_x = hip_x - lean
    shoulder_y = hip_y - TORSO

    # 肘は肩の後ろから前へ。手首は肘より遅れて回ってくる（腕のしなり）。
    elbow_angle = math.radians(200.0 - ease(progress) * 150.0)
    elbow = (
        shoulder_x + UPPER_ARM * math.cos(elbow_angle),
        shoulder_y - UPPER_ARM * math.sin(elbow_angle) * 0.75,
    )
    wrist_angle = math.radians(250.0 - ease(max(0.0, progress - 0.18) / 0.82) * 210.0)
    wrist = (
        elbow[0] + FOREARM * math.cos(wrist_angle),
        elbow[1] - FOREARM * math.sin(wrist_angle) * 0.85,
    )

    lead_knee_bend = 1.0 - ease(progress) * 0.55   # 接地後に前脚が伸びる
    lead_ankle = (hip_x - stride, hip_y + THIGH + SHIN)
    lead_knee = (
        (hip_x - stride * 0.55),
        hip_y + THIGH * (0.85 + 0.15 * lead_knee_bend),
    )

    return {
        "nose": (shoulder_x - 6.0, shoulder_y - 28.0),
        "right_shoulder": (shoulder_x + SHOULDER_WIDTH / 2 - rotation, shoulder_y + 2.0),
        "left_shoulder": (shoulder_x - SHOULDER_WIDTH / 2 - rotation, shoulder_y),
        "right_elbow": elbow,
        "left_elbow": (shoulder_x - 30.0, shoulder_y + 26.0),
        "right_wrist": wrist,
        "left_wrist": (shoulder_x - 34.0, shoulder_y + 60.0),
        "right_hip": (hip_x + HIP_WIDTH / 2, hip_y),
        "left_hip": (hip_x - HIP_WIDTH / 2, hip_y),
        "right_knee": (hip_x + 14.0 + stride * 0.15, hip_y + THIGH),
        "left_knee": lead_knee,
        "right_ankle": (hip_x + 26.0 + stride * 0.25, hip_y + THIGH + SHIN),
        "left_ankle": lead_ankle,
    }


def build_series(pitch_id: str) -> PoseSeries:
    frames = []
    for offset in range(FRAMES):
        points = pose_at(offset / (FRAMES - 1))
        frames.append(
            PoseFrame(
                frame_index=START_FRAME + offset,
                timestamp_sec=(START_FRAME + offset) / FPS,
                keypoints={
                    name: Keypoint(x, y, 0.92) for name, (x, y) in points.items()
                },
            )
        )
    return PoseSeries(frames=frames, fps=FPS, pitch_id=pitch_id, adapter="synthetic")


def build_config(pitch_id: str, label: str, release_frame: int) -> dict:
    return {
        "pitch_id": pitch_id,
        "throwing_hand": "right",
        "batter_direction": "left",
        "fps": FPS,
        "start_frame": START_FRAME,
        "end_frame": START_FRAME + FRAMES - 1,
        "events": {
            "stride_foot_contact_frame": FOOT_CONTACT_FRAME,
            "release_frame": release_frame,
        },
        "result": {"label": label, "description": "合成データ（実投球ではない）"},
        "preprocessing": {"smoothing_window": 7},
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 1

    root = Path(argv[1])
    for label, release_frame in RELEASE_FRAMES.items():
        directory = root / label
        pitch_id = f"sample_{label}"
        save_pose_json(build_series(pitch_id), directory / "pose.json")
        (directory / "config.yaml").write_text(
            yaml.safe_dump(
                build_config(pitch_id, label, release_frame),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        print(f"{directory}: pose.json / config.yaml")

    print("次はこれ:")
    print(f"  python -m pitching analyze --pose-data {root}/good/pose.json "
          f"--config {root}/good/config.yaml --output {root}/good")
    print(f"  python -m pitching analyze --pose-data {root}/bad/pose.json "
          f"--config {root}/bad/config.yaml --output {root}/bad")
    print(f"  python -m pitching viewer {root}/good {root}/bad --output {root}/viewer.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
