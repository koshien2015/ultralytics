from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PitcherFrameMetrics:
    """投球モーション 1 フレームの部位角度。None は信頼度不足で計算不可。"""
    frame_index: int

    # 肘の角度: 肩 → 肘 → 手首 のなす角 (deg)
    elbow_angle_deg: Optional[float]

    # 肩の傾き: 左肩 y - 右肩 y (正 = 右肩が下、投球側が下がっている)
    shoulder_tilt_deg: Optional[float]

    # 腰の回転角: 左腰 → 右腰 ベクトルの水平基準角 (deg)
    hip_angle_deg: Optional[float]

    # 前膝の屈曲角: 腰 → 膝 → 足首 のなす角 (deg)
    front_knee_angle_deg: Optional[float]

    # リリース手首座標 (px)
    wrist_x: Optional[float]
    wrist_y: Optional[float]


@dataclass(frozen=True)
class PitcherPitchMetrics:
    """1 投球分の投手メトリクス（フレーム列 + 代表値）。"""
    pitch_id: int
    release_frame: int

    # リリース時の手首座標
    release_wrist_x: Optional[float]
    release_wrist_y: Optional[float]

    # リリース時の肘角度
    release_elbow_angle_deg: Optional[float]

    # モーション中の腰回転幅（最大 - 最小）
    hip_rotation_range_deg: Optional[float]

    # フレームごとの詳細（比較・グラフ用）
    frames: Tuple[PitcherFrameMetrics, ...]
