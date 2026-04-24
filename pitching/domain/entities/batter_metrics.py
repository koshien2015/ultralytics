from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SwingFrameMetrics:
    """スイング中の 1 フレームの姿勢データ。"""
    frame_index: int

    # 手首軌跡（スイングパス）
    wrist_x: Optional[float]
    wrist_y: Optional[float]

    # 腰の回転角: 左腰 → 右腰 ベクトルの水平基準角 (deg)
    hip_angle_deg: Optional[float]

    # 肩の水平度: 左肩 y - 右肩 y の差 (px、0 に近いほど水平)
    shoulder_level_diff_px: Optional[float]

    # 頭（鼻）の座標（安定性チェック用）
    head_x: Optional[float]
    head_y: Optional[float]

    # 前膝の屈曲角 (deg)
    front_knee_angle_deg: Optional[float]


@dataclass(frozen=True)
class BatterSwingMetrics:
    """1 スイング分の打者メトリクス。"""
    pitch_id: int

    # スイング開始・終了フレーム（YOLO batter_swing クラスの区間）
    swing_start_frame: int
    swing_end_frame: int

    # スイングパスの全座標（手首軌跡）
    wrist_path: Tuple[Tuple[float, float], ...]   # ((x0,y0), (x1,y1), ...)

    # 腰回転幅（スイング中の最大 - 最小）
    hip_rotation_range_deg: Optional[float]

    # 肩の水平度（スイング中の平均 |left_y - right_y|）
    avg_shoulder_level_diff_px: Optional[float]

    # 頭の移動量（スイング開始→終了での鼻座標の変位 px）
    head_displacement_px: Optional[float]

    # フレームごとの詳細（グラフ用）
    frames: Tuple[SwingFrameMetrics, ...]
