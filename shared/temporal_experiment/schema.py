"""実験全体で共有するデータ型。検出と推定を同じ「成功」として扱わないための区分もここで定義する。"""

from dataclasses import dataclass
from typing import Optional

DETECTED = "detected"  # YOLO が実際に検出した位置
INTERPOLATED = "interpolated"  # 前後の観測値から補間した位置（未来フレームを参照する）
PREDICTED = "predicted"  # 状態推定で予測した位置（過去のみ参照する）
MISSING = "missing"  # 位置を確定できなかったフレーム

STATUSES = (DETECTED, INTERPOLATED, PREDICTED, MISSING)


@dataclass(frozen=True)
class Candidate:
    """1フレーム内の検出候補（元動画のピクセル座標）"""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def box(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class FrameRecord:
    """1フレーム分の推論結果。候補はしきい値を超えたものをすべて保持する"""

    frame: int
    t: float  # 動画先頭からの秒数
    candidates: tuple = ()
    infer_ms: Optional[float] = None


@dataclass(frozen=True)
class Estimate:
    """1フレーム分の手法の出力"""

    frame: int
    t: float
    status: str
    x: Optional[float] = None
    y: Optional[float] = None
    conf: Optional[float] = None  # detected のときだけ持つ
    box: Optional[tuple] = None
    track_id: Optional[int] = None

    @property
    def has_position(self) -> bool:
        return self.status != MISSING


@dataclass(frozen=True)
class GroundTruth:
    """人手の正解。visible=False は「蓋が写っていない」と確認したフレーム"""

    frame: int
    x: float
    y: float
    visible: bool = True


@dataclass(frozen=True)
class TrackEvent:
    """追跡中に起きた出来事。予測の破綻を後から分析するために残す"""

    frame: int
    kind: str  # gate_rejected / track_lost
    track_id: int
    distance: Optional[float] = None


@dataclass(frozen=True)
class TrackingResult:
    estimates: tuple
    events: tuple = ()
