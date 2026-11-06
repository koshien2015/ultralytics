"""実験条件。YAML に保存して再現できるようにする。

距離に関する値は reference_width（既定 1280px）の動画を基準に書き、動画の幅に合わせて換算する。
"""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class InferenceConfig:
    weights: str = "yolo26m-1280px-120epoch.pt"
    imgsz: int = 1280
    conf: float = 0.15
    preprocess: str = "enhanced"  # enhanced / raw
    cap_class: str = "cap"


@dataclass(frozen=True)
class TrackerConfig:
    max_speed_px_s: float = 2500.0
    """対応付けを許す最大の速さ。経過時間を掛けて最大移動距離にする"""
    max_missed_frames: int = 5
    """補完する連続欠損の上限。線形補間とカルマン予測で同じ値を使う"""
    min_hits: int = 2
    """この回数の観測で軌道として確定する。確定までの点は確定時に遡って出力する"""
    tentative_max_misses: int = 2
    min_track_extent_px: float = 5.0
    """直近 motion_window 回の観測の移動幅がこれ未満なら静止物として出力しない"""
    motion_window: int = 3
    max_prediction_error_px: float = 60.0
    """最近傍追跡で、移動予測位置からこの距離を超える候補は対応付けない"""


@dataclass(frozen=True)
class KalmanConfig:
    accel_noise_px_s2: float = 3000.0
    """プロセスノイズ（加速度の標準偏差）"""
    obs_noise_px: float = 3.0
    """観測ノイズ（検出中心の標準偏差）"""
    gate_chi2: float = 9.21
    """マハラノビス距離の2乗の上限（自由度2の99%点）"""
    init_speed_px_s: float = 1500.0
    """軌道開始時の速度の不確かさ"""


@dataclass(frozen=True)
class FitConfig:
    inlier_px: float = 12.0


@dataclass(frozen=True)
class EvalConfig:
    match_radius_px: float = 15.0
    """正解からこの距離以内なら同じ蓋を指しているとみなす"""
    error_cap_px: float = 100.0
    """全フレーム誤差で、欠損フレームと大外れに与える誤差"""


@dataclass(frozen=True)
class ExperimentConfig:
    reference_width: int = 1280
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def scaled_for(self, width: int) -> "ExperimentConfig":
        """動画の幅に合わせて距離の設定を換算した設定を返す"""
        scale = width / self.reference_width
        return replace(
            self,
            tracker=replace(
                self.tracker,
                max_speed_px_s=self.tracker.max_speed_px_s * scale,
                min_track_extent_px=self.tracker.min_track_extent_px * scale,
                max_prediction_error_px=self.tracker.max_prediction_error_px * scale,
            ),
            kalman=replace(
                self.kalman,
                accel_noise_px_s2=self.kalman.accel_noise_px_s2 * scale,
                obs_noise_px=self.kalman.obs_noise_px * scale,
                init_speed_px_s=self.kalman.init_speed_px_s * scale,
            ),
            fit=replace(self.fit, inlier_px=self.fit.inlier_px * scale),
            evaluation=replace(
                self.evaluation,
                match_radius_px=self.evaluation.match_radius_px * scale,
                error_cap_px=self.evaluation.error_cap_px * scale,
            ),
        )


def load_config(path: Path) -> ExperimentConfig:
    """YAML から設定を読む。未知のキーは書き間違いとしてエラーにする"""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    try:
        return ExperimentConfig(
            reference_width=raw.get("reference_width", 1280),
            inference=InferenceConfig(**raw.get("inference", {})),
            tracker=TrackerConfig(**raw.get("tracker", {})),
            kalman=KalmanConfig(**raw.get("kalman", {})),
            fit=FitConfig(**raw.get("fit", {})),
            evaluation=EvalConfig(**raw.get("evaluation", {})),
        )
    except TypeError as error:
        raise ValueError(f"設定ファイルに不明な項目があります: {path}: {error}") from error
