"""投球ごとの設定（YAML）の読み込みと検証。

閾値類はすべてここで設定可能にする。コードに数値を直接書かない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from pitching.models import EventSource, Hand, Side


class EventConfig(BaseModel):
    """人間が指定するイベントフレーム。未指定は None。"""

    stride_foot_contact_frame: int | None = None
    release_frame: int | None = None

    model_config = {"extra": "forbid", "validate_assignment": True}


class ResultConfig(BaseModel):
    """投球の結果ラベル。解析には使わず、出力の見出しに使う。"""

    label: str = "unlabeled"
    description: str = ""

    model_config = {"extra": "forbid", "validate_assignment": True}


class PreprocessingConfig(BaseModel):
    model_config = {"validate_assignment": True}

    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_gap_frames: int = Field(default=3, ge=0)
    smoothing_window: int = Field(default=9, ge=3)
    smoothing_polyorder: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _check_window(self) -> "PreprocessingConfig":
        if self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window は奇数で指定してください")
        if self.smoothing_window <= self.smoothing_polyorder:
            raise ValueError("smoothing_window は smoothing_polyorder より大きくしてください")
        return self


class MetricsConfig(BaseModel):
    model_config = {"validate_assignment": True}

    # 身体サイズの基準。ピクセル距離を実世界の距離として扱わないための正規化に使う。
    normalization: Literal["shoulder_width", "hip_width", "torso_length"] = "shoulder_width"
    # 前腕方向を「水平」とみなす許容角（度）
    horizontal_tolerance_deg: float = Field(default=15.0, ge=0.0, le=90.0)


class EventDetectionConfig(BaseModel):
    model_config = {"validate_assignment": True}

    # 局所最小を判定するときに前後何フレームと比べるか
    local_minimum_order: int = Field(default=2, ge=1)
    # 肘伸展開始とみなすのに必要な、増加が続くフレーム数
    extension_min_frames: int = Field(default=3, ge=1)
    # 伸展とみなす1フレームあたりの最小増加量（度）。ノイズ除け。
    extension_min_delta_deg: float = Field(default=0.5, ge=0.0)
    # 最大屈曲の探索区間を足接地の何フレーム前まで広げるか
    search_margin_frames: int = Field(default=10, ge=0)


class ExtractionConfig(BaseModel):
    model_config = {"validate_assignment": True}

    """動画から YOLO Pose を回すときの設定。解析本体はこれを参照しない。"""

    pose_model_path: str = "yolo11x-pose.pt"
    detection_model_path: str | None = None
    conf: float = Field(default=0.5, ge=0.0, le=1.0)
    imgsz: int = Field(default=960, ge=64)
    min_keypoint_score: float = Field(default=0.3, ge=0.0, le=1.0)
    min_overlap: float = Field(default=0.5, ge=0.0, le=1.0)
    device: str | None = None


class PitchConfig(BaseModel):
    """1投球分の設定。"""

    pitch_id: str
    video_path: str = ""
    throwing_hand: Hand
    # project.md の例には無いが、斜め後方カメラでは投げ手だけでは
    # 「打者方向に対して上向き何度」の符号が決まらないため追加した。
    batter_direction: Side = Side.LEFT
    fps: float = Field(default=60.0, gt=0.0)
    start_frame: int = Field(default=0, ge=0)
    end_frame: int | None = None

    events: EventConfig = Field(default_factory=EventConfig)
    result: ResultConfig = Field(default_factory=ResultConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    event_detection: EventDetectionConfig = Field(default_factory=EventDetectionConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)

    model_config = {"extra": "forbid", "validate_assignment": True}

    @model_validator(mode="after")
    def _check_range(self) -> "PitchConfig":
        if self.end_frame is not None and self.end_frame <= self.start_frame:
            raise ValueError("end_frame は start_frame より大きくしてください")
        release = self.events.release_frame
        contact = self.events.stride_foot_contact_frame
        if release is not None and contact is not None and release <= contact:
            raise ValueError("release_frame は stride_foot_contact_frame より後にしてください")
        return self

    @property
    def manual_event_source(self) -> EventSource:
        return EventSource.MANUAL

    def side_prefix(self, throwing: bool = True) -> str:
        """投球腕側 / 非投球腕側のキーポイント接頭辞。"""
        if throwing:
            return "right" if self.throwing_hand is Hand.RIGHT else "left"
        return "left" if self.throwing_hand is Hand.RIGHT else "right"


def load_pitch_config(path: str | Path) -> PitchConfig:
    """YAML を読み込んで検証する。"""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"設定ファイルのYAMLが不正です: {config_path}: {error}") from error

    if not isinstance(raw, dict):
        raise ValueError(f"設定ファイルの最上位はマッピングにしてください: {config_path}")

    try:
        return PitchConfig.model_validate(raw)
    except Exception as error:
        raise ValueError(f"設定ファイルの内容が不正です: {config_path}: {error}") from error
