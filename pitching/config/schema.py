from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class YoloConfig(BaseModel):
    model_path: str = "shared/yolo8m_20251109.pt"
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    target_classes: List[int] = Field(default=[0])


class FrameDiffConfig(BaseModel):
    threshold: int = Field(default=100, ge=0, le=255)
    gamma: float = Field(default=1.5, ge=0.1, le=5.0)
    dilation_size: int = Field(default=5, ge=0)
    erosion_kernel_size: int = Field(default=10, ge=0)
    area_min: float = Field(default=10.0, ge=0.0)
    area_max: float = Field(default=5000.0, ge=0.0)


class FusionConfig(BaseModel):
    max_jump_px: float = Field(default=80.0, ge=0.0)
    min_direction_cos: float = Field(default=0.3, ge=-1.0, le=1.0)


class TrackingConfig(BaseModel):
    max_trajectory_length: int = Field(default=60, ge=1)
    fade_frames: int = Field(default=60, ge=0)
    max_match_distance_px: float = Field(default=100.0, ge=0.0)


class StrikeZoneConfig(BaseModel):
    width_px: float = Field(default=50.0, ge=0.0)
    # None のとき初回捕手検出時に自動設定
    fixed_center_x: Optional[float] = None


class RenderingConfig(BaseModel):
    draw_trajectory: bool = True
    draw_strike_zone: bool = False
    glow_color_bgr: List[int] = Field(default=[0, 255, 255])   # シアン
    core_color_bgr: List[int] = Field(default=[255, 255, 255])  # 白
    glow_thickness: int = Field(default=15, ge=1)
    core_thickness: int = Field(default=3, ge=1)
    glow_blur: int = Field(default=25, ge=1)
    glow_intensity: float = Field(default=0.8, ge=0.0, le=1.0)


class PipelineConfig(BaseModel):
    yolo: YoloConfig = Field(default_factory=YoloConfig)
    frame_diff: FrameDiffConfig = Field(default_factory=FrameDiffConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    strike_zone: StrikeZoneConfig = Field(default_factory=StrikeZoneConfig)
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
