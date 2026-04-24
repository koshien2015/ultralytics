from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from pitching.config.schema import PipelineConfig
from pitching.domain.entities.batter_metrics import BatterSwingMetrics
from pitching.domain.entities.detection import DiffDetection, FusedDetection, YoloDetection
from pitching.domain.entities.pitch import Pitch, ReleaseEvent
from pitching.domain.entities.pitcher_metrics import PitcherPitchMetrics
from pitching.domain.entities.pose import PoseFrame
from pitching.domain.entities.pose_role import RoleAssignedPoseFrame
from pitching.domain.entities.strike_zone import StrikeZoneSeries
from pitching.domain.entities.track import Track


@dataclass(frozen=True)
class PipelineArtifacts:
    yolo_detections: Tuple[YoloDetection, ...] = ()
    diff_detections: Tuple[DiffDetection, ...] = ()
    fused_detections: Tuple[FusedDetection, ...] = ()
    pose_frames: Tuple[PoseFrame, ...] = ()
    role_assigned_pose_frames: Tuple[RoleAssignedPoseFrame, ...] = ()
    tracks: Tuple[Track, ...] = ()
    strike_zone_series: Optional[StrikeZoneSeries] = None
    release_events: Tuple[ReleaseEvent, ...] = ()
    pitches: Tuple[Pitch, ...] = ()
    pitcher_metrics: Tuple[PitcherPitchMetrics, ...] = ()
    batter_metrics: Tuple[BatterSwingMetrics, ...] = ()


@dataclass(frozen=True)
class PipelineContext:
    config: PipelineConfig
    video_path: Path
    output_dir: Path
    fps: float
    total_frames: int
    video_size: Tuple[int, int]       # (width, height)
    artifacts: PipelineArtifacts = field(default_factory=PipelineArtifacts)
    enhanced_video_path: Optional[Path] = None

    @property
    def base_name(self) -> str:
        return self.video_path.stem

    @property
    def output_video_path(self) -> Path:
        return self.output_dir / f"{self.base_name}_detected.mp4"

    @property
    def output_json_path(self) -> Path:
        return self.output_dir / f"{self.base_name}_trajectory.json"

    @property
    def output_pose_json_path(self) -> Path:
        return self.output_dir / f"{self.base_name}_pose_analysis.json"
