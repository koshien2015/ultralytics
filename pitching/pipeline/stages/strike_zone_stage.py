from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from typing import Dict, List, Optional

from pitching.domain.entities.bbox import BBox
from pitching.domain.entities.detection import YoloDetection
from pitching.domain.entities.strike_zone import StrikeZone, StrikeZoneSeries
from pitching.domain.services.camera_angle_estimator import estimate_camera_angle
from pitching.domain.services.strike_zone_estimator import estimate_strike_zone
from pitching.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

CLASS_BATTER = (2, 6)
CLASS_CATCHER = (4, 7)
CLASS_PITCHER_MOTION = 1


class StrikeZoneStage:
    """
    YOLO 検出の打者・捕手 bbox からストライクゾーンを推定し、
    投手・捕手 bbox からカメラ角度を推定する。
    """

    name = "strike_zone"

    def __init__(self, zone_width_px: float, fixed_center_x: Optional[float] = None) -> None:
        self._zone_width = zone_width_px
        self._locked_center_x = fixed_center_x
        self._camera_angle_locked = False
        self._camera_angle = 90.0

    def run(self, ctx: PipelineContext) -> PipelineContext:
        by_frame: Dict[int, List[YoloDetection]] = defaultdict(list)
        for det in ctx.artifacts.yolo_detections:
            by_frame[det.frame_index].append(det)

        zones: List[StrikeZone] = []
        locked_cx = self._locked_center_x
        camera_angle = self._camera_angle
        camera_angle_locked = self._camera_angle_locked

        for frame_index in range(ctx.total_frames):
            dets = by_frame.get(frame_index, [])
            batter_bbox = self._find_first(dets, CLASS_BATTER)
            catcher_bbox = self._find_first(dets, CLASS_CATCHER)
            pitcher_bbox = self._find_first(dets, (CLASS_PITCHER_MOTION,))

            if batter_bbox and catcher_bbox:
                zone, locked_cx = estimate_strike_zone(
                    frame_index=frame_index,
                    batter_bbox=batter_bbox,
                    catcher_center_x=catcher_bbox.center_x,
                    locked_center_x=locked_cx,
                    zone_width=self._zone_width,
                )
                zones.append(zone)

                if pitcher_bbox and not camera_angle_locked:
                    camera_angle = estimate_camera_angle(pitcher_bbox, catcher_bbox)
                    camera_angle_locked = True
                    logger.info("StrikeZoneStage: camera angle locked at %.1f°", camera_angle)

        series = StrikeZoneSeries(
            zones=tuple(zones),
            locked_center_x=locked_cx,
            camera_angle_deg=camera_angle,
        )
        logger.info("StrikeZoneStage: %d zone frames", len(zones))
        new_artifacts = dataclasses.replace(ctx.artifacts, strike_zone_series=series)
        return dataclasses.replace(ctx, artifacts=new_artifacts)

    @staticmethod
    def _find_first(dets: List[YoloDetection], class_ids: tuple) -> Optional[BBox]:
        for det in dets:
            if det.class_id in class_ids:
                return det.bbox
        return None
