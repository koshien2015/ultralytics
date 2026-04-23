from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict, deque
from typing import Dict, List, Tuple

from pitching.config.schema import FusionConfig, FrameDiffConfig
from pitching.domain.entities.bbox import BBox
from pitching.domain.entities.detection import (
    DiffDetection,
    FusedDetection,
    YoloDetection,
)
from pitching.domain.services.fusion_policy import fuse_frame
from pitching.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

CLASS_BALL = 0
_PERSON_CLASSES = {1, 2, 4, 5, 6, 7}  # 投手・打者・捕手


class FusionStage:
    """
    YOLO 検出と差分検出を融合して FusedDetection を生成する。
    Phase 4 の YoloToFusedStage を置き換えるステージ。

    融合ポリシー（domain/services/fusion_policy.py）:
      1. YOLO ボール検出が信頼度閾値を超えれば YOLO_PRIMARY として採用
      2. YOLO が取れないフレームは DiffDetection を 4 重フィルタで評価
         - 面積制約
         - 人物 bbox 内の誤検出除外
         - 直前採用座標からの距離制約
         - 軌跡方向の整合性
      3. どちらも条件を満たさない場合は欠損（後段 Tracker が補間）
    """

    name = "fusion"

    def __init__(
        self,
        fusion_cfg: FusionConfig,
        frame_diff_cfg: FrameDiffConfig,
        yolo_min_conf: float,
    ) -> None:
        self._fusion_cfg = fusion_cfg
        self._diff_cfg = frame_diff_cfg
        self._yolo_min_conf = yolo_min_conf

    def run(self, ctx: PipelineContext) -> PipelineContext:
        yolo_by_frame = self._group_by_frame(ctx.artifacts.yolo_detections)
        diff_by_frame = self._group_by_frame_diff(ctx.artifacts.diff_detections)
        body_by_frame = self._body_bboxes_by_frame(ctx.artifacts.yolo_detections)

        recent: deque[FusedDetection] = deque(maxlen=3)
        fused: List[FusedDetection] = []

        for frame_index in range(ctx.total_frames):
            result = fuse_frame(
                frame_index=frame_index,
                yolo_dets=yolo_by_frame.get(frame_index, ()),
                diff_dets=diff_by_frame.get(frame_index, ()),
                recent=tuple(recent),
                body_bboxes=body_by_frame.get(frame_index, ()),
                yolo_min_conf=self._yolo_min_conf,
                diff_area_min=self._diff_cfg.area_min,
                diff_area_max=self._diff_cfg.area_max,
                max_jump_px=self._fusion_cfg.max_jump_px,
                min_direction_cos=self._fusion_cfg.min_direction_cos,
            )
            if result is not None:
                fused.append(result)
                recent.append(result)

        yolo_count = sum(
            1 for f in fused if f.source.name == "FUSED_YOLO_PRIMARY"
        )
        diff_count = len(fused) - yolo_count
        logger.info(
            "FusionStage: %d fused (%d YOLO_PRIMARY / %d DIFF_FALLBACK)",
            len(fused), yolo_count, diff_count,
        )

        new_artifacts = dataclasses.replace(ctx.artifacts, fused_detections=tuple(fused))
        return dataclasses.replace(ctx, artifacts=new_artifacts)

    @staticmethod
    def _group_by_frame(
        dets: Tuple[YoloDetection, ...],
    ) -> Dict[int, Tuple[YoloDetection, ...]]:
        by_frame: Dict[int, List[YoloDetection]] = defaultdict(list)
        for d in dets:
            by_frame[d.frame_index].append(d)
        return {k: tuple(v) for k, v in by_frame.items()}

    @staticmethod
    def _group_by_frame_diff(
        dets: Tuple[DiffDetection, ...],
    ) -> Dict[int, Tuple[DiffDetection, ...]]:
        by_frame: Dict[int, List[DiffDetection]] = defaultdict(list)
        for d in dets:
            by_frame[d.frame_index].append(d)
        return {k: tuple(v) for k, v in by_frame.items()}

    @staticmethod
    def _body_bboxes_by_frame(
        yolo_dets: Tuple[YoloDetection, ...],
    ) -> Dict[int, Tuple[BBox, ...]]:
        by_frame: Dict[int, List[BBox]] = defaultdict(list)
        for d in yolo_dets:
            if d.class_id in _PERSON_CLASSES:
                by_frame[d.frame_index].append(d.bbox)
        return {k: tuple(v) for k, v in by_frame.items()}
