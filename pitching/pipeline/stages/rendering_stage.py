from __future__ import annotations

import dataclasses
import json
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

from pitching.config.schema import RenderingConfig
from pitching.domain.entities.track import Track
from pitching.infra.video.reader import VideoReader
from pitching.infra.video.renderer.neon_polyline import draw_neon_polyline
from pitching.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class RenderingStage:
    """
    元動画に軌跡・ストライクゾーンを描画して出力動画を生成し、
    軌跡 JSON を書き出す。
    """

    name = "rendering"

    def __init__(self, cfg: RenderingConfig) -> None:
        self._cfg = cfg

    def run(self, ctx: PipelineContext) -> PipelineContext:
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_video(ctx)
        self._write_json(ctx)
        self._write_pose_json(ctx)
        return ctx

    def _write_video(self, ctx: PipelineContext) -> None:
        cfg = self._cfg
        w, h = ctx.video_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(ctx.output_video_path), fourcc, ctx.fps, (w, h))

        # トラック座標をフレームごとにインデックス化
        track_points: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for track in ctx.artifacts.tracks:
            for pt in track.points:
                track_points[track.track_id].append((int(pt.x), int(pt.y)))

        with VideoReader(ctx.video_path) as reader:
            for meta, frame in reader:
                fi = meta.frame_index

                # ストライクゾーン描画
                if cfg.draw_strike_zone and ctx.artifacts.strike_zone_series:
                    zone = ctx.artifacts.strike_zone_series.at(fi)
                    if zone:
                        pt1 = (int(zone.left), int(zone.top))
                        pt2 = (int(zone.right), int(zone.bottom))
                        cv2.rectangle(frame, pt1, pt2, (255, 255, 0), 2)

                # 軌跡描画（ネオン効果）
                if cfg.draw_trajectory:
                    for track in ctx.artifacts.tracks:
                        pts_up_to_now = [
                            (int(p.x), int(p.y))
                            for p in track.points
                            if p.frame_index <= fi
                        ]
                        if len(pts_up_to_now) < 2:
                            continue

                        frames_since = fi - track.last_frame
                        fade = max(0.0, 1.0 - frames_since / max(cfg.glow_blur, 1))

                        frame = draw_neon_polyline(
                            frame,
                            pts_up_to_now[-cfg.glow_thickness:],  # 末尾のみ表示
                            glow_color=tuple(cfg.glow_color_bgr),
                            core_color=tuple(cfg.core_color_bgr),
                            glow_thickness=cfg.glow_thickness,
                            core_thickness=cfg.core_thickness,
                            glow_blur=cfg.glow_blur,
                            glow_intensity=cfg.glow_intensity * fade,
                        )

                writer.write(frame)

        writer.release()
        logger.info("RenderingStage: video saved to %s", ctx.output_video_path)

    def _write_json(self, ctx: PipelineContext) -> None:
        pitches_data = []
        for pitch in ctx.artifacts.pitches:
            pitches_data.append({
                "pitch_id": pitch.pitch_id,
                "release_frame": pitch.release.release_frame,
                "is_strike": pitch.is_strike,
                "trajectory": [
                    {
                        "frame": pt.frame_index,
                        "time": pt.elapsed_time_sec,
                        "x": pt.x_norm,
                        "y": pt.y_norm,
                        "z": pt.z,
                        "source": pt.source.name,
                    }
                    for pt in pitch.trajectory
                ],
            })

        output = {
            "metadata": {
                "video_file": str(ctx.video_path),
                "fps": ctx.fps,
                "camera_angle": (
                    ctx.artifacts.strike_zone_series.camera_angle_deg
                    if ctx.artifacts.strike_zone_series else 90.0
                ),
            },
            "pitches": pitches_data,
            "total_pitches": len(pitches_data),
        }

        with open(ctx.output_json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info("RenderingStage: JSON saved to %s", ctx.output_json_path)

    def _write_pose_json(self, ctx: PipelineContext) -> None:
        pitcher_data = []
        for pm in ctx.artifacts.pitcher_metrics:
            pitcher_data.append({
                "pitch_id": pm.pitch_id,
                "release_frame": pm.release_frame,
                "release_wrist_x": pm.release_wrist_x,
                "release_wrist_y": pm.release_wrist_y,
                "release_elbow_angle_deg": pm.release_elbow_angle_deg,
                "hip_rotation_range_deg": pm.hip_rotation_range_deg,
                "frames": [
                    {
                        "frame": f.frame_index,
                        "elbow_angle_deg": f.elbow_angle_deg,
                        "shoulder_tilt_deg": f.shoulder_tilt_deg,
                        "hip_angle_deg": f.hip_angle_deg,
                        "front_knee_angle_deg": f.front_knee_angle_deg,
                        "wrist_x": f.wrist_x,
                        "wrist_y": f.wrist_y,
                    }
                    for f in pm.frames
                ],
            })

        batter_data = []
        for bm in ctx.artifacts.batter_metrics:
            batter_data.append({
                "pitch_id": bm.pitch_id,
                "swing_start_frame": bm.swing_start_frame,
                "swing_end_frame": bm.swing_end_frame,
                "wrist_path": [{"x": x, "y": y} for x, y in bm.wrist_path],
                "hip_rotation_range_deg": bm.hip_rotation_range_deg,
                "avg_shoulder_level_diff_px": bm.avg_shoulder_level_diff_px,
                "head_displacement_px": bm.head_displacement_px,
                "frames": [
                    {
                        "frame": f.frame_index,
                        "wrist_x": f.wrist_x,
                        "wrist_y": f.wrist_y,
                        "hip_angle_deg": f.hip_angle_deg,
                        "shoulder_level_diff_px": f.shoulder_level_diff_px,
                        "head_x": f.head_x,
                        "head_y": f.head_y,
                        "front_knee_angle_deg": f.front_knee_angle_deg,
                    }
                    for f in bm.frames
                ],
            })

        output = {
            "metadata": {
                "video_file": str(ctx.video_path),
                "fps": ctx.fps,
            },
            "pitcher": pitcher_data,
            "batter": batter_data,
        }

        with open(ctx.output_pose_json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info("RenderingStage: pose JSON saved to %s", ctx.output_pose_json_path)
