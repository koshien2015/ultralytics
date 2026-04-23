from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

from pitching.domain.entities.detection import DiffDetection

logger = logging.getLogger(__name__)


class FrameDiffDetector:
    """
    連続3フレームの差分ANDから動体候補を検出する。
    tennis.py の検出ロジックをステージ化・純粋関数化したもの。
    """

    def __init__(
        self,
        threshold: int = 100,
        gamma: float = 1.5,
        dilation_size: int = 5,
        erosion_kernel_size: int = 10,
        area_min: float = 10.0,
        area_max: float = 5000.0,
    ) -> None:
        self._threshold = threshold
        self._gamma = gamma
        self._dilation_size = dilation_size
        self._erosion_kernel_size = erosion_kernel_size
        self._area_min = area_min
        self._area_max = area_max
        self._lut = self._build_lut(gamma)

    @staticmethod
    def _build_lut(gamma: float) -> np.ndarray:
        return np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
            dtype=np.uint8,
        )

    def detect_on_pair(
        self,
        prev: np.ndarray,
        curr: np.ndarray,
        next_frame: np.ndarray,
        frame_index: int,
    ) -> Tuple[DiffDetection, ...]:
        """
        prev/curr/next の3フレームを使い差分AND→膨張→収縮→輪郭で候補を検出。
        curr フレームに対応する検出として返す。
        """
        diff1 = cv2.absdiff(curr, prev)
        diff2 = cv2.absdiff(next_frame, curr)
        diff = cv2.bitwise_and(diff1, diff2)
        diff = cv2.LUT(diff, self._lut)

        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self._threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((self._dilation_size, self._dilation_size), np.uint8)
        dilated = cv2.dilate(binary, kernel)

        if self._erosion_kernel_size > 0:
            ek = np.ones((self._erosion_kernel_size, self._erosion_kernel_size), np.uint8)
            eroded = cv2.erode(dilated, ek, iterations=1)
        else:
            eroded = dilated

        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results: list[DiffDetection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self._area_min <= area <= self._area_max):
                continue
            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
            results.append(DiffDetection(
                frame_index=frame_index,
                center_x=float(cx),
                center_y=float(cy),
                area=float(area),
            ))

        return tuple(results)

    def make_enhanced_frame(
        self,
        prev: np.ndarray,
        curr: np.ndarray,
        next_frame: np.ndarray,
    ) -> np.ndarray:
        """YOLO 検出用の強調フレームを生成する（tennis.py の enhanced 相当）。"""
        diff1 = cv2.absdiff(curr, prev)
        diff2 = cv2.absdiff(next_frame, curr)
        diff = cv2.bitwise_and(diff1, diff2)
        diff = cv2.LUT(diff, self._lut)
        return cv2.addWeighted(curr, 0.4, diff, 0.6, 0)
