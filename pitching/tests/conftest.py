"""pytest フィクスチャ。合成データの生成は support.py にある。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# pitching パッケージのルート（ultralytics/）を import パスに足す
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pitching.config import PitchConfig  # noqa: E402
from pitching.models import PoseSeries  # noqa: E402
from pitching.tests.support import make_series  # noqa: E402

# 屈曲 → 伸展 の典型的な形。10フレーム目が最大屈曲、16フレーム目がリリース。
PITCH_ANGLES = [
    175.0, 170.0, 160.0, 148.0, 135.0, 120.0, 108.0, 98.0, 90.0, 85.0,
    82.0, 88.0, 100.0, 118.0, 138.0, 158.0, 172.0, 176.0, 177.0, 177.0,
]


@pytest.fixture
def pitch_angles() -> list[float]:
    return list(PITCH_ANGLES)


@pytest.fixture
def pitch_series() -> PoseSeries:
    return make_series(PITCH_ANGLES)


@pytest.fixture
def pitch_config() -> PitchConfig:
    return PitchConfig.model_validate(
        {
            "pitch_id": "synthetic_good",
            "throwing_hand": "right",
            "batter_direction": "left",
            "fps": 60,
            "start_frame": 0,
            "end_frame": 19,
            "events": {"stride_foot_contact_frame": 4, "release_frame": 16},
            "result": {"label": "good", "description": "合成データ"},
            "preprocessing": {"smoothing_window": 5, "smoothing_polyorder": 2},
        }
    )
