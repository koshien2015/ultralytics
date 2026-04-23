from __future__ import annotations

from pitching.domain.entities.bbox import BBox
from pitching.domain.entities.strike_zone import StrikeZone


def estimate_strike_zone(
    frame_index: int,
    batter_bbox: BBox,
    catcher_center_x: float,
    locked_center_x: float | None,
    zone_width: float,
) -> tuple[StrikeZone, float]:
    """
    打者のBBoxとキャッチャー中心X座標からストライクゾーンを推定する。

    Returns:
        (StrikeZone, used_center_x) — used_center_x はロック値として呼び出し元が保持する
    """
    batter_height = batter_bbox.height
    strike_top = batter_bbox.y1 + batter_height * 0.4
    strike_bottom = batter_bbox.y1 + batter_height * 0.75

    # 初回のみキャッチャー中心から設定し、以降はロックした値を使う
    center_x = locked_center_x if locked_center_x is not None else catcher_center_x

    zone = StrikeZone(
        frame_index=frame_index,
        left=center_x - zone_width / 2,
        right=center_x + zone_width / 2,
        top=strike_top,
        bottom=strike_bottom,
        center_x=center_x,
        center_y=(strike_top + strike_bottom) / 2,
        zone_width=zone_width,
    )
    return zone, center_x
