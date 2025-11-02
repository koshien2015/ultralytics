"""
カメラキャリブレーションのスキーマ定義

TypeScript版（bureau/cap-baseball-next/types/calibration.ts）と対応
"""

from typing import List, Optional, TypedDict


class CalibrationPoint(TypedDict):
    """キャリブレーション点（画像座標と実世界座標の対応）"""
    image_x: float  # 画像上のX座標（ピクセル）
    image_y: float  # 画像上のY座標（ピクセル）
    world_x: float  # 実世界のX座標（メートル）
    world_y: float  # 実世界のY座標（メートル）
    world_z: float  # 実世界のZ座標（メートル）
    label: str      # 点のラベル
    note: Optional[str]  # メモ・補足説明


class FrameDimensions(TypedDict):
    """フレーム寸法"""
    width: int   # 幅（ピクセル）
    height: int  # 高さ（ピクセル）


class CalibrationData(TypedDict):
    """カメラキャリブレーションデータ"""
    video_id: str                    # 動画ID
    calibration_date: str            # キャリブレーション実施日時（ISO 8601形式）
    frame_number: int                # キャリブレーションに使用したフレーム番号
    points: List[CalibrationPoint]   # キャリブレーション点（最低4点）
    homography_matrix: List[List[float]]  # ホモグラフィ行列（3x3）
    frame_dimensions: FrameDimensions     # フレームの寸法
    notes: Optional[str]             # 全体的な備考・メモ


def validate_calibration_data(data: dict) -> tuple[bool, List[str]]:
    """
    キャリブレーションデータを検証

    Args:
        data: キャリブレーションデータの辞書

    Returns:
        (is_valid, errors): 検証結果とエラーメッセージのリスト
    """
    errors = []

    # 必須フィールドの確認
    required_fields = ['video_id', 'calibration_date', 'frame_number',
                      'points', 'homography_matrix', 'frame_dimensions']
    for field in required_fields:
        if field not in data:
            errors.append(f"必須フィールド '{field}' が不足しています")

    # 点数の確認（最低4点必要）
    if 'points' in data:
        if len(data['points']) < 4:
            errors.append(f"キャリブレーション点は最低4点必要です（現在: {len(data['points'])}点）")

        # 各点の必須フィールド確認
        for i, point in enumerate(data['points']):
            point_required = ['image_x', 'image_y', 'world_x', 'world_y', 'world_z', 'label']
            for field in point_required:
                if field not in point:
                    errors.append(f"点{i+1}に必須フィールド '{field}' が不足しています")

    # ホモグラフィ行列の確認（3x3行列）
    if 'homography_matrix' in data:
        matrix = data['homography_matrix']
        if len(matrix) != 3:
            errors.append(f"ホモグラフィ行列は3x3である必要があります（現在: {len(matrix)}行）")
        else:
            for i, row in enumerate(matrix):
                if len(row) != 3:
                    errors.append(f"ホモグラフィ行列の{i+1}行目は3列である必要があります（現在: {len(row)}列）")

    return len(errors) == 0, errors


# JSONスキーマ例（ドキュメント用）
EXAMPLE_CALIBRATION_JSON = {
    "video_id": "game_2025_01_01_001",
    "calibration_date": "2025-11-02T13:45:00Z",
    "frame_number": 1,
    "points": [
        {
            "image_x": 640.0,
            "image_y": 480.0,
            "world_x": 0.0,
            "world_y": 0.0,
            "world_z": 0.0,
            "label": "ホームベース",
            "note": "ホームベースの中心"
        },
        {
            "image_x": 650.0,
            "image_y": 300.0,
            "world_x": 0.0,
            "world_y": 3.0,
            "world_z": 0.0,
            "label": "投手方向3m地点",
            "note": "床のライン交点"
        },
        {
            "image_x": 750.0,
            "image_y": 320.0,
            "world_x": 1.5,
            "world_y": 2.0,
            "world_z": 0.0,
            "label": "右側床ライン交点",
            "note": None
        },
        {
            "image_x": 550.0,
            "image_y": 320.0,
            "world_x": -1.5,
            "world_y": 2.0,
            "world_z": 0.0,
            "label": "左側床ライン交点",
            "note": None
        }
    ],
    "homography_matrix": [
        [1.2, 0.1, -100.0],
        [0.05, 1.5, -200.0],
        [0.0001, 0.0002, 1.0]
    ],
    "frame_dimensions": {
        "width": 1920,
        "height": 1080
    },
    "notes": "体育館での練習撮影"
}
