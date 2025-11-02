"""
ホモグラフィ変換ユーティリティ

画像座標と実世界座標の変換を行う
"""

import json
import numpy as np
import cv2
from typing import List, Tuple, Optional
from calibration_schema import CalibrationData, CalibrationPoint, validate_calibration_data


def calculate_homography(
    image_points: List[Tuple[float, float]],
    world_points: List[Tuple[float, float]]
) -> np.ndarray:
    """
    画像座標と実世界座標からホモグラフィ行列を計算

    Args:
        image_points: 画像上の点のリスト [(x, y), ...]（最低4点）
        world_points: 対応する実世界座標のリスト [(x, y), ...]（最低4点）

    Returns:
        homography_matrix: 3x3のホモグラフィ行列

    Raises:
        ValueError: 点数が4点未満、または対応点数が一致しない場合
    """
    if len(image_points) < 4:
        raise ValueError(f"画像座標は最低4点必要です（現在: {len(image_points)}点）")

    if len(image_points) != len(world_points):
        raise ValueError(
            f"画像座標と実世界座標の点数が一致しません "
            f"（画像: {len(image_points)}点、実世界: {len(world_points)}点）"
        )

    # NumPy配列に変換
    src_points = np.array(image_points, dtype=np.float32)
    dst_points = np.array(world_points, dtype=np.float32)

    # OpenCVでホモグラフィ行列を計算
    # findHomography は RANSAC を使用して外れ値に強い
    if len(image_points) == 4:
        # 4点の場合は直接計算
        H = cv2.getPerspectiveTransform(src_points, dst_points)
    else:
        # 5点以上の場合はRANSACで推定
        H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        if H is None:
            raise ValueError("ホモグラフィ行列の計算に失敗しました")

    return H


def calculate_homography_from_calibration(calib_data: CalibrationData) -> np.ndarray:
    """
    キャリブレーションデータからホモグラフィ行列を計算

    Args:
        calib_data: キャリブレーションデータ

    Returns:
        homography_matrix: 3x3のホモグラフィ行列
    """
    # 画像座標と実世界座標を抽出
    image_points = [(p['image_x'], p['image_y']) for p in calib_data['points']]
    world_points = [(p['world_x'], p['world_y']) for p in calib_data['points']]

    return calculate_homography(image_points, world_points)


def transform_point_to_world(
    image_x: float,
    image_y: float,
    homography_matrix: np.ndarray
) -> Tuple[float, float]:
    """
    画像座標を実世界座標に変換

    Args:
        image_x: 画像上のX座標（ピクセル）
        image_y: 画像上のY座標（ピクセル）
        homography_matrix: ホモグラフィ行列（3x3）

    Returns:
        (world_x, world_y): 実世界座標（メートル）
    """
    # OpenCVのperspectiveTransformを使用
    point = np.array([[[image_x, image_y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, homography_matrix)

    world_x, world_y = transformed[0][0]
    return float(world_x), float(world_y)


def transform_points_to_world(
    image_points: List[Tuple[float, float]],
    homography_matrix: np.ndarray
) -> List[Tuple[float, float]]:
    """
    複数の画像座標を実世界座標に変換（バッチ処理）

    Args:
        image_points: 画像上の点のリスト [(x, y), ...]
        homography_matrix: ホモグラフィ行列（3x3）

    Returns:
        world_points: 実世界座標のリスト [(x, y), ...]
    """
    if not image_points:
        return []

    # NumPy配列に変換してバッチ処理
    points_array = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points_array, homography_matrix)

    # リストに変換して返す
    return [(float(p[0][0]), float(p[0][1])) for p in transformed]


def load_calibration_from_json(json_path: str) -> CalibrationData:
    """
    JSONファイルからキャリブレーションデータを読み込み

    Args:
        json_path: キャリブレーションJSONファイルのパス

    Returns:
        calib_data: キャリブレーションデータ

    Raises:
        FileNotFoundError: ファイルが見つからない場合
        ValueError: データが不正な場合
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # バリデーション
    is_valid, errors = validate_calibration_data(data)
    if not is_valid:
        error_msg = "\n".join(errors)
        raise ValueError(f"キャリブレーションデータが不正です:\n{error_msg}")

    return data


def save_calibration_to_json(
    calib_data: CalibrationData,
    json_path: str
) -> None:
    """
    キャリブレーションデータをJSONファイルに保存

    Args:
        calib_data: キャリブレーションデータ
        json_path: 保存先のJSONファイルパス

    Raises:
        ValueError: データが不正な場合
    """
    # バリデーション
    is_valid, errors = validate_calibration_data(calib_data)
    if not is_valid:
        error_msg = "\n".join(errors)
        raise ValueError(f"キャリブレーションデータが不正です:\n{error_msg}")

    # ホモグラフィ行列をリストに変換（NumPy配列の場合）
    if isinstance(calib_data['homography_matrix'], np.ndarray):
        calib_data['homography_matrix'] = calib_data['homography_matrix'].tolist()

    # JSON保存
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(calib_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Calibration data saved to {json_path}")


def estimate_reprojection_error(
    image_points: List[Tuple[float, float]],
    world_points: List[Tuple[float, float]],
    homography_matrix: np.ndarray
) -> float:
    """
    再投影誤差を計算してキャリブレーションの精度を評価

    Args:
        image_points: 画像上の点のリスト
        world_points: 実世界座標のリスト
        homography_matrix: ホモグラフィ行列

    Returns:
        mean_error: 平均再投影誤差（ピクセル単位）
    """
    # 実世界座標を画像座標に逆変換
    world_array = np.array(world_points, dtype=np.float32).reshape(-1, 1, 2)
    H_inv = np.linalg.inv(homography_matrix)
    reprojected = cv2.perspectiveTransform(world_array, H_inv)

    # 元の画像座標との誤差を計算
    image_array = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
    errors = np.linalg.norm(image_array - reprojected, axis=2)

    mean_error = float(np.mean(errors))
    return mean_error


# デバッグ用のテスト関数
if __name__ == "__main__":
    # テスト用の4点
    image_pts = [
        (640.0, 480.0),   # ホームベース
        (650.0, 300.0),   # 投手方向3m
        (750.0, 320.0),   # 右側1.5m
        (550.0, 320.0),   # 左側1.5m
    ]

    world_pts = [
        (0.0, 0.0),      # ホームベース
        (0.0, 3.0),      # 投手方向3m
        (1.5, 2.0),      # 右側1.5m
        (-1.5, 2.0),     # 左側1.5m
    ]

    print("🧪 ホモグラフィ計算テスト")
    print(f"画像座標: {image_pts}")
    print(f"実世界座標: {world_pts}")
    print()

    # ホモグラフィ行列を計算
    H = calculate_homography(image_pts, world_pts)
    print("✅ ホモグラフィ行列:")
    print(H)
    print()

    # テスト: 画像座標を実世界座標に変換
    test_point = (640.0, 480.0)
    world_coord = transform_point_to_world(test_point[0], test_point[1], H)
    print(f"📍 テスト変換: {test_point} → {world_coord}")
    print()

    # 再投影誤差を計算
    error = estimate_reprojection_error(image_pts, world_pts, H)
    print(f"📊 平均再投影誤差: {error:.4f} ピクセル")
