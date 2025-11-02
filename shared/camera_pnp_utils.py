"""
カメラキャリブレーション・PnP（Perspective-n-Point）ユーティリティ

機能:
- 3D点と2D点からカメラパラメータを推定（solvePnP）
- 画像座標から3D座標を推定（地面平面との交点）
- 高さキャリブレーションによる高さ推定
"""

import json
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class CameraCalibration:
    """カメラキャリブレーションクラス（PnPベース）"""

    def __init__(
        self,
        calibration_data: Dict,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ):
        """
        Args:
            calibration_data: キャリブレーションデータ（JSON）
            camera_matrix: カメラ内部パラメータ行列 (3x3)、Noneの場合は推定
            dist_coeffs: レンズ歪み係数、Noneの場合は歪みなしと仮定
        """
        # データの検証
        required_keys = ['frame_dimensions', 'points']
        for key in required_keys:
            if key not in calibration_data:
                raise ValueError(f"キャリブレーションデータに必須キー '{key}' がありません")

        if len(calibration_data['points']) < 4:
            raise ValueError(f"キャリブレーション点は最低4点必要です（現在: {len(calibration_data['points'])}点）")

        self.calibration_data = calibration_data
        self.frame_width = calibration_data['frame_dimensions']['width']
        self.frame_height = calibration_data['frame_dimensions']['height']

        print(f"📐 Loading calibration data...")
        print(f"   Video ID: {calibration_data.get('video_id', 'unknown')}")
        print(f"   Frame dimensions: {self.frame_width}x{self.frame_height}")
        print(f"   Points: {len(calibration_data['points'])} points")

        # カメラ内部パラメータ（K行列）
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
        else:
            # デフォルト: 焦点距離=画像幅、主点=画像中心
            focal_length = self.frame_width
            cx = self.frame_width / 2
            cy = self.frame_height / 2
            self.camera_matrix = np.array([
                [focal_length, 0, cx],
                [0, focal_length, cy],
                [0, 0, 1]
            ], dtype=np.float32)

        # レンズ歪み係数
        if dist_coeffs is not None:
            self.dist_coeffs = dist_coeffs
        else:
            # デフォルト: 歪みなし
            self.dist_coeffs = np.zeros(5, dtype=np.float32)

        # キャリブレーション点から3D点と2D点を抽出
        self.object_points = []  # 3D点（実世界座標）
        self.image_points = []   # 2D点（画像座標）

        print(f"\n   📍 Calibration points:")
        for i, point in enumerate(calibration_data['points']):
            # 必須フィールドの確認
            required_point_keys = ['image_x', 'image_y', 'world_x', 'world_y', 'world_z']
            for key in required_point_keys:
                if key not in point:
                    raise ValueError(f"点{i+1}に必須フィールド '{key}' がありません")

            self.object_points.append([
                point['world_x'],
                point['world_y'],
                point['world_z']
            ])
            self.image_points.append([
                point['image_x'],
                point['image_y']
            ])

            label = point.get('label', f'Point {i+1}')
            print(f"      {i+1}. {label}: "
                  f"Image({point['image_x']:.1f}, {point['image_y']:.1f}) → "
                  f"World({point['world_x']:.2f}, {point['world_y']:.2f}, {point['world_z']:.2f})")

        self.object_points = np.array(self.object_points, dtype=np.float32)
        self.image_points = np.array(self.image_points, dtype=np.float32)

        # PnPでカメラの外部パラメータを推定
        success, self.rvec, self.tvec = cv2.solvePnP(
            self.object_points,
            self.image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            raise ValueError("PnP solver failed to find camera pose")

        # 回転ベクトルを回転行列に変換
        self.R, _ = cv2.Rodrigues(self.rvec)

        # 高さキャリブレーション
        self.height_calibration = calibration_data.get('height_calibration')

        print(f"✅ Camera calibration initialized (PnP)")
        print(f"   Camera position: {self.get_camera_position()}")
        print(f"   Reprojection error: {self.calculate_reprojection_error():.2f} pixels")

    def get_camera_position(self) -> np.ndarray:
        """
        カメラの世界座標系における位置を取得

        Returns:
            camera_pos: カメラ位置 (x, y, z)
        """
        # カメラ位置 = -R^T * t
        camera_pos = -self.R.T @ self.tvec
        return camera_pos.flatten()

    def calculate_reprojection_error(self) -> float:
        """
        再投影誤差を計算（キャリブレーション精度の評価）

        Returns:
            error: 平均再投影誤差（ピクセル）
        """
        # 3D点を画像に投影
        projected_points, _ = cv2.projectPoints(
            self.object_points,
            self.rvec,
            self.tvec,
            self.camera_matrix,
            self.dist_coeffs
        )

        # 元の画像点との差を計算
        projected_points = projected_points.reshape(-1, 2)
        errors = np.linalg.norm(self.image_points - projected_points, axis=1)

        return float(np.mean(errors))

    def project_to_image(self, world_x: float, world_y: float, world_z: float) -> Tuple[float, float]:
        """
        3D世界座標を画像座標に投影

        Args:
            world_x, world_y, world_z: 実世界座標（メートル）

        Returns:
            (image_x, image_y): 画像座標（ピクセル）
        """
        object_point = np.array([[world_x, world_y, world_z]], dtype=np.float32)

        image_point, _ = cv2.projectPoints(
            object_point,
            self.rvec,
            self.tvec,
            self.camera_matrix,
            self.dist_coeffs
        )

        return float(image_point[0][0][0]), float(image_point[0][0][1])

    def get_ray_direction(self, image_x: float, image_y: float) -> np.ndarray:
        """
        画像座標からカメラレイの方向ベクトルを取得

        Args:
            image_x, image_y: 画像座標（ピクセル）

        Returns:
            ray_dir: 正規化された方向ベクトル（カメラ座標系）
        """
        # 画像座標を正規化座標に変換
        point_2d = np.array([[image_x, image_y]], dtype=np.float32)
        point_normalized = cv2.undistortPoints(
            point_2d,
            self.camera_matrix,
            self.dist_coeffs
        )

        # 正規化座標からレイ方向を取得（カメラ座標系でZ=1の平面）
        x_norm, y_norm = point_normalized[0][0]
        ray_camera = np.array([x_norm, y_norm, 1.0])

        # カメラ座標系から世界座標系に変換
        ray_world = self.R.T @ ray_camera

        # 正規化
        ray_world = ray_world / np.linalg.norm(ray_world)

        return ray_world

    def intersect_ground_plane(
        self,
        image_x: float,
        image_y: float,
        plane_z: float = 0.0
    ) -> Optional[Tuple[float, float, float]]:
        """
        画像座標からレイを飛ばし、地面平面（Z=plane_z）との交点を求める

        Args:
            image_x, image_y: 画像座標（ピクセル）
            plane_z: 地面平面のZ座標（デフォルト0）

        Returns:
            (x, y, z): 交点の3D座標、または None（交点がない場合）
        """
        # カメラ位置
        camera_pos = self.get_camera_position()

        # レイの方向
        ray_dir = self.get_ray_direction(image_x, image_y)

        # 平面の法線ベクトル（Z軸に平行）
        plane_normal = np.array([0, 0, 1])

        # 平面上の点
        plane_point = np.array([0, 0, plane_z])

        # レイと平面の交点を計算
        # P = camera_pos + t * ray_dir
        # (P - plane_point) · plane_normal = 0
        # t = (plane_point - camera_pos) · plane_normal / (ray_dir · plane_normal)

        denom = np.dot(ray_dir, plane_normal)

        # レイが平面に平行な場合
        if abs(denom) < 1e-6:
            return None

        t = np.dot(plane_point - camera_pos, plane_normal) / denom

        # カメラの後ろの場合
        if t < 0:
            return None

        # 交点を計算
        intersection = camera_pos + t * ray_dir

        return float(intersection[0]), float(intersection[1]), float(intersection[2])

    def estimate_height_from_calibration(
        self,
        image_x: float,
        image_y: float
    ) -> Optional[float]:
        """
        高さキャリブレーションを使って画像座標から高さを推定

        Args:
            image_x, image_y: 画像座標（ピクセル）

        Returns:
            height: 推定された高さ（メートル）、または None
        """
        if not self.height_calibration:
            return None

        ground = self.height_calibration['ground_point']
        reference = self.height_calibration['reference_point']

        # 画像上でのY座標の差（ピクセル）
        pixel_diff_reference = ground['image_y'] - reference['image_y']

        # 実世界でのZ座標の差（メートル）
        real_height_diff = reference['world_z'] - ground['world_z']

        if abs(pixel_diff_reference) < 1e-6:
            return None

        # ピクセル/メートル比を計算
        pixels_per_meter = pixel_diff_reference / real_height_diff

        # 推定したい点の地面からのY座標の差（ピクセル）
        pixel_diff_target = ground['image_y'] - image_y

        # 高さを推定
        estimated_height = pixel_diff_target / pixels_per_meter

        # 地面より下の場合は0にクランプ
        return max(0.0, float(estimated_height))

    def transform_to_3d(
        self,
        image_x: float,
        image_y: float,
        use_height_calibration: bool = True
    ) -> Tuple[float, float, float]:
        """
        画像座標から3D世界座標を推定

        方法:
        1. 地面平面（Z=0）との交点でXY座標を取得
        2. 高さキャリブレーションがあれば、それを使ってZ座標を推定
           なければZ=0

        Args:
            image_x, image_y: 画像座標（ピクセル）
            use_height_calibration: 高さキャリブレーションを使用するか

        Returns:
            (x, y, z): 3D世界座標（メートル）
        """
        # 1. 地面平面との交点でXY座標を取得
        intersection = self.intersect_ground_plane(image_x, image_y, plane_z=0.0)

        if intersection is None:
            # 交点がない場合はカメラ正面を使う
            raise ValueError(f"No intersection with ground plane for pixel ({image_x}, {image_y})")

        world_x, world_y, _ = intersection

        # 2. Z座標（高さ）を推定
        world_z = 0.0
        if use_height_calibration and self.height_calibration:
            estimated_z = self.estimate_height_from_calibration(image_x, image_y)
            if estimated_z is not None:
                world_z = estimated_z

        return world_x, world_y, world_z


def load_calibration_from_json(filepath) -> Dict:
    """
    JSONファイルからキャリブレーションデータを読み込み

    Args:
        filepath: キャリブレーションJSONファイルのパス（文字列またはPath）

    Returns:
        calibration_data: キャリブレーションデータ
    """
    # 文字列の場合はPathに変換
    if isinstance(filepath, str):
        filepath = Path(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def create_camera_calibration(
    calibration_path,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None
) -> CameraCalibration:
    """
    キャリブレーションファイルからCameraCalibrationオブジェクトを作成

    Args:
        calibration_path: キャリブレーションJSONファイルのパス（文字列またはPath）
        camera_matrix: カメラ内部パラメータ（オプション）
        dist_coeffs: レンズ歪み係数（オプション）

    Returns:
        camera_calib: CameraCalibrationオブジェクト
    """
    calibration_data = load_calibration_from_json(calibration_path)
    return CameraCalibration(calibration_data, camera_matrix, dist_coeffs)


# テスト用
if __name__ == "__main__":
    print("Camera PnP Utils Test")

    # サンプルキャリブレーションデータ
    sample_calibration = {
        "video_id": "test_video",
        "calibration_date": "2025-11-02",
        "frame_number": 0,
        "points": [
            {"image_x": 640, "image_y": 480, "world_x": 0, "world_y": 0, "world_z": 0, "label": "ホームベース"},
            {"image_x": 650, "image_y": 300, "world_x": 0, "world_y": 3, "world_z": 0, "label": "投手方向3m"},
            {"image_x": 750, "image_y": 320, "world_x": 1.5, "world_y": 2, "world_z": 0, "label": "右側1.5m"},
            {"image_x": 550, "image_y": 320, "world_x": -1.5, "world_y": 2, "world_z": 0, "label": "左側1.5m"},
        ],
        "frame_dimensions": {"width": 1920, "height": 1080},
        "height_calibration": {
            "ground_point": {"image_x": 640, "image_y": 480, "world_z": 0, "label": "地面"},
            "reference_point": {"image_x": 640, "image_y": 300, "world_z": 1.8, "label": "頭頂"}
        }
    }

    # キャリブレーション作成
    camera = CameraCalibration(sample_calibration)

    # テスト1: ホームベースの座標
    print("\nTest 1: ホームベース (640, 480)")
    x, y, z = camera.transform_to_3d(640, 480)
    print(f"  → 3D座標: ({x:.2f}, {y:.2f}, {z:.2f})")

    # テスト2: 高さのある点
    print("\nTest 2: 頭頂部 (640, 300)")
    x, y, z = camera.transform_to_3d(640, 300)
    print(f"  → 3D座標: ({x:.2f}, {y:.2f}, {z:.2f})")

    print("\nTest completed!")
