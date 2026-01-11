"""
射影変換モジュール

機能:
- カメラ画像座標からフィールド座標（真上から見た位置）への変換
- キャリブレーション用のデータ管理
- 軌跡データの2D変換
"""

import cv2
import numpy as np
import json
import os


class PerspectiveTransformer:
    """射影変換クラス（画像座標 ⇔ フィールド座標）"""

    def __init__(self, calibration_file=None):
        """
        Args:
            calibration_file: キャリブレーションデータのJSONファイルパス（省略可）
        """
        self.H = None  # 射影変換行列（画像 → フィールド）
        self.H_inv = None  # 逆変換行列（フィールド → 画像）
        self.image_points = None  # キャリブレーション点（画像座標）
        self.real_points = None  # キャリブレーション点（フィールド座標）
        self.is_calibrated = False

        if calibration_file and os.path.exists(calibration_file):
            self.load_calibration(calibration_file)

    def calibrate(self, image_points, real_points):
        """
        キャリブレーションを実行

        Args:
            image_points: 画像上の対応点 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]（4点以上）
            real_points: フィールド上の対応点 [[x1, z1], [x2, z2], [x3, z3], [x4, z4]]（メートル単位）

        Returns:
            bool: 成功したらTrue
        """
        if len(image_points) < 4 or len(real_points) < 4:
            print("❌ Error: At least 4 points are required for calibration")
            return False

        if len(image_points) != len(real_points):
            print("❌ Error: Number of image points and real points must match")
            return False

        # numpy配列に変換
        self.image_points = np.array(image_points, dtype=np.float32)
        self.real_points = np.array(real_points, dtype=np.float32)

        try:
            # 射影変換行列を計算
            self.H = cv2.getPerspectiveTransform(self.image_points[:4], self.real_points[:4])
            self.H_inv = cv2.getPerspectiveTransform(self.real_points[:4], self.image_points[:4])
            self.is_calibrated = True
            print("✅ Perspective transformation calibrated successfully")
            return True
        except Exception as e:
            print(f"❌ Error during calibration: {e}")
            return False

    def image_to_field(self, x_pixel, y_pixel):
        """
        画像座標 → フィールド座標（真上から見た位置）

        Args:
            x_pixel, y_pixel: 画像上の座標（ピクセル）

        Returns:
            (x_real, z_real): フィールド座標（メートル）、または None（未キャリブレーション時）
                x_real: 横方向の位置（メートル）
                z_real: 奥行き方向の位置（メートル、投手→捕手方向）
        """
        if not self.is_calibrated:
            return None

        point = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.H)
        x_real, z_real = transformed[0][0]

        return (float(x_real), float(z_real))

    def field_to_image(self, x_real, z_real):
        """
        フィールド座標 → 画像座標

        Args:
            x_real, z_real: フィールド座標（メートル）

        Returns:
            (x_pixel, y_pixel): 画像座標（ピクセル）、または None（未キャリブレーション時）
        """
        if not self.is_calibrated:
            return None

        point = np.array([[[x_real, z_real]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.H_inv)
        x_pixel, y_pixel = transformed[0][0]

        return (float(x_pixel), float(y_pixel))

    def save_calibration(self, output_path):
        """
        キャリブレーションデータをJSONファイルに保存

        Args:
            output_path: 出力ファイルパス
        """
        if not self.is_calibrated:
            print("❌ Error: No calibration data to save")
            return False

        data = {
            'image_points': self.image_points.tolist(),
            'real_points': self.real_points.tolist(),
            'H': self.H.tolist(),
            'H_inv': self.H_inv.tolist()
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Calibration data saved to {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving calibration: {e}")
            return False

    def load_calibration(self, input_path):
        """
        キャリブレーションデータをJSONファイルから読み込み

        Args:
            input_path: 入力ファイルパス
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.image_points = np.array(data['image_points'], dtype=np.float32)
            self.real_points = np.array(data['real_points'], dtype=np.float32)
            self.H = np.array(data['H'], dtype=np.float32)
            self.H_inv = np.array(data['H_inv'], dtype=np.float32)
            self.is_calibrated = True
            print(f"✅ Calibration data loaded from {input_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            return False

    def transform_trajectory(self, trajectory_pixels):
        """
        軌跡データ（画像座標）をフィールド座標に一括変換

        Args:
            trajectory_pixels: [(x1, y1), (x2, y2), ...] 画像座標のリスト

        Returns:
            [(x1, z1), (x2, z2), ...] フィールド座標のリスト、または None
        """
        if not self.is_calibrated:
            return None

        if not trajectory_pixels:
            return []

        # numpy配列に変換
        points = np.array(trajectory_pixels, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, self.H)

        # リスト形式で返す
        return [(float(pt[0][0]), float(pt[0][1])) for pt in transformed]


# 野球場の標準的な距離（メートル）
BASEBALL_FIELD = {
    'pitcher_to_home': 9.22,  # 18.44の半分
    'base_distance': 4,    # 塁間
    'strike_zone_width': 0.432,  # 17インチ
    'home_plate_width': 0.432,   # 17インチ
}


def get_default_field_points():
    """
    野球場の標準的な基準点（真上から見た座標）を返す

    Returns:
        dict: {'name': (x, z), ...} フィールド座標（メートル、ホームベース中心が原点）
    """
    return {
        'home_plate': (0.0, 0.0),
        'pitcher_mound': (0.0, BASEBALL_FIELD['pitcher_to_home']),
        'first_baseの半分': (BASEBALL_FIELD['base_distance'], BASEBALL_FIELD['base_distance']),
        'second_base': (0.0, BASEBALL_FIELD['base_distance'] * np.sqrt(2)),
        'third_baseの半分': (-BASEBALL_FIELD['base_distance'], BASEBALL_FIELD['base_distance']),
    }
