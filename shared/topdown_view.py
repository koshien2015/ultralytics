"""
真上ビュー可視化モジュール

機能:
- フィールド座標を真上から見た画像に描画
- 野球場のマウンド、ベース、ストライクゾーンの描画
- 軌跡の可視化
"""

import cv2
import numpy as np
from perspective_transform import BASEBALL_FIELD


class TopDownView:
    """真上ビュー可視化クラス"""

    def __init__(self, width=800, height=1000, meters_per_pixel=0.02, view_range_x=5.0, view_range_z=15.0):
        """
        Args:
            width: 画像幅（ピクセル）
            height: 画像高さ（ピクセル）
            meters_per_pixel: 1ピクセルあたりのメートル数（スケール、デフォルト0.02=2cm/pixel）
            view_range_x: 横方向の表示範囲（メートル、ホームベースから左右）
            view_range_z: 奥行き方向の表示範囲（メートル、ホームベースから投手方向）
        """
        self.width = width
        self.height = height
        self.meters_per_pixel = meters_per_pixel
        self.view_range_x = view_range_x  # 横方向の表示範囲
        self.view_range_z = view_range_z  # 奥行き方向の表示範囲
        self.origin_x = width // 2  # ホームベースをX中央に配置
        self.origin_y = height - 100  # ホームベースをY下部に配置

        # 色設定（BGR）
        self.COLOR_FIELD = (34, 139, 34)  # フォレストグリーン
        self.COLOR_DIRT = (139, 90, 43)  # ブラウン
        self.COLOR_BASE = (255, 255, 255)  # 白
        self.COLOR_MOUND = (180, 120, 60)  # ライトブラウン
        self.COLOR_STRIKE_ZONE = (0, 255, 255)  # イエロー
        self.COLOR_TRAJECTORY = (255, 0, 255)  # マゼンタ
        self.COLOR_GRID = (100, 100, 100)  # グレー

        # 軌跡データ
        self.current_trajectory = []  # [(x, z), ...] フィールド座標

    def field_to_pixel(self, x_field, z_field):
        """
        フィールド座標 → 画像ピクセル座標

        Args:
            x_field: 横方向の位置（メートル、ホームベース中心が0）
            z_field: 奥行き方向の位置（メートル、ホームベース=0、投手方向=正）

        Returns:
            (x_pixel, y_pixel): 画像座標（ピクセル）
        """
        x_pixel = int(self.origin_x + x_field / self.meters_per_pixel)
        y_pixel = int(self.origin_y - z_field / self.meters_per_pixel)  # Y軸は上が正
        return (x_pixel, y_pixel)

    def draw_field(self):
        """野球場のフィールドを描画"""
        # 背景（芝生）
        field_img = np.full((self.height, self.width, 3), self.COLOR_FIELD, dtype=np.uint8)

        # グリッド線（1メートル間隔）
        # 横方向のグリッド（X軸）
        x_range = int(self.view_range_x) + 5  # 少し余裕を持たせる
        for i in range(-x_range, x_range + 1):
            x_pixel = self.origin_x + int(i / self.meters_per_pixel)
            if 0 <= x_pixel < self.width:
                cv2.line(field_img, (x_pixel, 0), (x_pixel, self.height), self.COLOR_GRID, 1)

        # 奥行き方向のグリッド（Z軸）
        z_range = int(self.view_range_z) + 5  # 少し余裕を持たせる
        for i in range(0, z_range + 1):
            y_pixel = self.origin_y - int(i / self.meters_per_pixel)
            if 0 <= y_pixel < self.height:
                cv2.line(field_img, (0, y_pixel), (self.width, y_pixel), self.COLOR_GRID, 1)

        # ホームベース
        home_x, home_y = self.field_to_pixel(0, 0)
        cv2.circle(field_img, (home_x, home_y), 8, self.COLOR_BASE, -1)
        cv2.putText(field_img, "HOME", (home_x + 12, home_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_BASE, 2)

        # ピッチャーマウンド
        mound_x, mound_y = self.field_to_pixel(0, BASEBALL_FIELD['pitcher_to_home'])
        # キャップ野球用にマウンド半径を調整（通常野球の半分程度）
        mound_radius_m = 1.4  # メートル（通常野球は2.74m）
        mound_radius = int(mound_radius_m / self.meters_per_pixel)
        cv2.circle(field_img, (mound_x, mound_y), mound_radius, self.COLOR_MOUND, -1)
        cv2.circle(field_img, (mound_x, mound_y), 8, self.COLOR_BASE, -1)
        cv2.putText(field_img, "MOUND", (mound_x + 12, mound_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_BASE, 2)

        # ストライクゾーン（簡易表示）
        strike_width = BASEBALL_FIELD['strike_zone_width']
        strike_left_x, strike_y = self.field_to_pixel(-strike_width / 2, 0)
        strike_right_x, _ = self.field_to_pixel(strike_width / 2, 0)
        strike_rect_width = strike_right_x - strike_left_x
        # キャップ野球用にストライクゾーン高さを調整（子供用）
        strike_rect_height = int(0.35 / self.meters_per_pixel)  # 高さ約35cm（キャップ野球用）
        cv2.rectangle(field_img,
                     (strike_left_x, strike_y - strike_rect_height),
                     (strike_right_x, strike_y),
                     self.COLOR_STRIKE_ZONE, 2)

        # スケール表示
        scale_text = f"Scale: {self.meters_per_pixel*100:.1f}cm/px"
        cv2.putText(field_img, scale_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return field_img

    def add_trajectory_point(self, x_field, z_field):
        """
        軌跡に点を追加

        Args:
            x_field: 横方向の位置（メートル）
            z_field: 奥行き方向の位置（メートル）
        """
        self.current_trajectory.append((x_field, z_field))

    def clear_trajectory(self):
        """軌跡をクリア"""
        self.current_trajectory = []

    def draw_trajectory(self, field_img, trajectory=None, color=None, thickness=2):
        """
        軌跡を描画

        Args:
            field_img: 描画対象の画像
            trajectory: [(x, z), ...] フィールド座標、Noneの場合はcurrent_trajectoryを使用
            color: 軌跡の色（BGR）、Noneの場合はデフォルト色
            thickness: 線の太さ

        Returns:
            軌跡を描画した画像
        """
        if trajectory is None:
            trajectory = self.current_trajectory

        if not trajectory or len(trajectory) < 2:
            return field_img

        if color is None:
            color = self.COLOR_TRAJECTORY

        # ピクセル座標に変換
        pixel_points = [self.field_to_pixel(x, z) for x, z in trajectory]

        # 線を描画
        for i in range(len(pixel_points) - 1):
            cv2.line(field_img, pixel_points[i], pixel_points[i+1], color, thickness)

        # 各点を円で描画
        for i, pt in enumerate(pixel_points):
            radius = 5 if i == len(pixel_points) - 1 else 3  # 最後の点は大きく
            cv2.circle(field_img, pt, radius, color, -1)

        return field_img

    def draw_ball_position(self, field_img, x_field, z_field, label=None):
        """
        ボールの現在位置を描画

        Args:
            field_img: 描画対象の画像
            x_field: 横方向の位置（メートル）
            z_field: 奥行き方向の位置（メートル）
            label: ラベルテキスト（省略可）

        Returns:
            描画した画像
        """
        x_pixel, y_pixel = self.field_to_pixel(x_field, z_field)

        # ボールを描画
        cv2.circle(field_img, (x_pixel, y_pixel), 10, (0, 0, 255), -1)  # 赤
        cv2.circle(field_img, (x_pixel, y_pixel), 10, (255, 255, 255), 2)  # 白枠

        # ラベル
        if label:
            cv2.putText(field_img, label, (x_pixel + 15, y_pixel - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 座標表示
        coord_text = f"({x_field:.2f}m, {z_field:.2f}m)"
        cv2.putText(field_img, coord_text, (x_pixel + 15, y_pixel + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return field_img

    def get_frame(self, ball_position=None, show_trajectory=True):
        """
        現在のフレームを取得

        Args:
            ball_position: (x_field, z_field) ボールの現在位置、Noneの場合は描画しない
            show_trajectory: 軌跡を表示するか

        Returns:
            描画されたフレーム
        """
        frame = self.draw_field()

        if show_trajectory and self.current_trajectory:
            frame = self.draw_trajectory(frame)

        if ball_position:
            x_field, z_field = ball_position
            frame = self.draw_ball_position(frame, x_field, z_field)

        return frame
