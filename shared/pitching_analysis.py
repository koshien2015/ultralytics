"""
ピッチング解析モジュール

機能:
- ストライクゾーン推定（打者と捕手のバウンディングボックスから）
- リリースポイント検出（クラス1→5の遷移）
- 3D座標計算（X: 横, Y: 高さ, Z: 経過時間）
"""

import cv2
import numpy as np
from ultralytics import YOLO


# 検出クラスID
CLASS_BALL = 0  # petbottle_cap
CLASS_PITCHER_MOTION = 1  # pitcher_motion
CLASS_PITCHER_RELEASE = 5  # pitcher_release


class PitchingAnalyzer:
    """ピッチング解析クラス"""

    def __init__(self, strike_zone_width_px=150, strike_zone_center_x=None):
        """
        Args:
            strike_zone_width_px: ストライクゾーンの幅（ピクセル）
            strike_zone_center_x: ストライクゾーンの中央X座標（ピクセル）、Noneの場合は初回検出時に自動設定
        """
        self.strike_zone = None
        self.release_frame = None
        self.fps = None

        # 投手の状態追跡
        self.pitcher_state = None  # 'motion' or 'release'
        self.prev_pitcher_state = None

        # ストライクゾーンの固定値
        self.STRIKE_ZONE_WIDTH_PX = strike_zone_width_px
        self.STRIKE_ZONE_CENTER_X = strike_zone_center_x  # Noneの場合は初回捕手検出時に設定

        # カメラ角度（度）- 投手-捕手ラインに対する回転角度
        self.camera_angle = 90.0  # デフォルトは正面（90度）
        self.camera_angle_locked = False  # カメラ角度を固定したか

        # 軌跡データの記録
        self.pitches = []  # 全投球のリスト
        self.current_pitch = None  # 現在の投球データ

    def estimate_strike_zone_from_batter_catcher(self, batter_bbox, catcher_bbox):
        """
        打者とキャッチャーのバウンディングボックスからストライクゾーンを推定
        中央X座標は固定値、高さは打者に追従

        Args:
            batter_bbox: (x1, y1, x2, y2) 打者のバウンディングボックス
            catcher_bbox: (x1, y1, x2, y2) キャッチャーのバウンディングボックス

        Returns:
            strike_zone: {'left': x1, 'right': x2, 'top': y1, 'bottom': y2, 'center_x': cx, 'center_y': cy, 'zone_width': width}
        """
        # 打者のバウンディングボックスから高さを算出（常時追従）
        batter_x1, batter_y1, batter_x2, batter_y2 = batter_bbox
        batter_height = batter_y2 - batter_y1

        # ストライクゾーンの高さ（打者のバウンディングボックスから）
        # 上端: バウンディングボックスの40%位置（肩あたり）
        strike_top = float(batter_y1 + batter_height * 0.4)
        # 下端: バウンディングボックスの75%位置（膝あたり）
        strike_bottom = float(batter_y1 + batter_height * 0.75)

        # ストライクゾーンの中央X座標を取得（初回のみ捕手から設定）
        if self.STRIKE_ZONE_CENTER_X is None:
            catcher_x1, catcher_y1, catcher_x2, catcher_y2 = catcher_bbox
            self.STRIKE_ZONE_CENTER_X = float((catcher_x1 + catcher_x2) / 2)
            print(f"✓ Strike zone center set to X={self.STRIKE_ZONE_CENTER_X:.1f}px")

        center_x = self.STRIKE_ZONE_CENTER_X
        strike_width = self.STRIKE_ZONE_WIDTH_PX

        # ストライクゾーンを更新（高さのみ常時更新、中央は固定）
        self.strike_zone = {
            'left': float(center_x - strike_width / 2),
            'right': float(center_x + strike_width / 2),
            'top': strike_top,
            'bottom': strike_bottom,
            'center_x': float(center_x),
            'center_y': float((strike_top + strike_bottom) / 2),
            'zone_width': float(strike_width)  # 正規化用
        }

        return self.strike_zone


    def estimate_camera_angle(self, pitcher_bbox, catcher_bbox, debug=False):
        """
        投手と捕手の位置からカメラ角度を推定

        Args:
            pitcher_bbox: (x1, y1, x2, y2) 投手のバウンディングボックス
            catcher_bbox: (x1, y1, x2, y2) 捕手のバウンディングボックス
            debug: デバッグ情報を表示するか

        Returns:
            angle: カメラ角度（度）、投手-捕手ラインと画面縦方向のなす角
                   90度 = 真横から（投手-捕手が画面上で縦に並ぶ）
                   0度 = 斜めから（投手-捕手が画面上で横に並ぶ）
        """
        if pitcher_bbox is None or catcher_bbox is None:
            return 90.0  # デフォルトは正面

        # 投手と捕手の位置を取得（X:中心、Y:足元=bottom）
        pitcher_x = (pitcher_bbox[0] + pitcher_bbox[2]) / 2  # X中心
        pitcher_y = pitcher_bbox[3]  # Y足元（bottom）
        catcher_x = (catcher_bbox[0] + catcher_bbox[2]) / 2  # X中心
        catcher_y = catcher_bbox[3]  # Y足元（bottom）

        # 投手から捕手へのベクトル
        dx = catcher_x - pitcher_x
        dy = catcher_y - pitcher_y

        # 画面縦方向（Y軸）からの傾き角度を計算
        # 理想的には dx=0, dy>0（真下）で90度
        # arctan2(dx, dy) はY軸からの角度（度単位）
        angle_from_vertical = np.degrees(np.arctan2(abs(dx), abs(dy)))

        # カメラ角度 = 90度 - Y軸からの傾き
        # dx=0（縦に並ぶ）→ angle_from_vertical=0° → camera_angle=90°
        # dx=dy（45度傾く）→ angle_from_vertical=45° → camera_angle=45°
        camera_angle = 90.0 - angle_from_vertical

        # 0～90度に制限
        camera_angle = np.clip(camera_angle, 0, 90)

        # デバッグ情報を表示
        if debug:
            print(f"🔍 Camera Angle Debug:")
            print(f"   Pitcher: ({pitcher_x:.1f}, {pitcher_y:.1f})")
            print(f"   Catcher: ({catcher_x:.1f}, {catcher_y:.1f})")
            print(f"   Vector: dx={dx:.1f}, dy={dy:.1f}")
            print(f"   Angle from vertical: {angle_from_vertical:.1f}°")
            print(f"   Camera angle: {camera_angle:.1f}°")

        return float(camera_angle)

    def detect_release(self, detection_results, frame_number):
        """
        リリースポイントを検出（クラス1→5の遷移）

        Args:
            detection_results: YOLOの検出結果
            frame_number: 現在のフレーム番号

        Returns:
            is_release: このフレームがリリースフレームならTrue
        """
        # 現在のフレームの投手状態を検出
        current_state = None

        for result in detection_results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])

                if cls == CLASS_PITCHER_MOTION:
                    current_state = 'motion'
                    break
                elif cls == CLASS_PITCHER_RELEASE:
                    current_state = 'release'
                    break

        # motion → release の遷移を検出
        is_release = False
        if self.prev_pitcher_state == 'motion' and current_state == 'release':
            self.release_frame = frame_number
            is_release = True

        self.prev_pitcher_state = current_state

        return is_release

    def get_elapsed_time(self, current_frame):
        """
        リリースからの経過時間を取得

        Args:
            current_frame: 現在のフレーム番号

        Returns:
            elapsed_time: 経過時間（秒）、またはNone
        """
        if self.fps is None or self.release_frame is None:
            return None

        if current_frame < self.release_frame:
            return None

        elapsed_frames = current_frame - self.release_frame
        elapsed_time = elapsed_frames / self.fps

        return elapsed_time

    def normalize_position(self, x_pixel, y_pixel):
        """
        ピクセル座標を正規化座標に変換（ストライクゾーン幅を基準、カメラ角度を考慮）

        Args:
            x_pixel, y_pixel: ピクセル座標

        Returns:
            (x_norm, y_norm): 正規化座標
                x_norm: ストライクゾーン中心からの距離をストライクゾーン幅で正規化
                        0 = ストライクゾーン中心, ±0.5 = ストライクゾーン端
                y_norm: ストライクゾーンを基準に正規化
                        0 = ストライクゾーン下端, 1 = ストライクゾーン上端
            または None（ストライクゾーンが未検出）
        """
        if self.strike_zone is None:
            return None

        zone_width = self.strike_zone['zone_width']
        center_x = self.strike_zone['center_x']
        strike_top = self.strike_zone['top']
        strike_bottom = self.strike_zone['bottom']

        # X座標: ストライクゾーン中心からの距離をストライクゾーン幅で正規化
        # カメラ角度を考慮して補正
        x_offset = x_pixel - center_x
        x_offset_corrected = x_offset / np.cos(np.radians(90 - self.camera_angle))
        x_norm = x_offset_corrected / zone_width

        # Y座標: ストライクゾーン高さで正規化
        strike_height = strike_bottom - strike_top
        y_norm = (strike_bottom - y_pixel) / strike_height  # 上下反転（画像座標は上が0）

        return (float(x_norm), float(y_norm))

    def is_strike(self, x_pixel, y_pixel):
        """
        ストライクかボールかを判定（ホームベース通過時点）

        Args:
            x_pixel, y_pixel: ボール位置（ピクセル座標）

        Returns:
            bool: ストライクならTrue、判定不可ならNone
        """
        if self.strike_zone is None:
            return None

        return (self.strike_zone['left'] <= x_pixel <= self.strike_zone['right'] and
                self.strike_zone['top'] <= y_pixel <= self.strike_zone['bottom'])

    def draw_strike_zone(self, frame):
        """
        ストライクゾーンを描画

        Args:
            frame: 描画対象のフレーム

        Returns:
            frame: ストライクゾーンを描画したフレーム
        """
        if self.strike_zone is None:
            return frame

        # ストライクゾーンを矩形で描画
        pt1 = (int(self.strike_zone['left']), int(self.strike_zone['top']))
        pt2 = (int(self.strike_zone['right']), int(self.strike_zone['bottom']))
        cv2.rectangle(frame, pt1, pt2, (255, 255, 0), 2)  # 黄色

        # 中心線を描画
        center_x = int(self.strike_zone['center_x'])
        center_y = int(self.strike_zone['center_y'])
        cv2.line(frame, (center_x, int(self.strike_zone['top'])),
                (center_x, int(self.strike_zone['bottom'])), (255, 255, 0), 1)

        return frame

    def draw_info(self, frame, elapsed_time=None, ball_3d=None):
        """
        解析情報をフレームに描画

        Args:
            frame: 描画対象のフレーム
            elapsed_time: リリースからの経過時間（秒）
            ball_3d: ボールの3D座標 (x_norm, y_norm, z)

        Returns:
            frame: 情報を描画したフレーム
        """
        y_pos = 30

        # リリース情報
        if self.release_frame is not None:
            cv2.putText(frame, f"Release Frame: {self.release_frame}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_pos += 25

        # 経過時間
        if elapsed_time is not None:
            cv2.putText(frame, f"Time: {elapsed_time:.3f}s", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_pos += 25

        # ストライクゾーン情報
        if self.strike_zone is not None:
            cv2.putText(frame, "Strike Zone: Detected", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 25

        # カメラ角度
        cv2.putText(frame, f"Camera Angle: {self.camera_angle:.1f}deg", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 25

        # 3D座標
        if ball_3d is not None:
            x_norm, y_norm, z = ball_3d
            cv2.putText(frame, f"Ball 3D: X={x_norm:.3f} Y={y_norm:.3f} Z={z:.3f}s", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

        return frame

    def set_fps(self, fps):
        """FPSを設定"""
        self.fps = fps

    def detect_batter_catcher_pitcher(self, detection_results):
        """
        検出結果からバッター、キャッチャー、投手を検出

        Args:
            detection_results: YOLOの検出結果

        Returns:
            (batter_bbox, catcher_bbox, pitcher_bbox): 検出されたバウンディングボックス（Noneの場合もあり）
        """
        batter_bbox = None
        catcher_bbox = None
        pitcher_bbox = None

        for result in detection_results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if (cls == 2 or cls == 6) and batter_bbox is None:  # batter
                    batter_bbox = (x1, y1, x2, y2)
                elif (cls == 4 or cls == 7) and catcher_bbox is None:  # catcher
                    catcher_bbox = (x1, y1, x2, y2)
                elif (cls == 1 or cls == 5) and pitcher_bbox is None:  # pitcher (motion or release)
                    pitcher_bbox = (x1, y1, x2, y2)

                # 全部見つかったら終了
                if batter_bbox is not None and catcher_bbox is not None and pitcher_bbox is not None:
                    break

        # ストライクゾーンを常に更新（高さは打者に追従）
        if batter_bbox and catcher_bbox:
            self.estimate_strike_zone_from_batter_catcher(batter_bbox, catcher_bbox)
        else:
            # どちらかが検出されない場合はクリア
            self.strike_zone = None

        # カメラ角度を推定（最初のmotion検出時にロック）
        if pitcher_bbox and catcher_bbox and not self.camera_angle_locked:
            # 投手がmotion状態の時のみカメラ角度を設定
            for result in detection_results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == CLASS_PITCHER_MOTION:  # motion状態
                        self.camera_angle = self.estimate_camera_angle(pitcher_bbox, catcher_bbox, debug=True)
                        self.camera_angle_locked = True
                        print(f"✓ Camera angle locked at {self.camera_angle:.1f}°")
                        break

        return batter_bbox, catcher_bbox, pitcher_bbox

    def update(self, detection_results, frame_number):
        """
        検出結果を受け取って全ての解析を実行

        Args:
            detection_results: YOLOの検出結果
            frame_number: 現在のフレーム番号

        Returns:
            dict: 解析結果
                - is_release: リリースが検出されたか
                - elapsed_time: リリースからの経過時間
                - batter_bbox: バッターのバウンディングボックス
                - catcher_bbox: キャッチャーのバウンディングボックス
                - pitcher_bbox: 投手のバウンディングボックス
                - ball_3d: ボールの3D座標 (x_norm, y_norm, z) または None
                - camera_angle: カメラ角度（度）
        """
        # バッター、キャッチャー、投手を検出してストライクゾーン推定＆カメラ角度推定
        # （常にストライクゾーンを更新、高さは打者に追従）
        batter_bbox, catcher_bbox, pitcher_bbox = self.detect_batter_catcher_pitcher(detection_results)

        # リリース検出
        is_release = self.detect_release(detection_results, frame_number)

        # リリース時に新しい投球を開始
        if is_release:
            if self.current_pitch is not None:
                # 前の投球を保存
                self.pitches.append(self.current_pitch)

            # 新しい投球を開始
            self.current_pitch = {
                'pitch_id': len(self.pitches) + 1,
                'release_frame': self.release_frame,
                'trajectory': []
            }

        # 経過時間を取得
        elapsed_time = self.get_elapsed_time(frame_number)

        # ボール検出と3D座標計算（クラス0）
        ball_3d = None
        for result in detection_results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:  # petbottle_cap (ball)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # XY正規化座標
                    normalized = self.normalize_position(center_x, center_y)
                    if normalized and elapsed_time is not None:
                        x_norm, y_norm = normalized
                        z = elapsed_time  # Z座標は経過時間
                        ball_3d = (x_norm, y_norm, z)

                        # 軌跡データを記録
                        if self.current_pitch is not None:
                            self.current_pitch['trajectory'].append({
                                'frame': frame_number,
                                'time': elapsed_time,
                                'x': x_norm,
                                'y': y_norm,
                                'z': z
                            })
                    break  # 最初のボールのみ

        return {
            'is_release': is_release,
            'elapsed_time': elapsed_time,
            'batter_bbox': batter_bbox,
            'catcher_bbox': catcher_bbox,
            'pitcher_bbox': pitcher_bbox,
            'ball_3d': ball_3d,
            'camera_angle': self.camera_angle
        }

    def draw(self, frame, frame_number, ball_3d=None, draw_strike_zone=True, draw_info=True):
        """
        解析結果をフレームに描画

        Args:
            frame: 描画対象のフレーム
            frame_number: 現在のフレーム番号
            ball_3d: ボールの3D座標 (x_norm, y_norm, z)
            draw_strike_zone: ストライクゾーンを描画するか
            draw_info: 解析情報を描画するか

        Returns:
            frame: 描画後のフレーム
        """
        # ストライクゾーンを描画
        if draw_strike_zone and self.strike_zone is not None:
            frame = self.draw_strike_zone(frame)

        # 解析情報を描画
        if draw_info:
            elapsed_time = self.get_elapsed_time(frame_number)
            frame = self.draw_info(frame, elapsed_time, ball_3d)

        return frame

    def export_to_json(self, output_path, video_file=None):
        """
        軌跡データをJSONファイルに出力

        Args:
            output_path: 出力ファイルパス
            video_file: 動画ファイル名（メタデータ用）
        """
        import json

        # 最後の投球を保存
        if self.current_pitch is not None and len(self.current_pitch['trajectory']) > 0:
            self.pitches.append(self.current_pitch)
            self.current_pitch = None

        # メタデータ
        metadata = {
            'video_file': video_file if video_file else 'unknown',
            'fps': self.fps,
            'camera_angle': self.camera_angle
        }

        # ストライクゾーン情報
        if self.strike_zone is not None:
            metadata['strike_zone'] = {
                'center_x': self.strike_zone['center_x'],
                'width': self.strike_zone['zone_width'],
                'top': self.strike_zone['top'],
                'bottom': self.strike_zone['bottom']
            }

        # JSON構造
        output_data = {
            'metadata': metadata,
            'pitches': self.pitches,
            'total_pitches': len(self.pitches)
        }

        # ファイルに書き込み
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Exported {len(self.pitches)} pitch(es) to {output_path}")

    def reset(self):
        """状態をリセット（新しい打席の開始時）"""
        self.release_frame = None
        self.pitcher_state = None
        self.prev_pitcher_state = None
        self.strike_zone = None
        self.STRIKE_ZONE_CENTER_X = None  # 次の打席で再設定
        self.camera_angle = 90.0  # デフォルトに戻す
        self.camera_angle_locked = False  # ロック解除
        self.pitches = []  # 軌跡データもリセット
        self.current_pitch = None
