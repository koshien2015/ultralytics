import cv2
import sys
import os
import numpy as np
from ultralytics import YOLO
import tennis
from pitching_analysis import PitchingAnalyzer
from topdown_view import TopDownView


def draw_neon_polyline(
    frame,
    points,
    glow_color=(0, 255, 255),   # BGR（ここではシアン寄り）
    core_color=(255, 255, 255), # 中心の線の色
    glow_thickness=12,          # 光の太さ
    core_thickness=2,           # 中心線の太さ
    glow_blur=25,               # ぼかし量（奇数）
    glow_intensity=0.8          # 光の強さ
):
    """
    ネオン風の軌跡を描画する

    Args:
        frame: 描画対象のフレーム（BGR）
        points: 軌跡の座標リスト [(x1, y1), (x2, y2), ...]
        glow_color: 光の色（BGR）
        core_color: 中心線の色（BGR）
        glow_thickness: 光の太さ
        core_thickness: 中心線の太さ
        glow_blur: ぼかし量（奇数）
        glow_intensity: 光の強さ（0.0〜1.0）

    Returns:
        ネオン効果を適用したフレーム
    """
    if len(points) < 2:
        return frame

    # numpy配列に変換
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    h, w = frame.shape[:2]

    # 1. 黒のレイヤーを作成（光用）
    glow_layer = np.zeros((h, w, 3), dtype=np.uint8)

    # 2. 太いラインを描画（光源）
    cv2.polylines(glow_layer, [pts], isClosed=False, color=glow_color, thickness=glow_thickness, lineType=cv2.LINE_AA)

    # 3. ぼかしてグロウを作る
    glow_layer = cv2.GaussianBlur(glow_layer, (glow_blur, glow_blur), 0)

    # 4. 元フレームと合成（addWeightedで軽く重ねる）
    # frame * 1.0 + glow_layer * glow_intensity
    neon_frame = cv2.addWeighted(frame, 1.0, glow_layer, glow_intensity, 0)

    # 5. 中心のシャープな線を上から引く
    cv2.polylines(neon_frame, [pts], isClosed=False, color=core_color, thickness=core_thickness, lineType=cv2.LINE_AA)

    return neon_frame


# コマンドライン引数から動画パスを取得
if len(sys.argv) > 1:
    original_video = sys.argv[1]
else:
    original_video = "test.mp4"

# マスク動画のパス（マスクを使う場合はここに指定、使わない場合はNone）
mask_video = None

# 検出に使用する動画を決定
detection_video = mask_video if mask_video else original_video

# ファイル名とディレクトリを取得
base_name = os.path.splitext(os.path.basename(detection_video))[0]
video_dir = os.path.dirname(detection_video) if os.path.dirname(detection_video) else "."

# 強調動画のパスを設定
enhance_video = os.path.join(video_dir, f"{base_name}_enhance.mp4")

if mask_video:
    print(f"Original video (for drawing): {original_video}")
    print(f"Mask video (for detection): {mask_video}")
else:
    print(f"Original video: {original_video}")
print(f"Generating enhanced video for detection...")

# tennisモジュールで強調動画を生成（検出用の動画から）
tennis.run(detection_video, enhance_video_path=enhance_video)

print(f"\nLoading YOLO model...")
# YOLOモデルをロード（同じディレクトリ内）
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "yolo8m_20251109.pt")
model = YOLO(model_path)

# 検出対象クラスを指定（空リストで全クラス、数字を指定で特定クラスのみ）
# 0: petbottle_cap (ボール)
# 1: pitcher_motion
# 5: pitcher_release
target_classes = [0]  # 検出したいクラスIDをここに指定

# 軌跡描画設定
DRAW_TRAJECTORY = True  # 軌跡を描画するか
MAX_TRAJECTORY_LENGTH = 60  # 軌跡の最大長さ（フレーム数）
TRAJECTORY_FADE_FRAMES = 60  # 検出されなくなってから何フレームで消えるか

# ネオン効果設定
TRAJECTORY_COLOR = (0, 255, 255)  # 軌跡の色（BGR: シアン）
NEON_CORE_COLOR = (255, 255, 255)  # 中心線の色（BGR: 白）
NEON_GLOW_THICKNESS = 15  # 光の太さ
NEON_CORE_THICKNESS = 3  # 中心線の太さ
NEON_GLOW_BLUR = 25  # ぼかし量（奇数推奨）
NEON_GLOW_INTENSITY = 0.8  # 光の強さ（0.0〜1.0）

# ピッチング解析設定
ENABLE_PITCHING_ANALYSIS = True  # ピッチング解析を有効にするか
DRAW_STRIKE_ZONE = False  # ストライクゾーンを描画するか
STRIKE_ZONE_WIDTH_PX = 50  # ストライクゾーンの幅（ピクセル）
STRIKE_ZONE_CENTER_X = None  # ストライクゾーンの中央X座標（ピクセル）、Noneで初回捕手検出時に自動設定

# 射影変換・真上ビュー設定
CALIBRATION_FILE = 'test2_calibration.json'  # キャリブレーションファイルのパス（例: "test8_calibration.json"）、Noneで無効
SAVE_TOPDOWN_VIDEO = True  # 真上ビューを動画として保存するか（CALIBRATION_FILEが必要、Docker環境推奨）
SHOW_TOPDOWN_VIEW = False  # 真上ビューをリアルタイム表示するか（Docker環境では動作しない）

# 各物体の軌跡を保存する辞書 {物体ID: {'points': [(x, y), ...], 'last_seen': frame_number}}
trajectories = {}
next_object_id = 0

# ピッチング解析の初期化
analyzer = None
if ENABLE_PITCHING_ANALYSIS:
    print("Initializing pitching analyzer...")

    # キャリブレーションファイルのパス解決
    calibration_path = None
    if CALIBRATION_FILE:
        if os.path.isabs(CALIBRATION_FILE):
            calibration_path = CALIBRATION_FILE
        else:
            # 動画と同じディレクトリからの相対パス
            calibration_path = os.path.join(video_dir, CALIBRATION_FILE)

        if not os.path.exists(calibration_path):
            print(f"⚠️  Warning: Calibration file not found: {calibration_path}")
            calibration_path = None

    analyzer = PitchingAnalyzer(
        strike_zone_width_px=STRIKE_ZONE_WIDTH_PX,
        strike_zone_center_x=STRIKE_ZONE_CENTER_X,
        calibration_file=calibration_path
    )

# 真上ビューの初期化
topdown_view = None
topdown_video_writer = None
if (SHOW_TOPDOWN_VIEW or SAVE_TOPDOWN_VIDEO) and analyzer and analyzer.transformer and analyzer.transformer.is_calibrated:
    print("Initializing top-down view...")
    # キャップ野球用のパラメータ
    topdown_view = TopDownView(
        width=800,
        height=1000,
        meters_per_pixel=0.05,  # 5cm/pixel（詳細表示）
        view_range_x=5.0,       # 横方向 ±5m
        view_range_z=15.0       # 奥行き方向 15m（ホームベースから投手方向）
    )

    if SHOW_TOPDOWN_VIEW:
        cv2.namedWindow('Top-Down View')

elif (SHOW_TOPDOWN_VIEW or SAVE_TOPDOWN_VIDEO):
    print("⚠️  Warning: Top-down view requires valid calibration file")

# 強調動画と元動画を開く
cap_enhance = cv2.VideoCapture(enhance_video)
cap_original = cv2.VideoCapture(original_video)

# 動画情報を取得
fps = cap_original.get(cv2.CAP_PROP_FPS)
width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 出力動画の設定
output_path = os.path.join(video_dir, f"{base_name}_detected.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if SAVE_TOPDOWN_VIDEO:
        topdown_output_path = os.path.join(video_dir, f"{base_name}_topdown.mp4")
        topdown_video_writer = cv2.VideoWriter(
            topdown_output_path,
            fourcc,
            fps,
            (topdown_view.width, topdown_view.height)
        )
        print(f"Top-down view will be saved to: {topdown_output_path}")

# ピッチング解析にFPSを設定
if analyzer:
    analyzer.set_fps(fps)

print(f"\nProcessing frames and detecting objects...")

import time
start_time = time.time()

frame_count = 0
while True:
    ret_enhance, frame_enhance = cap_enhance.read()
    ret_original, frame_original = cap_original.read()

    if not ret_enhance or not ret_original:
        break

    # 強調フレームで検出実行
    results = model(frame_enhance, verbose=False)

    # ピッチング解析
    if analyzer:
        # 解析実行
        analysis_result = analyzer.update(results, frame_count)

        # リリース検出時にログ出力と真上ビューのリセット
        if analysis_result['is_release']:
            print(f"🎯 Release detected at frame {frame_count}")
            if topdown_view:
                topdown_view.clear_trajectory()

        # 真上ビューを更新
        if topdown_view and analysis_result['ball_field_2d']:
            x_field, z_field = analysis_result['ball_field_2d']
            topdown_view.add_trajectory_point(x_field, z_field)

            # 真上ビューフレームを生成
            topdown_frame = topdown_view.get_frame(
                ball_position=(x_field, z_field),
                show_trajectory=True
            )

            # リアルタイム表示（Docker環境では動作しない）
            if SHOW_TOPDOWN_VIEW:
                cv2.imshow('Top-Down View', topdown_frame)

            # 動画として保存（Docker環境推奨）
            if topdown_video_writer:
                topdown_video_writer.write(topdown_frame)
        elif topdown_view:
            # ボールが検出されていない場合もフレームを出力（動画のフレーム数を合わせるため）
            topdown_frame = topdown_view.get_frame(show_trajectory=True)
            if topdown_video_writer:
                topdown_video_writer.write(topdown_frame)

        # 描画
        frame_original = analyzer.draw(frame_original, frame_count,
                                      ball_3d=analysis_result['ball_3d'],
                                      draw_strike_zone=DRAW_STRIKE_ZONE,
                                      draw_info=True)

    # 現在フレームの検出物体の中心座標を取得
    current_centers = []

    # 検出結果を元フレームに描画
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # target_classesが指定されている場合、該当クラスのみ描画
            if target_classes and cls not in target_classes:
                continue

            # 中心座標を計算
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            current_centers.append((center_x, center_y, cls))

            # バウンディングボックスとラベルを描画
            # cv2.rectangle(frame_original, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # label = f"{model.names[cls]} {conf:.2f}"
            # cv2.putText(frame_original, label, (x1, y1-10),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 軌跡を更新（簡易的なマッチング）
    if DRAW_TRAJECTORY:
        # 既存の軌跡を更新
        used_centers = set()
        for obj_id in list(trajectories.keys()):
            if not current_centers:
                break

            # 最も近い検出点を探す
            points = trajectories[obj_id]['points']
            last_point = points[-1] if points else None
            if last_point is None:
                continue

            min_dist = float('inf')
            best_match = None
            for i, (cx, cy, cls) in enumerate(current_centers):
                if i in used_centers:
                    continue
                dist = np.sqrt((last_point[0] - cx)**2 + (last_point[1] - cy)**2)
                if dist < min_dist and dist < 100:  # 距離閾値
                    min_dist = dist
                    best_match = i

            if best_match is not None:
                cx, cy, cls = current_centers[best_match]
                trajectories[obj_id]['points'].append((cx, cy))
                trajectories[obj_id]['last_seen'] = frame_count
                used_centers.add(best_match)
                # 最大長を超えたら古い点を削除
                if len(trajectories[obj_id]['points']) > MAX_TRAJECTORY_LENGTH:
                    trajectories[obj_id]['points'].pop(0)

        # 新しい検出物体を追加
        for i, (cx, cy, cls) in enumerate(current_centers):
            if i not in used_centers:
                trajectories[next_object_id] = {
                    'points': [(cx, cy)],
                    'last_seen': frame_count
                }
                next_object_id += 1

        # 古い軌跡を削除（TRAJECTORY_FADE_FRAMES経過したもの）
        trajectories_to_remove = []
        for obj_id, traj_data in trajectories.items():
            if frame_count - traj_data['last_seen'] > TRAJECTORY_FADE_FRAMES:
                trajectories_to_remove.append(obj_id)
        for obj_id in trajectories_to_remove:
            del trajectories[obj_id]

        # 軌跡を描画（ネオン効果）
        for obj_id, traj_data in trajectories.items():
            points = traj_data['points']
            frames_since_seen = frame_count - traj_data['last_seen']

            # フェードアウト効果（検出されなくなってから徐々に薄く）
            fade_alpha = max(0, 1 - (frames_since_seen / TRAJECTORY_FADE_FRAMES))

            if len(points) > 1 and fade_alpha > 0:
                # ネオン効果で軌跡を描画
                # glow_intensityにfade_alphaを適用してフェードアウト
                frame_original = draw_neon_polyline(
                    frame_original,
                    points,
                    glow_color=TRAJECTORY_COLOR,
                    core_color=NEON_CORE_COLOR,
                    glow_thickness=NEON_GLOW_THICKNESS,
                    core_thickness=NEON_CORE_THICKNESS,
                    glow_blur=NEON_GLOW_BLUR,
                    glow_intensity=NEON_GLOW_INTENSITY * fade_alpha  # フェードアウト効果を適用
                )

    out.write(frame_original)
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

cap_enhance.release()
cap_original.release()
out.release()

# 真上ビュー動画を閉じる
if topdown_video_writer:
    topdown_video_writer.release()
    print(f"✅ Top-down view video saved")
    #cv2.destroyAllWindows()

# 処理時間の計算
end_time = time.time()
processing_time = end_time - start_time
video_duration = frame_count / fps if fps > 0 else 0
processing_speed = video_duration / processing_time if processing_time > 0 else 0

print(f"\n{'='*60}")
print(f"Detection completed!")
print(f"{'='*60}")
print(f"Output saved to: {output_path}")
if topdown_video_writer:
    print(f"Top-down view saved to: {topdown_output_path}")
print(f"\n【Processing Statistics】")
print(f"  Total frames: {frame_count}")
print(f"  Video duration: {video_duration:.2f} seconds")
print(f"  Processing time: {processing_time:.2f} seconds")
print(f"  Processing speed: {processing_speed:.2f}x realtime")
if processing_speed < 1.0:
    print(f"  (処理は実時間の{1/processing_speed:.2f}倍かかっています)")
else:
    print(f"  (実時間の{processing_speed:.2f}倍速で処理できています)")

# ピッチング解析のJSON出力
if analyzer:
    json_output_path = os.path.join(video_dir, f"{base_name}_trajectory.json")
    analyzer.export_to_json(json_output_path, video_file=original_video)
    print(f"\nTrajectory data saved to: {json_output_path}")
