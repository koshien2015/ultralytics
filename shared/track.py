import cv2
import sys
import os
import numpy as np
from ultralytics import YOLO
import tennis
from pitching_analysis import PitchingAnalyzer

# コマンドライン引数から動画パスを取得
if len(sys.argv) > 1:
    original_video = sys.argv[1]
else:
    original_video = "test.mp4"

# ファイル名とディレクトリを取得
base_name = os.path.splitext(os.path.basename(original_video))[0]
video_dir = os.path.dirname(original_video) if os.path.dirname(original_video) else "."

# 強調動画のパスを設定
enhance_video = os.path.join(video_dir, f"{base_name}_enhance.mp4")

print(f"Original video: {original_video}")
print(f"Generating enhanced video for detection...")

# tennisモジュールで強調動画を生成
tennis.run(original_video, enhance_video_path=enhance_video)

print(f"\nLoading YOLO model...")
# YOLOモデルをロード（同じディレクトリ内）
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "yolo8m_20250510.pt")
model = YOLO(model_path)

# 検出対象クラスを指定（空リストで全クラス、数字を指定で特定クラスのみ）
# 0: petbottle_cap (ボール)
# 1: pitcher_motion
# 5: pitcher_release
target_classes = [0]  # 検出したいクラスIDをここに指定

# 軌跡描画設定
DRAW_TRAJECTORY = True  # 軌跡を描画するか
MAX_TRAJECTORY_LENGTH = 30  # 軌跡の最大長さ（フレーム数）
TRAJECTORY_FADE_FRAMES = 20  # 検出されなくなってから何フレームで消えるか
TRAJECTORY_COLOR = (0, 255 , 0)  # 軌跡の色（緑）
TRAJECTORY_THICKNESS = 4  # 軌跡の太さ

# ピッチング解析設定
ENABLE_PITCHING_ANALYSIS = True  # ピッチング解析を有効にするか
DRAW_STRIKE_ZONE = True  # ストライクゾーンを描画するか
STRIKE_ZONE_WIDTH_PX = 50  # ストライクゾーンの幅（ピクセル）
STRIKE_ZONE_CENTER_X = None  # ストライクゾーンの中央X座標（ピクセル）、Noneで初回捕手検出時に自動設定

# 各物体の軌跡を保存する辞書 {物体ID: {'points': [(x, y), ...], 'last_seen': frame_number}}
trajectories = {}
next_object_id = 0

# ピッチング解析の初期化
analyzer = None
if ENABLE_PITCHING_ANALYSIS:
    print("Initializing pitching analyzer...")

    # キャリブレーションファイルを自動検索
    calibration_path = None
    calibration_file = os.path.join(script_dir, "calibrations", f"{base_name}_calibration.json")
    if os.path.exists(calibration_file):
        calibration_path = calibration_file
        print(f"✅ Calibration file found: {calibration_file}")
    else:
        print(f"⚠️  No calibration file found (searched: {calibration_file})")
        print("   Continuing without calibration (visualization only)")

    analyzer = PitchingAnalyzer(
        calibration_path=calibration_path,
        strike_zone_width_px=STRIKE_ZONE_WIDTH_PX,
        strike_zone_center_x=STRIKE_ZONE_CENTER_X
    )

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

        # リリース検出時にログ出力
        if analysis_result['is_release']:
            print(f"🎯 Release detected at frame {frame_count}")

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
            cv2.rectangle(frame_original, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.putText(frame_original, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

        # 軌跡を描画
        for obj_id, traj_data in trajectories.items():
            points = traj_data['points']
            frames_since_seen = frame_count - traj_data['last_seen']

            # フェードアウト効果（検出されなくなってから徐々に薄く）
            fade_alpha = max(0, 1 - (frames_since_seen / TRAJECTORY_FADE_FRAMES))

            if len(points) > 1 and fade_alpha > 0:
                for i in range(1, len(points)):
                    # 古い点ほど薄く、さらにフェードアウト
                    alpha = (i / len(points)) * fade_alpha
                    thickness = max(1, int(TRAJECTORY_THICKNESS * alpha))

                    # フェードアウト時は色も薄く
                    color = tuple(int(c * fade_alpha) for c in TRAJECTORY_COLOR)

                    cv2.line(frame_original, points[i-1], points[i],
                            color, thickness)

    out.write(frame_original)
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

cap_enhance.release()
cap_original.release()
out.release()

# 処理時間の計算
end_time = time.time()
processing_time = end_time - start_time
video_duration = frame_count / fps if fps > 0 else 0
processing_speed = video_duration / processing_time if processing_time > 0 else 0

print(f"\n{'='*60}")
print(f"Detection completed!")
print(f"{'='*60}")
print(f"Output saved to: {output_path}")
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
