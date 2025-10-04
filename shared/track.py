import cv2
import sys
import os
from ultralytics import YOLO
import tennis

# コマンドライン引数から動画パスを取得
if len(sys.argv) > 1:
    original_video = sys.argv[1]
else:
    original_video = "test2.mp4"

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
# YOLOモデルをロード
model = YOLO("yolo8m_20250510.pt")

# 検出対象クラスを指定（空リストで全クラス、数字を指定で特定クラスのみ）
# 例: [0] = person, [32] = sports ball, [37] = tennis racket
# 複数指定: [0, 32, 37]
target_classes = [0]  # 検出したいクラスIDをここに指定

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

print(f"\nProcessing frames and detecting objects...")

frame_count = 0
while True:
    ret_enhance, frame_enhance = cap_enhance.read()
    ret_original, frame_original = cap_original.read()

    if not ret_enhance or not ret_original:
        break

    # 強調フレームで検出実行
    results = model(frame_enhance, verbose=False)

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

            # バウンディングボックスとラベルを描画
            cv2.rectangle(frame_original, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.putText(frame_original, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame_original)
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

cap_enhance.release()
cap_original.release()
out.release()

print(f"\nDetection completed!")
print(f"Output saved to: {output_path}")
