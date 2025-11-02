"""
実際の動画でキャリブレーションあり・なしの動作確認
"""

import sys
from pathlib import Path
from pitching_analysis import PitchingAnalyzer
from ultralytics import YOLO
import cv2
import json

def analyze_video_without_calibration():
    """キャリブレーションなしで動画解析"""
    print("=" * 80)
    print("動画解析テスト1: キャリブレーションなし")
    print("=" * 80)

    video_path = Path(__file__).parent / "test.mp4"
    trajectory_path = Path(__file__).parent / "test_no_calibration_trajectory.json"
    model_path = Path(__file__).parent / "yolo11n.pt"

    if not video_path.exists():
        print(f"❌ 動画ファイルが見つかりません: {video_path}")
        return

    if not model_path.exists():
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return

    # YOLOモデル読み込み
    model = YOLO(str(model_path))

    analyzer = PitchingAnalyzer()
    print(f"✅ アナライザー初期化成功")
    print(f"   Mode: {analyzer.use_calibration and 'Calibrated' or 'Visualization only'}")

    # 動画解析（最初の100フレームのみ）
    print(f"\n動画解析中（最初の100フレーム）...")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    analyzer.set_fps(fps)

    frame_count = 0
    max_frames = 300

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO検出
        results = model.track(frame, persist=True, verbose=False)

        # アナライザー更新
        analyzer.update(results, frame_count)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"   処理中: {frame_count}/{max_frames} フレーム")

    cap.release()

    # 軌跡データを保存
    if analyzer.pitches:
        analyzer.export_to_json(str(trajectory_path), video_file=str(video_path))

        print(f"\n✅ 解析完了")
        print(f"   検出ピッチ数: {len(analyzer.pitches)}")
        print(f"   軌跡データ: {trajectory_path}")

        # 最初のピッチの軌跡を表示
        if analyzer.pitches:
            first_pitch = analyzer.pitches[0]
            print(f"\n最初のピッチの軌跡データサンプル:")
            print(f"   ID: {first_pitch['id']}")
            print(f"   軌跡点数: {len(first_pitch['trajectory'])}")
            if first_pitch['trajectory']:
                sample = first_pitch['trajectory'][0]
                print(f"   サンプル点: {json.dumps(sample, ensure_ascii=False)}")
    else:
        print(f"\n⚠️  ピッチが検出されませんでした")
    print()

def analyze_video_with_calibration():
    """キャリブレーションありで動画解析"""
    print("=" * 80)
    print("動画解析テスト2: キャリブレーションあり")
    print("=" * 80)

    video_path = Path(__file__).parent / "test.mp4"
    calibration_path = Path(__file__).parent / "calibrations" / "test_calibration.json"
    trajectory_path = Path(__file__).parent / "test_with_calibration_trajectory.json"
    model_path = Path(__file__).parent / "yolo11n.pt"

    if not video_path.exists():
        print(f"❌ 動画ファイルが見つかりません: {video_path}")
        return

    if not calibration_path.exists():
        print(f"❌ キャリブレーションファイルが見つかりません: {calibration_path}")
        return

    if not model_path.exists():
        print(f"❌ モデルファイルが見つかりません: {model_path}")
        return

    # YOLOモデル読み込み
    model = YOLO(str(model_path))

    analyzer = PitchingAnalyzer(calibration_path=str(calibration_path))
    print(f"✅ アナライザー初期化成功")
    print(f"   Mode: {analyzer.use_calibration and 'Calibrated (Real-world coords)' or 'Visualization only'}")

    # 動画解析（最初の100フレームのみ）
    print(f"\n動画解析中（最初の100フレーム）...")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    analyzer.set_fps(fps)

    frame_count = 0
    max_frames = 300

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO検出
        results = model.track(frame, persist=True, verbose=False)

        # アナライザー更新
        analyzer.update(results, frame_count)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"   処理中: {frame_count}/{max_frames} フレーム")

    cap.release()

    # 軌跡データを保存
    if analyzer.pitches:
        analyzer.export_to_json(str(trajectory_path), video_file=str(video_path))

        print(f"\n✅ 解析完了")
        print(f"   検出ピッチ数: {len(analyzer.pitches)}")
        print(f"   軌跡データ: {trajectory_path}")

        # 最初のピッチの軌跡を表示
        if analyzer.pitches:
            first_pitch = analyzer.pitches[0]
            print(f"\n最初のピッチの軌跡データサンプル:")
            print(f"   ID: {first_pitch['id']}")
            print(f"   軌跡点数: {len(first_pitch['trajectory'])}")
            if first_pitch['trajectory']:
                sample = first_pitch['trajectory'][0]
                print(f"   サンプル点: {json.dumps(sample, ensure_ascii=False)}")
    else:
        print(f"\n⚠️  ピッチが検出されませんでした")
    print()

if __name__ == "__main__":
    analyze_video_without_calibration()
    analyze_video_with_calibration()

    print("=" * 80)
    print("全テスト完了")
    print("=" * 80)
    print("\n💡 軌跡データの違いを確認してください:")
    print("   - test_no_calibration_trajectory.json → pixel_x, pixel_y のみ")
    print("   - test_with_calibration_trajectory.json → x, y, z (メートル) + pixel_x, pixel_y")
