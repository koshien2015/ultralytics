"""
キャリブレーションファイルの指定方法サンプル
"""

from pathlib import Path
from pitching_analysis import PitchingAnalyzer
from ultralytics import YOLO
import cv2

# ===========================
# パターン1: 相対パス指定
# ===========================
def example_relative_path():
    """相対パスでキャリブレーションファイルを指定"""
    calibration_path = "calibrations/test_calibration.json"
    analyzer = PitchingAnalyzer(calibration_path=calibration_path)
    print(f"✅ キャリブレーション読み込み成功: {calibration_path}")


# ===========================
# パターン2: 絶対パス指定
# ===========================
def example_absolute_path():
    """絶対パスでキャリブレーションファイルを指定"""
    calibration_path = Path(__file__).parent / "calibrations" / "test_calibration.json"
    analyzer = PitchingAnalyzer(calibration_path=str(calibration_path))
    print(f"✅ キャリブレーション読み込み成功: {calibration_path}")


# ===========================
# パターン3: video_idから自動生成
# ===========================
def example_auto_path_from_video_id(video_id: str):
    """video_idからキャリブレーションファイル名を自動生成"""
    calibration_dir = Path(__file__).parent / "calibrations"
    calibration_path = calibration_dir / f"{video_id}_calibration.json"

    if calibration_path.exists():
        analyzer = PitchingAnalyzer(calibration_path=str(calibration_path))
        print(f"✅ キャリブレーション読み込み成功: {video_id}")
        return analyzer
    else:
        print(f"⚠️  キャリブレーションファイルが見つかりません: {calibration_path}")
        print(f"   キャリブレーションなしで続行")
        analyzer = PitchingAnalyzer()
        return analyzer


# ===========================
# パターン4: 完全な動画解析フロー
# ===========================
def analyze_video_with_calibration(video_path: str, calibration_path: str = None):
    """
    キャリブレーション付きで動画を解析

    Args:
        video_path: 動画ファイルのパス
        calibration_path: キャリブレーションファイルのパス（オプション）
    """
    video_path = Path(video_path)

    # キャリブレーションパスが指定されていない場合、自動検索
    if calibration_path is None:
        video_id = video_path.stem  # 拡張子なしのファイル名
        calibration_dir = video_path.parent / "calibrations"
        calibration_path = calibration_dir / f"{video_id}_calibration.json"

        if not calibration_path.exists():
            print(f"⚠️  キャリブレーションファイルが見つかりません")
            print(f"   探したパス: {calibration_path}")
            calibration_path = None

    # アナライザー初期化
    if calibration_path and Path(calibration_path).exists():
        analyzer = PitchingAnalyzer(calibration_path=str(calibration_path))
        print(f"✅ キャリブレーションモード: 実世界座標を算出")
    else:
        analyzer = PitchingAnalyzer()
        print(f"✅ 可視化モード: 軌跡描画のみ")

    # YOLOモデル読み込み
    model_path = Path(__file__).parent / "yolo11n.pt"
    model = YOLO(str(model_path))

    # 動画解析
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    analyzer.set_fps(fps)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO検出
        results = model.track(frame, persist=True, verbose=False)

        # アナライザー更新
        analyzer.update(results, frame_count)

        frame_count += 1

    cap.release()

    # 結果をエクスポート
    output_json = video_path.parent / f"{video_path.stem}_trajectory.json"
    analyzer.export_to_json(str(output_json), video_file=str(video_path))

    print(f"\n✅ 解析完了")
    print(f"   検出ピッチ数: {len(analyzer.pitches)}")
    print(f"   出力: {output_json}")

    return analyzer


# ===========================
# 使用例
# ===========================
if __name__ == "__main__":
    print("=" * 80)
    print("キャリブレーションファイルの指定方法")
    print("=" * 80)

    # 例1: 相対パス
    print("\n【例1】相対パス指定:")
    try:
        example_relative_path()
    except Exception as e:
        print(f"   エラー: {e}")

    # 例2: 絶対パス
    print("\n【例2】絶対パス指定:")
    try:
        example_absolute_path()
    except Exception as e:
        print(f"   エラー: {e}")

    # 例3: video_idから自動生成
    print("\n【例3】video_idから自動生成:")
    try:
        example_auto_path_from_video_id("test")
    except Exception as e:
        print(f"   エラー: {e}")

    # 例4: 完全な解析フロー（実際には実行しない）
    print("\n【例4】完全な動画解析フロー:")
    print("   analyze_video_with_calibration('test.mp4')")
    print("   → 自動的に 'calibrations/test_calibration.json' を探す")
    print()
    print("   analyze_video_with_calibration('test.mp4', 'custom_calibration.json')")
    print("   → 指定されたキャリブレーションファイルを使用")

    print("\n" + "=" * 80)
