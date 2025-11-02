"""
キャリブレーション機能のテストスクリプト
"""

import sys
from pathlib import Path
from pitching_analysis import PitchingAnalyzer

def test_without_calibration():
    """キャリブレーションなしでの動作確認"""
    print("=" * 80)
    print("テスト1: キャリブレーションなし")
    print("=" * 80)

    analyzer = PitchingAnalyzer()
    print(f"✅ アナライザー初期化成功")
    print(f"   use_calibration: {analyzer.use_calibration}")
    print()

def test_with_calibration():
    """キャリブレーションありでの動作確認"""
    print("=" * 80)
    print("テスト2: キャリブレーションあり")
    print("=" * 80)

    calibration_path = Path(__file__).parent / "calibrations" / "test_calibration.json"

    if not calibration_path.exists():
        print(f"❌ キャリブレーションファイルが見つかりません: {calibration_path}")
        return

    analyzer = PitchingAnalyzer(calibration_path=str(calibration_path))
    print(f"✅ アナライザー初期化成功")
    print(f"   use_calibration: {analyzer.use_calibration}")
    print(f"   video_id: {analyzer.calibration_data['video_id']}")
    print(f"   キャリブレーション点数: {len(analyzer.calibration_data['points'])}")
    print()

    # 座標変換のテスト
    print("座標変換テスト:")
    test_points = [
        (640, 480, "ホームベース"),
        (740, 480, "一塁側ライン"),
        (640, 380, "ピッチャー側"),
        (700, 430, "ランダムな点"),
    ]

    for px, py, label in test_points:
        result = analyzer.transform_to_real_world(px, py)
        if result:
            x_real, y_real = result
            print(f"   {label}: ({px}, {py}) px → ({x_real:.3f}, {y_real:.3f}) m")
        else:
            print(f"   {label}: 変換失敗")
    print()

def test_invalid_calibration():
    """無効なキャリブレーションファイルでの動作確認"""
    print("=" * 80)
    print("テスト3: 無効なキャリブレーションファイル")
    print("=" * 80)

    try:
        analyzer = PitchingAnalyzer(calibration_path="nonexistent.json")
        print(f"   use_calibration: {analyzer.use_calibration}")
        print(f"✅ エラーハンドリング成功（キャリブレーションなしモードで継続）")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
    print()

if __name__ == "__main__":
    test_without_calibration()
    test_with_calibration()
    test_invalid_calibration()

    print("=" * 80)
    print("全テスト完了")
    print("=" * 80)
