# Calibration Data Directory

このディレクトリは、動画のカメラキャリブレーションデータを保存します。

## ファイル命名規則

```
{video_id}_calibration.json
```

例: `game_2025_01_01_001_calibration.json`

## データ形式

各JSONファイルは以下の構造を持ちます：

- `video_id`: 動画の識別子
- `calibration_date`: キャリブレーション実施日時
- `frame_number`: キャリブレーションに使用したフレーム番号
- `points`: 画像座標と実世界座標の対応点（最低4点）
- `homography_matrix`: 計算されたホモグラフィ行列（3x3）
- `frame_dimensions`: フレームの幅・高さ

詳細なスキーマは `calibration_schema.py` を参照してください。

## 使用方法

```python
from pitching_analysis import PitchingAnalyzer

analyzer = PitchingAnalyzer()
analyzer.load_calibration('calibrations/game_2025_01_01_001_calibration.json')
```
