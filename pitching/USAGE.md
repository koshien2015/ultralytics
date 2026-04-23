# pitching パイプライン 使い方ガイド

## 概要

野球動画からキャップボールの軌跡を検出・可視化するパイプラインです。  
フレーム差分と YOLO の融合によって検出漏れを補完し、ネオン軌跡を描画した動画と JSON を出力します。

---

## インストール

```bash
pip install ultralytics opencv-python pydantic pyyaml numpy
```

YOLO モデルファイル（`.pt`）を用意して、設定ファイルの `model_path` に指定してください。

---

## 基本的な実行

```bash
python -m pitching.app.cli -i video.mp4 -o output/
```

| オプション | 説明 |
|-----------|------|
| `-i`, `--input` | 入力動画ファイル（必須） |
| `-o`, `--output-dir` | 出力ディレクトリ（必須） |
| `-c`, `--config` | YAML 設定ファイル（省略時はデフォルト値） |
| `--stop-after STAGE` | 指定ステージで停止しチェックポイントを保存 |
| `--resume-from STAGE` | 指定ステージから再実行（前ステージはチェックポイントからロード） |
| `--no-checkpoint` | チェックポイントの保存/ロードを無効にする |

---

## 出力ファイル

| ファイル | 内容 |
|---------|------|
| `result.mp4` | ネオン軌跡を重ねた出力動画 |
| `pitches.json` | 検出したピッチ一覧（フレーム番号・軌跡・リリース情報） |
| `<ステージ名>.json` | 各ステージのチェックポイント（resume 用） |

---

## ステージ一覧

パイプラインは以下の順序で実行されます。

```
frame_diff
  └─ フレーム差分で動体候補を検出し、enhanced 動画を生成

yolo_detect
  └─ enhanced 動画に YOLO を適用してボール候補を検出

fusion
  └─ YOLO 検出と差分検出を統合（YOLO 未検出フレームを差分で補完）

pose_estimation
  └─ 姿勢推定（現在は stub。MediaPipe/YOLOv8-Pose に差し替え可能）

tracking
  └─ フレーム間のボール候補を連結してトラック（軌跡）を生成

strike_zone
  └─ バッターとキャッチャーの bbox からストライクゾーンを推定

release_detection
  └─ ピッチャーの動作クラスからリリースフレームを検出

pitch_analysis
  └─ トラックとリリースイベントから各投球を構造化

rendering
  └─ ネオン軌跡を描画し result.mp4 と pitches.json を書き出し
```

---

## 途中停止と再実行

時間のかかる YOLO 検出だけ先に実行し、後段のロジックを繰り返し調整したい場合に便利です。

```bash
# Step 1: fusion まで実行して停止
python -m pitching.app.cli -i video.mp4 -o output/ --stop-after fusion

# Step 2: tracking 以降を再実行（fusion 以前はチェックポイントから復元）
python -m pitching.app.cli -i video.mp4 -o output/ --resume-from tracking
```

> `--stop-after` と `--resume-from` は独立したオプションです。  
> 同じ `-o` ディレクトリを指定することでチェックポイントが共有されます。

---

## 設定ファイル

`pitching/config/defaults/default.yaml` をコピーして値を調整し、`-c` で渡します。

```yaml
yolo:
  model_path: "shared/yolo8m_20251109.pt"  # YOLO モデルファイルのパス
  min_confidence: 0.3                       # 検出信頼度の下限

frame_diff:
  threshold: 100      # 差分二値化の閾値（小さいほど敏感）
  area_min: 10.0      # 検出する最小面積 (px²)
  area_max: 5000.0    # 検出する最大面積 (px²)

fusion:
  max_jump_px: 80.0        # フレーム間の最大移動距離 (px)
  min_direction_cos: 0.3   # 方向コサイン下限（軌跡の連続性フィルタ）

tracking:
  max_trajectory_length: 60   # トラック保持フレーム数
  max_match_distance_px: 100.0

strike_zone:
  width_px: 50.0       # ストライクゾーンの幅 (px)
  fixed_center_x: null # null で自動推定、数値で固定

rendering:
  draw_trajectory: true
  draw_strike_zone: false
  glow_color_bgr: [0, 255, 255]   # ネオン発光色（BGR）
  core_color_bgr: [255, 255, 255] # 中心線の色（BGR）
  glow_thickness: 15
  core_thickness: 3
```

---

## 姿勢推定を差し替える

現在の `PoseEstimationStage` は `NullPoseEstimator`（何も返さない stub）を使用しています。  
`PoseEstimator` Protocol に準拠した実装を用意することで、変更なしに差し替えられます。

```python
# infra/ml/my_pose_estimator.py
import numpy as np
from typing import Tuple
from pitching.domain.entities.pose import Keypoint, PoseFrame

class MyPoseEstimator:
    def estimate(self, frame: np.ndarray, frame_index: int) -> Tuple[PoseFrame, ...]:
        # MediaPipe や YOLOv8-Pose の呼び出しをここに実装
        ...
```

```python
# app/run_pipeline.py の build_stages() 内で差し替え
pose_estimator = MyPoseEstimator()   # NullPoseEstimator() の代わりに
```

---

## ディレクトリ構成

```
pitching/
├── app/           # エントリポイント（CLI・パイプライン組み立て）
├── config/        # YAML スキーマとデフォルト設定
├── domain/        # ビジネスロジック（entities・services）
├── infra/         # 外部依存の実装（YOLO・VideoReader・ストレージ）
├── pipeline/      # ステージ定義・Runner・Context
└── tests/         # ユニット・インテグレーションテスト

legacy/            # リファクタ前のコード（参照専用）
```
