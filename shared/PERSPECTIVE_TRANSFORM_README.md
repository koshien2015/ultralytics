# 射影変換機能の使い方

真上からの2D軌道を再現する射影変換機能の使用方法を説明します。

## 📋 概要

この機能を使うと：
- ✅ カメラ画像座標 → 真上から見たフィールド座標に変換
- ✅ 遠近感による歪みを補正した正確な軌跡
- ✅ 真上ビュー動画の自動生成（Docker/Colab対応）

## ⚡ クイックスタート（Docker/Colab環境）

```bash
# 1. フレーム画像を保存
python calibrate_perspective_headless.py test8.mp4 --frame 100

# 2. 保存された test8_frame100.jpg を確認

# 3. 座標を入力（対話形式）
# または JSONテンプレートを使用：
python calibrate_perspective_headless.py test8.mp4 --create-template
# → test8_calibration_template.json を編集
python calibrate_perspective_headless.py test8.mp4 --json test8_calibration_template.json

# 4. track.py で CALIBRATION_FILE を設定
# CALIBRATION_FILE = "test8_calibration.json"
# SAVE_TOPDOWN_VIDEO = True

# 5. 実行
python track.py test8.mp4

# 6. 出力確認
# - test8_detected.mp4（元動画）
# - test8_topdown.mp4（真上ビュー）
```

## 🚀 使用手順

### ステップ1: キャリブレーション（初回のみ）

キャリブレーションには以下の3つの方法があります：

#### 方法A: ヘッドレス環境（Docker/Colab対応、推奨）

```bash
cd ultralytics/shared
python calibrate_perspective_headless.py test8.mp4 --frame 100
```

**操作方法:**
1. フレーム画像が保存されます（`test8_frame100.jpg`）
2. 画像をダウンロードして確認
3. ターミナルで基準点の座標を入力
   ```
   --- 点 1 ---
   画像上のX座標（ピクセル）: 640
   画像上のY座標（ピクセル）: 720

   この点の種類を選択してください:
     1. home_plate
     2. pitcher_mound
     3. first_base
     4. third_base
     5. custom
   選択 (1-5): 1
   ```
4. 4点以上入力
5. 確認用画像（`test8_calibration_preview.jpg`）とキャリブレーションファイル（`test8_calibration.json`）が生成されます

**JSONテンプレート方式（より簡単）:**
```bash
# 1. テンプレート作成
python calibrate_perspective_headless.py test8.mp4 --create-template

# 2. test8_calibration_template.json を編集

# 3. テンプレートから読み込み
python calibrate_perspective_headless.py test8.mp4 --json test8_calibration_template.json
```

#### 方法B: ローカルマシンでGUI使用（GUIが使える場合）

```bash
cd ultralytics/shared
python calibrate_perspective.py test8.mp4 --frame 100
```

**操作方法:**
1. 野球場のフレームが表示されます
2. 基準点を**4点以上**クリック
   - 推奨: ホームベース、マウンド、1塁、3塁など
3. 各点をクリック後、ターミナルで点の種類を選択
   ```
   点 1 を追加: (640, 720)
   この点の種類を選択してください:
     1. home_plate (X=0.00m, Z=0.00m)
     2. pitcher_mound (X=0.00m, Z=18.44m)
     3. first_base (X=27.43m, Z=27.43m)
     4. third_base (X=-27.43m, Z=27.43m)
     5. custom (手動入力)
   選択 (1-5): 1
   ```
4. 4点以上選択したら `s` キーで保存
5. `test8_calibration.json` が生成されます

**キー操作:**
- **クリック**: 基準点を追加
- **u**: 最後の点を削除（Undo）
- **r**: 全ての点をリセット
- **s**: 保存（4点以上必要）
- **q**: 終了

#### 方法C: JSONファイルを手動作成（上級者向け）

以下のようなJSONを手動で作成：

```json
{
  "image_points": [
    [640, 720],
    [640, 300],
    [500, 150],
    [780, 150]
  ],
  "real_points": [
    [0.0, 0.0],
    [0.0, 18.44],
    [27.43, 27.43],
    [-27.43, 27.43]
  ]
}
```

ただし、`H`と`H_inv`行列は自動計算されないので、一度Pythonで計算する必要があります。

### ステップ2: track.py の設定

`ultralytics/shared/track.py` の設定を変更：

```python
# 射影変換・真上ビュー設定
CALIBRATION_FILE = "test8_calibration.json"  # キャリブレーションファイル名
SAVE_TOPDOWN_VIDEO = True  # 真上ビュー動画を保存（Docker/Colab推奨）
SHOW_TOPDOWN_VIEW = False  # リアルタイム表示（ローカルのみ、Docker/Colabでは False）
```

**パラメータ説明:**
- `CALIBRATION_FILE`: キャリブレーションファイルのパス
  - 相対パス: 動画と同じディレクトリから探す
  - 絶対パス: フルパス指定
  - `None`: 射影変換を使わない（従来通り）
- `SAVE_TOPDOWN_VIDEO`: 真上ビューを動画ファイルとして保存
  - Docker/Colab環境では `True` 推奨
- `SHOW_TOPDOWN_VIEW`: リアルタイムでウィンドウ表示
  - Docker/Colab環境では動作しないので `False`

### ステップ3: 実行

#### Docker環境の場合

```bash
# Dockerコンテナ内で実行
cd /path/to/ultralytics/shared
python track.py test8.mp4
```

#### Colab環境の場合

```python
# Colab上で実行
!cd /content/ultralytics/shared && python track.py /content/test8.mp4
```

#### ローカル環境の場合

```bash
cd ultralytics/shared
python track.py test8.mp4
```

### ステップ4: 出力ファイルの確認

実行後、以下のファイルが生成されます：

```
test8_detected.mp4         # 元の検出動画（ネオン軌跡付き）
test8_topdown.mp4          # 真上ビュー動画（NEW！）
test8_trajectory.json      # 軌跡データ（拡張版）
test8_enhance.mp4          # 強調動画（中間ファイル）
```

## 📊 出力データの詳細

### `test8_topdown.mp4`

真上から見た野球場の動画：
- 🟢 緑色のフィールド
- 🟤 マウンド・ベース
- 🟡 ストライクゾーン
- 🔴 ボールの現在位置
- 🟣 マゼンタの軌跡

### `test8_trajectory.json` の拡張

射影変換を有効にすると、各軌跡点に以下が追加されます：

```json
{
  "pitches": [
    {
      "pitch_id": 1,
      "release_frame": 150,
      "trajectory": [
        {
          "frame": 150,
          "time": 0.0,
          "x": 0.123,        // 正規化座標（従来）
          "y": 0.456,        // 正規化座標（従来）
          "z": 0.0,          // 経過時間（従来）
          "pixel_x": 640,    // 画像座標（NEW）
          "pixel_y": 360,    // 画像座標（NEW）
          "field_x": 0.15,   // フィールド座標X（NEW、メートル）
          "field_z": 18.2    // フィールド座標Z（NEW、メートル）
        }
      ]
    }
  ]
}
```

**新規追加フィールド:**
- `pixel_x`, `pixel_y`: 元画像のピクセル座標
- `field_x`: 横方向の位置（メートル、ホームベース中心=0）
- `field_z`: 奥行き方向の位置（メートル、ホームベース=0、投手方向=正）

## 🔧 トラブルシューティング

### Q0: "The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support" エラー

**原因:** Docker/Colab環境でGUIが使えない

**解決策:**
- `calibrate_perspective.py`（GUI版）の代わりに `calibrate_perspective_headless.py`（ヘッドレス版）を使用
```bash
python calibrate_perspective_headless.py test8.mp4 --frame 100
```

### Q1: "Calibration file not found" エラー

**原因:** キャリブレーションファイルが見つからない

**解決策:**
1. ファイルパスが正しいか確認
2. 動画と同じディレクトリにキャリブレーションファイルを配置
3. または絶対パスで指定

### Q2: 真上ビューが表示されない（Docker/Colab）

**原因:** Docker/ColabはGUI表示に対応していない

**解決策:**
- `SHOW_TOPDOWN_VIEW = False` に設定
- `SAVE_TOPDOWN_VIDEO = True` で動画として保存
- 実行後 `test8_topdown.mp4` を確認

### Q3: 真上ビューの位置がずれている

**原因:** キャリブレーションの精度が低い

**解決策:**
1. 基準点を**正確に**クリック
2. できるだけ**4隅に近い点**を選ぶ
3. **4点以上**選択（多いほど精度向上）
4. キャリブレーション用フレームを変更（`--frame` オプション）

### Q4: "At least 4 points are required" エラー

**原因:** 基準点が4点未満

**解決策:** 4点以上クリックしてから `s` キーで保存

## 📐 座標系の説明

### フィールド座標系
- **原点**: ホームベース中心
- **X軸**: 横方向（右が正、左が負）
- **Z軸**: 奥行き方向（投手方向が正）
- **単位**: メートル

### 基準点の例
```
home_plate:     (0.00,  0.00)   # ホームベース
pitcher_mound:  (0.00, 18.44)   # マウンド（60.5フィート）
first_base:    (27.43, 27.43)   # 1塁（90フィート）
third_base:   (-27.43, 27.43)   # 3塁（90フィート）
```

## 🎯 使用例

### 例1: Docker環境で実行（完全版）

```bash
# 1. Docker内でキャリブレーション（ヘッドレス版）
docker exec -it <container_id> python calibrate_perspective_headless.py test8.mp4 --frame 100

# 2. フレーム画像を取得して確認
docker cp <container_id>:/workspace/test8_frame100.jpg ./

# 3. 対話的に座標を入力（Docker内）
docker exec -it <container_id> python calibrate_perspective_headless.py test8.mp4 --frame 100
# または JSONテンプレート方式（推奨）:
docker exec -it <container_id> python calibrate_perspective_headless.py test8.mp4 --create-template
docker cp <container_id>:/workspace/test8_calibration_template.json ./
# → ローカルで編集
docker cp test8_calibration_template.json <container_id>:/workspace/
docker exec -it <container_id> python calibrate_perspective_headless.py test8.mp4 --json test8_calibration_template.json

# 4. 確認用画像を確認
docker cp <container_id>:/workspace/test8_calibration_preview.jpg ./

# 5. track.pyの設定変更
# CALIBRATION_FILE = "test8_calibration.json"
# SAVE_TOPDOWN_VIDEO = True

# 6. Docker内で実行
docker exec -it <container_id> python track.py test8.mp4

# 7. 結果を取得
docker cp <container_id>:/workspace/test8_detected.mp4 ./
docker cp <container_id>:/workspace/test8_topdown.mp4 ./
docker cp <container_id>:/workspace/test8_trajectory.json ./
```

### 例2: 複数動画の一括処理

```bash
# キャリブレーションファイルを共用（同じカメラアングルの場合）
for video in *.mp4; do
    python track.py "$video"
done
```

同じカメラ設定の動画なら、キャリブレーションは1回でOKです。

## 💡 ヒント

- **キャリブレーションの精度が最重要**: 時間をかけて正確に基準点を選択
- **フレーム選択**: できるだけ全体が見えるフレームを選択
- **基準点の配置**: 画面の四隅に近い点を選ぶと精度向上
- **同じカメラなら再利用**: キャリブレーションファイルは使い回し可能

## 🔄 既存機能との互換性

**重要:** 射影変換機能は完全にオプショナルです。

- `CALIBRATION_FILE = None` なら従来通り動作
- 既存のJSONファイルも互換性あり（`field_x`, `field_z`が追加されるだけ）
- 既存のワークフローを壊しません
