# pitching — 投球フォーム解析

YOLO Pose のキーポイント時系列から関節角度・位置・タイミングを数値化し、
同一投手の2球（例: 良かった球と悪かった球）のフォームを比較する。

**フォームの良否は判定しない。** 出すのは観測値と、比較のための差分だけ。

## 何が分かるか

project.md の仮説に対応する観測値を出す。

| 仮説 | 対応する値 |
|---|---|
| 最大肘屈曲からリリースまでにどれだけ伸展したか | `elbow_extension_range_deg`, `flexion_to_release_time_sec` |
| 伸展途中で早くリリースしていないか | `elbow_angle_at_release_deg`, `peak_extension_velocity_deg_per_sec` |
| リリース時に前腕が上を向きすぎていないか | `forearm_angle_at_release_deg`, `forearm_direction_at_release` |
| リリース位置や体幹の傾きが違うか | `relative_to_shoulder_mid_x/y`, `trunk_lean_at_release_deg` |
| 骨盤・肩・肘・手首の動作タイミングが違うか | `shoulder_hip_separation_deg` 系列, `wrist_passes_elbow_frame_offset` |

## 構成

```text
pitching/
  models.py              共通データ構造（Keypoint / PoseFrame / PoseSeries / イベント）
  config.py              投球ごとの設定（YAML）の検証
  analysis.py            解析パイプライン本体
  adapters/              入力アダプタ（ここだけが動画・YOLO・shared/ に依存する）
  preprocessing/         信頼度フィルタ・補間・平滑化
  metrics/               角度・正規化・特徴量（副作用のない関数のみ）
  events/                イベント検出とリリース供給元インターフェース
  comparison/            同期（時間正規化・イベント基準）と比較
  reporting/             JSON / CSV 出力
  visualization/         グラフ・解析動画・比較画像
  cli.py                 CLI
  examples/              設定例と、動画なしで試すための合成データ生成
  tests/                 合成座標による単体テスト（実動画・GPU 不要）
```

`shared/pose.py`（`PoseEstimator` / `RoleTracker` / `KEYPOINT_NAMES`）はそのまま流用する。
`shared/*.py` はフラット import なので、`adapters/ultralytics_adapter.py` が
`sys.path` を足して遅延 import する。**`shared/` 側は変更していない。**

## 実行方法

```bash
cd ultralytics   # パッケージルート

# 1. 動画から YOLO Pose のキーポイントを抽出（GPU 推奨、ultralytics が必要）
python -m pitching extract \
  --video input/good.mp4 \
  --config pitching/examples/good.yaml \
  --output output/good

# 2. 解析（動画も YOLO も不要。pose.json だけで動く）
python -m pitching analyze \
  --pose-data output/good/pose.json \
  --config pitching/examples/good.yaml \
  --output output/good

# 3. 2球を比較
python -m pitching compare \
  --pitch-a output/good/metrics.json \
  --pitch-b output/bad/metrics.json \
  --output output/comparison
```

主なオプション:

- `analyze --overlay-video` … 骨格・肘角度・前腕角度・イベント名を重ねた動画を出す
- `analyze --no-charts` … グラフを出さない
- `compare --frames` … 足接地・最大肘屈曲・リリースの同一イベント比較画像を出す

実動画が無い場合は、合成データで一通り試せる。

```bash
python pitching/examples/generate_sample_pose.py output/sample
```

## 入力

### 設定 YAML（`examples/good.yaml`）

```yaml
pitch_id: good_001
video_path: input/good.mp4
throwing_hand: right
batter_direction: left    # 画像上で打者がどちら側にいるか
fps: 60
start_frame: 100
end_frame: 180

events:
  stride_foot_contact_frame: null   # 手動指定を優先する
  release_frame: 152

result:
  label: good
  description: "縦回転58km/h"
```

`preprocessing` / `metrics` / `event_detection` / `extraction` は省略可能
（閾値・補間可能な最大フレーム数・平滑化ウィンドウはすべて設定できる）。

> `batter_direction` は project.md の例には無いが追加した。斜め後方からの2D映像では、
> 投げ手だけでは「打者方向に対して上向き何度」の符号が決まらないため。既定は `left`。

### キーポイント JSON

`extract` が出す形式。外部で作ったものを渡してもよい（`adapters/json_adapter.py`）。

```json
{
  "schema_version": 1,
  "meta": {"pitch_id": "good_001", "fps": 60.0, "adapter": "ultralytics"},
  "frames": [
    {"frame_index": 100, "timestamp_sec": 1.667,
     "keypoints": {"right_elbow": [140.2, 120.5, 0.93], "right_wrist": [null, null, 0.11]}}
  ]
}
```

座標が `null` は欠損。信頼度は残すので、「低信頼度で落とした」と「未検出」を区別できる。

CSV は目視確認用（座標は小数3桁に丸めて書く）。読み込み直すときは JSON を使う。

## 出力

`analyze --output <dir>` が作るもの:

| ファイル | 内容 |
|---|---|
| `metrics.json` | 投球単位の要約特徴量・イベント・信頼性情報 |
| `summary.csv` | 要約特徴量（表計算で開く用） |
| `events.csv` | 決定したイベントと確度（`manual` / `cap_tracking` / `pose_heuristic`） |
| `angles.csv` | フレーム単位の関節角度・角速度・進行率（足接地〜リリースの区間外は空欄） |
| `pose_raw.json` / `.csv` | 生キーポイント |
| `pose_smoothed.json` / `.csv` | 平滑化後キーポイント |
| `charts/*.png` | 肘角度・伸展角速度・前腕角度・体幹・前脚のグラフ |
| `overlay.mp4` | 解析動画（`--overlay-video` 指定時） |

`compare --output <dir>` が作るもの: `comparison.json` / `comparison.csv` /
`charts/compare_*.png`（時間正規化して重ねたグラフ）/ `frames/compare_*.png`。

グラフ内のラベルは英語。日本語フォントが無い環境で豆腐になるのを避けるため。

## 座標と角度の約束事

- 画像座標の **Y軸は下向き**。上向きを正にする反転は `metrics/geometry.py` だけで行う。
- 肘角度・膝角度は「なす角」（0〜180度）。完全伸展が約180度。左右反転しても変わらない。
- 前腕角度・上腕角度は「打者方向の水平線からの角度」。+90 が真上、0 が打者方向の水平、負が下向き。
- ピクセル量は身体サイズ（既定は両肩間距離の中央値）で正規化する。**実世界の長さではない。**
- 体幹の角度はすべて2D画像上の見かけの値。**3Dの回旋角ではない**（出力にも明記される）。

## 前処理

1. 信頼度が閾値未満のキーポイントを欠損（NaN）にする。信頼度自体は残す。
2. `max_gap_frames` 以下の欠損だけを線形補間する。長い欠損と端の欠損は埋めない。
3. Savitzky–Golay で平滑化する。対称フィルタなので位相遅れがなく、イベントのタイミングをずらさない。
4. 生データと平滑化後データの両方を保持する。
5. 低信頼度区間・補間した区間は `metrics.json` の `quality` に残る。

## イベントの扱い

| イベント | 決め方 |
|---|---|
| 投球区間開始 / 終了 | 設定の `start_frame` / `end_frame` |
| 踏み出し足接地 | **手動指定のみ**（自動検出は誤りやすい） |
| 最大肘屈曲 | 平滑化後の肘角度の局所最小。探索区間は足接地の少し前〜リリースに限定 |
| 肘伸展開始 | 最大屈曲後、角度の増加が一定フレーム以上続いた最初の地点 |
| リリース | **手動指定 > キャップ追跡**。Pose の動きだけからは決めない |

確度は `manual` / `cap_tracking` / `detection_class` / `pose_heuristic` で区別し、
手動指定は常に自動推定を上書きする。

キャップ追跡が使えるようになったら、`events/release_source.py` の
`CapSeparationRelease`（手首とキャップが分離した最初のフレーム）を
`analyze_pitch(..., release_source=...)` に渡す。

## 比較

同じフレーム番号どうしは比べない。

- **イベント基準**: 足接地 / 最大肘屈曲 / リリースのいずれかで揃える（`align_by_event`）
- **時間正規化**: 足接地=0%、リリース=100% に引き伸ばして比べる（`time_normalize`）

元のフレーム番号と正規化後の進行率は両方保持する。進行率は足接地〜リリースの区間内だけに出す
（区間外に外挿した -33% のような値は、進行率として読めないので出さない）。
`comparison.json` には差分だけでなく、各投球の実測値も入る。

## テスト

```bash
cd ultralytics/pitching
python -m pytest tests -q                     # 実動画・GPU 不要
python -m pytest tests -q --cov=pitching      # カバレッジ
```

合成した座標系列だけで検証する（`tests/support.py`）。
YOLO 推論そのものは実素材が要るので対象外（`shared/tests/test_pose.py` と同じ方針）。

## 既知の制約

- **実素材での検証は未実施。** `input/` が空のため、合成データでしか通していない。
- `extract` は解析区間を1フレームずつ推論する。`track.py` の prefilter（推論の間引き）は使わない。
  間引くと「推論していないフレーム」と「信頼度が低いフレーム」が区別できず、
  短い欠損だけを補間する前処理が誤動作するため。区間が長いと時間がかかる。
- `extraction.detection_model_path` を指定しないと、役割bboxで投手を特定できない。
  その場合は「最も大きい骨格」を投手とみなし、`pose.json` の `notes` に警告を残す。
  捕手や打者を拾っていないか確認すること。
- 2D座標から肩の内外旋や奥行きは推定しない。
- 1球対1球の比較は「観測された差」であり、統計的な結論でも因果関係でもない。
- `pitching_analysis.py` の `pitcher_motion` → `pitcher_release` クラス遷移による
  リリース検出（確度 `detection_class`）は、まだ配線していない。
- 体幹の傾き・前腕角度の符号は `batter_direction` に依存する。設定を間違えると符号が反転するので、
  最初の1球で `charts/trunk_angle.png` の向きが直感と合うか確認すること。
