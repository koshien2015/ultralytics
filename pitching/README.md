# pitching — 投球フォーム解析

YOLO Pose のキーポイント時系列から関節角度・位置・タイミングを数値化し、
同一投手の2球のフォームを比較する。

**フォームの良否は判定しない。** 出すのは観測値と、比較のための差分だけ。
投球に付ける `label` も良否の分類ではなく、"1球目" のような区別のための覚え書き。

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
  visualization/         グラフ・解析動画・比較画像・棒人間ビューア
  pipeline.py            run（抽出→解析→比較→ビューア）の段取り
  cli.py                 CLI
  tools/extract-pose.py  推論環境用の抽出台本（track.py を使わない場合）。単体で完結する
  tools/make-viewer.py   ビューア生成スクリプト（単体で実行できる）
  examples/              設定例と、動画なしで試すための合成データ生成
  tests/                 合成座標による単体テスト（実動画・GPU 不要）
```

`shared/pose.py`（`PoseEstimator` / `RoleTracker` / `KEYPOINT_NAMES`）はそのまま流用する。
`shared/*.py` はフラット import なので、`adapters/ultralytics_adapter.py` が
`sys.path` を足して遅延 import する。**`shared/` 側は変更していない。**

## 実行方法

### いちばん短い手順

`run` は 抽出 → 解析 → 比較 → ビューア をまとめて実行する。

```bash
cd ultralytics   # パッケージルート

python -m pitching run --output output/compare \
  --video input/a.mov --release 152 --contact 140 --start 100 --end 180 \
  --video input/b.mov --release 160 --contact 148 --start 110 --end 190 \
  --hand right --batter left \
  --detection-model shared/yolo8m_20250510.pt
```

イベントの指定（`--release` など）は**直前の `--video` に付く**。
`--id` を省くと動画のファイル名が投球名になる。1球だけ渡せば比較なしで解析とビューアを作る。

出力は `output/compare/<投球名>/`（metrics.json・CSV・グラフ）、
`output/compare/comparison/`、`output/compare/viewer.html`。

推論をやり直すと時間がかかるので、`<投球名>/pose.json` があれば既定で使い回す。
取り直すときは `--force-extract`。

主なオプション: `--config <yaml>`（直前の `--video` の設定を YAML で渡す。
コマンドラインの指定が優先）/ `--label`（覚え書き）/ `--no-charts` / `--no-viewer` /
`--overlay-video` / `--fps` / `--pose-model`。

### 推論だけ別環境（Docker）で回す

YOLO Pose は GPU のあるコンテナ、解析とビューアは手元、という分け方ができる。
やり取りするのは **pose.json 1ファイルだけ**。

**すでに `track.py` を回しているなら**、その先頭のフラグを1つ立てるだけでよい。
キャップ検出と同じ1コマンドのまま、`{動画名}_pose.json` が増える。

```python
# shared/track.py
ENABLE_POSE = True     # ← これだけ。POSE_EXPORT は既定で True
```

書き出し先は起動時にも表示される（`Pose keypoints will be saved to: ...`）。
骨格を描くだけでファイルが要らなければ `POSE_EXPORT = False`。

```bash
# 推論環境（いつもどおり）
cd shared && python track.py input/a.mov

# 手元（GPU 不要）
python -m pitching run --output output/compare \
  --pose shared/input/a_pose.json --release 152 --contact 140 \
  --pose shared/input/b_pose.json --release 160 --contact 148
```

書き出されるのは**姿勢推定の窓の中だけ**（窓の外は推論していないので含まれない）。
どの区間が入ったかは実行時のログと pose.json の `meta.notes` に出るので、
1投球ぶんを `--start` / `--end` で切り出す。

投球部分だけを切り出した短いクリップでは、前段フィルタが窓を立てられないことがある
（活動量の中央値を基準にするので、全編が動いている映像では閾値を超えない）。
その場合は自動で全フレームを対象にする。ログに「全フレームを対象にします」と出る。

**`track.py` を使わずに抽出だけしたいとき**は `tools/extract-pose.py` を使う。
この1ファイルと `shared/pose.py` だけで動き、pydantic も matplotlib も import しない。
`docker/docker-compose.yml` は `../pitching` を読み取り専用で入れてある。

```bash
docker compose -f docker/docker-compose.yml exec yolov8 \
  python /pitching/tools/extract-pose.py /shared/input/a.mov \
    -o /shared/out/a_pose.json \
    --start 100 --end 180 \
    --detection-model /shared/yolo8m_20250510.pt
```

`shared/pose.py` の場所が違うときは `--shared-dir` か環境変数 `PITCHING_SHARED_DIR` で指定する。

**解析とビューアはコンテナ側では動かないことがある。** `pitching` パッケージ本体は
pydantic を使うので、素の推論コンテナには入っていない（`pip install pydantic` で足りる）。
入れない場合は、pose.json を手元に持ち帰ってから `run` / `viewer` を実行する。

`--pose` を使うと推論は一切走らないので、イベントのフレーム番号を直しながら
何度でも解析し直せる。

### 段階ごとに実行する

途中結果を差し替えたいときは個別に呼ぶ。`run` と結果は同じ。

```bash
# 1. 動画から YOLO Pose のキーポイントを抽出（GPU 推奨、ultralytics が必要）
python -m pitching extract \
  --video input/a.mov \
  --config pitching/examples/pitch_a.yaml \
  --output output/a

# 2. 解析（動画も YOLO も不要。pose.json だけで動く）
#    設定YAMLの代わりに --release / --contact などを直接指定してもよい
python -m pitching analyze \
  --pose-data output/a/pose.json \
  --config pitching/examples/pitch_a.yaml \
  --output output/a

# 3. 2球を比較
python -m pitching compare \
  --pitch-a output/a/metrics.json \
  --pitch-b output/b/metrics.json \
  --output output/comparison

# 4. 棒人間ビューア（HTML 1枚。サーバ不要、ブラウザで開くだけ）
python -m pitching viewer output/a output/b --output output/viewer.html
```

`analyze --overlay-video` で解析動画、`compare --frames` で同一イベントの比較画像も出る。

実動画が無い場合は、合成データで一通り試せる。2投球分の `pose.json` と `config.yaml` が
作られ、続けて打つ `run` のコマンドも表示される。

```bash
python pitching/examples/generate_sample_pose.py output/sample
```

## 棒人間ビューア

```bash
# analyze の出力ディレクトリから
python -m pitching viewer output/good output/bad --output output/viewer.html

# track.py が書いた {動画名}_pose.json から直接（イベントはここで指定する）
python -m pitching viewer shared/a_pose.json --release 152 --contact 140 \
  --hand right --batter right --output output/viewer.html

# 同じものが単体スクリプトからも作れる（依存が入った python で実行すること）
pitching/.venv/bin/python pitching/tools/make-viewer.py output/good output/bad -o output/viewer.html
```

渡せるのは「analyze の出力ディレクトリ」「metrics.json」「キーポイントJSON」のいずれでもよい。
`--release` などを2投球に指定するときは、投球を並べた順に対応する（1つだけなら両方に効く）。

キーポイントを線で結んだ棒人間を描く HTML を1枚だけ作る。データは HTML に埋め込むので、
`file://` で開けばよくサーバは要らない。元動画も再エンコードもしない。

**2投球の比較**

- 表示: 並べて / 重ねて
- そろえ方: 進行率0〜100%（足接地〜リリースを引き伸ばす）/ リリース基準 / フレーム番号
- 再生・コマ送り（←→キー、スペースで再生）、投球腕の軌跡
- 右の表に、その時点の肘角度・前腕角度・体幹傾き・前脚膝角度と、2球の差

**力の向きと大きさ**

連続フレームの差分をベクトルとして関節に重ねる。対象は 全関節 / 投球腕 / 単一関節。

| 表示 | 計算 | 意味 |
|---|---|---|
| 速度 | 連続2フレームの変位 | どちらへどれだけ動いたか |
| 加速度 | 3フレームの2階差分 | **向きが力の向きに相当**（F = ma） |

> 矢印は力そのものではない。質量が不明なので大きさはニュートンではなく「身体長/秒²」の相対値。
> また2D画像から得た見かけの動きなので、奥行き方向の成分は含まれない。
> 突出した値で画面が埋まらないよう、矢印の長さには上限がある（頭打ちの矢印は先端を白く縁取る）。

座標は身体サイズを1とした相対値で、原点は足接地時の股関節中点、
Xは打者方向が正、Yは上が正。撮影距離も左右の向きも違う2投球を同じ土俵に載せるため。

## 入力

### 設定 YAML（`examples/pitch_a.yaml`）

`run` にコマンドラインで渡すなら不要。細かく詰めるときだけ使う。

```yaml
pitch_id: pitch_a
video_path: input/a.mov
throwing_hand: right
batter_direction: left    # 画像上で打者がどちら側にいるか
fps: 60
start_frame: 100
end_frame: 180

events:
  stride_foot_contact_frame: null   # 手動指定を優先する
  release_frame: 152

result:
  label: "1球目"          # 良否の分類ではなく、区別のための覚え書き
  description: "縦回転58km/h"
```

`preprocessing` / `metrics` / `event_detection` / `extraction` は省略可能
（閾値・補間可能な最大フレーム数・平滑化ウィンドウはすべて設定できる）。

> `batter_direction` は project.md の例には無いが追加した。斜め後方からの2D映像では、
> 投げ手だけでは「打者方向に対して上向き何度」の符号が決まらないため。既定は `left`。

### キーポイント JSON

`extract` と `tools/extract-pose.py` が出す形式。外部で作ったものを渡してもよい
（`adapters/json_adapter.py`）。

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

`viewer` が作るもの: `viewer.html` 1枚（データ埋め込み済み）。

グラフの軸名・表題は英語（日本語フォントが無い環境で豆腐になるのを避けるため）。
投球名と覚え書きは日本語のまま出す。日本語フォント（Hiragino Sans / Noto Sans CJK JP など）が
見つからない環境では、その部分だけ表示できない。

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
