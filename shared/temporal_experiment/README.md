# 時系列追跡・補完の検証

YOLO の検出結果に時系列の追跡と補完を組み合わせて、再学習なしで検出抜けをどこまで補えるかを検証する実験モジュールです。
既存の `track.py` や学習処理は変更していません。

結果と考察は `../docs/temporal-tracking-experiment.md` にまとめます。

## 比較する手法

| 名前 | 内容 | 未来フレームの参照 |
|---|---|---|
| `yolo` | 各フレームで信頼度が最も高い候補を採る | なし |
| `nearest` | 手法A: 最近傍追跡。欠損は補完しない | 動いた点の直前2点を遡る。軌道の選択もオフライン |
| `linear` | `nearest` の軌道を前後の検出で線形補間する | あり（オフライン） |
| `kalman` | 手法B: 等速モデルのカルマンフィルタ。欠損は予測する | 動いた点の直前2点を遡る。軌道の選択もオフライン |
| `ransac` | `nearest` の軌道に既存の `trajectory_fitter` で2次式を当てる | あり（オフライン） |

差分強調（`tennis.py`）は次のフレームも使うので、どの手法も入力の時点で1フレーム先を見ています。

各フレームの出力は次の4つに分かれ、同じ「検出成功」としては扱いません。

| 状態 | 意味 |
|---|---|
| `detected` | YOLO が実際に検出した位置 |
| `interpolated` | 前後の観測値から補間した位置 |
| `predicted` | 状態推定で予測した位置 |
| `missing` | 位置を確定できなかったフレーム |

## 実行手順

`shared/` で実行します。Python は `../.venv/bin/python` を使います。

```bash
# 1. 全フレームを1回だけ推論する（runs/<動画名>/detections.json に候補をすべて保存）
python -m temporal_experiment infer bXETW0GxBbkZoyHO.mp4

# 2. 全手法を適用する（再推論なし）
python -m temporal_experiment track bXETW0GxBbkZoyHO

# 3. 調整用の区間だけを見て experiment.yaml のしきい値を決める
python -m temporal_experiment evaluate bXETW0GxBbkZoyHO \
  --truth-yolo <人が確認済みのラベル> --clips clips.yaml --split tuning

# 4. 評価用の区間で指標とグラフを出す（runs/<動画名>/eval/）
python -m temporal_experiment evaluate bXETW0GxBbkZoyHO \
  --truth-yolo <人が確認済みのラベル> --clips clips.yaml --split eval

# 5. 比較動画を作る
python -m temporal_experiment video bXETW0GxBbkZoyHO --start 550 --end 580
```

テストは次のコマンドで実行します。

```bash
python -m pytest tests/test_temporal_tracking.py tests/test_temporal_metrics.py \
  tests/test_temporal_io.py tests/test_temporal_outputs.py -q
```

## 正解データ

次のどちらかを渡します。

- `--truth`: `TrackNet/src/annotate.py` の CSV（`frame_id,x,y,visible`）
- `--truth-yolo`: 人が確認した YOLO ラベルのディレクトリ（ファイル名は `<動画名>_<フレーム番号>.txt`）

`datasets/prelabel_trial/labels` は使えません。`prelabel.py` が出力した下書きで、中身は YOLO の予測と補間です。これを正解にすると、YOLO と補完の結果を自分自身と比べることになります。

`--truth-yolo` では、蓋の行が無いラベルを「蓋が写っていないと確認したフレーム」として扱います。これは区間内の全フレームを人が蓋について確認した場合にだけ成り立つ前提です。

`prelabel.py` は何も検出しなかったフレームに `.txt` を作りません。そのまま使うと、検出抜けのフレームが正解から消えて Recall が高く出ます。そのため `evaluate` は、区間内にラベルファイルの無いフレームがあると止まります。蓋が写っていないと確認したフレームには、空の `.txt` を置いてください。

評価区間は `clips.yaml` に書きます。`split` は `tuning`（しきい値の調整用）、`eval`（評価用）、`reference`（学習データに含まれる動画など、参考値）のいずれかです。調整用と評価用の区間が重なるとエラーになります。

```yaml
clips:
  - {id: pitch01, start: 550, end: 580, split: tuning}
  - {id: pitch02, start: 700, end: 730, split: eval}
```

区間は YOLO の検出結果を見ずに選びます。検出できた投球だけを選ぶと Recall とカバー率が高く出ます。

区間は投球の終わりより長く取ります。カルマンフィルタの末尾の予測は、蓋が写っていないと確認したフレームが区間に無いと架空出力に数えられません。

## 評価指標

| 指標 | 定義 |
|---|---|
| Recall | 正解から `match_radius_px` 以内に候補が1つでもあったフレームの割合 |
| カバー率 | 蓋が見えているフレームのうち、位置を出力した割合 |
| 正しいカバー率 | 蓋が見えているフレームのうち、正解から `match_radius_px` 以内に位置を出力した割合 |
| MAE / RMSE / P90 / P95（出力のみ） | 位置を出力したフレームだけの中心位置誤差（ユークリッド距離） |
| MAE / RMSE / P95（全フレーム） | 欠損を `error_cap_px` とし、誤差も同じ値で頭打ちにして全フレームで計算した値 |
| 最大連続欠損 | 正解が連続するフレームの中で、位置を出力しなかった最長の長さ |
| 誤対応 | 正解から `match_radius_px` を超えて離れた出力の件数 |
| 架空出力 | 蓋が写っていないと確認したフレームに位置を出力した件数 |
| 末尾の予測 | 正解のあるフレームのうち、軌道の最後の検出より後に出した予測の件数 |

距離の設定は幅 1280px の動画を基準に書き、動画の幅に合わせて換算します。

## 設定

実験条件はすべて `experiment.yaml` にあります。未知の項目はエラーになります。
`max_missed_frames` は線形補間とカルマン予測で共通の、補完する連続欠損の上限です。
