"""YOLOv8 学習スクリプト（小物体=キャップ検出向け設定）

キャップは1080p上で10px前後の小物体のため、以下がデフォルト:
- imgsz=1280: 従来の640では縮小時にキャップが3px程度になり検出限界を下回る
- yolov8m-p2.yaml: P2ヘッド付き構成。stride=4の高解像度特徴マップで
  小物体を検出する（通常のyolov8mはstride=8が最小）

使用例:
    # 推奨設定（P2ヘッド + 1280px）
    python train.py --data data.yaml

    # VRAMが足りない場合はバッチを自動調整
    python train.py --data data.yaml --batch -1

    # 従来と同じ構成で学習（比較実験用）
    python train.py --data data.yaml --model yolov8m.yaml --imgsz 640
"""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="キャップ検出モデルの学習")
    parser.add_argument("--data", default="data.yaml", help="データセット定義")
    parser.add_argument(
        "--model", default="yolov8m-p2.yaml",
        help="モデル構成（yolov8m-p2.yaml=小物体向けP2ヘッド / yolov8m.yaml=従来）",
    )
    parser.add_argument(
        "--weights", default="yolov8m.pt",
        help="転移元の学習済み重み（空文字でスクラッチ学習）",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280,
        help="学習解像度。キャップの見かけサイズを保つため1280以上を推奨",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch", type=int, default=4,
        help="バッチサイズ。-1でVRAMに合わせて自動決定",
    )
    parser.add_argument("--name", default=None, help="実験名（runs/detect/配下）")
    parser.add_argument(
        "--project", default=None,
        help="run出力先の親ディレクトリ。Colab等の揮発環境ではDrive上のパスを"
             "指定すると切断時も成果物が残り、resume=True での再開も可能になる",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)
    if args.weights:
        model = model.load(args.weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.project,
        # --- 小物体向けの拡張設定 ---
        # モザイクは小物体をさらに縮小して有害なことがあるため縮小率を抑える
        scale=0.3,
        # 学習終盤はモザイクを切って実解像度の分布に寄せる
        close_mosaic=15,
        # 左右反転は投球方向が反転するだけで有効。上下反転は無効のまま
        fliplr=0.5,
        flipud=0.0,
        # 回転・シアーはブラー方向とbboxの整合が崩れるため無効化
        degrees=0.0,
        shear=0.0,
    )


if __name__ == "__main__":
    main()
